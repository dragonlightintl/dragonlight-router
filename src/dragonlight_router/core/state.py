"""Mutable runtime state for backend tracking.

Tracks rate limit consumption, error history, and circuit breaker state.
One instance per registered backend.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

from dragonlight_router.core.types import BackendStatus

logger = structlog.get_logger()


def invariant(condition: bool, message: str) -> None:
    """Inline invariant check — always enforced, even under python -O.

    Follows the NASA Power-of-10 rule #5: assert invariants at the point of
    use. Unlike the `assert` keyword, this function is never stripped by the
    Python optimiser (-O flag), so invariants are guaranteed in production.
    Raises AssertionError on violation, identical to a bare assert.
    """
    if not condition:
        raise AssertionError(message)


@dataclass
class BackendState:
    """Mutable runtime state for a single backend.

    Tracks rate limit consumption, error history, and circuit breaker state.
    One instance per registered backend. Not frozen — mutated by the router
    on every dispatch and every response.

    DEVIATION_RECORD:
      rule violated: dragonlight-coding-standards-v2.md#frozen-dataclasses
        (all data objects must be frozen dataclass)
      justification: Runtime state must be mutable to track changing
        backend health, error counts, and circuit breaker state
      approved by: Korrigon @ Dragonlight International
      mitigations: All mutations are encapsulated in methods with
        invariants, and the state is not exposed outside the health
        tracker
      scope: This class
      expiration: 2026-06-30 (to be revisited)
    """

    status: BackendStatus = BackendStatus.AVAILABLE

    request_timestamps: deque[float] = field(default_factory=deque)
    requests_today: int = 0
    tokens_today: int = 0
    day_reset_at: float = 0.0

    consecutive_errors: int = 0
    last_error_time: float = 0.0
    circuit_open_until: float = 0.0
    error_threshold: int = 3
    error_window: float = 120.0
    circuit_cooldown: float = 60.0

    avg_latency_ms: float = 0.0
    latency_alpha: float = 0.1

    def has_rpm_capacity(self, limit: int) -> bool:
        """Check if requests-per-minute capacity is available."""
        invariant(limit > 0, "RPM limit must be positive")
        now = time.time()
        cutoff = now - 60.0
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        return len(self.request_timestamps) < limit

    def has_rpd_capacity(self, limit: int) -> bool:
        """Check if requests-per-day capacity is available."""
        invariant(limit > 0, "RPD limit must be positive")
        self._maybe_reset_daily()
        return self.requests_today < limit

    def has_token_capacity(self, limit: int) -> bool:
        """Check if daily token capacity is available. 0 = unlimited."""
        invariant(limit >= 0, "Token limit must be non-negative")
        if limit == 0:
            return True
        self._maybe_reset_daily()
        return self.tokens_today < limit

    def is_circuit_open(self) -> bool:
        """Return True if this backend's circuit breaker is open."""
        return time.time() < self.circuit_open_until

    def record_request(self) -> None:
        """Record that a request was dispatched."""
        now = time.time()
        self.request_timestamps.append(now)
        self.requests_today += 1

    def record_success(self, tokens_in: int, tokens_out: int, latency_ms: float) -> None:
        """Record a successful response."""
        invariant(tokens_in >= 0, "tokens_in must be non-negative")
        invariant(tokens_out >= 0, "tokens_out must be non-negative")
        self.tokens_today += tokens_in + tokens_out
        self.consecutive_errors = 0
        self.status = BackendStatus.AVAILABLE
        if self.avg_latency_ms == 0.0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                self.latency_alpha * latency_ms + (1.0 - self.latency_alpha) * self.avg_latency_ms
            )

    def record_error(self) -> bool:
        """Record an error. Returns True if circuit just tripped."""
        now = time.time()
        if now - self.last_error_time > self.error_window:
            self.consecutive_errors = 1
        else:
            self.consecutive_errors += 1
        self.last_error_time = now

        if self.consecutive_errors >= self.error_threshold:
            self.circuit_open_until = now + self.circuit_cooldown
            self.status = BackendStatus.CIRCUIT_OPEN
            return True
        self.status = BackendStatus.ERROR
        return False

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if the day has rolled over."""
        now = time.time()
        if now >= self.day_reset_at:
            self.requests_today = 0
            self.tokens_today = 0
            tomorrow = dt.datetime.now(dt.UTC).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            ) + dt.timedelta(days=1)
            self.day_reset_at = tomorrow.timestamp()
