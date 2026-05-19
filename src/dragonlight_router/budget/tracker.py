"""Budget tracker — per-provider rate limit tracking with sliding windows.

Tracks RPM (requests per minute) via a sliding window of timestamps,
and RPD (requests per day) via a simple counter with daily reset.
"""
from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict, deque

import structlog

from dragonlight_router.core.types import ProviderConfig

logger = structlog.get_logger()


class BudgetTracker:
    """Tracks rate limit budget for all configured providers."""

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self._providers: dict[str, ProviderConfig] = {p.name: p for p in providers}
        self._rpm_windows: dict[str, deque[float]] = defaultdict(deque)
        self._rpd_counts: dict[str, int] = defaultdict(int)
        self._day_reset_at: float = self._next_day_boundary()

    def score(self, provider_name: str) -> float:
        """Budget availability score (0-100) for a provider.

        Returns min(rpm_ratio, rpd_ratio) * 100.
        Unknown providers return 100.0 (assume unlimited).
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            return 100.0

        rpm_remaining = self._rpm_remaining(provider_name)
        rpm_limit = provider.rpm_limit

        rpd_remaining: int | None = None
        rpd_limit: int | None = provider.rpd_limit

        if rpd_limit is not None:
            self._maybe_reset_daily()
            rpd_remaining = max(0, rpd_limit - self._rpd_counts[provider_name])

        # Compute ratios
        rpm_ratio = rpm_remaining / rpm_limit if rpm_limit > 0 else 1.0

        if rpd_remaining is None or rpd_limit is None:
            rpd_ratio = 1.0
        elif rpd_limit == 0:
            rpd_ratio = 1.0
        else:
            rpd_ratio = rpd_remaining / rpd_limit

        return min(rpm_ratio, rpd_ratio) * 100.0

    def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
        """Record a request against a provider's budget."""
        now = time.time()
        self._rpm_windows[provider_name].append(now)
        self._rpd_counts[provider_name] = self._rpd_counts.get(provider_name, 0) + 1

    def has_capacity(self, provider_name: str) -> bool:
        """Quick check: does this provider have RPM and RPD headroom?"""
        provider = self._providers.get(provider_name)
        if provider is None:
            return True

        # Check RPM
        if self._rpm_remaining(provider_name) <= 0:
            return False

        # Check RPD
        if provider.rpd_limit is not None:
            self._maybe_reset_daily()
            if self._rpd_counts[provider_name] >= provider.rpd_limit:
                return False

        return True

    def _rpm_remaining(self, provider_name: str) -> int:
        """Count remaining RPM capacity after pruning expired timestamps."""
        provider = self._providers.get(provider_name)
        if provider is None:
            return 999

        now = time.time()
        cutoff = now - 60.0
        window = self._rpm_windows[provider_name]
        while window and window[0] < cutoff:
            window.popleft()

        return max(0, provider.rpm_limit - len(window))

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if the day boundary has passed."""
        now = time.time()
        if now >= self._day_reset_at:
            self._rpd_counts.clear()
            self._day_reset_at = self._next_day_boundary()

    @staticmethod
    def _next_day_boundary() -> float:
        """Compute the next UTC midnight timestamp."""
        tomorrow = dt.datetime.now(dt.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) + dt.timedelta(days=1)
        return tomorrow.timestamp()
