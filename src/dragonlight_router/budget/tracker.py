"""Budget tracker — per-provider rate limit tracking with sliding windows.

Tracks RPM (requests per minute) via a sliding window of timestamps,
and RPD (requests per day) via a simple counter with daily reset.
"""
from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict, deque

import structlog

from dragonlight_router.core.errors import ProviderNotFoundError
from dragonlight_router.core.types import Err, Ok, ProviderConfig, Result

logger = structlog.get_logger()


def invariant(condition: bool, message: str) -> None:
    """Inline invariant check — enforced even under python -O."""
    if not condition:
        raise AssertionError(message)


class BudgetTracker:
    """Tracks rate limit budget for all configured providers."""

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self._providers: dict[str, ProviderConfig] = {p.name: p for p in providers}
        self._rpm_windows: dict[str, deque[float]] = defaultdict(deque)
        self._rpd_counts: dict[str, int] = defaultdict(int)
        self._tpm_windows: dict[str, deque[float]] = defaultdict(deque)
        self._tokens_today: dict[str, int] = defaultdict(int)
        self._day_reset_at: float = self._next_day_boundary()

    def score(self, provider_name: str) -> Result[float, ProviderNotFoundError]:
        """Budget availability score (0-100) for a provider."""
        provider = self._providers.get(provider_name)
        if provider is None:
            logger.debug("provider_not_found", provider=provider_name)
            return Err(ProviderNotFoundError(provider=provider_name))

        rpm_remaining = self._rpm_remaining(provider_name)
        rpm_limit = provider.rpm_limit

        rpd_remaining: int | None = None
        rpd_limit: int | None = provider.rpd_limit

        if rpd_limit is not None:
            self._maybe_reset_daily()
            rpd_remaining = max(0, rpd_limit - self._rpd_counts[provider_name])

        # TPM tracking
        tpm_remaining = self._tpm_remaining(provider_name)
        tpm_limit: int | None = provider.tpm_limit

        # Daily token cap tracking
        tokens_today = self._tokens_today[provider_name]
        token_cap: int | None = provider.daily_token_cap

        rpm_ratio = rpm_remaining / rpm_limit if rpm_limit > 0 else 1.0

        if rpd_remaining is None or rpd_limit is None or rpd_limit == 0:
            rpd_ratio = 1.0
        else:
            rpd_ratio = rpd_remaining / rpd_limit

        if tpm_remaining is None or tpm_limit is None or tpm_limit == 0:
            tpm_ratio = 1.0
        else:
            tpm_ratio = tpm_remaining / tpm_limit

        # Daily token cap ratio
        if tokens_today is None or token_cap is None or token_cap == 0:
            token_ratio = 1.0
        else:
            token_ratio = max(0, token_cap - tokens_today) / token_cap

        # Assertions for coding standard (>=2 assertions)
        assert 0.0 <= rpm_ratio <= 1.0, f"rpm_ratio out of bounds: {rpm_ratio}"
        assert 0.0 <= rpd_ratio <= 1.0, f"rpd_ratio out of bounds: {rpd_ratio}"
        assert 0.0 <= tpm_ratio <= 1.0, f"tpm_ratio out of bounds: {tpm_ratio}"
        assert 0.0 <= token_ratio <= 1.0, f"token_ratio out of bounds: {token_ratio}"

        return Ok(min(rpm_ratio, rpd_ratio, tpm_ratio, token_ratio) * 100.0)

    def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
        """Record that a request was dispatched."""
        now = time.time()
        self._rpm_windows[provider_name].append(now)
        self._rpd_counts[provider_name] += 1
        # TPM tracking
        self._tpm_windows[provider_name].append(now)
        # Daily token cap tracking
        self._tokens_today[provider_name] += tokens_used

    def has_capacity(self, provider_name: str) -> bool:
        """Quick check: does this provider have RPM, RPD, TPM, and token cap headroom?"""
        provider = self._providers.get(provider_name)
        if provider is None:
            return True
        if not self._rpm_remaining(provider_name):
            return False
        if not self._rpd_remaining(provider_name):
            return False
        if not self._tpm_remaining(provider_name):
            return False
        return self._tokens_today[provider_name] < provider.daily_token_cap if provider.daily_token_cap and provider.daily_token_cap > 0 else True

    def _rpm_remaining(self, provider_name: str) -> int:
        """Remaining RPM in the current minute window."""
        provider = self._providers.get(provider_name)
        invariant(
            provider is not None,
            f"_rpm_remaining called for unknown provider: {provider_name}",
        )
        assert provider is not None, f"_rpm_remaining called for unknown provider: {provider_name}"
        limit = provider.rpm_limit
        if limit <= 0:
            return 1
        now = time.time()
        cutoff = now - 60.0
        window = self._rpm_windows.get(provider_name, deque())
        while window and window[0] < cutoff:
            window.popleft()
        return max(0, limit - len(window))

    def _rpd_remaining(self, provider_name: str) -> int:
        """Remaining RPD in the current day."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.rpd_limit is None:
            return 0
        self._maybe_reset_daily()
        return max(0, provider.rpd_limit - self._rpd_counts[provider_name])

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if a day boundary has passed."""
        now = time.time()
        if now >= self._day_reset_at:
            self._rpd_counts.clear()
            self._tokens_today.clear()
            self._day_reset_at = self._next_day_boundary()

    @staticmethod
    def _next_day_boundary() -> float:
        """Compute the next UTC midnight timestamp."""
        tomorrow = dt.datetime.now(dt.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) + dt.timedelta(days=1)
        return tomorrow.timestamp()

    def _tpm_remaining(self, provider_name: str) -> int:
        """Remaining TPM in the current minute window."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.tpm_limit is None:
            return 0
        limit = provider.tpm_limit
        if limit <= 0:
            return 1
        now = time.time()
        cutoff = now - 60.0
        window = self._tpm_windows.get(provider_name, deque())
        # Remove old entries outside the 60-second window
        while window and window[0][0] < cutoff:
            window.popleft()
        # Sum the tokens in the window
        tokens_in_window = sum(tokens for _, tokens in window)
        return max(0, limit - tokens_in_window)
