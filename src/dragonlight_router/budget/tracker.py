"""Budget tracker -- per-provider rate limit tracking with sliding windows.

Tracks RPM (requests per minute) via a sliding window of timestamps,
and RPD (requests per day) via a simple counter with daily reset.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections import defaultdict, deque
from typing import Any

import structlog

from dragonlight_router.core.errors import ProviderNotFoundError
from dragonlight_router.core.types import Ok, ProviderConfig, Result

logger = structlog.get_logger()


def invariant(condition: bool, message: str) -> None:
    """Inline invariant check -- enforced even under python -O."""
    if not condition:
        raise AssertionError(message)


class BudgetTracker:
    """Tracks rate limit budget for all configured providers."""

    def __init__(self, providers: list[ProviderConfig]) -> None:
        assert isinstance(providers, list), "providers must be a list"
        assert all(isinstance(p, ProviderConfig) for p in providers), (
            "all providers must be ProviderConfig instances"
        )
        self._providers: dict[str, ProviderConfig] = {p.name: p for p in providers}
        self._rpm_windows: dict[str, deque[float]] = defaultdict(deque)
        self._rpd_counts: dict[str, int] = defaultdict(int)
        self._tpm_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._daily_token_counts: dict[str, int] = defaultdict(int)
        self._day_reset_at: float = self._next_day_boundary()
        # HAZ-002: asyncio.Lock for atomic check-then-record under concurrency
        self._lock = asyncio.Lock()

    def score(self, provider_name: str) -> Result[float, ProviderNotFoundError]:
        """Budget availability score (0-100) for a provider.

        Considers RPM, RPD, TPM, and daily token cap limits.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        provider = self._providers.get(provider_name)
        if provider is None:
            logger.debug("provider_not_found", provider=provider_name)
            return Ok(100.0)

        ratios = self._compute_budget_ratios(provider_name, provider)
        score_value = min(ratios) * 100.0

        assert 0.0 <= score_value <= 100.0, f"score must be in [0, 100], got {score_value}"
        return Ok(score_value)

    def _compute_budget_ratios(self, provider_name: str, provider: ProviderConfig) -> list[float]:
        """Compute budget utilization ratios for all limit dimensions."""
        rpm_ratio = self._rpm_ratio(provider_name, provider)
        rpd_ratio = self._rpd_ratio(provider_name, provider)
        tpm_ratio = self._tpm_ratio(provider_name, provider)
        daily_token_ratio = self._daily_token_ratio(provider_name, provider)

        ratios = [rpm_ratio, rpd_ratio, tpm_ratio, daily_token_ratio]
        assert all(0.0 <= r <= 1.0 for r in ratios), f"all ratios must be in [0, 1], got {ratios}"
        return ratios

    def _rpm_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute RPM remaining ratio."""
        rpm_remaining = self._rpm_remaining(provider_name)
        return rpm_remaining / provider.rpm_limit if provider.rpm_limit > 0 else 1.0

    def _rpd_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute RPD remaining ratio."""
        if provider.rpd_limit is None or provider.rpd_limit == 0:
            return 1.0
        self._maybe_reset_daily()
        rpd_remaining = max(0, provider.rpd_limit - self._rpd_counts[provider_name])
        return rpd_remaining / provider.rpd_limit

    def _tpm_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute TPM remaining ratio."""
        if provider.tpm_limit is None or provider.tpm_limit <= 0:
            return 1.0
        tpm_remaining = self._tpm_remaining(provider_name)
        return tpm_remaining / provider.tpm_limit

    def _daily_token_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute daily token cap remaining ratio."""
        if provider.daily_token_cap is None or provider.daily_token_cap == 0:
            return 1.0
        self._maybe_reset_daily()
        remaining = max(0, provider.daily_token_cap - self._daily_token_counts[provider_name])
        return remaining / provider.daily_token_cap

    def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
        """Record that a request was dispatched."""
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(tokens_used, int) and tokens_used >= 0, (
            f"tokens_used must be a non-negative integer, got {tokens_used}"
        )
        now = time.time()
        self._rpm_windows[provider_name].append(now)
        self._rpd_counts[provider_name] += 1
        self._tpm_windows[provider_name].append((now, tokens_used))
        self._daily_token_counts[provider_name] += tokens_used

    async def check_and_reserve(self, provider_name: str, estimated_tokens: int = 0) -> bool:
        """Atomically check capacity and reserve budget under the async lock.

        HAZ-002 mitigation: Prevents concurrent requests from passing budget
        checks simultaneously before either records its spend. Returns True
        if capacity was available and the reservation was recorded, False if
        the provider has no remaining capacity.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(estimated_tokens, int) and estimated_tokens >= 0, (
            f"estimated_tokens must be a non-negative integer, got {estimated_tokens}"
        )
        async with self._lock:
            if not self.has_capacity(provider_name):
                return False
            self.record_request(provider_name, estimated_tokens)
            return True

    def has_capacity(self, provider_name: str) -> bool:
        """Quick check: does this provider have RPM, RPD, TPM, and daily token headroom?"""
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert provider_name, "provider_name must be non-empty"
        provider = self._providers.get(provider_name)
        if provider is None:
            return True
        if not self._rpm_remaining(provider_name):
            return False
        if provider.rpd_limit is not None and self._rpd_remaining(provider_name) <= 0:
            return False
        if (
            provider.tpm_limit is not None
            and provider.tpm_limit > 0
            and self._tpm_remaining(provider_name) <= 0
        ):
            return False
        return not (
            provider.daily_token_cap is not None
            and provider.daily_token_cap > 0
            and self._daily_token_remaining(provider_name) <= 0
        )

    def daily_spend_usd(self, provider_name: str, avg_cost_per_token: float = 0.0) -> float:
        """Estimated daily spend for a provider in USD.

        Calculates from tokens_today * avg_cost_per_token.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(avg_cost_per_token, (int, float)) and avg_cost_per_token >= 0, (
            f"avg_cost_per_token must be a non-negative number, got {avg_cost_per_token}"
        )
        self._maybe_reset_daily()
        tokens_today = self._daily_token_counts.get(provider_name, 0)
        spend = tokens_today * avg_cost_per_token
        assert spend >= 0.0, f"daily spend must be non-negative, got {spend}"
        return spend

    def monthly_spend_usd(self, provider_name: str, avg_cost_per_token: float = 0.0) -> float:
        """Estimated monthly spend for a provider in USD.

        Approximates monthly spend as daily_spend * 30.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(avg_cost_per_token, (int, float)) and avg_cost_per_token >= 0, (
            f"avg_cost_per_token must be a non-negative number, got {avg_cost_per_token}"
        )
        daily = self.daily_spend_usd(provider_name, avg_cost_per_token)
        monthly = daily * 30.0
        assert monthly >= 0.0, f"monthly spend must be non-negative, got {monthly}"
        return monthly

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
        remaining = max(0, limit - len(window))
        assert remaining >= 0, f"Remaining RPM must be non-negative, got {remaining}"
        return remaining

    def _rpd_remaining(self, provider_name: str) -> int:
        """Remaining RPD in the current day."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.rpd_limit is None:
            return 0
        self._maybe_reset_daily()
        remaining = max(0, provider.rpd_limit - self._rpd_counts[provider_name])
        assert remaining >= 0, f"Remaining RPD must be non-negative, got {remaining}"
        return remaining

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if a day boundary has passed."""
        now = time.time()
        if now >= self._day_reset_at:
            self._rpd_counts.clear()
            self._daily_token_counts.clear()
            self._day_reset_at = self._next_day_boundary()
        assert self._day_reset_at > now, (
            f"Day reset time must be in the future, got {self._day_reset_at}"
        )

    @staticmethod
    def _next_day_boundary() -> float:
        """Compute the next UTC midnight timestamp."""
        tomorrow = dt.datetime.now(dt.UTC).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) + dt.timedelta(days=1)
        result = tomorrow.timestamp()
        assert result > time.time(), f"Next day boundary must be in the future, got {result}"
        return result

    def _tpm_remaining(self, provider_name: str) -> int:
        """Remaining TPM in the current minute window."""
        provider = self._providers.get(provider_name)
        invariant(
            provider is not None,
            f"_tpm_remaining called for unknown provider: {provider_name}",
        )
        assert provider is not None, f"_tpm_remaining called for unknown provider: {provider_name}"
        limit = provider.tpm_limit
        if limit is None or limit <= 0:
            return 1
        now = time.time()
        cutoff = now - 60.0
        window = self._tpm_windows.get(provider_name, deque())
        # Remove outdated entries
        while window and window[0][0] < cutoff:
            window.popleft()
        # Sum tokens used in the window
        tokens_used = sum(tokens for _, tokens in window)
        remaining = max(0, limit - tokens_used)
        assert remaining >= 0, f"Remaining TPM must be non-negative, got {remaining}"
        return remaining

    def _daily_token_remaining(self, provider_name: str) -> int:
        """Remaining daily token cap for the provider."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.daily_token_cap is None:
            return 0
        self._maybe_reset_daily()
        remaining = max(0, provider.daily_token_cap - self._daily_token_counts[provider_name])
        assert remaining >= 0, f"Remaining daily token cap must be non-negative, got {remaining}"
        return remaining

    def get_state(self) -> dict[str, Any]:
        """Export serializable budget state for persistence (HAZ-012 mitigation).

        Returns daily counters and reset timestamp. Sliding windows (RPM/TPM)
        are intentionally excluded -- they represent sub-minute state that
        becomes stale immediately on restore.
        """
        assert isinstance(self._rpd_counts, (dict, defaultdict)), "_rpd_counts must be a dict"
        assert isinstance(self._day_reset_at, float), "_day_reset_at must be a float"
        return {
            "rpd_counts": dict(self._rpd_counts),
            "daily_token_counts": dict(self._daily_token_counts),
            "day_reset_at": self._day_reset_at,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore budget state from persistence (HAZ-012 mitigation).

        Only restores daily counters if the persisted reset boundary has not
        passed (i.e., we are still within the same UTC day). If the boundary
        has passed, counters start fresh.
        """
        assert isinstance(state, dict), "state must be a dict"
        assert "day_reset_at" not in state or isinstance(state["day_reset_at"], (int, float)), (
            "day_reset_at must be numeric if present"
        )
        persisted_reset = state.get("day_reset_at", 0.0)
        now = time.time()

        if now >= persisted_reset:
            # Day boundary passed since save -- start fresh
            logger.info("budget_state_stale_skipping_restore")
            return

        self._day_reset_at = persisted_reset
        rpd = state.get("rpd_counts", {})
        for provider_name, count in rpd.items():
            self._rpd_counts[provider_name] = count
        dtc = state.get("daily_token_counts", {})
        for provider_name, count in dtc.items():
            self._daily_token_counts[provider_name] = count
        logger.info(
            "budget_state_restored",
            providers_restored=len(rpd),
        )
