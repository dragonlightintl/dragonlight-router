"""Tests for budget/tracker.py — per-provider budget tracking.

Spec traceability: TM-012 (BudgetTracker)
"""
from __future__ import annotations

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import Ok, ProviderConfig


def _provider(
    name: str = "groq",
    rpm: int = 30,
    rpd: int | None = 14400,
    tpm: int | None = None,
    daily_token_cap: int | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="http://localhost",
        catalog_url=None,
        env_key=None,
        model_prefix=name,
        rpm_limit=rpm,
        rpd_limit=rpd,
        tpm_limit=tpm,
        daily_token_cap=daily_token_cap,
    )


class TestBudgetTrackerInit:
    def test_initializes_with_providers(self):
        """[TM-012 AC-1] BudgetTracker initializes capacity tracking for each provider."""
        bt = BudgetTracker(providers=[_provider("groq"), _provider("nvidia")])
        assert bt.has_capacity("groq")
        assert bt.has_capacity("nvidia")

    def test_empty_providers(self):
        """[TM-012 AC-1] Empty provider list yields default 100 score for unknown providers."""
        bt = BudgetTracker(providers=[])
        result = bt.score("unknown")
        assert isinstance(result, Ok)
        assert result.value == 100.0


class TestBudgetScore:
    def test_full_capacity_is_100(self):
        """[TM-012 AC-2] Full capacity yields a budget score of 100."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(100.0)

    def test_score_decreases_with_requests(self):
        """[TM-012 AC-2] Score decreases as requests consume capacity."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        for _ in range(15):
            bt.record_request("groq")
        result = bt.score("groq")
        assert isinstance(result, Ok)
        score = result.value
        assert score > 0.0
        assert score < 100.0

    def test_unknown_provider_returns_100(self):
        """[TM-012 AC-2] Unknown provider returns default score of 100."""
        bt = BudgetTracker(providers=[_provider("groq")])
        result = bt.score("unknown")
        assert isinstance(result, Ok)
        assert result.value == 100.0

    def test_unlimited_rpd_none(self):
        """[TM-012 AC-2] None rpd_limit means unlimited — only RPM matters."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=10, rpd=None)])
        for _ in range(5):
            bt.record_request("groq")
        result = bt.score("groq")
        assert isinstance(result, Ok)
        score = result.value
        assert score == pytest.approx(50.0)


class TestRecordRequest:
    def test_records_count(self):
        """[TM-012 AC-3] Recording requests decrements remaining capacity."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        bt.record_request("groq")
        bt.record_request("groq")
        # RPM: 98/100, RPD: 998/1000 → min(0.98, 0.998) * 100 = 98.0
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(98.0)

    def test_records_tokens(self):
        """[TM-012 AC-3] Recording requests with tokens decrements TPM and daily token capacity."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=100, rpd=1000, tpm=500, daily_token_cap=10000,
        )])
        bt.record_request("groq", tokens_used=100)
        bt.record_request("groq", tokens_used=200)
        # RPM: 98/100, RPD: 998/1000, TPM: 200/500 → 60% remaining, daily: 9700/10000 → 97%
        # min(0.98, 0.998, 0.4, 0.97) = 0.4 → 40.0
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(40.0)

    def test_unknown_provider_no_error(self):
        """[TM-012 AC-3] Recording a request for an unknown provider does not raise."""
        bt = BudgetTracker(providers=[_provider("groq")])
        bt.record_request("unknown")  # Should not raise
        result = bt.score("unknown")
        assert isinstance(result, Ok)
        assert result.value == 100.0


class TestHasCapacity:
    def test_has_capacity_initially(self):
        """[TM-012 AC-4] Fresh tracker reports capacity available."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        assert bt.has_capacity("groq") is True

    def test_no_capacity_after_rpm_exhausted(self):
        """[TM-012 AC-4] has_capacity returns False when RPM limit exhausted."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        bt.record_request("groq")
        bt.record_request("groq")
        assert bt.has_capacity("groq") is False

    def test_no_capacity_after_rpd_exhausted(self):
        """[TM-012 AC-4] has_capacity returns False when RPD limit exhausted."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=2)])
        bt.record_request("groq")
        bt.record_request("groq")
        assert bt.has_capacity("groq") is False

    def test_unknown_provider_has_capacity(self):
        """[TM-012 AC-4] Unknown provider is treated as having capacity."""
        bt = BudgetTracker(providers=[])
        assert bt.has_capacity("unknown") is True


class TestSlidingWindow:
    """TM-012 AC-5: Sliding window correctly expires old requests."""

    def test_old_requests_expire(self):
        """[TM-012 AC-5] Requests older than 60s are evicted from the RPM window."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        # Simulate two requests in the past (>60s ago)
        bt._rpm_windows["groq"].append(time.time() - 70.0)
        bt._rpm_windows["groq"].append(time.time() - 70.0)
        # Those should be expired, so capacity is available
        assert bt.has_capacity("groq") is True

    def test_tpm_window_expires(self):
        """[TM-012 AC-5] Token usage entries older than 60s are evicted from the TPM window."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, tpm=100)])
        # Add a token usage from 70 seconds ago
        bt._tpm_windows["groq"].append((time.time() - 70.0, 50))
        # Add another from 70 seconds ago
        bt._tpm_windows["groq"].append((time.time() - 70.0, 50))
        # Now record a request with 10 tokens (should be allowed because old ones expired)
        bt.record_request("groq", tokens_used=10)
        # TPM remaining should be 90 (limit 100 - 10 used in last minute)
        assert bt._tpm_remaining("groq") == 90
        # Score should reflect TPM usage
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(90.0)  # RPM and RPD still unlimited

    def test_tpm_window_accumulates_within_minute(self):
        """[TM-012 AC-5] Token usage within the same minute accumulates correctly."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, tpm=100)])
        # Two requests within the same minute
        bt.record_request("groq", tokens_used=30)
        bt.record_request("groq", tokens_used=40)
        # Total tokens used in last minute: 70
        assert bt._tpm_remaining("groq") == 30
        # Record another request that would exceed TPM
        bt.record_request("groq", tokens_used=20)  # total 90
        assert bt._tpm_remaining("groq") == 10
        # One more request would exceed
        bt.record_request("groq", tokens_used=5)  # total 95
        assert bt._tpm_remaining("groq") == 5
        # Next request would go over limit
        # total 96 -> limit 100, 96 used -> 4 remaining
        bt.record_request("groq", tokens_used=1)
        # Actually, let's do exact: 30+40+20+5+1 = 96 -> remaining 4
        assert bt._tpm_remaining("groq") == 4
        # One more token would make 97
        bt.record_request("groq", tokens_used=1)  # total 97
        assert bt._tpm_remaining("groq") == 3
        # Continue until we hit limit
        bt.record_request("groq", tokens_used=3)  # total 100
        assert bt._tpm_remaining("groq") == 0


class TestDailyTokenCap:
    """TM-012 AC-6: Daily token cap tracking and enforcement."""

    def test_daily_token_cap_unlimited_when_none(self):
        """[TM-012 AC-6] None daily_token_cap means unlimited daily token usage."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=None)])
        # Use lots of tokens
        bt.record_request("groq", tokens_used=10000)
        bt.record_request("groq", tokens_used=10000)
        # Score should be based on RPM/RPD only
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(98.0)  # 2 requests out of 100 RPM

    def test_daily_token_cap_tracking(self):
        """[TM-012 AC-6] Daily token usage is tracked and reflected in score."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=10000, rpd=10000, daily_token_cap=1000,
        )])
        bt.record_request("groq", tokens_used=300)
        bt.record_request("groq", tokens_used=400)
        assert bt._daily_token_remaining("groq") == 300
        bt.record_request("groq", tokens_used=200)
        assert bt._daily_token_remaining("groq") == 100
        bt.record_request("groq", tokens_used=50)
        assert bt._daily_token_remaining("groq") == 50
        bt.record_request("groq", tokens_used=25)
        assert bt._daily_token_remaining("groq") == 25
        bt.record_request("groq", tokens_used=10)
        assert bt._daily_token_remaining("groq") == 15
        # Score should reflect daily token usage
        result = bt.score("groq")
        assert isinstance(result, Ok)
        score = result.value
        # RPM and RPD are unlimited (large limits), so score is based on daily token cap
        # 15/1000 = 0.015 -> 1.5%
        assert score == pytest.approx(1.5)

    def test_daily_token_cap_zero_means_unlimited(self):
        """[TM-012 AC-6] Zero daily_token_cap is treated as unlimited."""
        # According to code, if daily_token_cap == 0, it's considered unlimited
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=100, daily_token_cap=0)])
        bt.record_request("groq", tokens_used=1000000)
        # Should still have full capacity for RPM/RPD
        assert bt.has_capacity("groq") is True
        result = bt.score("groq")
        assert isinstance(result, Ok)
        score = result.value
        # RPM: 99/100, RPD: 99/100 -> 99%
        assert score == pytest.approx(99.0)

    def test_daily_token_reset_at_day_boundary(self):
        """[TM-012 AC-6] Daily token counters reset at day boundary."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=10000, rpd=10000, daily_token_cap=1000,
        )])
        # Use some tokens
        bt.record_request("groq", tokens_used=500)
        assert bt._daily_token_remaining("groq") == 500
        # Manually set day reset to past to simulate reset
        bt._day_reset_at = time.time() - 10  # in the past
        bt._maybe_reset_daily()
        # Counters should be cleared
        assert bt._daily_token_counts["groq"] == 0
        assert bt._daily_token_remaining("groq") == 1000


class TestScoreWithAllLimits:
    """TM-012 AC-7: Score considers all four limit dimensions."""

    def test_score_considers_all_limits(self):
        """[TM-012 AC-7] Score is the minimum ratio across RPM, RPD, TPM, and daily token cap."""
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10,      # RPM limit
            rpd=100,     # RPD limit
            tpm=1000,    # TPM limit
            daily_token_cap=10000  # daily token cap
        )])
        # Use 5 RPM (50%), 5 RPD (95%), 500 TPM (50%), 500 daily (95%)
        for _ in range(5):
            bt.record_request("groq", tokens_used=100)
        # Each request: 100 tokens, 5 requests -> 500 tokens
        # RPM: 5 used -> 5 remaining -> 0.5
        # RPD: 5 used -> 95 remaining -> 0.95
        # TPM: 500 used -> 500 remaining -> 0.5
        # Daily: 500 used -> 9500 remaining -> 0.95
        # min = 0.5 -> score 50.0
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(50.0)

    def test_score_zero_when_any_limit_exhausted(self):
        """[TM-012 AC-7] Score drops to zero when any single limit dimension is exhausted."""
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10,
            rpd=100,
            tpm=1000,
            daily_token_cap=10000
        )])
        # Exhaust RPM
        for _ in range(10):
            bt.record_request("groq", tokens_used=1)
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == 0.0

        # Reset and exhaust RPD
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10000,
            rpd=10,
            tpm=1000,
            daily_token_cap=10000
        )])
        for _ in range(10):
            bt.record_request("groq", tokens_used=1)
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == 0.0

        # Reset and exhaust TPM
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10000,
            rpd=10000,
            tpm=10,
            daily_token_cap=10000
        )])
        bt.record_request("groq", tokens_used=10)  # exactly at limit
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == 0.0  # TPM remaining 0 -> ratio 0

        # Reset and exhaust daily token cap
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10000,
            rpd=10000,
            tpm=10000,
            daily_token_cap=10
        )])
        bt.record_request("groq", tokens_used=10)
        result = bt.score("groq")
        assert isinstance(result, Ok)
        assert result.value == 0.0


class TestHasCapacityTPMAndDailyTokenCap:
    """Tests for has_capacity checking TPM and daily_token_cap (TM-012 AC4)."""

    def test_has_capacity_false_when_tpm_exceeded(self):
        """[TM-012 AC-4] has_capacity returns False when TPM limit is exhausted."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=1000, rpd=10000, tpm=100)])
        # Use 100 tokens in a single request — exactly at limit
        bt.record_request("groq", tokens_used=100)
        assert bt.has_capacity("groq") is False

    def test_has_capacity_false_when_daily_token_cap_exceeded(self):
        """[TM-012 AC-4] has_capacity returns False when daily_token_cap is exhausted."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=1000, rpd=10000, daily_token_cap=500)])
        bt.record_request("groq", tokens_used=250)
        bt.record_request("groq", tokens_used=250)
        assert bt.has_capacity("groq") is False

    def test_has_capacity_true_when_all_limits_within_bounds(self):
        """[TM-012 AC-4] has_capacity True when all limits have headroom."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=100, rpd=1000, tpm=10000, daily_token_cap=100000,
        )])
        bt.record_request("groq", tokens_used=50)
        assert bt.has_capacity("groq") is True

    def test_has_capacity_true_when_tpm_none(self):
        """[TM-012 AC-4] has_capacity returns True when tpm_limit is None (unlimited)."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, tpm=None)])
        bt.record_request("groq", tokens_used=999999)
        assert bt.has_capacity("groq") is True

    def test_has_capacity_true_when_daily_token_cap_none(self):
        """[TM-012 AC-4] has_capacity returns True when daily_token_cap is None (unlimited)."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=100, rpd=1000, daily_token_cap=None,
        )])
        bt.record_request("groq", tokens_used=999999)
        assert bt.has_capacity("groq") is True

    def test_has_capacity_true_when_tpm_zero(self):
        """[TM-012 AC-4] has_capacity True when tpm_limit is 0 (unlimited)."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, tpm=0)])
        bt.record_request("groq", tokens_used=999999)
        assert bt.has_capacity("groq") is True

    def test_has_capacity_true_when_daily_token_cap_zero(self):
        """[TM-012 AC-4] has_capacity True when daily_token_cap is 0 (unlimited)."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=100, rpd=1000, daily_token_cap=0,
        )])
        bt.record_request("groq", tokens_used=999999)
        assert bt.has_capacity("groq") is True

    def test_tpm_exhaustion_blocks_even_with_rpm_rpd_available(self):
        """[TM-012 AC-4] TPM limit blocking should work independently of RPM/RPD headroom."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=1000, rpd=100000, tpm=50)])
        bt.record_request("groq", tokens_used=50)
        assert bt.has_capacity("groq") is False
        # RPM and RPD still have plenty of headroom
        assert bt._rpm_remaining("groq") == 999
        assert bt._rpd_remaining("groq") == 99999

    def test_daily_token_exhaustion_blocks_even_with_rpm_rpd_tpm_available(self):
        """[TM-012 AC-4] daily_token_cap blocking works independently of RPM/RPD/TPM headroom."""
        bt = BudgetTracker(providers=[_provider(
            "groq", rpm=1000, rpd=100000, tpm=100000, daily_token_cap=200,
        )])
        bt.record_request("groq", tokens_used=200)
        assert bt.has_capacity("groq") is False


class TestDailySpendUsd:
    """[TM-012 AC-8] Tests for daily_spend_usd and monthly_spend_usd methods."""

    def test_daily_spend_zero_with_no_requests(self):
        """[TM-012 AC-8] Zero requests yields zero daily spend."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        spend = bt.daily_spend_usd("groq", avg_cost_per_token=0.00001)
        assert spend == 0.0

    def test_daily_spend_calculates_from_tokens(self):
        """[TM-012 AC-8] Daily spend is calculated from total tokens * cost per token."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        bt.record_request("groq", tokens_used=1000)
        bt.record_request("groq", tokens_used=500)
        # 1500 tokens * 0.00001 per token = 0.015 USD
        spend = bt.daily_spend_usd("groq", avg_cost_per_token=0.00001)
        assert spend == pytest.approx(0.015)

    def test_monthly_spend_is_daily_times_30(self):
        """[TM-012 AC-8] Monthly spend is estimated as daily spend * 30."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        bt.record_request("groq", tokens_used=1000)
        # daily: 1000 * 0.00001 = 0.01
        # monthly: 0.01 * 30 = 0.3
        monthly = bt.monthly_spend_usd("groq", avg_cost_per_token=0.00001)
        assert monthly == pytest.approx(0.3)

    def test_daily_spend_unknown_provider_is_zero(self):
        """[TM-012 AC-8] Unknown provider daily spend is zero."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100)])
        spend = bt.daily_spend_usd("unknown", avg_cost_per_token=0.001)
        assert spend == 0.0

    def test_daily_spend_zero_cost_is_zero(self):
        """[TM-012 AC-8] Zero cost per token yields zero daily spend."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100)])
        bt.record_request("groq", tokens_used=10000)
        spend = bt.daily_spend_usd("groq", avg_cost_per_token=0.0)
        assert spend == 0.0


class TestInvariantHelper:
    def test_invariant_raises_on_false(self):
        """[TM-012 AC-1] invariant() raises AssertionError when condition is False (line 25)."""
        from dragonlight_router.budget.tracker import invariant
        with pytest.raises(AssertionError, match="test message"):
            invariant(False, "test message")

    def test_invariant_passes_on_true(self):
        """[TM-012 AC-1] invariant() does not raise when condition is True."""
        from dragonlight_router.budget.tracker import invariant
        invariant(True, "should not raise")


class TestRpdRemainingEdgeCases:
    def test_rpd_remaining_returns_zero_for_none_provider(self):
        """[TM-012 AC-5] _rpd_remaining returns 0 when provider is None (line 180)."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        result = bt._rpd_remaining("unregistered")
        assert result == 0

    def test_tpm_remaining_returns_one_for_none_limit(self):
        """[TM-012 AC-5] _tpm_remaining returns 1 when tpm_limit is None (line 215)."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, tpm=None)])
        result = bt._tpm_remaining("groq")
        assert result == 1

    def test_daily_token_remaining_returns_zero_for_none_provider(self):
        """[TM-012 AC-6] _daily_token_remaining returns 0 when provider is None (line 232)."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100)])
        result = bt._daily_token_remaining("unregistered")
        assert result == 0


class TestCheckAndReserve:
    """HAZ-002: Atomic check-then-reserve under async lock."""

    @pytest.mark.asyncio
    async def test_check_and_reserve_succeeds_with_capacity(self):
        """[TM-012 AC-4] check_and_reserve returns True when capacity is available."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=10, rpd=1000)])
        result = await bt.check_and_reserve("groq", estimated_tokens=100)
        assert result is True
        # Verify the request was recorded
        assert bt._rpd_counts["groq"] == 1

    @pytest.mark.asyncio
    async def test_check_and_reserve_fails_without_capacity(self):
        """[TM-012 AC-4] check_and_reserve returns False when RPM is exhausted."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=1000)])
        bt.record_request("groq")
        bt.record_request("groq")
        result = await bt.check_and_reserve("groq", estimated_tokens=100)
        assert result is False
        # Verify no additional request was recorded
        assert bt._rpd_counts["groq"] == 2

    @pytest.mark.asyncio
    async def test_check_and_reserve_prevents_race_condition(self):
        """[TM-012 AC-4] Concurrent check_and_reserve calls do not exceed capacity."""
        import asyncio
        bt = BudgetTracker(providers=[_provider("groq", rpm=3, rpd=100000)])
        # Launch 10 concurrent reserve attempts, only 3 should succeed
        results = await asyncio.gather(
            *[bt.check_and_reserve("groq", estimated_tokens=1) for _ in range(10)]
        )
        successes = sum(1 for r in results if r is True)
        assert successes == 3

    @pytest.mark.asyncio
    async def test_check_and_reserve_unknown_provider(self):
        """[TM-012 AC-4] check_and_reserve returns True for unknown provider (default capacity)."""
        bt = BudgetTracker(providers=[])
        result = await bt.check_and_reserve("unknown", estimated_tokens=50)
        assert result is True


class TestGetAndRestoreState:
    """HAZ-012: Budget state serialization and restoration."""

    def test_get_state_returns_daily_counters(self):
        """[TM-012 AC-9] get_state includes rpd_counts, daily_token_counts, and day_reset_at."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=10000)])
        bt.record_request("groq", tokens_used=500)
        bt.record_request("groq", tokens_used=300)
        state = bt.get_state()
        assert state["rpd_counts"]["groq"] == 2
        assert state["daily_token_counts"]["groq"] == 800
        assert "day_reset_at" in state
        assert state["day_reset_at"] > time.time()

    def test_restore_state_reloads_counters(self):
        """[TM-012 AC-9] restore_state reloads daily counters from persisted state."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=10000)])
        state = {
            "rpd_counts": {"groq": 5},
            "daily_token_counts": {"groq": 2000},
            "day_reset_at": time.time() + 3600,  # 1 hour in the future
        }
        bt.restore_state(state)
        assert bt._rpd_counts["groq"] == 5
        assert bt._daily_token_counts["groq"] == 2000

    def test_restore_state_skips_stale_data(self):
        """[TM-012 AC-9] restore_state ignores state whose reset boundary has passed."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        state = {
            "rpd_counts": {"groq": 99},
            "daily_token_counts": {"groq": 9999},
            "day_reset_at": time.time() - 100,  # already passed
        }
        bt.restore_state(state)
        # Should NOT have restored the stale counters
        assert bt._rpd_counts["groq"] == 0

    def test_round_trip_get_restore(self):
        """[TM-012 AC-9] get_state -> restore_state preserves counters."""
        bt1 = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=10000)])
        bt1.record_request("groq", tokens_used=750)
        bt1.record_request("groq", tokens_used=250)
        state = bt1.get_state()

        bt2 = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=10000)])
        bt2.restore_state(state)
        assert bt2._rpd_counts["groq"] == 2
        assert bt2._daily_token_counts["groq"] == 1000


class TestPropertyTests:
    @given(
        rpm=st.integers(min_value=0, max_value=1000),
        rpd=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        tpm=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        daily_token_cap=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
        requests=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=1000),
            ),
            max_size=10,
        )
    )
    def test_score_is_between_0_and_100(self, rpm, rpd, tpm, daily_token_cap, requests):
        """Score should always be between 0 and 100 inclusive."""
        provider = _provider(
            name="test",
            rpm=rpm,
            rpd=rpd,
            tpm=tpm,
            daily_token_cap=daily_token_cap
        )
        bt = BudgetTracker(providers=[provider])
        for tokens_used, _ in requests:
            bt.record_request("test", tokens_used=tokens_used)
        result = bt.score("test")
        assert isinstance(result, Ok)
        score = result.value
        assert 0.0 <= score <= 100.0