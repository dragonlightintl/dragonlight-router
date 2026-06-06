"""Tests for budget/tracker.py — per-provider budget tracking."""
from __future__ import annotations

import time

import pytest
from hypothesis import given, strategies as st

from dragonlight_router.core.types import ProviderConfig
from dragonlight_router.budget.tracker import BudgetTracker


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
        bt = BudgetTracker(providers=[_provider("groq"), _provider("nvidia")])
        assert bt.has_capacity("groq")
        assert bt.has_capacity("nvidia")

    def test_empty_providers(self):
        bt = BudgetTracker(providers=[])
        assert bt.score("unknown").unwrap() == 100.0


class TestBudgetScore:
    def test_full_capacity_is_100(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        assert bt.score("groq").unwrap() == pytest.approx(100.0)

    def test_score_decreases_with_requests(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        for _ in range(15):
            bt.record_request("groq")
        score = bt.score("groq").unwrap()
        assert score > 0.0
        assert score < 100.0

    def test_unknown_provider_returns_100(self):
        bt = BudgetTracker(providers=[_provider("groq")])
        assert bt.score("unknown").unwrap() == 100.0

    def test_unlimited_rpd_none(self):
        """None rpd_limit means unlimited — only RPM matters."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=10, rpd=None)])
        for _ in range(5):
            bt.record_request("groq")
        score = bt.score("groq").unwrap()
        assert score == pytest.approx(50.0)


class TestRecordRequest:
    def test_records_count(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        bt.record_request("groq")
        bt.record_request("groq")
        # RPM: 98/100, RPD: 998/1000 → min(0.98, 0.998) * 100 = 98.0
        assert bt.score("groq").unwrap() == pytest.approx(98.0)

    def test_records_tokens(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, tpm=500, daily_token_cap=10000)])
        bt.record_request("groq", tokens_used=100)
        bt.record_request("groq", tokens_used=200)
        # RPM: 98/100, RPD: 998/1000, TPM: 200/500 → 60% remaining, daily: 9700/10000 → 97%
        # min(0.98, 0.998, 0.6, 0.97) = 0.6 → 60.0
        assert bt.score("groq").unwrap() == pytest.approx(60.0)

    def test_unknown_provider_no_error(self):
        bt = BudgetTracker(providers=[_provider("groq")])
        bt.record_request("unknown")  # Should not raise
        assert bt.score("unknown").unwrap() == 100.0


class TestHasCapacity:
    def test_has_capacity_initially(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        assert bt.has_capacity("groq") is True

    def test_no_capacity_after_rpm_exhausted(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        bt.record_request("groq")
        bt.record_request("groq")
        assert bt.has_capacity("groq") is False

    def test_no_capacity_after_rpd_exhausted(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=2)])
        bt.record_request("groq")
        bt.record_request("groq")
        assert bt.has_capacity("groq") is False

    def test_unknown_provider_has_capacity(self):
        bt = BudgetTracker(providers=[])
        assert bt.has_capacity("unknown") is True


class TestSlidingWindow:
    def test_old_requests_expire(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        # Simulate two requests in the past (>60s ago)
        bt._rpm_windows["groq"].append(time.time() - 70.0)
        bt._rpm_windows["groq"].append(time.time() - 70.0)
        # Those should be expired, so capacity is available
        assert bt.has_capacity("groq") is True

    def test_tpm_window_expires(self):
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
        score = bt.score("groq").unwrap()
        assert score == pytest.approx(90.0)  # RPM and RPD still unlimited

    def test_tpm_window_accumulates_within_minute(self):
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
        bt.record_request("groq", tokens_used=1)  # total 96 -> still under? Wait, limit 100, 96 used -> 4 remaining
        # Actually, let's do exact: 30+40+20+5+1 = 96 -> remaining 4
        assert bt._tpm_remaining("groq") == 4
        # One more token would make 97
        bt.record_request("groq", tokens_used=1)  # total 97
        assert bt._tpm_remaining("groq") == 3
        # Continue until we hit limit
        bt.record_request("groq", tokens_used=3)  # total 100
        assert bt._tpm_remaining("groq") == 0


class TestDailyTokenCap:
    def test_daily_token_cap_unlimited_when_none(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000, daily_token_cap=None)])
        # Use lots of tokens
        bt.record_request("groq", tokens_used=10000)
        bt.record_request("groq", tokens_used=10000)
        # Score should be based on RPM/RPD only
        assert bt.score("groq").unwrap() == pytest.approx(98.0)  # 2 requests out of 100 RPM

    def test_daily_token_cap_tracking(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=10000, rpd=10000, daily_token_cap=1000)])
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
        score = bt.score("groq").unwrap()
        # RPM and RPD are unlimited (large limits), so score is based on daily token cap
        # 15/1000 = 0.015 -> 1.5%
        assert score == pytest.approx(1.5)

    def test_daily_token_cap_zero_means_unlimited(self):
        # According to code, if daily_token_cap == 0, it's considered unlimited
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=100, daily_token_cap=0)])
        bt.record_request("groq", tokens_used=1000000)
        # Should still have full capacity for RPM/RPD
        assert bt.has_capacity("groq") is True
        score = bt.score("groq").unwrap()
        # RPM: 99/100, RPD: 99/100 -> 99%
        assert score == pytest.approx(99.0)

    def test_daily_token_reset_at_day_boundary(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=10000, rpd=10000, daily_token_cap=1000)])
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
    def test_score_considers_all_limits(self):
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10,      # RPM limit
            rpd=100,     # RPD limit
            tpm=1000,    # TPM limit
            daily_token_cap=10000  # daily token cap
        )])
        # Use 5 RPM (50% remaining), 50 RPD (50% remaining), 500 TPM (50% remaining), 5000 tokens (50% remaining)
        for i in range(5):
            bt.record_request("groq", tokens_used=1000)
        # Each request: 1000 tokens, 5 requests -> 5000 tokens
        # RPM: 5 used -> 5 remaining -> 0.5
        # RPD: 5 used -> 95 remaining -> 0.95
        # TPM: 5000 used -> 5000 remaining -> 0.5
        # Daily: 5000 used -> 5000 remaining -> 0.5
        # min = 0.5 -> score 50.0
        score = bt.score("groq").unwrap()
        assert score == pytest.approx(50.0)

    def test_score_zero_when_any_limit_exhausted(self):
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
        assert bt.score("groq").unwrap() == 0.0

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
        assert bt.score("groq").unwrap() == 0.0

        # Reset and exhaust TPM
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10000,
            rpd=10000,
            tpm=10,
            daily_token_cap=10000
        )])
        bt.record_request("groq", tokens_used=10)  # exactly at limit
        assert bt.score("groq").unwrap() == 0.0  # TPM remaining 0 -> ratio 0

        # Reset and exhaust daily token cap
        bt = BudgetTracker(providers=[_provider(
            name="groq",
            rpm=10000,
            rpd=10000,
            tpm=10000,
            daily_token_cap=10
        )])
        bt.record_request("groq", tokens_used=10)
        assert bt.score("groq").unwrap() == 0.0


class TestPropertyTests:
    @given(
        rpm=st.integers(min_value=0, max_value=1000),
        rpd=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        tpm=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        daily_token_cap=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
        requests=st.lists(st.tuples(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=1000)), max_size=10)
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
        assert result.is_ok()
        score = result.unwrap()
        assert 0.0 <= score <= 100.0


# Note: The existing test file had a TestSlidingWindow class for RPM only.
# We've extended it to include TPM tests above.
# We'll keep the original RPM sliding window test and add TPM ones.