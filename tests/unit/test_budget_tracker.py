"""Tests for budget/tracker.py — per-provider budget tracking."""
from __future__ import annotations

import time

import pytest

from dragonlight_router.core.types import ProviderConfig
from dragonlight_router.budget.tracker import BudgetTracker


def _provider(name: str = "groq", rpm: int = 30, rpd: int | None = 14400) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="http://localhost",
        catalog_url=None,
        env_key=None,
        model_prefix=name,
        rpm_limit=rpm,
        rpd_limit=rpd,
        tpm_limit=None,
    )


class TestBudgetTrackerInit:
    def test_initializes_with_providers(self):
        bt = BudgetTracker(providers=[_provider("groq"), _provider("nvidia")])
        assert bt.has_capacity("groq")
        assert bt.has_capacity("nvidia")

    def test_empty_providers(self):
        bt = BudgetTracker(providers=[])
        assert bt.score("unknown") == 100.0


class TestBudgetScore:
    def test_full_capacity_is_100(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        assert bt.score("groq") == pytest.approx(100.0)

    def test_score_decreases_with_requests(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=30, rpd=14400)])
        for _ in range(15):
            bt.record_request("groq")
        score = bt.score("groq")
        assert score < 100.0
        assert score > 0.0

    def test_unknown_provider_returns_100(self):
        bt = BudgetTracker(providers=[_provider("groq")])
        assert bt.score("unknown") == 100.0

    def test_unlimited_rpd_none(self):
        """None rpd_limit means unlimited — only RPM matters."""
        bt = BudgetTracker(providers=[_provider("groq", rpm=10, rpd=None)])
        for _ in range(5):
            bt.record_request("groq")
        score = bt.score("groq")
        assert score == pytest.approx(50.0)


class TestRecordRequest:
    def test_records_count(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=100, rpd=1000)])
        bt.record_request("groq")
        bt.record_request("groq")
        # RPM: 98/100, RPD: 998/1000 → min(0.98, 0.998) * 100 = 98.0
        assert bt.score("groq") == pytest.approx(98.0)

    def test_unknown_provider_no_error(self):
        bt = BudgetTracker(providers=[_provider("groq")])
        bt.record_request("unknown")  # Should not raise
        assert bt.score("unknown") == 100.0


class TestHasCapacity:
    def test_has_capacity_initially(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
        assert bt.has_capacity("groq") is True

    def test_no_capacity_after_rpm_exhausted(self):
        bt = BudgetTracker(providers=[_provider("groq", rpm=2, rpd=100)])
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
