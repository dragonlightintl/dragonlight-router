"""Tests for selection/scoring.py — composite scoring functions."""
from __future__ import annotations

import pytest

from dragonlight_router.selection.scoring import (
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
)


class TestCompositeScore:
    def test_perfect_scores(self):
        """Rank 100, budget 100, health 100 → weighted composite."""
        result = compute_composite_score(rank=100, budget_score=100.0, health_score=100.0)
        assert result == pytest.approx(100.0)

    def test_zero_scores(self):
        result = compute_composite_score(rank=0, budget_score=0.0, health_score=0.0)
        assert result == pytest.approx(0.0)

    def test_weight_distribution(self):
        """Rank dominates (60%), budget second (25%), health third (15%)."""
        result = compute_composite_score(rank=50, budget_score=80.0, health_score=60.0)
        expected = 50 * 0.6 + 80.0 * 0.25 + 60.0 * 0.15
        assert result == pytest.approx(expected)

    def test_high_rank_low_health(self):
        result = compute_composite_score(rank=100, budget_score=100.0, health_score=0.0)
        assert result == pytest.approx(85.0)


class TestBudgetScore:
    def test_full_capacity(self):
        """All limits full → 100."""
        score = compute_budget_score(rpm_remaining=100, rpm_limit=100, rpd_remaining=1000, rpd_limit=1000)
        assert score == pytest.approx(100.0)

    def test_half_capacity(self):
        score = compute_budget_score(rpm_remaining=50, rpm_limit=100, rpd_remaining=500, rpd_limit=1000)
        assert score == pytest.approx(50.0)

    def test_rpm_limiting(self):
        """RPM is more constrained than RPD."""
        score = compute_budget_score(rpm_remaining=10, rpm_limit=100, rpd_remaining=900, rpd_limit=1000)
        assert score == pytest.approx(10.0)

    def test_rpd_limiting(self):
        """RPD is more constrained than RPM."""
        score = compute_budget_score(rpm_remaining=90, rpm_limit=100, rpd_remaining=100, rpd_limit=1000)
        assert score == pytest.approx(10.0)

    def test_none_rpd_unlimited(self):
        """None RPD limits treated as unlimited (100)."""
        score = compute_budget_score(rpm_remaining=50, rpm_limit=100, rpd_remaining=None, rpd_limit=None)
        assert score == pytest.approx(50.0)

    def test_none_rpd_with_low_rpm(self):
        """None RPD + low RPM → RPM drives score."""
        score = compute_budget_score(rpm_remaining=5, rpm_limit=100, rpd_remaining=None, rpd_limit=None)
        assert score == pytest.approx(5.0)

    def test_zero_remaining(self):
        score = compute_budget_score(rpm_remaining=0, rpm_limit=100, rpd_remaining=0, rpd_limit=1000)
        assert score == pytest.approx(0.0)


class TestHealthScore:
    def test_healthy(self):
        """No errors, circuit closed, recent success → 100."""
        score = compute_health_score(error_count=0, circuit_open=False, last_success_age_s=1.0)
        assert score == pytest.approx(100.0)

    def test_circuit_open_zero(self):
        """Circuit open → always 0."""
        score = compute_health_score(error_count=0, circuit_open=True, last_success_age_s=0.0)
        assert score == pytest.approx(0.0)

    def test_one_error(self):
        score = compute_health_score(error_count=1, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(70.0)

    def test_two_errors(self):
        score = compute_health_score(error_count=2, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(70.0)

    def test_three_plus_errors(self):
        score = compute_health_score(error_count=3, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(30.0)

    def test_many_errors(self):
        score = compute_health_score(error_count=10, circuit_open=False, last_success_age_s=100.0)
        assert score == pytest.approx(30.0)
