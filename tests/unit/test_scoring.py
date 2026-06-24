"""Tests for selection/scoring.py — composite scoring functions.

Spec traceability: TM-007 (Scoring weights)
"""

from __future__ import annotations

import pytest

from dragonlight_router.selection.scoring import (
    ScoringWeights,
    ScoringWeightsConfig,
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
    cost_adjusted_weights,
    score_candidate,
)

pytestmark = pytest.mark.unit


class TestCompositeScore:
    def test_perfect_scores(self):
        """[TM-007 AC-1] Rank 100, budget 100, health 100 → weighted composite."""
        result = compute_composite_score(rank=100, budget_score=100.0, health_score=100.0)
        assert result == pytest.approx(100.0)

    def test_zero_scores(self):
        """[TM-007 AC-1] All-zero inputs produce zero composite score."""
        result = compute_composite_score(rank=0, budget_score=0.0, health_score=0.0)
        assert result == pytest.approx(0.0)

    def test_weight_distribution(self):
        """[TM-007 AC-2] Rank dominates (60%), budget second (25%), health third (15%)."""
        result = compute_composite_score(rank=50, budget_score=80.0, health_score=60.0)
        expected = 50 * 0.6 + 80.0 * 0.25 + 60.0 * 0.15
        assert result == pytest.approx(expected)

    def test_high_rank_low_health(self):
        """[TM-007 AC-2] High rank with zero health still produces a positive composite score."""
        result = compute_composite_score(rank=100, budget_score=100.0, health_score=0.0)
        assert result == pytest.approx(85.0)


class TestBudgetScore:
    def test_full_capacity(self):
        """[TM-007 AC-3] All limits full -> 100."""
        score = compute_budget_score(
            rpm_remaining=100,
            rpm_limit=100,
            rpd_remaining=1000,
            rpd_limit=1000,
        )
        assert score == pytest.approx(100.0)

    def test_half_capacity(self):
        """[TM-007 AC-3] Half remaining on both RPM and RPD yields 50."""
        score = compute_budget_score(
            rpm_remaining=50,
            rpm_limit=100,
            rpd_remaining=500,
            rpd_limit=1000,
        )
        assert score == pytest.approx(50.0)

    def test_rpm_limiting(self):
        """[TM-007 AC-3] RPM is more constrained than RPD -- score follows the minimum."""
        score = compute_budget_score(
            rpm_remaining=10,
            rpm_limit=100,
            rpd_remaining=900,
            rpd_limit=1000,
        )
        assert score == pytest.approx(10.0)

    def test_rpd_limiting(self):
        """[TM-007 AC-3] RPD is more constrained than RPM -- score follows the minimum."""
        score = compute_budget_score(
            rpm_remaining=90,
            rpm_limit=100,
            rpd_remaining=100,
            rpd_limit=1000,
        )
        assert score == pytest.approx(10.0)

    def test_none_rpd_unlimited(self):
        """[TM-007 AC-3] None RPD limits treated as unlimited (100)."""
        score = compute_budget_score(
            rpm_remaining=50,
            rpm_limit=100,
            rpd_remaining=None,
            rpd_limit=None,
        )
        assert score == pytest.approx(50.0)

    def test_none_rpd_with_low_rpm(self):
        """[TM-007 AC-3] None RPD + low RPM -> RPM drives score."""
        score = compute_budget_score(
            rpm_remaining=5,
            rpm_limit=100,
            rpd_remaining=None,
            rpd_limit=None,
        )
        assert score == pytest.approx(5.0)

    def test_zero_remaining(self):
        """[TM-007 AC-3] Zero remaining on both dimensions yields zero score."""
        score = compute_budget_score(
            rpm_remaining=0,
            rpm_limit=100,
            rpd_remaining=0,
            rpd_limit=1000,
        )
        assert score == pytest.approx(0.0)


class TestHealthScore:
    def test_healthy(self):
        """[TM-007 AC-4] No errors, circuit closed, recent success → 100."""
        score = compute_health_score(error_count=0, circuit_open=False, last_success_age_s=1.0)
        assert score == pytest.approx(100.0)

    def test_circuit_open_zero(self):
        """[TM-007 AC-4] Circuit open → always 0."""
        score = compute_health_score(error_count=0, circuit_open=True, last_success_age_s=0.0)
        assert score == pytest.approx(0.0)

    def test_one_error(self):
        """[TM-007 AC-4] One error yields degraded health score of 70."""
        score = compute_health_score(error_count=1, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(70.0)

    def test_two_errors(self):
        """[TM-007 AC-4] Two errors still yield degraded health score of 70."""
        score = compute_health_score(error_count=2, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(70.0)

    def test_three_plus_errors(self):
        """[TM-007 AC-4] Three or more errors yield severely degraded health score of 30."""
        score = compute_health_score(error_count=3, circuit_open=False, last_success_age_s=5.0)
        assert score == pytest.approx(30.0)

    def test_many_errors(self):
        """[TM-007 AC-4] Many errors (10) still yield the 3+ errors score of 30."""
        score = compute_health_score(error_count=10, circuit_open=False, last_success_age_s=100.0)
        assert score == pytest.approx(30.0)


class TestScoringWeightsEnum:
    def test_enum_members_are_floats(self):
        """[TM-007 AC-5] ScoringWeights enum members have expected float values (IBR-active defaults)."""
        assert ScoringWeights.COST.value == pytest.approx(0.20)
        assert ScoringWeights.LATENCY.value == pytest.approx(0.25)
        assert ScoringWeights.PRIORITY.value == pytest.approx(0.20)
        assert ScoringWeights.QUEUE.value == pytest.approx(0.10)
        assert ScoringWeights.HEALTH.value == pytest.approx(0.10)
        assert ScoringWeights.SPECTROGRAPH_MATCH.value == pytest.approx(0.15)

    def test_cost_adjusted_weights_returns_governor_config(self):
        """[TM-007 AC-5] cost_adjusted_weights returns ScoringWeightsConfig with IBR governor weights."""
        base = ScoringWeightsConfig()
        adjusted = cost_adjusted_weights(base)
        # With IBR active (spectrograph_match > 0), cost governor uses IBR-SCORE-05 weights
        assert adjusted.cost == pytest.approx(0.65)
        assert adjusted.latency == pytest.approx(0.10)
        assert adjusted.spectrograph_match == pytest.approx(0.05)
        total = (
            adjusted.cost + adjusted.latency + adjusted.priority
            + adjusted.queue + adjusted.health + adjusted.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9


class TestScoreCandidateWithHealthTracker:
    def test_score_candidate_with_health_tracker(self, make_backend_config):
        """[TM-007 AC-6] score_candidate incorporates health_tracker score (lines 299-300)."""
        from unittest.mock import MagicMock

        from dragonlight_router.budget.tracker import BudgetTracker
        from dragonlight_router.core.types import Ok, ProviderConfig

        config = make_backend_config(name="test", provider="prov")
        provider = ProviderConfig(
            name="prov",
            base_url="http://test",
            catalog_url=None,
            env_key=None,
            model_prefix="prov",
            rpm_limit=100,
            rpd_limit=None,
            tpm_limit=None,
            daily_token_cap=None,
        )
        bt = BudgetTracker(providers=[provider])
        weights = ScoringWeightsConfig()

        health_tracker = MagicMock()
        health_tracker.score.return_value = Ok(90.0)

        result = score_candidate(
            config=config,
            order=None,  # type: ignore[arg-type]
            weights=weights,
            budget_tracker=bt,
            health_tracker=health_tracker,
        )
        assert 0.0 <= result <= 1.0
        health_tracker.score.assert_called_once()
