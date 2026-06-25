"""Tests for health/tracker.py — per-model health tracking.

Spec traceability: TM-014 (Health tracker per-model scoring)
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import Ok
from dragonlight_router.health.tracker import HealthTracker

pytestmark = pytest.mark.unit


class TestHealthTrackerInit:
    def test_fresh_model_score_100(self):
        """[TM-014 AC-1] Fresh model starts with score 100."""
        ht = HealthTracker()
        result = ht.score("some-model")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(100.0)

    def test_fresh_model_is_available(self):
        """[TM-014 AC-1] Fresh model is available."""
        ht = HealthTracker()
        assert ht.is_available("some-model") is True


class TestRecordSuccess:
    def test_clears_error_count(self):
        """[TM-014 AC-2] Success clears error count and restores score to 100."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_success("m1", latency_ms=50.0)
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(100.0)

    def test_tracks_latency(self):
        """[TM-014 AC-2] Success tracks latency via EMA smoothing."""
        ht = HealthTracker()
        ht.record_success("m1", latency_ms=200.0)
        ht.record_success("m1", latency_ms=100.0)
        # EMA should be between 100 and 200
        assert ht.get_avg_latency("m1") > 100.0
        assert ht.get_avg_latency("m1") < 200.0


class TestRecordError:
    def test_one_error_score_70(self):
        """[TM-014 AC-3] One error reduces score to 70."""
        ht = HealthTracker()
        ht.record_error("m1")
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(70.0)

    def test_two_errors_score_70(self):
        """[TM-014 AC-3] Two errors keep score at 70 (below circuit threshold)."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(70.0)

    def test_three_errors_trips_circuit_score_zero(self):
        """3 errors trips the circuit breaker → score goes to 0."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        # Circuit is now open, so score = 0
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(0.0)

    def test_circuit_opens_at_three(self):
        """[TM-014 AC-4] Three errors trips circuit, model becomes unavailable."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is False

    def test_circuit_open_score_zero(self):
        """[TM-014 AC-4] Circuit open yields score of zero."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(0.0)


class TestIsAvailable:
    def test_available_initially(self):
        """[TM-014 AC-1] Model is available initially."""
        ht = HealthTracker()
        assert ht.is_available("m1") is True

    def test_available_with_errors_below_threshold(self):
        """[TM-014 AC-3] Model remains available with errors below threshold."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is True

    def test_unavailable_after_circuit_open(self):
        """[TM-014 AC-4] Model becomes unavailable after circuit opens."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is False


class TestRetirement:
    def test_score_retired_model_returns_zero(self):
        """[TM-008 AC-1] Retired model score is 0."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(0.0)

    def test_record_error_404_retires_model(self):
        """[TM-008 AC-2] record_error with http_status=404 retires the model."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        assert ht.is_retired("m1") is True

    def test_retire_model_sets_timestamp(self):
        """[TM-008 AC-3] _retire_model stores a float timestamp in _retired."""
        import time

        ht = HealthTracker()
        before = time.time()
        ht._retire_model("m1")
        after = time.time()
        assert "m1" in ht._retired
        assert isinstance(ht._retired["m1"], float)
        assert before <= ht._retired["m1"] <= after

    def test_is_available_returns_false_for_retired(self):
        """[TM-008 AC-4] Retired model is not available."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        assert ht.is_available("m1") is False

    def test_get_retired_models_returns_copy(self):
        """[TM-008 AC-5] get_retired_models returns a dict with all retired models."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        ht.record_error("m2", http_status=404)
        retired = ht.get_retired_models()
        assert "m1" in retired
        assert "m2" in retired
        # Verify it's a copy — mutating it does not affect internal state
        retired["extra"] = 9999.0
        assert "extra" not in ht.get_retired_models()


class TestReinstatement:
    def test_reinstate_model_restores_availability(self):
        """[TM-008 AC-6] Reinstating a retired model makes it available again."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        assert ht.is_available("m1") is False
        ht.reinstate_model("m1")
        assert ht.is_available("m1") is True

    def test_reinstate_model_resets_errors(self):
        """[TM-008 AC-7] Reinstating a model resets its error count to 0."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        ht.reinstate_model("m1")
        assert ht.get_error_count("m1") == 0

    def test_reinstate_nonexistent_model_noop(self):
        """[TM-008 AC-8] Reinstating a model that was never retired is a no-op."""
        ht = HealthTracker()
        # Should not raise
        ht.reinstate_model("never-retired")
        assert ht.is_available("never-retired") is True


class TestRetirement403:
    def test_record_error_403_retires_model(self):
        """[TM-008] record_error with http_status=403 retires the model."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.is_retired("m1") is True

    def test_score_403_retired_model_returns_zero(self):
        """[TM-008] 403-retired model score is 0."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(0.0)

    def test_403_retired_model_not_available(self):
        """[TM-008] 403-retired model is not available."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.is_available("m1") is False

    def test_403_suspended_model_is_unavailable(self):
        """[TM-008] 403 suspends the model (not retired), making it unavailable."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert "m1" in ht._suspended
        assert "m1" not in ht._retired

    def test_403_increments_error_count(self):
        """[TM-008] 403 suspends and also increments error count."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.get_error_count("m1") == 1

    def test_reinstate_after_403_works(self):
        """[TM-008] 403-suspended model can be reinstated."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.is_available("m1") is False
        ht.reinstate_model("m1")
        assert ht.is_available("m1") is True

    def test_mixed_403_and_404_both_retire(self):
        """[TM-008] Both 403 and 404 trigger retirement for different models."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        ht.record_error("m2", http_status=404)
        assert ht.is_retired("m1") is True
        assert ht.is_retired("m2") is True

    def test_other_4xx_do_not_permanently_retire(self):
        """[TM-008] Other HTTP 4xx codes (e.g. 429) do NOT permanently retire.

        429 now triggers a temporary suspension (TTL-based cooldown) rather
        than permanent retirement. The model is unavailable during the
        cooldown but returns to the pool after it expires.
        """
        ht = HealthTracker()
        ht.record_error("m1", http_status=429)
        # Not permanently retired, but temporarily suspended
        assert "m1" not in ht._retired
        assert "m1" in ht._suspended
        assert ht.get_error_count("m1") == 1


class TestScoreEdgeCases:
    def test_score_three_plus_errors_returns_30(self):
        """[TM-008 AC-9] score() returns 30 when error_count >= 3 but circuit is still closed.

        Uses a high error_threshold so the circuit breaker does not open,
        allowing the error_count >= 3 branch to be reached.
        """
        ht = HealthTracker(error_threshold=10, error_window_s=120.0, cooldown_s=60.0)
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(30.0)

    def test_is_available_returns_false_when_circuit_open(self):
        """[TM-008 AC-10] is_available returns False when circuit breaker is open."""
        ht = HealthTracker(error_threshold=3, error_window_s=120.0, cooldown_s=60.0)
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is False
