"""Tests for health/tracker.py — per-model health tracking.

Spec traceability: TM-014 (Health tracker per-model scoring)
"""
from __future__ import annotations

import pytest

from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.core.types import Ok


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