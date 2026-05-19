"""Tests for health/tracker.py — per-model health tracking."""
from __future__ import annotations

import pytest

from dragonlight_router.health.tracker import HealthTracker


class TestHealthTrackerInit:
    def test_fresh_model_score_100(self):
        ht = HealthTracker()
        assert ht.score("some-model") == pytest.approx(100.0)

    def test_fresh_model_is_available(self):
        ht = HealthTracker()
        assert ht.is_available("some-model") is True


class TestRecordSuccess:
    def test_clears_error_count(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_success("m1", latency_ms=50.0)
        assert ht.score("m1") == pytest.approx(100.0)

    def test_tracks_latency(self):
        ht = HealthTracker()
        ht.record_success("m1", latency_ms=200.0)
        ht.record_success("m1", latency_ms=100.0)
        # EMA should be between 100 and 200
        assert ht.get_avg_latency("m1") > 100.0
        assert ht.get_avg_latency("m1") < 200.0


class TestRecordError:
    def test_one_error_score_70(self):
        ht = HealthTracker()
        ht.record_error("m1")
        assert ht.score("m1") == pytest.approx(70.0)

    def test_two_errors_score_70(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.score("m1") == pytest.approx(70.0)

    def test_three_errors_trips_circuit_score_zero(self):
        """3 errors trips the circuit breaker → score goes to 0."""
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        # Circuit is now open, so score = 0
        assert ht.score("m1") == pytest.approx(0.0)

    def test_circuit_opens_at_three(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is False

    def test_circuit_open_score_zero(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.score("m1") == pytest.approx(0.0)


class TestIsAvailable:
    def test_available_initially(self):
        ht = HealthTracker()
        assert ht.is_available("m1") is True

    def test_available_with_errors_below_threshold(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is True

    def test_unavailable_after_circuit_open(self):
        ht = HealthTracker()
        ht.record_error("m1")
        ht.record_error("m1")
        ht.record_error("m1")
        assert ht.is_available("m1") is False
