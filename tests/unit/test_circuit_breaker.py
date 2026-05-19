"""Tests for health/circuit_breaker.py — circuit breaker state machine."""
from __future__ import annotations

import time

import pytest

from dragonlight_router.health.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInit:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_allows_request_initially(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True


class TestTripping:
    def test_trips_after_threshold_errors(self):
        cb = CircuitBreaker(error_threshold=3, error_window_s=120.0, cooldown_s=60.0)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        assert cb.state == CircuitState.OPEN

    def test_does_not_trip_below_threshold(self):
        cb = CircuitBreaker(error_threshold=3)
        cb.record_error()
        cb.record_error()
        assert cb.state == CircuitState.CLOSED

    def test_blocks_requests_when_open(self):
        cb = CircuitBreaker(error_threshold=3, cooldown_s=60.0)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        assert cb.allow_request() is False


class TestHalfOpen:
    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(error_threshold=3, cooldown_s=0.01)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        time.sleep(0.02)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes(self):
        cb = CircuitBreaker(error_threshold=3, cooldown_s=0.01)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        time.sleep(0.02)
        cb.allow_request()  # transitions to half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_error_in_half_open_reopens(self):
        cb = CircuitBreaker(error_threshold=3, cooldown_s=0.01)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        time.sleep(0.02)
        cb.allow_request()  # transitions to half-open
        cb.record_error()
        assert cb.state == CircuitState.OPEN


class TestErrorWindow:
    def test_errors_outside_window_dont_accumulate(self):
        cb = CircuitBreaker(error_threshold=3, error_window_s=0.01, cooldown_s=60.0)
        cb.record_error()
        cb.record_error()
        time.sleep(0.02)  # errors expire
        cb.record_error()  # starts fresh
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_error_count(self):
        cb = CircuitBreaker(error_threshold=3)
        cb.record_error()
        cb.record_error()
        cb.record_success()
        cb.record_error()
        # Only 1 error since last success
        assert cb.state == CircuitState.CLOSED
