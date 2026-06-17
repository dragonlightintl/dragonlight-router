"""Tests for dragonlight_router.core.state — BackendState rate tracking and circuit breaker.

Spec traceability: TM-023 (Backend state tracking)
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import BackendStatus


class TestRPMCapacity:
    def test_empty_state_has_capacity(self):
        """[TM-023 AC-1] Fresh state has RPM capacity."""
        state = BackendState()
        assert state.has_rpm_capacity(30) is True

    def test_at_limit_no_capacity(self):
        """[TM-023 AC-1] At RPM limit, has_rpm_capacity returns False."""
        state = BackendState()
        now = time.time()
        for i in range(30):
            state.request_timestamps.append(now - i)
        assert state.has_rpm_capacity(30) is False

    def test_old_timestamps_evicted(self):
        """[TM-023 AC-1] Timestamps older than 60s are evicted from RPM window."""
        state = BackendState()
        now = time.time()
        for i in range(30):
            state.request_timestamps.append(now - 61 - i)
        assert state.has_rpm_capacity(30) is True

    def test_mixed_timestamps(self):
        """[TM-023 AC-1] Only recent timestamps count toward RPM limit."""
        state = BackendState()
        now = time.time()
        for i in range(20):
            state.request_timestamps.append(now - 61 - i)
        for i in range(10):
            state.request_timestamps.append(now - i)
        assert state.has_rpm_capacity(30) is True
        assert state.has_rpm_capacity(10) is False

    def test_zero_limit_asserts(self):
        """[TM-023 AC-1] Zero RPM limit raises AssertionError."""
        state = BackendState()
        with pytest.raises(AssertionError):
            state.has_rpm_capacity(0)


class TestRPDCapacity:
    def test_empty_has_capacity(self):
        """[TM-023 AC-2] Fresh state has RPD capacity."""
        state = BackendState()
        assert state.has_rpd_capacity(1000) is True

    def test_at_limit_no_capacity(self):
        """[TM-023 AC-2] At RPD limit, has_rpd_capacity returns False."""
        state = BackendState()
        state.requests_today = 1000
        state.day_reset_at = time.time() + 86400
        assert state.has_rpd_capacity(1000) is False

    def test_day_rollover_resets(self):
        """[TM-023 AC-2] Day rollover resets daily counters."""
        state = BackendState()
        state.requests_today = 999
        state.day_reset_at = time.time() - 1
        assert state.has_rpd_capacity(1000) is True
        assert state.requests_today == 0


class TestTokenCapacity:
    def test_zero_limit_means_unlimited(self):
        """[TM-023 AC-3] Zero token limit means unlimited capacity."""
        state = BackendState()
        state.tokens_today = 999_999_999
        assert state.has_token_capacity(0) is True

    def test_under_limit(self):
        """[TM-023 AC-3] Under token limit has capacity."""
        state = BackendState()
        state.tokens_today = 5000
        state.day_reset_at = time.time() + 86400
        assert state.has_token_capacity(6000) is True

    def test_at_limit(self):
        """[TM-023 AC-3] At token limit, has_token_capacity returns False."""
        state = BackendState()
        state.tokens_today = 6000
        state.day_reset_at = time.time() + 86400
        assert state.has_token_capacity(6000) is False


class TestCircuitBreaker:
    def test_fresh_state_circuit_closed(self):
        """[TM-023 AC-4] Fresh state has circuit closed."""
        state = BackendState()
        assert state.is_circuit_open() is False

    def test_single_error_no_trip(self):
        """[TM-023 AC-4] Single error does not trip the circuit."""
        state = BackendState()
        tripped = state.record_error()
        assert tripped is False
        assert state.status == BackendStatus.ERROR
        assert state.consecutive_errors == 1

    def test_three_errors_trips_circuit(self):
        """[TM-023 AC-4] Three consecutive errors trip the circuit."""
        state = BackendState()
        state.record_error()
        state.record_error()
        tripped = state.record_error()
        assert tripped is True
        assert state.status == BackendStatus.CIRCUIT_OPEN
        assert state.is_circuit_open() is True

    def test_circuit_recovers_after_cooldown(self):
        """[TM-023 AC-4] Circuit recovers after cooldown period."""
        state = BackendState(circuit_cooldown=0.01)
        state.record_error()
        state.record_error()
        state.record_error()
        assert state.is_circuit_open() is True
        time.sleep(0.02)
        assert state.is_circuit_open() is False

    def test_errors_outside_window_reset_count(self):
        """[TM-023 AC-4] Errors outside the error window reset the count."""
        state = BackendState(error_window=0.01)
        state.record_error()
        state.record_error()
        time.sleep(0.02)
        tripped = state.record_error()
        assert tripped is False
        assert state.consecutive_errors == 1

    def test_success_resets_consecutive_errors(self):
        """[TM-023 AC-4] Successful request resets consecutive error count."""
        state = BackendState()
        state.record_error()
        state.record_error()
        state.record_success(100, 200, 500.0)
        assert state.consecutive_errors == 0
        assert state.status == BackendStatus.AVAILABLE


class TestRecordRequest:
    def test_increments_daily_and_timestamps(self):
        """[TM-023 AC-5] record_request increments daily count and timestamps."""
        state = BackendState()
        state.record_request()
        assert state.requests_today == 1
        assert len(state.request_timestamps) == 1

    def test_multiple_requests(self):
        """[TM-023 AC-5] Multiple requests accumulate correctly."""
        state = BackendState()
        for _ in range(5):
            state.record_request()
        assert state.requests_today == 5
        assert len(state.request_timestamps) == 5


class TestRecordSuccess:
    def test_updates_tokens_and_latency(self):
        """[TM-023 AC-5] record_success updates tokens and latency."""
        state = BackendState()
        state.record_success(100, 200, 1000.0)
        assert state.tokens_today == 300
        assert state.avg_latency_ms == 1000.0

    def test_ema_latency_smoothing(self):
        """[TM-023 AC-5] Latency uses EMA smoothing."""
        state = BackendState(latency_alpha=0.5)
        state.record_success(0, 0, 100.0)
        state.record_success(0, 0, 200.0)
        assert state.avg_latency_ms == 150.0

    def test_negative_tokens_asserts(self):
        """[TM-023 AC-5] Negative token values raise AssertionError."""
        state = BackendState()
        with pytest.raises(AssertionError):
            state.record_success(-1, 0, 100.0)


class TestDayReset:
    def test_reset_zeroes_counters(self):
        """[TM-023 AC-2] Day reset zeroes request and token counters."""
        state = BackendState()
        state.requests_today = 500
        state.tokens_today = 100_000
        state.day_reset_at = time.time() - 1
        state.has_rpd_capacity(1000)
        assert state.requests_today == 0
        assert state.tokens_today == 0

    def test_reset_sets_future_boundary(self):
        """[TM-023 AC-2] Day reset sets next boundary in the future."""
        state = BackendState()
        state.day_reset_at = time.time() - 1
        state.has_rpd_capacity(1000)
        assert state.day_reset_at > time.time()
