"""Unit tests for the HealthCheckLoop class.

Spec traceability: TM-008 (Health check loop)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.dragonlight_router.health.check_loop import HealthCheckLoop, CircuitBreaker
from dragonlight_router.core.types import BackendStatus, GenerativeBackend, LatencySLO
from dragonlight_router.core.state import BackendState
from dragonlight_router.health.circuit_breaker import CircuitState


@pytest.fixture
def mock_backends():
    """Create mock backends."""
    return {
        "backend1": MagicMock(spec=GenerativeBackend),
        "backend2": MagicMock(spec=GenerativeBackend),
    }


@pytest.fixture
def mock_states():
    """Create mock backend states."""
    return {
        "backend1": BackendState(),
        "backend2": BackendState(),
    }


@pytest.fixture
def mock_latency_slos():
    """Create mock latency SLOs."""
    return {
        "backend1": LatencySLO(latency_ms=50.0),
        "backend2": LatencySLO(latency_ms=50.0),
    }


@pytest.fixture
def health_check_loop(mock_backends, mock_states, mock_latency_slos):
    """Create a HealthCheckLoop instance."""
    return HealthCheckLoop(
        backends=mock_backends,
        states=mock_states,
        latency_slos=mock_latency_slos,
        interval_s=0.1,  # Short interval for testing
        timeout_s=1.0,
    )


def test_loop_initialization(health_check_loop, mock_backends, mock_states):
    """[TM-008 AC-1] Loop initializes with correct backends, states, and breakers."""
    assert health_check_loop._backends == mock_backends
    assert health_check_loop._states == mock_states
    assert health_check_loop._interval == 0.1
    assert health_check_loop._timeout == 1.0
    assert health_check_loop._task is None
    assert len(health_check_loop._breakers) == 2
    for name in mock_backends:
        assert name in health_check_loop._breakers
        assert isinstance(health_check_loop._breakers[name], CircuitBreaker)


@pytest.mark.asyncio
async def test_loop_start_stop(health_check_loop):
    """[TM-008 AC-2] Loop can be started and stopped cleanly."""
    # Initially not running
    assert health_check_loop._task is None

    # Start the loop
    await health_check_loop.start()
    assert health_check_loop._task is not None
    assert not health_check_loop._task.done()

    # Stop the loop
    await health_check_loop.stop()
    assert health_check_loop._task is None


@pytest.mark.asyncio
async def test_probe_all_backends_calls_probe_backend(
    health_check_loop, mock_backends
):
    """[TM-008 AC-3] _probe_all_backends calls _probe_backend for each backend."""
    with patch.object(
        health_check_loop, "_probe_backend", new_callable=AsyncMock
    ) as mock_probe:
        await health_check_loop._probe_all_backends()
        # Should be called once for each backend
        assert mock_probe.call_count == len(mock_backends)
        for name in mock_backends:
            mock_probe.assert_any_call(name, mock_backends[name])


@pytest.mark.asyncio
async def test_probe_backend_success_updates_state(
    health_check_loop, mock_backends, mock_states
):
    """[TM-008 AC-4] Successful probe resets errors, sets AVAILABLE, closes circuit."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Set up initial state to simulate some errors
    state.consecutive_errors = 2
    state.status = BackendStatus.ERROR
    # Force circuit breaker to OPEN state by adding errors
    breaker._error_timestamps = [1000.0, 1000.0, 1000.0]  # Three errors
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 1000.0

    # Mock the probe to succeed (no exception)
    # We need to set up the session to avoid the RuntimeError
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    mock_response = MagicMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    # Mock the session.get to return an async context manager
    mock_session_context = MagicMock()
    mock_session_context.__aenter__.return_value = mock_response
    mock_session_context.__aexit__.return_value = None
    health_check_loop._session.get.return_value = mock_session_context

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await health_check_loop._probe_backend(name, backend)

    # After success: consecutive errors reset, status AVAILABLE, breaker CLOSED
    assert state.consecutive_errors == 0
    assert state.status == BackendStatus.AVAILABLE
    assert breaker.state == CircuitState.CLOSED
    assert len(breaker._error_timestamps) == 0


@pytest.mark.asyncio
async def test_probe_backend_failure_increments_errors_and_updates_status(
    health_check_loop, mock_backends, mock_states
):
    """[TM-008 AC-5] Failed probe increments error count and sets ERROR status."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Reset to clean state
    state.consecutive_errors = 0
    state.status = BackendStatus.AVAILABLE
    breaker._state = CircuitState.CLOSED
    breaker._error_timestamps = []

    # Mock the probe to raise an exception during the sleep
    # We need to set up the session to avoid the RuntimeError
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    # Make the session.get raise an exception
    health_check_loop._session.get.side_effect = Exception("Probe failed")

    # Mock asyncio.sleep to avoid waiting (but the exception will be raised in the sleep?
    # Actually, the exception is raised in the session.get call, so we don't need to sleep)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await health_check_loop._probe_backend(name, backend)

    # After failure: consecutive errors incremented, status ERROR
    assert state.consecutive_errors == 1
    assert state.status == BackendStatus.ERROR
    # Circuit breaker should have recorded the error
    assert len(breaker._error_timestamps) == 1
    # State should still be CLOSED (threshold is 3)
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_probe_backend_failure_opens_circuit(
    health_check_loop, mock_backends, mock_states
):
    """[TM-008 AC-5] Consecutive errors trip the circuit breaker to OPEN."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Set up state with 2 errors already (so next error will trip)
    state.consecutive_errors = 2
    state.status = BackendStatus.AVAILABLE
    breaker._state = CircuitState.CLOSED
    now = time.time()
    breaker._error_timestamps = [now, now]  # Two recent errors within the window

    # Mock the probe to raise an exception during the sleep
    # We need to set up the session to avoid the RuntimeError
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    # Make the session.get raise an exception
    health_check_loop._session.get.side_effect = Exception("Probe failed")

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await health_check_loop._probe_backend(name, backend)

    # After this failure: consecutive errors = 3, status ERROR, circuit OPEN
    assert state.consecutive_errors == 3
    assert state.status == BackendStatus.ERROR
    assert breaker.state == CircuitState.OPEN
    assert len(breaker._error_timestamps) == 3
    assert breaker._opened_at > 1000.0


@pytest.mark.asyncio
async def test_loop_respects_interval(health_check_loop):
    """[TM-008 AC-2] Loop waits for the specified interval between cycles."""
    # Mock _probe_all_backends to track calls without doing real HTTP probing.
    # Do NOT patch asyncio.sleep — the loop interval is 0.1s, so we let it
    # run naturally and wait long enough for at least 2 full iterations.
    with patch.object(
        health_check_loop, "_probe_all_backends", new_callable=AsyncMock
    ) as mock_probe:
        # Start the loop
        await health_check_loop.start()
        # Wait for at least 2 full iterations (interval=0.1s, so 0.35s is safe)
        await asyncio.sleep(0.35)
        # Stop the loop
        await health_check_loop.stop()

        # Should have run at least 2 probe cycles in 0.35s with 0.1s interval
        assert mock_probe.call_count >= 2


@pytest.mark.asyncio
async def test_slo_violation_transitions_to_degraded_via_latency(health_check_loop, mock_backends, mock_states):
    """[TM-008 AC-6] Exceeding latency SLO for 3 consecutive checks transitions to DEGRADED."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Set up an SLO of 50ms
    health_check_loop._latency_slos[name] = LatencySLO(latency_ms=50.0)

    # Reset state to clean
    state.consecutive_errors = 0
    state.status = BackendStatus.AVAILABLE
    breaker._state = CircuitState.CLOSED
    breaker._error_timestamps = []
    health_check_loop._slo_violation_counts[name] = 0

    # Mock the probe to succeed with latency 60ms (> SLO of 50ms)
    # Simulate 60ms latency by making time.time return values 60ms apart.
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    mock_response = MagicMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_session_context = MagicMock()
    mock_session_context.__aenter__.return_value = mock_response
    mock_session_context.__aexit__.return_value = None
    health_check_loop._session.get.return_value = mock_session_context

    base_time = 1000.0

    def fake_time():
        # Each call advances by 60ms. The probe measures:
        #   start_time = time.time()         <- call N
        #   latency_ms = (time.time() - start_time) * 1000  <- call N+1
        # With 0.060s per call: latency = 60ms > 50ms SLO.
        fake_time.calls += 1
        return base_time + (fake_time.calls * 0.060)  # 60ms per call

    fake_time.calls = 0

    with patch("dragonlight_router.health.check_loop.time.time", side_effect=fake_time):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # First violation
            await health_check_loop._probe_backend(name, backend)
            assert health_check_loop._slo_violation_counts[name] == 1
            assert state.status == BackendStatus.AVAILABLE  # Not yet degraded

            # Second violation
            await health_check_loop._probe_backend(name, backend)
            assert health_check_loop._slo_violation_counts[name] == 2
            assert state.status == BackendStatus.AVAILABLE

            # Third violation -> should transition to degraded
            await health_check_loop._probe_backend(name, backend)
            assert health_check_loop._slo_violation_counts[name] == 3
            assert state.status == BackendStatus.DEGRADED


@pytest.mark.asyncio
async def test_slo_violation_transitions_to_degraded_via_failure(health_check_loop, mock_backends, mock_states):
    """[TM-008 AC-6] Three consecutive failed checks transition to DEGRADED."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Set up an SLO (any value, since we are testing failure)
    health_check_loop._latency_slos[name] = LatencySLO(latency_ms=50.0)

    # Reset state to clean
    state.consecutive_errors = 0
    state.status = BackendStatus.AVAILABLE
    breaker._state = CircuitState.CLOSED
    breaker._error_timestamps = []
    health_check_loop._slo_violation_counts[name] = 0

    # Mock the probe to fail (exception during sleep)
    # We need to set up the session to avoid the RuntimeError
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    # Make the session.get raise an exception
    health_check_loop._session.get.side_effect = Exception("Probe failed")

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # First failure
        await health_check_loop._probe_backend(name, backend)
        assert health_check_loop._slo_violation_counts[name] == 1
        assert state.status == BackendStatus.ERROR  # Failed, circuit still CLOSED

        # Second failure
        await health_check_loop._probe_backend(name, backend)
        assert health_check_loop._slo_violation_counts[name] == 2
        assert state.status == BackendStatus.ERROR  # Failed, circuit still CLOSED

        # Third failure -> SLO violations >= 3, should transition to degraded
        await health_check_loop._probe_backend(name, backend)
        assert health_check_loop._slo_violation_counts[name] == 3
        assert state.status == BackendStatus.DEGRADED


@pytest.mark.asyncio
async def test_health_check_failures_do_not_crash_loop(health_check_loop, mock_backends):
    """[TM-008 AC-7] Health check failures do not crash the loop."""
    name = "backend1"
    backend = mock_backends[name]

    # We need to set up the session to avoid the RuntimeError
    health_check_loop._session = MagicMock(spec=aiohttp.ClientSession)
    # Make the session.get raise an exception
    health_check_loop._session.get.side_effect = Exception("Probe failed")

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Start the loop
        await health_check_loop.start()
        # Give it a moment to run several iterations
        await asyncio.sleep(0.5)  # Wait for 5 intervals (0.1s each)
        # The loop should still be running (not crashed)
        assert health_check_loop._task is not None
        assert not health_check_loop._task.done()
        # Stop the loop
        await health_check_loop.stop()

    # The loop should have stopped without raising an exception
    assert health_check_loop._task is None


if __name__ == "__main__":
    pytest.main([__file__])