"""Unit tests for the HealthCheckLoop class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.dragonlight_router.health.check_loop import HealthCheckLoop, CircuitBreaker
from dragonlight_router.core.types import BackendStatus, GenerativeBackend
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
def health_check_loop(mock_backends, mock_states):
    """Create a HealthCheckLoop instance."""
    return HealthCheckLoop(
        backends=mock_backends,
        states=mock_states,
        interval_s=0.1,  # Short interval for testing
        timeout_s=1.0,
    )


def test_loop_initialization(health_check_loop, mock_backends, mock_states):
    """Test that the loop initializes correctly."""
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
    """Test that the loop can be started and stopped."""
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
    """Test that _probe_all_backends calls _probe_backend for each backend."""
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
    """Test that _probe_backend updates state on success."""
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
    """Test that _probe_backend updates state on failure."""
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
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = Exception("Probe failed")
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
    """Test that consecutive errors trip the circuit breaker."""
    name = "backend1"
    backend = mock_backends[name]
    state = mock_states[name]
    breaker = health_check_loop._breakers[name]

    # Set up state with 2 errors already (so next error will trip)
    state.consecutive_errors = 2
    state.status = BackendStatus.AVAILABLE  # Actually, after 2 errors it might be ERROR, but let's set
    breaker._state = CircuitState.CLOSED
    breaker._error_timestamps = [1000.0, 1000.0]  # Two errors

    # Mock the probe to raise an exception during the sleep
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = Exception("Probe failed")
        await health_check_loop._probe_backend(name, backend)

    # After this failure: consecutive errors = 3, status ERROR, circuit OPEN
    assert state.consecutive_errors == 3
    assert state.status == BackendStatus.ERROR
    assert breaker.state == CircuitState.OPEN
    assert len(breaker._error_timestamps) == 3
    assert breaker._opened_at > 1000.0


@pytest.mark.asyncio
async def test_loop_respects_interval(health_check_loop):
    """Test that the loop waits for the specified interval between cycles."""
    # Mock _probe_all_backends to do nothing but track calls
    with patch.object(
        health_check_loop, "_probe_all_backends", new_callable=AsyncMock
    ) as mock_probe:
        # Mock asyncio.sleep to track when it's called and with what argument
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Start the loop
            await health_check_loop.start()
            # Give it a moment to run one iteration
            await asyncio.sleep(0.2)  # Wait for ~2 intervals (0.1s each)
            # Stop the loop
            await health_check_loop.stop()

            # Check that _probe_all_backends was called at least twice
            assert mock_probe.call_count >= 2
            # Check that asyncio.sleep was called with the interval
            mock_sleep.assert_any_call(0.1)
            # Ensure it was called multiple times (once per loop iteration)
            assert mock_sleep.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])