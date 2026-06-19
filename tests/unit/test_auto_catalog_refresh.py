"""Tests for HAZ-008 mitigation — automatic periodic catalog refresh.

Validates that the health check loop supports an on_cycle callback
for automatic catalog refresh, preventing stale catalog routing.

Spec traceability: HAZ-008 (Stale Catalog Routing to Deprecated Models)
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import BackendStatus, LatencySLO
from dragonlight_router.health.check_loop import HealthCheckLoop

pytestmark = pytest.mark.unit


def _make_stub_backend() -> MagicMock:
    """Create a stub GenerativeBackend for testing."""
    backend = MagicMock()
    backend.health_check = AsyncMock(return_value=True)
    backend.status = BackendStatus.AVAILABLE
    return backend


class TestOnCycleCallback:
    """HAZ-008: on_cycle callback fires periodically."""

    @pytest.mark.asyncio
    async def test_on_cycle_called_every_cycle(self):
        """[HAZ-008 AC-1] on_cycle callback fires every cycle by default."""
        callback = AsyncMock()
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=callback,
            on_cycle_interval=1,
        )

        # Run exactly 3 cycles
        await loop._probe_all_backends()
        loop._cycle_count += 1
        await loop._invoke_on_cycle()

        await loop._probe_all_backends()
        loop._cycle_count += 1
        await loop._invoke_on_cycle()

        await loop._probe_all_backends()
        loop._cycle_count += 1
        await loop._invoke_on_cycle()

        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_on_cycle_interval_respected(self):
        """[HAZ-008 AC-2] on_cycle callback fires only every N cycles."""
        callback = AsyncMock()
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=callback,
            on_cycle_interval=3,
        )

        # Simulate 6 cycles
        for _ in range(6):
            loop._cycle_count += 1
            await loop._invoke_on_cycle()

        # Should fire on cycles 3 and 6
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_on_cycle_none_is_noop(self):
        """[HAZ-008 AC-3] No callback configured means no invocation."""
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=None,
        )

        loop._cycle_count = 1
        # Should not raise
        await loop._invoke_on_cycle()

    @pytest.mark.asyncio
    async def test_on_cycle_failure_does_not_crash_loop(self):
        """[HAZ-008 AC-4] Callback failure does not crash the health check loop."""
        callback = AsyncMock(side_effect=RuntimeError("catalog refresh failed"))
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=callback,
            on_cycle_interval=1,
        )

        # Should not raise
        loop._cycle_count = 1
        await loop._invoke_on_cycle()
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_on_cycle_os_error_handled(self):
        """[HAZ-008 AC-4] OSError in callback is caught gracefully."""
        callback = AsyncMock(side_effect=OSError("disk full"))
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=callback,
            on_cycle_interval=1,
        )

        loop._cycle_count = 1
        await loop._invoke_on_cycle()
        # No exception propagated

    def test_on_cycle_interval_validation(self):
        """[HAZ-008 AC-5] on_cycle_interval must be positive."""
        backend = _make_stub_backend()
        state = BackendState()

        with pytest.raises(AssertionError, match="on_cycle_interval must be positive"):
            HealthCheckLoop(
                backends={"b1": backend},
                states={"b1": state},
                latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
                on_cycle_interval=0,
            )

    @pytest.mark.asyncio
    async def test_cycle_count_increments_in_run_loop(self):
        """[HAZ-008 AC-6] _run_loop increments cycle_count and invokes callback."""
        callback = AsyncMock()
        backend = _make_stub_backend()
        state = BackendState()

        loop = HealthCheckLoop(
            backends={"b1": backend},
            states={"b1": state},
            latency_slos={"b1": LatencySLO(latency_ms=5000.0)},
            interval_s=0.01,
            on_cycle=callback,
            on_cycle_interval=1,
        )

        # Start loop and let it run a few cycles
        task = asyncio.create_task(loop._run_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        # _run_loop catches CancelledError internally via try/except break,
        # so the task completes normally rather than raising.
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert loop._cycle_count > 0
        assert callback.call_count > 0
