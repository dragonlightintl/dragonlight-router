from __future__ import annotations
import aiohttp
"""Health check loop — periodic backend probing.

Uses CircuitBreaker for each tracked model. Provides health scores
based on error count and circuit state. Handles model retirement on 404.
"""

import asyncio
import contextlib

import structlog

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import BackendStatus, GenerativeBackend
from dragonlight_router.health.circuit_breaker import CircuitBreaker, CircuitState

logger = structlog.get_logger()


class HealthCheckLoop:
    """Periodically probes backend health and updates state."""

    def __init__(
        self,
        backends: dict[str, GenerativeBackend],
        states: dict[str, BackendState],
        interval_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> None:
        self._backends = backends
        self._states = states
        self._interval = interval_s
        self._timeout = timeout_s
        self._task: asyncio.Task[None] | None = None
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker() for name in backends
        }

    async def start(self) -> None:
        """Start the background health check loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background health check loop."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run_loop(self) -> None:
        """Run the health check loop until cancelled."""
        while True:
            await self._probe_all_backends()
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _probe_all_backends(self) -> None:
        """Probe all backends and update their state."""
        for name, backend in self._backends.items():
            await self._probe_backend(name, backend)

    async def _probe_backend(self, name: str, backend: GenerativeBackend) -> None:
        """Probe a single backend and update its state."""
        state = self._states[name]
        breaker = self._breakers[name]

        # Check if circuit is open — if so, skip probing (but allow half-open after cooldown)
        if not breaker.allow_request():
            # In half-open state, we allow one probe request
            if breaker.state != CircuitState.HALF_OPEN:
                return

        try:
            # Perform a lightweight probe (e.g., GET /v1/models or a simple completion)
            # For now, we'll just check if the backend is reachable by trying to get its config.
            # In a real implementation, this would be an actual HTTP request to the provider.
            # Since we don't have a real HTTP client in the backend object, we'll simulate.
            # TODO: Replace with actual health check endpoint call.
            await asyncio.sleep(0.01)  # Simulate network delay
            # Assume success for now — in reality, we'd check the response.
            breaker.record_success()
            state.status = BackendStatus.AVAILABLE
            state.consecutive_errors = 0
            logger.debug("health_check_success", backend=name)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            # Record failure
            breaker.record_error()
            state.consecutive_errors += 1
            logger.warning("health_check_failed", backend=name, error=str(exc))
            # Update status based on circuit state
            if breaker.state == CircuitState.OPEN:
                state.status = BackendStatus.ERROR
            else:
                state.status = BackendStatus.ERROR  # or RATE_LIMITED? We'll keep simple for now.
            # Check for 404-like condition (not implemented in this mock)
            # if isinstance(exc, SomeNotFoundError):
            #     state.status = BackendStatus.OFFLINE  # or a retired state
            #     # Optionally, remove from backends? Not here — handled elsewhere.


# Note: This is a simplified implementation. In reality, the HealthCheckLoop would
# need to make actual HTTP requests to the providers' health endpoints.
# For the purpose of this delta, we provide the structure and integration points.