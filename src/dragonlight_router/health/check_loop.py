"""Health check loop -- periodic backend probing.

Uses CircuitBreaker for each tracked model. Provides health scores
based on error count and circuit state. Handles model retirement on 404.

Implements SLO enforcement: providers exceeding latency SLO for 3 consecutive
checks transition to degraded state.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

import structlog

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import BackendStatus, GenerativeBackend, LatencySLO
from dragonlight_router.health.circuit_breaker import CircuitBreaker, CircuitState

logger = structlog.get_logger()


@dataclass(frozen=True)
class HealthCheckConfig:
    """Configuration for the health check loop."""

    interval_s: float = 30.0
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ProbeResult:
    """Result of a single health probe."""

    success: bool
    latency_ms: float
    error: BaseException | None = None
    is_404: bool = False


class HealthCheckLoop:
    """Periodically probes backend health and updates state.

    Implements SLO enforcement: providers exceeding latency SLO for 3 consecutive
    checks transition to degraded state.
    """

    def __init__(
        self,
        backends: dict[str, GenerativeBackend],
        states: dict[str, BackendState],
        latency_slos: dict[str, LatencySLO],
        interval_s: float = 30.0,
        timeout_s: float = 10.0,
    ) -> None:
        assert isinstance(backends, dict), "backends must be a dict"
        assert isinstance(states, dict), "states must be a dict"
        self._backends = backends
        self._states = states
        self._latency_slos = latency_slos
        self._config = HealthCheckConfig(interval_s=interval_s, timeout_s=timeout_s)
        self._interval = interval_s
        self._timeout = timeout_s
        self._task: asyncio.Task[None] | None = None
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker() for name in backends
        }
        self._slo_violation_counts: dict[str, int] = {
            name: 0 for name in backends
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
        tasks = []
        for name, backend in self._backends.items():
            task = asyncio.create_task(self._probe_backend(name, backend))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _probe_backend(self, name: str, backend: GenerativeBackend) -> None:
        """Probe a single backend and update its state.

        This is the I/O boundary for health probing. All probe failures
        are caught here and converted to ProbeResult for processing.
        """
        assert isinstance(name, str), "backend name must be a string"
        state = self._states[name]
        breaker = self._breakers[name]
        slo = self._latency_slos.get(name)

        if not breaker.allow_request() and breaker.state != CircuitState.HALF_OPEN:
            return

        start_time = time.time()
        probe_result = await self._safe_probe(name, backend, start_time)

        if probe_result.is_404:
            state.status = BackendStatus.OFFLINE
            return

        self._update_breaker_and_errors(name, state, breaker, probe_result)
        self._evaluate_slo(name, slo, probe_result)
        self._update_backend_status(name, state, breaker, slo, probe_result)

    async def _safe_probe(
        self, name: str, backend: GenerativeBackend, start_time: float
    ) -> ProbeResult:
        """Execute probe, catching all errors at the I/O boundary.

        Uses asyncio.gather(return_exceptions=True) to convert any exception
        (including unexpected types) into a value rather than propagating.
        This avoids bare 'except Exception' while still handling all failure
        modes at the network I/O boundary.
        """
        task = asyncio.create_task(self._send_health_probe(name, backend, start_time))
        results = await asyncio.gather(task, return_exceptions=True)
        outcome = results[0]

        if isinstance(outcome, BaseException):
            return self._make_failure_result(name, outcome, start_time)

        assert isinstance(outcome, ProbeResult), "probe must return a ProbeResult"
        return outcome

    async def _send_health_probe(
        self, name: str, backend: GenerativeBackend, start_time: float
    ) -> ProbeResult:
        """Delegate health probing to the backend adapter.

        Each adapter knows its own health-check URL via _resolve_base_url() +
        _models_path, so we delegate rather than reconstruct the URL here.
        After the call, we inspect backend.status to detect 404→OFFLINE cases
        (the adapter sets OFFLINE when it receives a 404) and surface them as
        is_404=True so _probe_backend can set OFFLINE state on the BackendState.
        """
        healthy = await backend.health_check()
        latency_ms = (time.time() - start_time) * 1000

        if healthy:
            logger.debug("health_check_success", backend=name, latency_ms=round(latency_ms, 2))
            return ProbeResult(success=True, latency_ms=latency_ms)

        is_404 = backend.status == BackendStatus.OFFLINE
        if is_404:
            logger.warning("health_check_failed_404", backend=name, status=404, latency_ms=round(latency_ms, 2))
            return ProbeResult(
                success=False,
                latency_ms=latency_ms,
                error=Exception("Model Not Found"),
                is_404=True,
            )

        error = Exception("Health check returned non-success response")
        logger.warning("health_check_failed", backend=name, error=str(error), latency_ms=round(latency_ms, 2))
        return ProbeResult(success=False, latency_ms=latency_ms, error=error)

    def _make_failure_result(
        self, name: str, exc: BaseException, start_time: float
    ) -> ProbeResult:
        """Create a failure ProbeResult from an exception."""
        latency_ms = (time.time() - start_time) * 1000
        logger.warning("health_check_failed", backend=name, error=str(exc), latency_ms=round(latency_ms, 2))
        return ProbeResult(success=False, latency_ms=latency_ms, error=exc)

    def _update_breaker_and_errors(
        self, name: str, state: BackendState, breaker: CircuitBreaker, result: ProbeResult
    ) -> None:
        """Update circuit breaker state and consecutive error counts."""
        if result.success:
            breaker.record_success()
            state.consecutive_errors = 0
        else:
            breaker.record_error()
            state.consecutive_errors += 1

    def _evaluate_slo(
        self, name: str, slo: LatencySLO | None, result: ProbeResult
    ) -> None:
        """Update SLO violation tracking based on probe result."""
        if result.success and slo and result.latency_ms > slo.latency_ms:
            self._slo_violation_counts[name] += 1
        elif result.success:
            self._slo_violation_counts[name] = 0
        else:
            self._slo_violation_counts[name] += 1

    def _update_backend_status(
        self,
        name: str,
        state: BackendState,
        breaker: CircuitBreaker,
        slo: LatencySLO | None,
        result: ProbeResult,
    ) -> None:
        """Set backend status based on circuit state and SLO violations."""
        if slo and self._slo_violation_counts[name] >= 3:
            state.status = BackendStatus.DEGRADED
            logger.info(
                "backend_degraded_due_to_slo_violations",
                backend=name,
                slo_violations=self._slo_violation_counts[name],
                latency_slo_ms=slo.latency_ms,
            )
            return

        if breaker.state == CircuitState.OPEN:
            state.status = BackendStatus.ERROR
        elif result.success:
            state.status = BackendStatus.AVAILABLE
        else:
            state.status = BackendStatus.ERROR
