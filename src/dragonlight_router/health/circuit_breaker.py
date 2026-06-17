"""Circuit breaker — prevents cascading failures to unhealthy backends.

State machine: CLOSED → OPEN → HALF_OPEN → CLOSED (or back to OPEN).
3 consecutive errors within error_window_s → OPEN for cooldown_s.
After cooldown: HALF_OPEN (allow 1 probe request).
Success in HALF_OPEN → CLOSED. Failure → re-OPEN.

HAZ-009 mitigation: Jittered cooldown prevents synchronized recovery
across breakers tripped simultaneously. Each breaker adds a random
offset (0 to jitter_factor * cooldown_s) so HALF_OPEN probes are
staggered, reducing the risk of correlated flapping.
"""
from __future__ import annotations

import random
import time
from enum import Enum, unique
from typing import Any


@unique
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-model circuit breaker with configurable thresholds.

    HAZ-009 mitigation: jitter_factor adds randomized offset to cooldown
    so breakers tripped simultaneously do not recover in lockstep.
    """

    def __init__(
        self,
        error_threshold: int = 3,
        error_window_s: float = 120.0,
        cooldown_s: float = 60.0,
        jitter_factor: float = 0.25,
    ) -> None:
        """Per-model circuit breaker with configurable thresholds.

        Args:
            error_threshold: Number of errors within window to trip circuit.
            error_window_s: Time window for error accumulation.
            cooldown_s: Base cooldown before HALF_OPEN probe.
            jitter_factor: Random jitter as fraction of cooldown_s (0.0-1.0).
                           HAZ-009: prevents synchronized recovery flapping.
        """
        # Precondition assertions
        assert error_threshold > 0, "error_threshold must be positive"
        assert error_window_s > 0, "error_window_s must be positive"
        assert cooldown_s > 0, "cooldown_s must be positive"
        assert 0.0 <= jitter_factor <= 1.0, "jitter_factor must be in [0.0, 1.0]"

        self._error_threshold = error_threshold
        self._error_window_s = error_window_s
        self._cooldown_s = cooldown_s
        self._jitter_factor = jitter_factor

        self._state = CircuitState.CLOSED
        self._error_timestamps: list[float] = []
        self._opened_at: float = 0.0
        # HAZ-009: jittered cooldown for this specific breaker instance
        self._effective_cooldown_s = self._compute_jittered_cooldown()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _compute_jittered_cooldown(self) -> float:
        """Compute a jittered cooldown duration.

        HAZ-009 mitigation: adds random offset so multiple breakers
        tripped at the same time don't all recover simultaneously.
        """
        jitter = random.uniform(0, self._jitter_factor * self._cooldown_s)
        result = self._cooldown_s + jitter
        assert result >= self._cooldown_s, "jittered cooldown must be >= base cooldown"
        return result

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns True if CLOSED or if cooldown has elapsed (transitions to HALF_OPEN).
        Uses jittered cooldown (HAZ-009) to prevent synchronized recovery.
        """
        if self._state == CircuitState.CLOSED or self._state == CircuitState.HALF_OPEN:
            result = True
        else:
            # OPEN — check if jittered cooldown elapsed (HAZ-009)
            if time.time() >= self._opened_at + self._effective_cooldown_s:
                self._state = CircuitState.HALF_OPEN
                result = True
            else:
                result = False
        assert isinstance(result, bool), "allow_request must return a bool"
        return result

    def record_success(self) -> None:
        """Record a successful request — resets circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._error_timestamps.clear()
        # Postcondition assertions
        assert self._state == CircuitState.CLOSED, "state must be CLOSED after record_success"
        assert len(self._error_timestamps) == 0, "error_timestamps must be empty after record_success"

    def record_error(self) -> None:
        """Record a failed request — may trip the circuit.

        HAZ-009: recomputes jittered cooldown each time the circuit opens
        so repeated flaps do not synchronize across breakers.
        """
        now = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failure during probe — re-open with fresh jitter
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._effective_cooldown_s = self._compute_jittered_cooldown()
            return

        # Prune errors outside the window
        cutoff = now - self._error_window_s
        self._error_timestamps = [t for t in self._error_timestamps if t >= cutoff]
        self._error_timestamps.append(now)

        if len(self._error_timestamps) >= self._error_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._effective_cooldown_s = self._compute_jittered_cooldown()

    def get_state(self) -> dict[str, Any]:
        """Export circuit breaker state for persistence (HAZ-012).

        Returns the state name, opened_at timestamp, and error timestamps
        so OPEN circuit breakers survive process restarts.
        """
        return {
            "state": self._state.value,
            "opened_at": self._opened_at,
            "error_timestamps": list(self._error_timestamps),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore circuit breaker state from persistence (HAZ-012).

        Only restores OPEN state -- CLOSED and HALF_OPEN start fresh.
        Error timestamps are pruned to the current window.
        Uses jittered cooldown (HAZ-009) for restored OPEN state.
        """
        assert isinstance(state, dict), "state must be a dict"
        saved_state_name = state.get("state", "closed")
        if saved_state_name == CircuitState.OPEN.value:
            opened_at = state.get("opened_at", 0.0)
            now = time.time()
            # Use jittered cooldown for consistency with HAZ-009
            if now < opened_at + self._effective_cooldown_s:
                self._state = CircuitState.OPEN
                self._opened_at = opened_at
            else:
                self._state = CircuitState.HALF_OPEN

        saved_timestamps = state.get("error_timestamps", [])
        now = time.time()
        cutoff = now - self._error_window_s
        self._error_timestamps = [t for t in saved_timestamps if t >= cutoff]
