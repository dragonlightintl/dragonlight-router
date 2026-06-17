"""Circuit breaker — prevents cascading failures to unhealthy backends.

State machine: CLOSED → OPEN → HALF_OPEN → CLOSED (or back to OPEN).
3 consecutive errors within error_window_s → OPEN for cooldown_s.
After cooldown: HALF_OPEN (allow 1 probe request).
Success in HALF_OPEN → CLOSED. Failure → re-OPEN.
"""
from __future__ import annotations

import time
from enum import Enum, unique


@unique
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-model circuit breaker with configurable thresholds."""

    def __init__(
        self,
        error_threshold: int = 3,
        error_window_s: float = 120.0,
        cooldown_s: float = 60.0,
    ) -> None:
        """Per-model circuit breaker with configurable thresholds."""
        # Precondition assertions
        assert error_threshold > 0, "error_threshold must be positive"
        assert error_window_s > 0, "error_window_s must be positive"
        assert cooldown_s > 0, "cooldown_s must be positive"

        self._error_threshold = error_threshold
        self._error_window_s = error_window_s
        self._cooldown_s = cooldown_s

        self._state = CircuitState.CLOSED
        self._error_timestamps: list[float] = []
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns True if CLOSED or if cooldown has elapsed (transitions to HALF_OPEN).
        """
        if self._state == CircuitState.CLOSED or self._state == CircuitState.HALF_OPEN:
            result = True
        else:
            # OPEN — check if cooldown elapsed
            if time.time() >= self._opened_at + self._cooldown_s:
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
        """Record a failed request — may trip the circuit."""
        now = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failure during probe — re-open
            self._state = CircuitState.OPEN
            self._opened_at = now
            return

        # Prune errors outside the window
        cutoff = now - self._error_window_s
        self._error_timestamps = [t for t in self._error_timestamps if t >= cutoff]
        self._error_timestamps.append(now)

        if len(self._error_timestamps) >= self._error_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now

    def get_state(self) -> dict:
        """Export circuit breaker state for persistence (HAZ-012).

        Returns the state name, opened_at timestamp, and error timestamps
        so OPEN circuit breakers survive process restarts.
        """
        return {
            "state": self._state.value,
            "opened_at": self._opened_at,
            "error_timestamps": list(self._error_timestamps),
        }

    def restore_state(self, state: dict) -> None:
        """Restore circuit breaker state from persistence (HAZ-012).

        Only restores OPEN state -- CLOSED and HALF_OPEN start fresh.
        Error timestamps are pruned to the current window.
        """
        assert isinstance(state, dict), "state must be a dict"
        saved_state_name = state.get("state", "closed")
        if saved_state_name == CircuitState.OPEN.value:
            opened_at = state.get("opened_at", 0.0)
            now = time.time()
            if now < opened_at + self._cooldown_s:
                self._state = CircuitState.OPEN
                self._opened_at = opened_at
            else:
                self._state = CircuitState.HALF_OPEN

        saved_timestamps = state.get("error_timestamps", [])
        now = time.time()
        cutoff = now - self._error_window_s
        self._error_timestamps = [t for t in saved_timestamps if t >= cutoff]
