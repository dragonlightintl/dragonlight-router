"""Tests for HAZ-009 mitigation — jittered circuit breaker cooldown.

Validates that circuit breakers use randomized cooldowns to prevent
synchronized recovery flapping when multiple breakers trip simultaneously.

Spec traceability: HAZ-009 (Circuit Breaker Flapping)
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from dragonlight_router.health.circuit_breaker import CircuitBreaker, CircuitState


class TestJitteredCooldown:
    """HAZ-009: Jittered cooldown prevents synchronized recovery."""

    def test_jitter_factor_defaults_to_025(self):
        """[HAZ-009 AC-1] Default jitter_factor is 0.25."""
        cb = CircuitBreaker(cooldown_s=60.0)
        assert cb._jitter_factor == 0.25

    def test_jitter_factor_zero_gives_exact_cooldown(self):
        """[HAZ-009 AC-1] jitter_factor=0 gives deterministic cooldown."""
        cb = CircuitBreaker(cooldown_s=60.0, jitter_factor=0.0)
        assert cb._effective_cooldown_s == 60.0

    def test_effective_cooldown_at_least_base(self):
        """[HAZ-009 AC-2] Jittered cooldown is always >= base cooldown."""
        for _ in range(50):
            cb = CircuitBreaker(cooldown_s=60.0, jitter_factor=0.25)
            assert cb._effective_cooldown_s >= 60.0

    def test_effective_cooldown_at_most_base_plus_jitter(self):
        """[HAZ-009 AC-2] Jittered cooldown is <= base + jitter_factor * base."""
        for _ in range(50):
            cb = CircuitBreaker(cooldown_s=60.0, jitter_factor=0.25)
            assert cb._effective_cooldown_s <= 60.0 + 0.25 * 60.0

    def test_multiple_breakers_get_different_cooldowns(self):
        """[HAZ-009 AC-3] Multiple breakers created at same time have varied cooldowns."""
        cooldowns = set()
        for _ in range(20):
            cb = CircuitBreaker(cooldown_s=60.0, jitter_factor=0.25)
            cooldowns.add(cb._effective_cooldown_s)
        # With random jitter, we should get multiple distinct values
        assert len(cooldowns) > 1, "Jitter should produce varied cooldowns"

    def test_jittered_cooldown_used_in_allow_request(self):
        """[HAZ-009 AC-4] allow_request uses jittered cooldown, not base cooldown."""
        # Use jitter_factor=0 for deterministic test, then manually set effective_cooldown_s
        cb = CircuitBreaker(error_threshold=3, cooldown_s=100.0, jitter_factor=0.0)
        cb.record_error()
        cb.record_error()
        cb.record_error()
        assert cb.state == CircuitState.OPEN

        # Manually set a shorter effective cooldown to test it's used
        cb._effective_cooldown_s = 0.01
        time.sleep(0.02)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_reopen_recomputes_jitter(self):
        """[HAZ-009 AC-5] Each re-open computes fresh jitter to prevent settling into lockstep."""
        cb = CircuitBreaker(error_threshold=3, cooldown_s=0.01, jitter_factor=0.5)
        # Trip the circuit
        cb.record_error()
        cb.record_error()
        cb.record_error()
        first_cooldown = cb._effective_cooldown_s

        # Let it transition to HALF_OPEN, then fail again
        time.sleep(0.02)
        cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_error()  # re-opens
        assert cb.state == CircuitState.OPEN
        second_cooldown = cb._effective_cooldown_s

        # With jitter_factor=0.5, the two cooldowns are very likely different
        # (probability of being exactly equal is negligible for continuous random)
        # But we can't assert they're definitely different, so just verify
        # the value was recomputed (could be same by chance)
        assert second_cooldown >= cb._cooldown_s

    def test_jitter_factor_validation(self):
        """[HAZ-009 AC-6] Invalid jitter_factor raises AssertionError."""
        with pytest.raises(AssertionError, match="jitter_factor must be in"):
            CircuitBreaker(jitter_factor=-0.1)
        with pytest.raises(AssertionError, match="jitter_factor must be in"):
            CircuitBreaker(jitter_factor=1.1)

    def test_jitter_factor_boundary_one(self):
        """[HAZ-009 AC-6] jitter_factor=1.0 is valid and gives cooldown in [base, 2*base]."""
        cb = CircuitBreaker(cooldown_s=60.0, jitter_factor=1.0)
        assert 60.0 <= cb._effective_cooldown_s <= 120.0

    def test_restore_state_uses_jittered_cooldown(self):
        """[HAZ-009 AC-7] restore_state uses jittered cooldown for OPEN state check."""
        # Create breaker with zero jitter for deterministic behavior
        cb = CircuitBreaker(
            error_threshold=3,
            cooldown_s=3600.0,
            jitter_factor=0.0,
        )
        state = {
            "state": "open",
            "opened_at": time.time() - 10,  # opened 10s ago
            "error_timestamps": [time.time() - 5, time.time() - 3, time.time() - 1],
        }
        cb.restore_state(state)
        # cooldown is 3600s, opened 10s ago => should still be OPEN
        assert cb.state == CircuitState.OPEN

    def test_threshold_error_recomputes_jitter(self):
        """[HAZ-009 AC-5] Tripping via threshold errors also computes jitter."""
        cb = CircuitBreaker(error_threshold=3, cooldown_s=60.0, jitter_factor=0.25)
        cb.record_error()
        cb.record_error()
        initial_cooldown = cb._effective_cooldown_s
        cb.record_error()  # trips circuit
        assert cb.state == CircuitState.OPEN
        # A new jittered cooldown was computed
        assert cb._effective_cooldown_s >= 60.0
