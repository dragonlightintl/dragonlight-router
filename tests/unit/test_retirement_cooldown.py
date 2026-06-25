"""Tests for TTL-based model retirement and cooldown logic.

Verifies that transient HTTP failures (403, 429, 5xx) use time-limited
suspensions while truly permanent failures (404, 410) use permanent
retirement. Models must return to the pool after their cooldown expires.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from dragonlight_router.core.types import Ok
from dragonlight_router.health.tracker import HealthTracker

pytestmark = pytest.mark.unit


class TestPermanentRetirement:
    """404 and 410 errors permanently retire a model."""

    def test_404_retires_permanently(self):
        """Model receiving 404 stays retired indefinitely."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        assert ht.is_retired("m1") is True
        assert ht.is_available("m1") is False

    def test_410_retires_permanently(self):
        """Model receiving 410 (Gone) stays retired indefinitely."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=410)
        assert ht.is_retired("m1") is True
        assert ht.is_available("m1") is False

    def test_404_does_not_expire(self):
        """404 retirement does not expire even after long time."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=404)
        # Simulate time far in the future
        with patch("dragonlight_router.health.tracker.time") as mock_time:
            mock_time.time.return_value = time.time() + 999999
            assert ht.is_retired("m1") is True


class TestTransientCooldown403:
    """403 errors use temporary suspension with TTL-based cooldown."""

    def test_403_suspends_model(self):
        """403 suspends the model (not permanent retirement)."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert "m1" in ht._suspended
        assert "m1" not in ht._retired

    def test_403_model_unavailable_during_cooldown(self):
        """Model is unavailable during 403 cooldown period."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.is_available("m1") is False
        assert ht.is_retired("m1") is True  # is_retired checks both

    def test_403_model_returns_after_cooldown(self):
        """Model returns to pool after 403 cooldown expires."""
        ht = HealthTracker(suspend_ttl_403_s=10.0)
        ht.record_error("m1", http_status=403)
        assert ht.is_available("m1") is False

        # Fast-forward past the cooldown
        ht._suspended["m1"] = time.time() - 11.0
        assert ht.is_retired("m1") is False
        # Circuit breaker may still be tripped from the error, so check
        # that the suspension itself has expired
        assert "m1" not in ht._suspended

    def test_403_score_zero_during_cooldown(self):
        """Score is 0 during 403 cooldown."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        result = ht.score("m1")
        assert isinstance(result, Ok)
        assert result.value == pytest.approx(0.0)

    def test_403_increments_error_count(self):
        """403 also increments error count for circuit breaker."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert ht.get_error_count("m1") == 1


class TestTransientCooldown429:
    """429 errors use short temporary suspension."""

    def test_429_suspends_model(self):
        """429 suspends the model temporarily."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=429)
        assert "m1" in ht._suspended
        assert "m1" not in ht._retired

    def test_429_model_unavailable_during_cooldown(self):
        """Model is unavailable during 429 cooldown."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=429)
        assert ht.is_available("m1") is False

    def test_429_model_returns_after_cooldown(self):
        """Model returns to pool after 429 cooldown expires."""
        ht = HealthTracker(suspend_ttl_429_s=5.0)
        ht.record_error("m1", http_status=429)
        assert ht.is_available("m1") is False

        # Fast-forward past the short cooldown
        ht._suspended["m1"] = time.time() - 6.0
        assert ht.is_retired("m1") is False

    def test_429_uses_shorter_ttl_than_403(self):
        """429 cooldown is shorter than 403 cooldown by default."""
        ht = HealthTracker()
        assert ht._suspend_ttl_429_s < ht._suspend_ttl_403_s

    def test_429_does_not_trigger_provider_suspension(self):
        """429 does not count toward provider-level suspension threshold."""
        ht = HealthTracker()
        ht.record_error("openrouter/model-a", http_status=429)
        ht.record_error("openrouter/model-b", http_status=429)
        ht.record_error("openrouter/model-c", http_status=429)
        assert "openrouter" not in ht._suspended_providers

    def test_429_increments_error_count(self):
        """429 also increments error count for circuit breaker."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=429)
        assert ht.get_error_count("m1") == 1


class TestTransientCooldown5xx:
    """5xx errors use medium temporary suspension."""

    def test_500_suspends_model(self):
        """500 suspends the model temporarily."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=500)
        assert "m1" in ht._suspended
        assert "m1" not in ht._retired

    def test_502_suspends_model(self):
        """502 suspends the model temporarily."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=502)
        assert "m1" in ht._suspended

    def test_503_suspends_model(self):
        """503 suspends the model temporarily."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=503)
        assert "m1" in ht._suspended

    def test_5xx_model_returns_after_cooldown(self):
        """Model returns to pool after 5xx cooldown expires."""
        ht = HealthTracker(suspend_ttl_5xx_s=5.0)
        ht.record_error("m1", http_status=500)
        assert ht.is_available("m1") is False

        # Fast-forward past the medium cooldown
        ht._suspended["m1"] = time.time() - 6.0
        assert ht.is_retired("m1") is False

    def test_5xx_ttl_between_429_and_403(self):
        """5xx cooldown is between 429 (short) and 403 (long) by default."""
        ht = HealthTracker()
        assert ht._suspend_ttl_429_s < ht._suspend_ttl_5xx_s
        assert ht._suspend_ttl_5xx_s < ht._suspend_ttl_403_s

    def test_5xx_does_not_trigger_provider_suspension(self):
        """5xx does not count toward provider-level suspension threshold."""
        ht = HealthTracker()
        ht.record_error("openrouter/model-a", http_status=500)
        ht.record_error("openrouter/model-b", http_status=502)
        ht.record_error("openrouter/model-c", http_status=503)
        assert "openrouter" not in ht._suspended_providers


class TestPerModelTTL:
    """Different failure types store and use their own TTLs."""

    def test_different_ttls_stored_per_model(self):
        """Each suspended model stores its own TTL."""
        ht = HealthTracker(
            suspend_ttl_403_s=300.0,
            suspend_ttl_429_s=60.0,
            suspend_ttl_5xx_s=120.0,
        )
        ht.record_error("m-403", http_status=403)
        ht.record_error("m-429", http_status=429)
        ht.record_error("m-500", http_status=500)

        assert ht._suspended_ttls["m-403"] == pytest.approx(300.0)
        assert ht._suspended_ttls["m-429"] == pytest.approx(60.0)
        assert ht._suspended_ttls["m-500"] == pytest.approx(120.0)

    def test_429_expires_before_403(self):
        """429 model returns to pool before 403 model."""
        ht = HealthTracker(
            suspend_ttl_403_s=300.0,
            suspend_ttl_429_s=10.0,
        )
        now = time.time()
        ht.record_error("m-403", http_status=403)
        ht.record_error("m-429", http_status=429)

        # Fast-forward 15 seconds: 429 expired, 403 still suspended
        ht._suspended["m-403"] = now - 15.0
        ht._suspended["m-429"] = now - 15.0

        assert ht.is_retired("m-429") is False  # expired
        assert ht.is_retired("m-403") is True   # still cooling down


class TestCooldownConfigurable:
    """Cooldown TTLs are configurable via constructor."""

    def test_custom_403_ttl(self):
        """Custom 403 TTL is respected."""
        ht = HealthTracker(suspend_ttl_403_s=10.0)
        assert ht._suspend_ttl_403_s == pytest.approx(10.0)

    def test_custom_429_ttl(self):
        """Custom 429 TTL is respected."""
        ht = HealthTracker(suspend_ttl_429_s=30.0)
        assert ht._suspend_ttl_429_s == pytest.approx(30.0)

    def test_custom_5xx_ttl(self):
        """Custom 5xx TTL is respected."""
        ht = HealthTracker(suspend_ttl_5xx_s=60.0)
        assert ht._suspend_ttl_5xx_s == pytest.approx(60.0)

    def test_custom_provider_suspend_ttl(self):
        """Custom provider suspension TTL is respected."""
        ht = HealthTracker(provider_suspend_ttl_s=600.0)
        assert ht._provider_suspend_ttl_s == pytest.approx(600.0)


class TestNonRetirableStatuses:
    """Other HTTP statuses do not cause retirement or suspension."""

    def test_400_no_suspension(self):
        """400 (bad request) only feeds circuit breaker, no suspension."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=400)
        assert "m1" not in ht._suspended
        assert "m1" not in ht._retired
        assert ht.get_error_count("m1") == 1

    def test_401_no_suspension(self):
        """401 (unauthorized) only feeds circuit breaker, no suspension."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=401)
        assert "m1" not in ht._suspended
        assert "m1" not in ht._retired

    def test_none_status_no_suspension(self):
        """None http_status only feeds circuit breaker."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=None)
        assert "m1" not in ht._suspended
        assert "m1" not in ht._retired
        assert ht.get_error_count("m1") == 1


class TestProviderSuspension403Only:
    """Provider-level suspension only triggers from 403 errors."""

    def test_403_triggers_provider_suspension(self):
        """Two 403s from same provider triggers provider suspension."""
        ht = HealthTracker()
        ht.record_error("openrouter/model-a", http_status=403)
        ht.record_error("openrouter/model-b", http_status=403)
        assert "openrouter" in ht._suspended_providers

    def test_provider_suspension_expires(self):
        """Provider suspension expires after TTL."""
        ht = HealthTracker(provider_suspend_ttl_s=10.0)
        ht.record_error("openrouter/model-a", http_status=403)
        ht.record_error("openrouter/model-b", http_status=403)
        assert "openrouter" in ht._suspended_providers

        # Fast-forward past provider TTL
        ht._suspended_providers["openrouter"] = time.time() - 11.0
        # Clear model-level suspensions too
        ht._suspended.clear()
        ht._suspended_ttls.clear()

        # A model from this provider should now be available
        # (provider suspension expired, model suspension cleared)
        assert ht._is_retired_or_suspended("openrouter/model-c") is False


class TestReinstatementCleansUpTTLs:
    """Reinstatement clears both suspension timestamp and TTL."""

    def test_reinstate_clears_suspended_ttl(self):
        """Reinstating a suspended model clears its TTL entry."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=403)
        assert "m1" in ht._suspended_ttls
        ht.reinstate_model("m1")
        assert "m1" not in ht._suspended_ttls
        assert "m1" not in ht._suspended

    def test_reinstate_429_model(self):
        """Reinstating a 429-suspended model works correctly."""
        ht = HealthTracker()
        ht.record_error("m1", http_status=429)
        assert ht.is_available("m1") is False
        ht.reinstate_model("m1")
        assert ht.is_available("m1") is True
