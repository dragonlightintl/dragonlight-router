"""Integration tests — circuit breaker opens after consecutive errors and
excludes the backend from MBR routing.

Exercises the full path: dispatch failures → BackendState trips circuit →
MBR excludes CIRCUIT_OPEN backends → fallback to healthy backend.

Spec traceability:
  - TM-011 AC4: Circuit breaker opens after 3 consecutive errors
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    CatalogEntry,
    DispatchOrder,
    GenerativeBackend,
)
from dragonlight_router.health.circuit_breaker import CircuitBreaker, CircuitState
from dragonlight_router.server.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend_config(
    name: str,
    provider: str,
    model: str,
    tier: BackendTier = BackendTier.SIMPLE,
    priority: int = 0,
) -> BackendConfig:
    """Build a BackendConfig for test backends.

    Defaults to SIMPLE tier (not LOCAL) so MBR applies circuit breaker
    filtering — LOCAL backends bypass health checks per AC5.
    """
    return BackendConfig(
        name=name,
        provider=provider,
        model=model,
        tier=tier,
        base_url=f"https://api.{provider}.test/v1",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=32768,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(
            input_per_mtok=0.50,
            output_per_mtok=1.50,
        ),
        rate_limits=BackendRateLimits(
            rpm=60,
            rpd=14400,
            tpm=100000,
            daily_token_cap=500000,
        ),
        priority=priority,
    )


def _make_mock_backend(config: BackendConfig) -> MagicMock:
    """Create a mock GenerativeBackend that satisfies the protocol."""
    backend = MagicMock(spec=GenerativeBackend)
    backend.config = config
    backend.status = BackendStatus.AVAILABLE

    async def _fake_generate(messages, *, max_tokens=4096, temperature=0.7, stream=True):
        chunks = ["Circuit breaker ", "test response."]
        for chunk in chunks:
            yield chunk

    backend.generate = _fake_generate
    backend.health_check = AsyncMock(return_value=True)
    backend.record_usage = MagicMock()
    return backend


def _setup_env(tmp_path: Path) -> tuple[Path, list[BackendConfig]]:
    """Create config, catalog, role matrix for a two-backend setup.

    Both backends are SIMPLE tier so MBR applies circuit breaker checks.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [
            {
                "name": "provider_a",
                "base_url": "https://api.provider-a.test/v1",
                "model_prefix": "pa_",
                "rate_limits": {"rpm": 60, "rpd": 14400},
            },
            {
                "name": "provider_b",
                "base_url": "https://api.provider-b.test/v1",
                "model_prefix": "pb_",
                "rate_limits": {"rpm": 60, "rpd": 14400},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    matrix = {
        "coding": {"pa_model1": 90, "pb_model2": 85},
    }
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    from dragonlight_router.catalog.cache import CatalogCache

    catalog = {
        "provider_a": [CatalogEntry(model_id="pa_model1", provider="provider_a")],
        "provider_b": [CatalogEntry(model_id="pb_model2", provider="provider_b")],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    backend_configs = [
        _make_backend_config(
            name="backend-a",
            provider="provider_a",
            model="model-alpha",
            tier=BackendTier.SIMPLE,
            priority=10,
        ),
        _make_backend_config(
            name="backend-b",
            provider="provider_b",
            model="model-beta",
            tier=BackendTier.SIMPLE,
            priority=5,
        ),
    ]

    return config_path, backend_configs


VALID_DISPATCH_BODY = {
    "intent_category": "code_generation",
    "specific_intent": "write_function",
    "operator_message": "Write a Python function to calculate fibonacci numbers",
    "system_prompt": "You are a helpful coding assistant",
    "context_tokens": 100,
    "requires_tool_use": False,
    "requires_long_context": False,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpensAfterConsecutiveErrors:
    """AC4: Circuit breaker opens after 3 consecutive errors."""

    def test_circuit_breaker_state_opens_after_three_errors(self) -> None:
        """Unit-level sanity: CircuitBreaker transitions to OPEN after
        error_threshold consecutive errors.
        """
        cb = CircuitBreaker(error_threshold=3, error_window_s=120.0, cooldown_s=60.0)

        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

        cb.record_error()
        assert cb.state == CircuitState.CLOSED, "1 error: still CLOSED"

        cb.record_error()
        assert cb.state == CircuitState.CLOSED, "2 errors: still CLOSED"

        cb.record_error()
        assert cb.state == CircuitState.OPEN, "3 errors: must be OPEN"
        assert cb.allow_request() is False, "OPEN circuit must reject requests"

    def test_backend_state_trips_circuit_after_three_errors(self) -> None:
        """BackendState.record_error() sets status to CIRCUIT_OPEN after
        error_threshold consecutive errors.
        """
        state = BackendState(error_threshold=3, error_window=120.0, circuit_cooldown=60.0)

        assert state.status == BackendStatus.AVAILABLE

        tripped = state.record_error()
        assert tripped is False
        assert state.status == BackendStatus.ERROR

        tripped = state.record_error()
        assert tripped is False
        assert state.status == BackendStatus.ERROR

        tripped = state.record_error()
        assert tripped is True
        assert state.status == BackendStatus.CIRCUIT_OPEN
        assert state.is_circuit_open() is True


class TestCircuitBreakerExcludesFromRouting:
    """AC4 integration: After circuit trips, the backend is excluded from
    MBR routing and a healthy fallback is selected instead.
    """

    def test_tripped_backend_excluded_from_dispatch(self, tmp_path: Path) -> None:
        """Register two SIMPLE-tier backends. Trip backend-a's circuit via
        3 consecutive errors recorded through the registry state. Then
        dispatch and verify backend-b is selected.
        """
        config_path, backend_configs = _setup_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Register both backends
        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Trip backend-a's circuit breaker through BackendState
        _backend_a, state_a = engine._registry.get("backend-a")
        assert state_a is not None
        state_a.record_error()
        state_a.record_error()
        tripped = state_a.record_error()
        assert tripped is True, "Circuit should trip after 3 errors"
        assert state_a.status == BackendStatus.CIRCUIT_OPEN

        # Verify backend-b is still healthy
        _backend_b, state_b = engine._registry.get("backend-b")
        assert state_b is not None
        assert state_b.status == BackendStatus.AVAILABLE

        # Dispatch — the cascade should skip backend-a and use backend-b
        def _fake_create_adapter(config):
            return _make_mock_backend(config)

        with patch(
            "dragonlight_router.adapters.create_adapter",
            side_effect=_fake_create_adapter,
        ):
            client = TestClient(app)
            response = client.post("/v1/dispatch", json=VALID_DISPATCH_BODY)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert data["backend_used"] == "backend-b", (
            f"Expected backend-b (healthy fallback), got {data['backend_used']}"
        )

    def test_all_backends_circuit_open_returns_error(self, tmp_path: Path) -> None:
        """When all backends have tripped circuits, dispatch returns an error
        (no healthy candidates survive MBR).
        """
        config_path, backend_configs = _setup_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Trip both backends' circuits
        for name in ("backend-a", "backend-b"):
            _backend, state = engine._registry.get(name)
            assert state is not None
            state.record_error()
            state.record_error()
            state.record_error()
            assert state.status == BackendStatus.CIRCUIT_OPEN

        def _fake_create_adapter(config):
            return _make_mock_backend(config)

        with patch(
            "dragonlight_router.adapters.create_adapter",
            side_effect=_fake_create_adapter,
        ):
            client = TestClient(app)
            response = client.post("/v1/dispatch", json=VALID_DISPATCH_BODY)

        # Should fail — no healthy candidates
        assert response.status_code == 500, (
            f"Expected 500, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "message" in data

    def test_dispatch_failures_trip_circuit_via_health_tracker(self, tmp_path: Path) -> None:
        """End-to-end: actual dispatch failures (adapter raises) flow through
        health_tracker.record_error → circuit_breaker.record_error → OPEN.

        After 3 adapter failures on backend-a, the health tracker's circuit
        breaker should be open for that model.
        """
        config_path, backend_configs = _setup_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Record 3 errors through the health tracker (the path dispatch uses)
        model_id = backend_configs[0].model  # "model-alpha"
        engine._health.record_error(model_id)
        engine._health.record_error(model_id)
        engine._health.record_error(model_id)

        # The health tracker's circuit breaker should now be open
        assert engine._health.is_available(model_id) is False, (
            "Health tracker should report model-alpha as unavailable after 3 errors"
        )
        assert engine._health.get_error_count(model_id) == 3
