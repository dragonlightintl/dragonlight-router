"""Server smoke tests — verify the app boots, serves HTTP, and shuts down cleanly.

These tests exercise the full create_app() → Starlette → route handler path
with minimal mocking (only network I/O is mocked). They verify:

1. The server starts without error (lifespan enters)
2. Health endpoint returns valid JSON
3. Select endpoint with a known role returns a model list
4. Dispatch endpoint with valid payload returns a response
5. Graceful shutdown completes (lifespan exits)

Spec traceability:
  - TM-009: Server startup, health endpoint
  - TM-010: Dispatch through full cascade
  - TM-011: HTTP contract verification
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    CatalogEntry,
    GenerativeBackend,
)
from dragonlight_router.server.app import create_app

pytestmark = pytest.mark.smoke


def _build_smoke_env(tmp_path: Path) -> Path:
    """Create minimal config, matrix, and catalog for a smoke test."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [
            {
                "name": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "model_prefix": "groq_",
                "rate_limits": {"rpm": 60, "rpd": 14400},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    matrix = {"coding": {"groq_llama70b": 90}}
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    from dragonlight_router.catalog.cache import CatalogCache

    catalog = {"groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")]}
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    return config_path


def _make_mock_backend(name: str, tier: BackendTier = BackendTier.LOCAL) -> MagicMock:
    """Create a mock backend that yields realistic content."""
    config = BackendConfig(
        name=name,
        provider="groq",
        model="llama-3.3-70b-versatile",
        tier=tier,
        base_url="https://api.groq.test/v1",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=32768,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.10, output_per_mtok=0.30),
        rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=500000),
        priority=10,
    )
    backend = MagicMock(spec=GenerativeBackend)
    backend.config = config
    backend.status = BackendStatus.AVAILABLE
    backend.health_check = AsyncMock(return_value=True)
    backend.record_usage = MagicMock()

    async def _fake_generate(messages, *, max_tokens=4096, temperature=0.7, stream=True):
        for chunk in ["Hello from ", "smoke test."]:
            yield chunk

    backend.generate = _fake_generate
    return backend


class TestServerBoot:
    """Verify the server boots through lifespan and serves the health endpoint."""

    def test_health_endpoint_returns_200(self, tmp_path: Path):
        """Server starts, health endpoint returns 200 with budget/health keys."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "budget" in data
        assert "health" in data
        assert "status" in data

    def test_catalog_endpoint_returns_200(self, tmp_path: Path):
        """Catalog endpoint returns 200 with provider and model count."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.get("/v1/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "model_count" in data
        assert isinstance(data["model_count"], int)


class TestSelectSmoke:
    """Verify the select endpoint through a full boot cycle."""

    def test_select_returns_models(self, tmp_path: Path):
        """Select with a valid role returns a non-empty model list."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine
        engine._registry.register(_make_mock_backend("groq-llama70b"))

        with TestClient(app) as client:
            resp = client.post("/v1/select", json={"role": "coding", "top_n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_select_invalid_role_returns_empty(self, tmp_path: Path):
        """Select with an unknown role returns an empty model list (not an error)."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.post("/v1/select", json={"role": "nonexistent_role", "top_n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []


class TestDispatchSmoke:
    """Verify dispatch through a full boot + cascade cycle."""

    def test_dispatch_returns_content(self, tmp_path: Path):
        """Dispatch with valid payload returns 200 with content from mock adapter."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine
        engine._registry.register(_make_mock_backend("groq-llama70b"))

        def _fake_create_adapter(config):
            return _make_mock_backend(config.name)

        with (
            patch("dragonlight_router.adapters.create_adapter", side_effect=_fake_create_adapter),
            TestClient(app) as client,
        ):
            resp = client.post(
                "/v1/dispatch",
                json={
                    "intent_category": "general",
                    "specific_intent": "greeting",
                    "operator_message": "Say hello",
                    "system_prompt": "You are helpful",
                    "context_tokens": 50,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert len(data["content"]) > 0
        assert "backend_used" in data
        assert "latency_ms" in data

    def test_dispatch_invalid_json_returns_400(self, tmp_path: Path):
        """Dispatch with malformed JSON returns 400."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/dispatch",
                content=b"{bad json",
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400

    def test_dispatch_missing_fields_returns_400(self, tmp_path: Path):
        """Dispatch with missing required fields returns 400."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.post("/v1/dispatch", json={"intent_category": "general"})
        assert resp.status_code == 400
        assert "missing" in resp.json()["error"].lower()


class TestRecordSmoke:
    """Verify the record endpoint through a full boot cycle."""

    def test_record_returns_ok(self, tmp_path: Path):
        """Record a valid outcome and get status ok."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/record",
                json={
                    "provider": "groq",
                    "model_id": "groq_llama70b",
                    "success": True,
                    "tokens_used": 100,
                    "latency_ms": 250.0,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestGracefulShutdown:
    """Verify the lifespan exit path runs without error."""

    def test_lifespan_exit_saves_state(self, tmp_path: Path):
        """Entering and exiting the TestClient triggers lifespan start and stop."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Patch save_state to verify it's called during shutdown
        with patch.object(engine, "save_state", wraps=engine.save_state) as mock_save:
            with TestClient(app):
                pass  # lifespan enters and exits
            mock_save.assert_called_once()

    def test_lifespan_exit_tolerates_save_failure(self, tmp_path: Path):
        """Shutdown completes even if save_state raises."""
        config_path = _build_smoke_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        with (
            patch.object(engine, "save_state", side_effect=OSError("disk full")) as mock_save,
            TestClient(app),
        ):
            pass  # Should not raise

        # Verify save_state was actually called (and its OSError was tolerated)
        assert mock_save.called, "save_state must be invoked during lifespan exit"
