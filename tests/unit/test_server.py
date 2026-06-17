"""Tests for server/routes.py — HTTP API endpoints.

Spec traceability: TM-009 (HTTP API endpoints)
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

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
)
from dragonlight_router.server.app import create_app


def _setup_test_env(tmp_path: Path) -> Path:
    """Create a full test config + state for server testing."""
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
                "rate_limits": {"rpm": 30, "rpd": 14400},
            },
            {
                "name": "nvidia",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "model_prefix": "nvidia_",
                "rate_limits": {"rpm": 60, "rpd": 5000},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    # Role matrix
    matrix = {
        "coding": {
            "groq_llama70b": 90,
            "nvidia_nemotron": 85,
            "groq_mixtral": 75,
        },
    }
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    # Catalog cache
    from dragonlight_router.catalog.cache import CatalogCache
    from dragonlight_router.core.types import CatalogEntry

    catalog = {
        "groq": [
            CatalogEntry(model_id="groq_llama70b", provider="groq"),
            CatalogEntry(model_id="groq_mixtral", provider="groq"),
        ],
        "nvidia": [
            CatalogEntry(model_id="nvidia_nemotron", provider="nvidia"),
        ],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    return config_path


class TestSelectEndpoint:
    def test_select_returns_models(self, tmp_path: Path):
        """[TM-009 AC-1] POST /v1/select returns model list."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding", "top_n": 5})
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        assert len(data["models"]) <= 5

    def test_select_includes_scores(self, tmp_path: Path):
        """[TM-009 AC-1] Select response includes scores matching model count."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding"})
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == len(data["models"])

    def test_select_unknown_role_empty(self, tmp_path: Path):
        """[TM-009 AC-1] Unknown role returns empty model list."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "nonexistent"})
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    def test_select_includes_trust_and_complexity_tiers(self, tmp_path: Path):
        """[TM-009 AC-1] Select response includes valid trust and complexity tiers."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) > 0
        valid_complexity = {"simple", "moderate", "complex", "local"}
        valid_trust = {"local", "trusted", "semi_trusted"}
        for entry in data["scores"]:
            assert "complexity_tier" in entry, f"missing complexity_tier in {entry}"
            assert "trust_tier" in entry, f"missing trust_tier in {entry}"
            assert entry["complexity_tier"] in valid_complexity, (
                f"invalid complexity_tier: {entry['complexity_tier']}"
            )
            assert entry["trust_tier"] in valid_trust, (
                f"invalid trust_tier: {entry['trust_tier']}"
            )

    def test_select_tiers_match_registered_backend(self, tmp_path: Path):
        """When a backend is registered, its tier should flow through to the select response."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine
        # Register a backend with COMPLEX tier whose name matches a matrix model_id
        complex_config = BackendConfig(
            name="groq_llama70b",
            provider="groq",
            model="llama-3.3-70b",
            tier=BackendTier.COMPLEX,
            base_url="http://localhost:9999",
            env_key=None,
            capabilities=BackendCapabilities(
                max_context_tokens=4096,
                supports_tool_use=False,
                supports_streaming=False,
                supports_json_mode=False,
                supports_system_prompts=True,
            ),
            cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
            rate_limits=BackendRateLimits(rpm=60, rpd=1000, tpm=100000, daily_token_cap=0),
        )
        stub = _StubBackend(_config=complex_config)
        engine._registry.register(stub)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding"})
        data = response.json()
        llama_scores = [s for s in data["scores"] if s["model_id"] == "groq_llama70b"]
        assert len(llama_scores) == 1
        assert llama_scores[0]["complexity_tier"] == "complex"
        assert llama_scores[0]["trust_tier"] == "trusted"

    def test_select_missing_role_400(self, tmp_path: Path):
        """[TM-009 AC-2] Missing role field returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={})
        assert response.status_code == 400


class TestRecordEndpoint:
    def test_record_success(self, tmp_path: Path):
        """[TM-009 AC-3] POST /v1/record success returns 200."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/record", json={
            "provider": "groq",
            "model_id": "groq_llama70b",
            "success": True,
            "tokens_used": 150,
            "latency_ms": 45.0,
        })
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_record_failure(self, tmp_path: Path):
        """[TM-009 AC-3] POST /v1/record failure returns 200."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/record", json={
            "provider": "groq",
            "model_id": "groq_llama70b",
            "success": False,
        })
        assert response.status_code == 200

    def test_record_missing_fields_400(self, tmp_path: Path):
        """[TM-009 AC-2] Missing required fields in record returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/record", json={"provider": "groq"})
        assert response.status_code == 400


class TestHealthEndpoint:
    def test_health_returns_200(self, tmp_path: Path):
        """[TM-009 AC-4] GET /v1/health returns 200 with budget and health data."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "budget" in data
        assert "health" in data


class TestCatalogEndpoint:
    def test_catalog_status(self, tmp_path: Path):
        """[TM-009 AC-5] GET /v1/catalog returns 200 with stale indicator."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/catalog")
        assert response.status_code == 200
        data = response.json()
        assert "stale" in data


def _make_backend_config(name: str) -> BackendConfig:
    """Create a minimal BackendConfig for testing."""
    return BackendConfig(
        name=name,
        provider="test_provider",
        model="test_model",
        tier=BackendTier.SIMPLE,
        base_url="http://localhost:9999",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=False,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(rpm=60, rpd=1000, tpm=100000, daily_token_cap=0),
    )


@dataclass
class _StubBackend:
    """Minimal GenerativeBackend stub for retire/reinstate tests."""

    _config: BackendConfig
    _status: BackendStatus = BackendStatus.AVAILABLE

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        yield ""  # pragma: no cover

    async def health_check(self) -> bool:
        return True  # pragma: no cover

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        pass  # pragma: no cover


def _create_app_with_backend(tmp_path: Path, backend_name: str) -> TestClient:
    """Create a test app with a single stub backend registered."""
    config_path = _setup_test_env(tmp_path)
    app = create_app(config_path=config_path)
    engine = app.state.engine
    stub = _StubBackend(_config=_make_backend_config(backend_name))
    engine._registry.register(stub)
    return TestClient(app)


class TestRetireEndpoint:
    def test_retire_backend_success(self, tmp_path: Path):
        """[TM-009 AC-6] POST /v1/retire successfully retires a known backend."""
        client = _create_app_with_backend(tmp_path, "my-backend")
        response = client.post("/v1/retire", json={"backend": "my-backend"})
        assert response.status_code == 200
        data = response.json()
        assert data["retired"] is True
        assert data["backend"] == "my-backend"

    def test_retire_unknown_backend_returns_404(self, tmp_path: Path):
        """[TM-009 AC-6] Retiring an unknown backend returns 404."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/retire", json={"backend": "nonexistent"})
        assert response.status_code == 404
        assert response.json()["error"] == "backend not found"

    def test_retire_missing_field_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Missing backend field in retire returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/retire", json={})
        assert response.status_code == 400
        assert "missing required field" in response.json()["error"]


class TestReinstateEndpoint:
    def test_reinstate_backend_success(self, tmp_path: Path):
        """[TM-009 AC-7] POST /v1/reinstate successfully reinstates a retired backend."""
        client = _create_app_with_backend(tmp_path, "my-backend")
        # First retire it
        retire_resp = client.post("/v1/retire", json={"backend": "my-backend"})
        assert retire_resp.status_code == 200
        # Then reinstate it
        response = client.post("/v1/reinstate", json={"backend": "my-backend"})
        assert response.status_code == 200
        data = response.json()
        assert data["reinstated"] is True
        assert data["backend"] == "my-backend"

    def test_reinstate_unknown_backend_returns_404(self, tmp_path: Path):
        """[TM-009 AC-7] Reinstating an unknown backend returns 404."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/reinstate", json={"backend": "nonexistent"})
        assert response.status_code == 404
        assert response.json()["error"] == "backend not found"
