"""Tests for server/routes.py — HTTP API endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from starlette.testclient import TestClient

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
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding"})
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == len(data["models"])

    def test_select_unknown_role_empty(self, tmp_path: Path):
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "nonexistent"})
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    def test_select_missing_role_400(self, tmp_path: Path):
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={})
        assert response.status_code == 400


class TestRecordEndpoint:
    def test_record_success(self, tmp_path: Path):
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
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/record", json={"provider": "groq"})
        assert response.status_code == 400


class TestHealthEndpoint:
    def test_health_returns_200(self, tmp_path: Path):
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
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/catalog")
        assert response.status_code == 200
        data = response.json()
        assert "stale" in data
