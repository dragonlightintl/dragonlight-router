"""Tests for HAZ-011 mitigation — admin endpoint authentication.

Validates that admin endpoints (retire, reinstate, catalog/refresh)
require valid bearer token authentication when admin_api_key is configured.

Spec traceability: HAZ-011 (Unauthenticated Admin Endpoints)
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


def _setup_test_env_with_admin_key(tmp_path: Path, admin_key: str | None = None) -> Path:
    """Create a test config with optional admin_api_key."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config: dict = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [],
    }
    if admin_key is not None:
        config["admin_api_key"] = admin_key

    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))
    (state_dir / "model_role_matrix.json").write_text(json.dumps({}))
    return config_path


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
    """Minimal GenerativeBackend stub for admin auth tests."""

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


class TestAdminAuthRetire:
    """HAZ-011: Retire endpoint authentication."""

    def test_retire_no_admin_key_configured_allows_access(self, tmp_path: Path):
        """[HAZ-011 AC-1] Without admin_api_key, retire works without auth."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key=None)
        app = create_app(config_path=config_path)
        engine = app.state.engine
        stub = _StubBackend(_config=_make_backend_config("test-backend"))
        engine._registry.register(stub)
        client = TestClient(app)
        response = client.post("/v1/retire", json={"backend": "test-backend"})
        assert response.status_code == 200
        assert response.json()["retired"] is True

    def test_retire_with_valid_auth(self, tmp_path: Path):
        """[HAZ-011 AC-2] Valid bearer token allows retire."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="my-secret-key")
        app = create_app(config_path=config_path)
        engine = app.state.engine
        stub = _StubBackend(_config=_make_backend_config("test-backend"))
        engine._registry.register(stub)
        client = TestClient(app)
        response = client.post(
            "/v1/retire",
            json={"backend": "test-backend"},
            headers={"Authorization": "Bearer my-secret-key"},
        )
        assert response.status_code == 200
        assert response.json()["retired"] is True

    def test_retire_missing_auth_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-3] Missing auth header returns 401 when admin_api_key is set."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="my-secret-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/retire", json={"backend": "test-backend"})
        assert response.status_code == 401
        assert "Authorization" in response.json()["error"]

    def test_retire_wrong_key_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-4] Wrong bearer token returns 401."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="my-secret-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/retire",
            json={"backend": "test-backend"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401
        assert "Invalid admin API key" in response.json()["error"]

    def test_retire_malformed_auth_header_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-3] Non-Bearer auth header returns 401."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="my-secret-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/retire",
            json={"backend": "test-backend"},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert response.status_code == 401


class TestAdminAuthReinstate:
    """HAZ-011: Reinstate endpoint authentication."""

    def test_reinstate_no_admin_key_allows_access(self, tmp_path: Path):
        """[HAZ-011 AC-1] Without admin_api_key, reinstate works without auth."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key=None)
        app = create_app(config_path=config_path)
        engine = app.state.engine
        stub = _StubBackend(_config=_make_backend_config("test-backend"))
        engine._registry.register(stub)
        # First retire, then reinstate
        client = TestClient(app)
        client.post("/v1/retire", json={"backend": "test-backend"})
        response = client.post("/v1/reinstate", json={"backend": "test-backend"})
        assert response.status_code == 200
        assert response.json()["reinstated"] is True

    def test_reinstate_with_valid_auth(self, tmp_path: Path):
        """[HAZ-011 AC-2] Valid bearer token allows reinstate."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-token")
        app = create_app(config_path=config_path)
        engine = app.state.engine
        stub = _StubBackend(_config=_make_backend_config("test-backend"))
        engine._registry.register(stub)
        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer admin-token"}
        client.post("/v1/retire", json={"backend": "test-backend"}, headers=auth_headers)
        response = client.post(
            "/v1/reinstate",
            json={"backend": "test-backend"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_reinstate_missing_auth_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-3] Missing auth header returns 401 for reinstate."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-token")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/reinstate", json={"backend": "test-backend"})
        assert response.status_code == 401

    def test_reinstate_wrong_key_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-4] Wrong bearer token returns 401 for reinstate."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-token")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/reinstate",
            json={"backend": "test-backend"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401


class TestAdminAuthCatalogRefresh:
    """HAZ-011: Catalog refresh endpoint authentication."""

    def test_catalog_refresh_no_admin_key_allows_access(self, tmp_path: Path):
        """[HAZ-011 AC-1] Without admin_api_key, catalog/refresh works without auth."""
        from unittest.mock import AsyncMock, patch
        from dragonlight_router.result import Ok
        from dragonlight_router.core.types import CatalogEntry

        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key=None)
        app = create_app(config_path=config_path)

        mock_refresher = AsyncMock()
        mock_refresher.refresh.return_value = Ok({"test": [CatalogEntry(model_id="m1", provider="test")]})

        client = TestClient(app)
        with patch("dragonlight_router.server.routes._refresher_mod.CatalogRefresher", return_value=mock_refresher):
            response = client.post("/v1/catalog/refresh")
        assert response.status_code == 200

    def test_catalog_refresh_missing_auth_returns_401(self, tmp_path: Path):
        """[HAZ-011 AC-3] Missing auth header returns 401 for catalog/refresh."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="refresh-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/catalog/refresh")
        assert response.status_code == 401
        assert "Authorization" in response.json()["error"]

    def test_catalog_refresh_with_valid_auth(self, tmp_path: Path):
        """[HAZ-011 AC-2] Valid bearer token allows catalog/refresh."""
        from unittest.mock import AsyncMock, patch
        from dragonlight_router.result import Ok
        from dragonlight_router.core.types import CatalogEntry

        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="refresh-key")
        app = create_app(config_path=config_path)

        mock_refresher = AsyncMock()
        mock_refresher.refresh.return_value = Ok({"test": [CatalogEntry(model_id="m1", provider="test")]})

        client = TestClient(app)
        with patch("dragonlight_router.server.routes._refresher_mod.CatalogRefresher", return_value=mock_refresher):
            response = client.post(
                "/v1/catalog/refresh",
                headers={"Authorization": "Bearer refresh-key"},
            )
        assert response.status_code == 200


class TestNonAdminEndpointsUnaffected:
    """HAZ-011: Non-admin endpoints should NOT require auth."""

    def test_health_endpoint_no_auth_required(self, tmp_path: Path):
        """[HAZ-011 AC-5] GET /v1/health works without auth even when admin_api_key is set."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_catalog_status_no_auth_required(self, tmp_path: Path):
        """[HAZ-011 AC-5] GET /v1/catalog works without auth even when admin_api_key is set."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/catalog")
        assert response.status_code == 200

    def test_select_no_auth_required(self, tmp_path: Path):
        """[HAZ-011 AC-5] POST /v1/select works without auth even when admin_api_key is set."""
        config_path = _setup_test_env_with_admin_key(tmp_path, admin_key="admin-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/select", json={"role": "coding"})
        assert response.status_code == 200
