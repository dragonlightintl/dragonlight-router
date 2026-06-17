"""Tests for server/routes.py — HTTP API endpoints.

Spec traceability: TM-009 (HTTP API endpoints)
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
    StreamChunk,
)
from dragonlight_router.result import Ok, Err
from dragonlight_router.server.app import create_app
from dragonlight_router.server.routes import (
    _backend_tier_to_trust,
    _format_stream_chunk,
    _validate_select_request,
    _validate_dispatch_request,
    _execute_catalog_refresh,
)


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

    def test_reinstate_invalid_json_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid JSON body in reinstate returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/reinstate",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "invalid JSON" in response.json()["error"]

    def test_reinstate_missing_backend_field_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Missing backend field in reinstate returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post("/v1/reinstate", json={})
        assert response.status_code == 400
        assert "missing required field" in response.json()["error"]


class TestRetireInvalidJson:
    def test_retire_invalid_json_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid JSON body in retire returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/retire",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "invalid JSON" in response.json()["error"]


class TestRecordInvalidJson:
    def test_record_invalid_json_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid JSON body in record returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/record",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "invalid JSON" in response.json()["error"]


class TestValidateSelectRequest:
    def test_invalid_role_type_returns_error(self):
        """[TM-009 AC-2] Non-string role returns validation error."""
        error = _validate_select_request({"role": 123})
        assert error is not None
        assert "invalid role" in error

    def test_role_too_long_returns_error(self):
        """[TM-009 AC-2] Role exceeding max length returns validation error."""
        error = _validate_select_request({"role": "x" * 100_001})
        assert error is not None
        assert "invalid role" in error

    def test_invalid_top_n_string_returns_error(self):
        """[TM-009 AC-2] Non-integer top_n returns validation error."""
        error = _validate_select_request({"role": "coding", "top_n": "five"})
        assert error is not None
        assert "invalid top_n" in error

    def test_top_n_zero_returns_error(self):
        """[TM-009 AC-2] top_n of 0 is below minimum, returns error."""
        error = _validate_select_request({"role": "coding", "top_n": 0})
        assert error is not None
        assert "invalid top_n" in error

    def test_top_n_above_max_returns_error(self):
        """[TM-009 AC-2] top_n above 500 returns validation error."""
        error = _validate_select_request({"role": "coding", "top_n": 501})
        assert error is not None
        assert "invalid top_n" in error


class TestBackendTierToTrust:
    def test_complex_tier_maps_to_trusted(self):
        """[TM-009 AC-1] COMPLEX tier maps to 'trusted'."""
        assert _backend_tier_to_trust(BackendTier.COMPLEX) == "trusted"

    def test_local_tier_maps_to_local(self):
        """[TM-009 AC-1] LOCAL tier maps to 'local'."""
        assert _backend_tier_to_trust(BackendTier.LOCAL) == "local"

    def test_simple_tier_maps_to_semi_trusted(self):
        """[TM-009 AC-1] SIMPLE tier maps to 'semi_trusted'."""
        assert _backend_tier_to_trust(BackendTier.SIMPLE) == "semi_trusted"

    def test_moderate_tier_maps_to_semi_trusted(self):
        """[TM-009 AC-1] MODERATE tier maps to 'semi_trusted'."""
        assert _backend_tier_to_trust(BackendTier.MODERATE) == "semi_trusted"


class TestSelectExcludeProviders:
    def test_select_with_exclude_providers(self, tmp_path: Path):
        """[TM-009 AC-1] exclude_providers filters out the specified provider."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/select",
            json={"role": "coding", "exclude_providers": ["groq"]},
        )
        assert response.status_code == 200
        data = response.json()
        for model_id in data["models"]:
            assert not model_id.startswith("groq_")


class TestValidateDispatchRequest:
    def test_missing_intent_category_returns_error(self):
        """[TM-009 AC-2] Missing intent_category returns validation error."""
        body = {
            "specific_intent": "x",
            "operator_message": "hi",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_invalid_string_field_too_long(self):
        """[TM-009 AC-2] String field exceeding max length returns validation error."""
        body = {
            "intent_category": "x" * 100_001,
            "specific_intent": "x",
            "operator_message": "hi",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_invalid_system_prompt_too_long(self):
        """[TM-009 AC-2] system_prompt exceeding max length returns validation error."""
        body = {
            "intent_category": "general",
            "specific_intent": "spec",
            "operator_message": "msg",
            "context_tokens": 0,
            "system_prompt": "x" * 100_001,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "system_prompt" in error

    def test_invalid_context_tokens_negative(self):
        """[TM-009 AC-2] Negative context_tokens returns validation error."""
        body = {
            "intent_category": "general",
            "specific_intent": "spec",
            "operator_message": "msg",
            "context_tokens": -1,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "context_tokens" in error

    def test_invalid_context_tokens_string(self):
        """[TM-009 AC-2] Non-integer context_tokens returns validation error."""
        body = {
            "intent_category": "general",
            "specific_intent": "spec",
            "operator_message": "msg",
            "context_tokens": "many",
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "context_tokens" in error


class TestExecuteCatalogRefresh:
    @pytest.mark.asyncio
    async def test_execute_catalog_refresh_success(self, tmp_path: Path):
        """[TM-009 AC-5] _execute_catalog_refresh returns ok response on success."""
        from dragonlight_router.core.types import CatalogEntry

        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        mock_catalog = {
            "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
        }

        mock_refresher = AsyncMock()
        mock_refresher.refresh.return_value = Ok(mock_catalog)

        with patch("dragonlight_router.server.routes._refresher_mod.CatalogRefresher", return_value=mock_refresher):
            response = await _execute_catalog_refresh(engine)

        data = response.body
        parsed = json.loads(data)
        assert parsed["status"] == "ok"
        assert parsed["model_count"] == 1
        assert "groq" in parsed["providers_refreshed"]


class TestDispatchEndpoint:
    def test_dispatch_invalid_json_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid JSON body in dispatch returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "invalid JSON" in response.json()["error"]

    def test_dispatch_missing_field_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Missing required field in dispatch returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={"intent_category": "general"},
        )
        assert response.status_code == 400

    def test_dispatch_engine_exception_returns_500(self, tmp_path: Path):
        """[TM-009 AC-8] Engine RuntimeError during dispatch returns 500."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        engine.dispatch = _raise
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
            },
        )
        assert response.status_code == 500
        assert "Internal server error" in response.json()["error"]


class TestLifespan:
    def test_lifespan_refresh_failure_is_swallowed(self, tmp_path: Path):
        """[TM-009 AC-9] Lifespan startup continues even if catalog refresh fails."""
        config_path = _setup_test_env(tmp_path)

        async def _bad_refresh(self):
            raise RuntimeError("network unavailable")

        with patch(
            "dragonlight_router.router.RouterEngine._async_refresh_catalog",
            new=_bad_refresh,
        ), patch(
            "dragonlight_router.router.RouterEngine.start_health_check_loop",
            new=AsyncMock(),
        ):
            app = create_app(config_path=config_path)
            with TestClient(app) as client:
                response = client.get("/v1/health")
        assert response.status_code == 200

    def test_lifespan_starts_health_check_loop(self, tmp_path: Path):
        """[TM-009 AC-9] Lifespan creates a health check loop task that is cancelled on shutdown."""
        config_path = _setup_test_env(tmp_path)

        async def _instant_loop(self_engine):
            pass

        with patch(
            "dragonlight_router.router.RouterEngine._async_refresh_catalog",
            new=AsyncMock(),
        ), patch(
            "dragonlight_router.router.RouterEngine.start_health_check_loop",
            new=_instant_loop,
        ):
            app = create_app(config_path=config_path)
            with TestClient(app) as client:
                response = client.get("/v1/health")
                assert response.status_code == 200

    def test_lifespan_saves_state_on_shutdown(self, tmp_path: Path):
        """[TM-009 AC-10] HAZ-012: Lifespan calls save_state on shutdown."""
        config_path = _setup_test_env(tmp_path)

        async def _instant_loop(self_engine):
            pass

        with patch(
            "dragonlight_router.router.RouterEngine._async_refresh_catalog",
            new=AsyncMock(),
        ), patch(
            "dragonlight_router.router.RouterEngine.start_health_check_loop",
            new=_instant_loop,
        ):
            app = create_app(config_path=config_path)
            engine = app.state.engine
            with patch.object(engine, "save_state") as mock_save:
                with TestClient(app):
                    pass  # enter and exit triggers lifespan shutdown
            mock_save.assert_called_once()

    def test_lifespan_shutdown_save_error_does_not_crash(self, tmp_path: Path):
        """[TM-009 AC-10] HAZ-012: save_state failure at shutdown does not crash the server."""
        config_path = _setup_test_env(tmp_path)

        async def _instant_loop(self_engine):
            pass

        with patch(
            "dragonlight_router.router.RouterEngine._async_refresh_catalog",
            new=AsyncMock(),
        ), patch(
            "dragonlight_router.router.RouterEngine.start_health_check_loop",
            new=_instant_loop,
        ):
            app = create_app(config_path=config_path)
            engine = app.state.engine
            with patch.object(engine, "save_state", side_effect=OSError("disk full")):
                # Should not raise even though save_state fails
                with TestClient(app):
                    pass


class TestMain:
    def test_main_uses_default_host_and_port(self, tmp_path: Path):
        """[TM-009 AC-9] main() calls uvicorn.run with default host 127.0.0.1 port 8100."""
        import dragonlight_router.server.app as app_module

        with patch.dict("os.environ", {}, clear_env := False):
            for key in ("DRAGONLIGHT_ROUTER_CONFIG", "DRAGONLIGHT_HOST", "DRAGONLIGHT_PORT"):
                patch.object
            with patch("dragonlight_router.server.app.uvicorn.run") as mock_run, \
                 patch("dragonlight_router.server.app.create_app") as mock_create:
                mock_create.return_value = MagicMock()
                import os
                env_backup = {k: os.environ.pop(k, None) for k in ("DRAGONLIGHT_ROUTER_CONFIG", "DRAGONLIGHT_HOST", "DRAGONLIGHT_PORT")}
                try:
                    app_module.main()
                finally:
                    for k, v in env_backup.items():
                        if v is not None:
                            os.environ[k] = v
                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs.get("host", mock_run.call_args[0][1] if len(mock_run.call_args[0]) > 1 else "127.0.0.1") == "127.0.0.1"

    def test_main_uses_env_config_path(self, tmp_path: Path):
        """[TM-009 AC-9] main() passes config path from DRAGONLIGHT_ROUTER_CONFIG env var."""
        import os
        import dragonlight_router.server.app as app_module

        config_path = str(tmp_path / "router.yaml")
        captured_path = []

        def _fake_create_app(config_path=None, **overrides):
            captured_path.append(config_path)
            return MagicMock()

        with patch("dragonlight_router.server.app.create_app", side_effect=_fake_create_app), \
             patch("dragonlight_router.server.app.uvicorn.run"):
            old = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
            os.environ["DRAGONLIGHT_ROUTER_CONFIG"] = config_path
            try:
                app_module.main()
            finally:
                if old is None:
                    del os.environ["DRAGONLIGHT_ROUTER_CONFIG"]
                else:
                    os.environ["DRAGONLIGHT_ROUTER_CONFIG"] = old

        assert captured_path and str(captured_path[0]) == config_path

    def test_main_uses_custom_host_and_port(self, tmp_path: Path):
        """[TM-009 AC-9] main() reads DRAGONLIGHT_HOST and DRAGONLIGHT_PORT env vars."""
        import os
        import dragonlight_router.server.app as app_module

        captured = {}

        def _fake_run(app, host="127.0.0.1", port=8100, **kwargs):
            captured["host"] = host
            captured["port"] = port
            captured["timeout_graceful_shutdown"] = kwargs.get("timeout_graceful_shutdown")

        with patch("dragonlight_router.server.app.create_app", return_value=MagicMock()), \
             patch("dragonlight_router.server.app.uvicorn.run", side_effect=_fake_run):
            env_keys = {"DRAGONLIGHT_HOST": "0.0.0.0", "DRAGONLIGHT_PORT": "9000"}
            old = {k: os.environ.get(k) for k in env_keys}
            os.environ.update(env_keys)
            for key in ("DRAGONLIGHT_ROUTER_CONFIG",):
                os.environ.pop(key, None)
            try:
                app_module.main()
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        assert captured.get("host") == "0.0.0.0"
        assert captured.get("port") == 9000


class TestSelectInvalidJson:
    def test_select_invalid_json_returns_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid JSON body in select returns 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/select",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "invalid JSON" in response.json()["error"]


class TestExecuteCatalogRefreshError:
    @pytest.mark.asyncio
    async def test_execute_catalog_refresh_error_path(self, tmp_path: Path):
        """[TM-009 AC-5] _execute_catalog_refresh returns 500 on Err result."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        mock_refresher = AsyncMock()
        mock_refresher.refresh.return_value = Err(RuntimeError("network down"))

        with patch("dragonlight_router.server.routes._refresher_mod.CatalogRefresher", return_value=mock_refresher):
            response = await _execute_catalog_refresh(engine)

        import json as _json
        parsed = _json.loads(response.body)
        assert parsed["status"] == "error"
        assert response.status_code == 500

    def test_catalog_refresh_handler_exception_returns_500(self, tmp_path: Path):
        """[TM-009 AC-5] catalog_refresh_handler catches OSError and returns 500."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)

        with patch(
            "dragonlight_router.server.routes._execute_catalog_refresh",
            side_effect=OSError("connection refused"),
        ):
            response = client.post("/v1/catalog/refresh")

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"


class TestDispatchOkPath:
    def test_dispatch_returns_engine_response(self, tmp_path: Path):
        """[TM-009 AC-8] dispatch_handler returns 200 with EngineResponse fields on Ok."""
        from dragonlight_router.core.types import EngineResponse

        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        fake_response = EngineResponse(
            content="Hello from model",
            backend_used="groq_llama70b",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=123.4,
            was_fallback=False,
            fallback_chain=[],
        )

        async def _ok_dispatch(order):
            return Ok(fake_response)

        engine.dispatch = _ok_dispatch
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Hello from model"
        assert data["backend_used"] == "groq_llama70b"
        assert data["was_fallback"] is False

    def test_dispatch_returns_dispatch_failure(self, tmp_path: Path):
        """[TM-009 AC-8] dispatch_handler returns 500 with DispatchFailure fields on Err."""
        from dragonlight_router.core.types import DispatchFailure

        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        failure = DispatchFailure(
            message="all backends exhausted",
            attempted_backends=["groq_llama70b"],
            error_details={"groq_llama70b": "timeout"},
        )

        async def _err_dispatch(order):
            return Err(failure)

        engine.dispatch = _err_dispatch
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert data["message"] == "all backends exhausted"
        assert "groq_llama70b" in data["attempted_backends"]

    def test_dispatch_err_non_dispatch_failure(self, tmp_path: Path):
        """[TM-009 AC-8] dispatch_handler returns 500 for generic Err with non-DispatchFailure error."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _generic_err_dispatch(order):
            return Err(ValueError("unexpected"))

        engine.dispatch = _generic_err_dispatch
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert data["message"] == "Dispatch failed"


# ---------------------------------------------------------------------------
# Streaming dispatch — _format_stream_chunk
# ---------------------------------------------------------------------------


class TestFormatStreamChunk:
    def test_token_chunk_format(self):
        """[TM-009 AC-8] Token chunk serializes with event and content fields."""
        chunk = StreamChunk(event_type="token", content="Hello")
        result = _format_stream_chunk(chunk)
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        parsed = json.loads(result[6:].strip())
        assert parsed["event"] == "token"
        assert parsed["content"] == "Hello"

    def test_metadata_chunk_format(self):
        """[TM-009 AC-8] Metadata chunk serializes with all response fields."""
        chunk = StreamChunk(
            event_type="metadata",
            backend_used="groq_llama70b",
            backend_tier="complex",
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=123.4,
            was_fallback=True,
            fallback_chain=["openai_gpt4"],
        )
        result = _format_stream_chunk(chunk)
        parsed = json.loads(result[6:].strip())
        assert parsed["event"] == "metadata"
        assert parsed["backend_used"] == "groq_llama70b"
        assert parsed["backend_tier"] == "complex"
        assert parsed["tokens_in"] == 10
        assert parsed["tokens_out"] == 20
        assert parsed["was_fallback"] is True
        assert parsed["fallback_chain"] == ["openai_gpt4"]

    def test_error_chunk_format(self):
        """[TM-009 AC-8] Error chunk serializes with error_message."""
        chunk = StreamChunk(event_type="error", error_message="boom")
        result = _format_stream_chunk(chunk)
        parsed = json.loads(result[6:].strip())
        assert parsed["event"] == "error"
        assert parsed["error_message"] == "boom"

    def test_metadata_chunk_none_fallback_chain(self):
        """[TM-009 AC-8] Metadata chunk with None fallback_chain serializes as empty list."""
        chunk = StreamChunk(
            event_type="metadata",
            backend_used="b1",
            backend_tier="simple",
            fallback_chain=None,
        )
        result = _format_stream_chunk(chunk)
        parsed = json.loads(result[6:].strip())
        assert parsed["fallback_chain"] == []


# ---------------------------------------------------------------------------
# Streaming dispatch — endpoint integration
# ---------------------------------------------------------------------------


class TestStreamingDispatchEndpoint:
    def test_stream_true_returns_sse(self, tmp_path: Path):
        """[TM-009 AC-8] POST /v1/dispatch with stream=true returns text/event-stream."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _mock_stream(order):
            yield StreamChunk(event_type="token", content="Hello")
            yield StreamChunk(event_type="token", content=" world")
            yield StreamChunk(
                event_type="metadata",
                backend_used="groq_llama70b",
                backend_tier="simple",
                tokens_in=5,
                tokens_out=3,
                estimated_cost_usd=0.0001,
                latency_ms=50.0,
                was_fallback=False,
                fallback_chain=[],
            )

        engine.dispatch_stream = _mock_stream
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        events = [
            json.loads(line[6:].strip())
            for line in response.text.split("\n\n")
            if line.strip().startswith("data: ")
        ]
        assert len(events) == 3
        assert events[0]["event"] == "token"
        assert events[0]["content"] == "Hello"
        assert events[1]["event"] == "token"
        assert events[1]["content"] == " world"
        assert events[2]["event"] == "metadata"
        assert events[2]["backend_used"] == "groq_llama70b"

    def test_stream_false_returns_json(self, tmp_path: Path):
        """[TM-009 AC-8] POST /v1/dispatch with stream=false returns normal JSON."""
        from dragonlight_router.core.types import EngineResponse

        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        fake_response = EngineResponse(
            content="Hello from model",
            backend_used="groq_llama70b",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=123.4,
            was_fallback=False,
            fallback_chain=[],
        )

        async def _ok_dispatch(order):
            return Ok(fake_response)

        engine.dispatch = _ok_dispatch
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["content"] == "Hello from model"

    def test_stream_error_event(self, tmp_path: Path):
        """[TM-009 AC-8] Streaming dispatch error yields error SSE event."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _error_stream(order):
            yield StreamChunk(event_type="error", error_message="cascade failed")

        engine.dispatch_stream = _error_stream
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
                "stream": True,
            },
        )
        assert response.status_code == 200
        events = [
            json.loads(line[6:].strip())
            for line in response.text.split("\n\n")
            if line.strip().startswith("data: ")
        ]
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert events[0]["error_message"] == "cascade failed"

    def test_stream_engine_exception_yields_error(self, tmp_path: Path):
        """[TM-009 AC-8] Engine exception during streaming yields internal error event."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _exploding_stream(order):
            raise RuntimeError("unexpected")
            yield  # noqa: RET503 — makes this an async generator

        engine.dispatch_stream = _exploding_stream
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "write a function",
                "operator_message": "hello",
                "context_tokens": 100,
                "stream": True,
            },
        )
        assert response.status_code == 200
        events = [
            json.loads(line[6:].strip())
            for line in response.text.split("\n\n")
            if line.strip().startswith("data: ")
        ]
        assert any(e["event"] == "error" for e in events)
        error_event = next(e for e in events if e["event"] == "error")
        assert error_event["error_message"] == "Internal server error"

    def test_stream_validation_error_returns_json_400(self, tmp_path: Path):
        """[TM-009 AC-2] Invalid body with stream=true still returns JSON 400."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "stream": True,
            },
        )
        assert response.status_code == 400
        assert response.headers["content-type"] == "application/json"

    def test_stream_sse_headers(self, tmp_path: Path):
        """[TM-009 AC-8] Streaming response includes correct SSE headers."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        async def _mock_stream(order):
            yield StreamChunk(event_type="token", content="ok")
            yield StreamChunk(event_type="metadata", backend_used="b1", backend_tier="simple")

        engine.dispatch_stream = _mock_stream
        client = TestClient(app)
        response = client.post(
            "/v1/dispatch",
            json={
                "intent_category": "general",
                "specific_intent": "test",
                "operator_message": "hi",
                "context_tokens": 0,
                "stream": True,
            },
        )
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"
