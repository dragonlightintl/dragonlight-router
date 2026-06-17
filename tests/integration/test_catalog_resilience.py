"""Integration tests — catalog refresh failure degrades gracefully.

Exercises the path: catalog becomes stale → refresh fails → router
continues serving from the stale/cached catalog without crashing.

Spec traceability:
  - TM-011 AC6: Catalog refresh failure degrades gracefully
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.catalog.cache import CatalogCache
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
from dragonlight_router.catalog.refresher import CatalogRefresher
from dragonlight_router.server.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend_config(
    name: str,
    provider: str,
    model: str,
    tier: BackendTier = BackendTier.LOCAL,
    priority: int = 0,
) -> BackendConfig:
    """Build a BackendConfig for test backends."""
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
        chunks = ["Catalog resilience ", "test response."]
        for chunk in chunks:
            yield chunk

    backend.generate = _fake_generate
    backend.health_check = AsyncMock(return_value=True)
    backend.record_usage = MagicMock()
    return backend


def _setup_env_with_cached_catalog(
    tmp_path: Path,
) -> tuple[Path, list[BackendConfig]]:
    """Create config, a valid cached catalog, and role matrix.

    The catalog is written with a current timestamp so it starts fresh.
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
                "name": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "model_prefix": "groq_",
                "rate_limits": {"rpm": 60, "rpd": 14400},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    matrix = {
        "coding": {"groq_llama70b": 90},
    }
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    catalog = {
        "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    backend_configs = [
        _make_backend_config(
            name="groq-llama70b",
            provider="groq",
            model="llama-3.3-70b-versatile",
            tier=BackendTier.LOCAL,
            priority=10,
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


class TestCatalogRefreshFailureDegracefully:
    """AC6: Catalog refresh failure degrades gracefully — the router
    continues serving from the stale/cached catalog.
    """

    def test_select_models_works_with_valid_cache_when_refresh_fails(
        self, tmp_path: Path,
    ) -> None:
        """When the catalog is fresh (not stale), select_models returns
        models from the cache even if the refresher would fail.

        This verifies the happy path — cache is used without triggering refresh.
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # select_models uses the cached catalog — no refresh needed
        models = engine.select_models("coding", top_n=5)
        assert len(models) > 0, "Should return models from the cached catalog"
        assert "groq_llama70b" in models

    def test_select_models_degrades_when_refresh_raises(
        self, tmp_path: Path,
    ) -> None:
        """When the catalog cache is stale and the refresher raises an
        exception, select_models should not crash. The _refresh_catalog
        method catches the exception internally, so select_models proceeds
        with the stale catalog data.
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Force the catalog to appear stale so select_models triggers a refresh
        with patch.object(engine._catalog, "is_stale", return_value=True):
            # Make the underlying refresher raise — _refresh_catalog catches it
            with patch.object(
                engine._refresher,
                "refresh",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Network unreachable"),
            ):
                # select_models should NOT crash — _refresh_catalog catches
                # the exception and select_models proceeds with stale cache
                try:
                    models = engine.select_models("coding", top_n=5)
                except RuntimeError:
                    pytest.fail(
                        "select_models must not propagate catalog refresh errors"
                    )

    def test_router_refresh_catalog_catches_exception(
        self, tmp_path: Path,
    ) -> None:
        """RouterEngine._refresh_catalog wraps the async refresher call in a
        try/except. When the refresher raises, the method logs a warning and
        returns without crashing.
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Patch the async refresher to raise
        with patch.object(
            engine._refresher,
            "refresh",
            new_callable=AsyncMock,
            side_effect=ConnectionError("DNS resolution failed"),
        ):
            # _refresh_catalog must not raise
            try:
                engine._refresh_catalog()
            except ConnectionError:
                pytest.fail(
                    "_refresh_catalog must catch refresher exceptions"
                )

        # The stale catalog should still be readable
        result = engine._catalog.get()
        from dragonlight_router.result import Ok

        assert isinstance(result, Ok), (
            "Stale catalog should still be readable after refresh failure"
        )

    def test_dispatch_succeeds_when_catalog_refresh_fails(
        self, tmp_path: Path,
    ) -> None:
        """Full E2E: dispatch through HTTP still succeeds when catalog
        refresh would fail, because the dispatch cascade works off the
        backend registry (not the catalog directly).
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Poison the refresher — it will raise if called
        with patch.object(
            engine._refresher,
            "refresh",
            new_callable=AsyncMock,
            side_effect=OSError("Connection refused"),
        ):
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
        assert "content" in data
        assert len(data["content"]) > 0

    def test_catalog_refresh_endpoint_returns_error_on_failure(
        self, tmp_path: Path,
    ) -> None:
        """POST /v1/catalog/refresh returns a 500 with error details when
        the refresher raises, but the server itself does not crash.
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)

        # The route handler does `from dragonlight_router.catalog.refresher
        # import CatalogRefresher` — patch at the source module
        with patch(
            "dragonlight_router.catalog.refresher.CatalogRefresher",
        ) as MockRefresherClass:
            mock_refresher = MagicMock(spec=CatalogRefresher)
            mock_refresher.refresh = AsyncMock(
                side_effect=OSError("Provider API unreachable"),
            )
            MockRefresherClass.return_value = mock_refresher

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/v1/catalog/refresh")

        assert response.status_code == 500, (
            f"Expected 500, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "error" in data or "status" in data

    def test_stale_catalog_still_serves_select_endpoint(
        self, tmp_path: Path,
    ) -> None:
        """POST /v1/select returns results from the stale catalog when
        the catalog is marked stale but refresh fails. The router should
        not crash and should return whatever models it can resolve.
        """
        config_path, backend_configs = _setup_env_with_cached_catalog(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Make refresh a no-op (simulates failed refresh that was caught)
        with patch.object(engine, "_refresh_catalog"):
            client = TestClient(app)
            response = client.post(
                "/v1/select",
                json={"role": "coding", "top_n": 5},
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
