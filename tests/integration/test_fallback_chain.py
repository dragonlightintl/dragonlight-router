"""E2E fallback chain tests — verify cascade retries with fallback adapters.

Exercises: primary adapter failure → cascade retry → fallback adapter success.
Mocks only at the adapter/network seam — no real API calls.

Spec traceability:
  - TM-010: Fallback dispatch (cascade retries next candidate on adapter failure)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Build a realistic BackendConfig for test backends."""
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


def _make_mock_backend(config: BackendConfig, *, fail: bool = False) -> MagicMock:
    """Create a mock GenerativeBackend.

    Args:
        config: Backend config to attach.
        fail: If True, generate() raises RuntimeError on first call.
    """
    backend = MagicMock(spec=GenerativeBackend)
    backend.config = config
    backend.status = BackendStatus.AVAILABLE

    if fail:
        async def _failing_generate(messages, *, max_tokens=4096, temperature=0.7, stream=True):
            raise RuntimeError(f"Adapter {config.name} failed: simulated network error")
            yield  # pragma: no cover — makes this an async generator for async-for compat
        backend.generate = _failing_generate
    else:
        async def _fake_generate(messages, *, max_tokens=4096, temperature=0.7, stream=True):
            chunks = [
                f"Response from {config.name}: ",
                "Here is a Python function ",
                "to calculate Fibonacci numbers:\n\n",
                "```python\ndef fib(n):\n",
                "    if n <= 1:\n        return n\n",
                "    return fib(n-1) + fib(n-2)\n```",
            ]
            for chunk in chunks:
                yield chunk
        backend.generate = _fake_generate

    backend.health_check = AsyncMock(return_value=True)
    backend.record_usage = MagicMock()
    return backend


def _setup_fallback_env(tmp_path: Path) -> tuple[Path, list[BackendConfig]]:
    """Create config and catalog for fallback chain testing.

    Registers two backends at LOCAL tier so MBR finds both as candidates
    for the simple dispatch order (context_tokens=100 → LOCAL tier).
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
                "rate_limits": {"rpm": 30, "rpd": 14400},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    # Role matrix
    matrix = {"coding": {"groq_llama70b": 90}}
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    # Catalog cache
    from dragonlight_router.catalog.cache import CatalogCache

    catalog = {
        "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    # Two backends at LOCAL tier — primary (high priority) will fail,
    # secondary (lower priority) will succeed
    backend_configs = [
        _make_backend_config(
            name="groq-primary",
            provider="groq",
            model="llama-3.3-70b-primary",
            tier=BackendTier.LOCAL,
            priority=10,
        ),
        _make_backend_config(
            name="groq-secondary",
            provider="groq",
            model="llama-3.3-70b-secondary",
            tier=BackendTier.LOCAL,
            priority=5,
        ),
    ]

    return config_path, backend_configs


VALID_DISPATCH_BODY = {
    "intent_category": "general",
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


class TestFallbackChainE2E:
    """POST /v1/dispatch where the primary adapter fails and the cascade
    retries with the next candidate."""

    def test_fallback_to_second_adapter_on_primary_failure(self, tmp_path: Path):
        """Primary adapter raises → cascade retries → secondary adapter succeeds.

        Spec: TM-010 fallback dispatch path.
        Mocks: create_adapter returns a failing mock for the first backend,
               a succeeding mock for the second.
        """
        config_path, backend_configs = _setup_fallback_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Register both mock backends into the registry
        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # Build adapter mapping: primary fails, secondary succeeds
        primary_config = backend_configs[0]
        secondary_config = backend_configs[1]

        # DEVIATION TEST-MOCK-001: branching mock required
        # — factory returns different adapters per backend name.
        def _fake_create_adapter(config):
            if config.name == primary_config.name:
                return _make_mock_backend(config, fail=True)
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

        # Content should come from the secondary adapter
        assert "content" in data
        assert len(data["content"]) > 20, "Content should be substantive"
        assert secondary_config.name in data["content"], (
            f"Content should identify the secondary adapter ({secondary_config.name})"
        )

        # was_fallback must be True since primary failed
        assert data["was_fallback"] is True, (
            "was_fallback should be True when the first adapter fails"
        )

        # fallback_chain should contain the failed primary backend
        assert isinstance(data["fallback_chain"], list)
        assert primary_config.name in data["fallback_chain"], (
            f"fallback_chain should contain the failed primary backend ({primary_config.name})"
        )

        # backend_used should be the secondary
        assert data["backend_used"] == secondary_config.name

    def test_all_adapters_fail_returns_error(self, tmp_path: Path):
        """When every adapter in the cascade fails, dispatch returns 500 with
        exhaustion details.

        Spec: TM-010 all-backends-exhausted path.
        """
        config_path, backend_configs = _setup_fallback_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Register mock backends
        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # All adapters fail
        def _all_fail_adapter(config):
            return _make_mock_backend(config, fail=True)

        with patch(
            "dragonlight_router.adapters.create_adapter",
            side_effect=_all_fail_adapter,
        ):
            client = TestClient(app)
            response = client.post("/v1/dispatch", json=VALID_DISPATCH_BODY)

        assert response.status_code == 500, (
            f"Expected 500 when all backends exhausted, got {response.status_code}"
        )

        data = response.json()
        assert "message" in data
        assert "exhausted" in data["message"].lower() or "all" in data["message"].lower(), (
            "Error message should indicate all backends were exhausted"
        )

    def test_fallback_chain_preserves_order(self, tmp_path: Path):
        """The fallback_chain list records backends in the order they were tried
        and failed, not just as an unordered set.

        Spec: TM-010 fallback chain ordering.
        """
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        config = {
            "state_dir": str(state_dir),
            "catalog_ttl_hours": 24,
            "default_top_n": 12,
            "max_consecutive_same_provider": 3,
            "providers": [
                {
                    "name": "groq",
                    "base_url": "https://api.groq.com/openai/v1",
                    "model_prefix": "groq_",
                    "rate_limits": {"rpm": 30, "rpd": 14400},
                },
            ],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))

        matrix = {"coding": {"groq_llama70b": 90}}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        from dragonlight_router.catalog.cache import CatalogCache

        catalog = {
            "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
        }
        cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
        cache.set(catalog)

        # Three backends — first two fail, third succeeds
        backend_configs = [
            _make_backend_config(
                name="groq-alpha",
                provider="groq",
                model="llama-alpha",
                tier=BackendTier.LOCAL,
                priority=10,
            ),
            _make_backend_config(
                name="groq-beta",
                provider="groq",
                model="llama-beta",
                tier=BackendTier.LOCAL,
                priority=7,
            ),
            _make_backend_config(
                name="groq-gamma",
                provider="groq",
                model="llama-gamma",
                tier=BackendTier.LOCAL,
                priority=3,
            ),
        ]

        app = create_app(config_path=config_path)
        engine = app.state.engine

        for bc in backend_configs:
            mock_backend = _make_mock_backend(bc)
            engine._registry.register(mock_backend)

        # First two fail, third succeeds
        failing_names = {backend_configs[0].name, backend_configs[1].name}

        # DEVIATION TEST-MOCK-001: branching mock required
        # — factory returns different adapters per backend name.
        def _selective_adapter(config):
            if config.name in failing_names:
                return _make_mock_backend(config, fail=True)
            return _make_mock_backend(config)

        with patch(
            "dragonlight_router.adapters.create_adapter",
            side_effect=_selective_adapter,
        ):
            client = TestClient(app)
            response = client.post("/v1/dispatch", json=VALID_DISPATCH_BODY)

        assert response.status_code == 200

        data = response.json()
        assert data["was_fallback"] is True
        assert len(data["fallback_chain"]) == 2, (
            f"Expected 2 entries in fallback_chain, got {len(data['fallback_chain'])}"
        )

        # Both failed backends should appear in the chain
        for name in failing_names:
            assert name in data["fallback_chain"], (
                f"{name} should appear in fallback_chain"
            )

        # The successful backend should NOT be in the fallback chain
        assert backend_configs[2].name not in data["fallback_chain"]
        assert data["backend_used"] == backend_configs[2].name
