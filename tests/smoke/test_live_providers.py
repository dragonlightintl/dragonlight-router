"""Live provider smoke tests -- validate end-to-end paths against real APIs.

These tests hit real provider endpoints and are gated behind:
  - The ``live`` pytest marker (deselect with ``-m "not live"``)
  - A skipif guard that checks for at least one API key in the environment

Run on-demand only:
    python3 -m pytest tests/smoke/test_live_providers.py -m live -v

Spec traceability:
  - TM-012: Live provider validation (select, catalog, health)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dragonlight_router.catalog.refresher import CatalogRefresher
from dragonlight_router.result import Ok
from dragonlight_router.router import RouterEngine

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ALL_API_KEYS = [
    "NVIDIA_NIM_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "TOGETHER_API_KEY",
]

_has_any_key = any(os.environ.get(k) for k in _ALL_API_KEYS)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "router.yaml"
_MATRIX_PATH = _PROJECT_ROOT / "config" / "model_role_matrix.json"

skip_no_keys = pytest.mark.skipif(
    not _has_any_key,
    reason="No API keys available -- skipping live provider tests",
)


def _prepare_state_dir(tmp_path: Path) -> Path:
    """Create a temp state_dir pre-populated with the model role matrix.

    RouterEngine.__init__ creates the RoleMatrix in _init_subsystems()
    before _ensure_matrix_in_state_dir() copies the file, so when using
    a fresh tmp_path the matrix loads empty and no backends get registered.
    Pre-copying the matrix avoids this ordering issue.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    if _MATRIX_PATH.exists():
        shutil.copy2(_MATRIX_PATH, state_dir / "model_role_matrix.json")
    return state_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_no_keys
class TestLiveSelectModels:
    """Validate that select_models returns results against real provider state."""

    def test_live_select_models(self, tmp_path: Path) -> None:
        """Create a RouterEngine with real config and prove the cascade returns models."""
        engine = RouterEngine(
            config_path=_CONFIG_PATH,
            state_dir=_prepare_state_dir(tmp_path),
        )
        models = engine.select_models("coding", top_n=5)

        # We just need at least 1 model to prove the full pipeline works:
        # config load -> matrix -> catalog -> scoring -> interleave
        assert isinstance(models, list)
        assert len(models) >= 1, (
            "select_models('coding') should return at least 1 model "
            "when API keys are configured"
        )
        # Every entry should be a non-empty string
        for model_id in models:
            assert isinstance(model_id, str)
            assert len(model_id) > 0


@pytest.mark.live
@skip_no_keys
class TestLiveCatalogRefresh:
    """Validate catalog refresh against real provider APIs."""

    @pytest.mark.asyncio
    async def test_live_catalog_refresh(self) -> None:
        """Refresh catalogs and verify at least one provider returns models."""
        from dragonlight_router.config.loader import load_config

        config_result = load_config(_CONFIG_PATH)
        assert isinstance(config_result, Ok), f"Config load failed: {config_result}"
        config = config_result.value

        refresher = CatalogRefresher(timeout_s=15.0)
        result = await refresher.refresh(config.providers)

        assert isinstance(result, Ok), f"Catalog refresh failed: {result}"
        refresh_result = result.value

        catalog = refresh_result.catalog
        auth_failures = refresh_result.auth_failures

        # At least one provider should return models
        providers_with_models = {
            name: len(entries) for name, entries in catalog.items() if entries
        }
        assert len(providers_with_models) >= 1, (
            f"Expected at least 1 provider with models, got 0. "
            f"Auth failures: {auth_failures}"
        )

        # Auth failures should only contain providers whose keys are missing/invalid
        for provider_name, status_code in auth_failures.items():
            assert status_code in (401, 403), (
                f"Unexpected auth failure status for {provider_name}: {status_code}"
            )


@pytest.mark.live
@skip_no_keys
class TestLiveHealthCheck:
    """Validate health_check() against at least one real backend."""

    @pytest.mark.asyncio
    async def test_live_health_check(self, tmp_path: Path) -> None:
        """Probe health on backends with valid keys and verify at least one passes."""
        engine = RouterEngine(
            config_path=_CONFIG_PATH,
            state_dir=_prepare_state_dir(tmp_path),
        )

        # Collect all backends that have a valid API key set
        healthy_count = 0
        checked_count = 0
        for _name, backend, _state in engine._registry.all_backends():
            env_key = backend.config.env_key
            if not env_key or not os.environ.get(env_key):
                continue

            checked_count += 1
            is_healthy = await backend.health_check()
            if is_healthy:
                healthy_count += 1

        assert checked_count >= 1, (
            "Expected at least 1 backend with a valid API key to check"
        )
        assert healthy_count >= 1, (
            f"Expected at least 1 healthy backend out of {checked_count} checked"
        )
