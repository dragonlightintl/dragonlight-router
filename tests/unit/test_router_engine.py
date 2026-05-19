"""Tests for router.py — RouterEngine wiring."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.router import RouterEngine, get_router


def _setup_config(tmp_path: Path) -> Path:
    """Create a minimal config + matrix for testing."""
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

    # Create a role matrix
    matrix = {
        "coding": {
            "groq_llama70b": 90,
            "nvidia_nemotron": 85,
            "groq_mixtral": 75,
            "nvidia_llama8b": 60,
            "groq_llama8b": 50,
        },
        "testing": {
            "groq_llama70b": 80,
            "nvidia_nemotron": 70,
        },
    }
    matrix_path = state_dir / "model_role_matrix.json"
    matrix_path.write_text(json.dumps(matrix))

    # Create a catalog cache (so select_models has live models to filter)
    from dragonlight_router.catalog.cache import CatalogCache
    from dragonlight_router.core.types import CatalogEntry

    catalog = {
        "groq": [
            CatalogEntry(model_id="groq_llama70b", provider="groq"),
            CatalogEntry(model_id="groq_mixtral", provider="groq"),
            CatalogEntry(model_id="groq_llama8b", provider="groq"),
        ],
        "nvidia": [
            CatalogEntry(model_id="nvidia_nemotron", provider="nvidia"),
            CatalogEntry(model_id="nvidia_llama8b", provider="nvidia"),
        ],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    return config_path


class TestSelectModels:
    def test_returns_ranked_model_ids(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        result = engine.select_models("coding")
        assert isinstance(result, list)
        assert len(result) > 0
        # Top model should be groq_llama70b (rank 90)
        assert result[0] == "groq_llama70b"

    def test_respects_top_n(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        result = engine.select_models("coding", top_n=3)
        assert len(result) <= 3

    def test_unknown_role_returns_empty(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        result = engine.select_models("nonexistent_role")
        assert result == []

    def test_exclude_providers(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        result = engine.select_models("coding", exclude_providers=frozenset({"groq"}))
        for model_id in result:
            assert not model_id.startswith("groq_")

    def test_interleaving_applied(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        result = engine.select_models("coding", top_n=12)
        # The coding matrix has groq dominant; interleaving should prevent
        # 3 consecutive groq models
        providers = []
        for model_id in result:
            if model_id.startswith("groq_"):
                providers.append("groq")
            elif model_id.startswith("nvidia_"):
                providers.append("nvidia")
        for i in range(len(providers) - 2):
            assert not (providers[i] == providers[i + 1] == providers[i + 2])


class TestRecordRequest:
    def test_record_success(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        # Should not raise
        engine.record_request("groq", "groq_llama70b", success=True, tokens_used=100, latency_ms=50.0)

    def test_record_failure(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        engine.record_request("groq", "groq_llama70b", success=False)

    def test_failure_affects_health_score(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        # Record multiple failures
        engine.record_request("groq", "groq_llama70b", success=False)
        engine.record_request("groq", "groq_llama70b", success=False)
        engine.record_request("groq", "groq_llama70b", success=False)
        # The model should be penalized in selection
        result = engine.select_models("coding")
        # groq_llama70b should no longer be first (circuit open = score 0)
        if result:
            assert result[0] != "groq_llama70b"

    def test_success_affects_budget(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        engine.record_request("groq", "groq_llama70b", success=True, tokens_used=500)
        # Budget should reflect the request
        snapshot = engine.budget_snapshot()
        assert "groq" in snapshot


class TestBudgetSnapshot:
    def test_returns_dict(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        snapshot = engine.budget_snapshot()
        assert isinstance(snapshot, dict)
        assert "groq" in snapshot
        assert "nvidia" in snapshot

    def test_score_field_present(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        snapshot = engine.budget_snapshot()
        for provider_name, data in snapshot.items():
            assert "score" in data
            assert "has_capacity" in data


class TestHealthSnapshot:
    def test_returns_dict(self, tmp_path: Path):
        config_path = _setup_config(tmp_path)
        engine = RouterEngine(config_path=config_path)
        snapshot = engine.health_snapshot()
        assert isinstance(snapshot, dict)


class TestCatalogRefreshOnStale:
    """select_models() must trigger catalog refresh when cache is stale."""

    def _setup_no_catalog(self, tmp_path: Path) -> Path:
        """Config with no pre-populated catalog (simulates first boot)."""
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

        # Matrix includes a retired model (nvidia_retired) not in live catalog
        matrix = {
            "coding": {
                "groq_llama70b": 90,
                "nvidia_nemotron": 85,
                "nvidia_retired": 80,
            },
        }
        matrix_path = state_dir / "model_role_matrix.json"
        matrix_path.write_text(json.dumps(matrix))

        return config_path

    def test_stale_cache_triggers_refresh(self, tmp_path: Path):
        """When catalog cache is empty, select_models triggers refresh."""
        config_path = self._setup_no_catalog(tmp_path)
        engine = RouterEngine(config_path=config_path)

        live_catalog = {
            "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
            "nvidia": [CatalogEntry(model_id="nvidia_nemotron", provider="nvidia")],
        }

        with patch.object(
            engine._refresher, "refresh", new_callable=AsyncMock, return_value=live_catalog
        ):
            result = engine.select_models("coding")

        assert "groq_llama70b" in result
        assert "nvidia_nemotron" in result
        assert "nvidia_retired" not in result

    def test_refresh_failure_returns_all_candidates(self, tmp_path: Path):
        """If refresh fails, all matrix models pass through (graceful degradation)."""
        config_path = self._setup_no_catalog(tmp_path)
        engine = RouterEngine(config_path=config_path)

        with patch.object(
            engine._refresher, "refresh", new_callable=AsyncMock, side_effect=Exception("network down")
        ):
            result = engine.select_models("coding")

        # Graceful degradation: all candidates pass through unfiltered
        assert "groq_llama70b" in result
        assert "nvidia_nemotron" in result
        assert "nvidia_retired" in result

    def test_fresh_cache_skips_refresh(self, tmp_path: Path):
        """When catalog cache is fresh, no refresh is triggered."""
        config_path = self._setup_no_catalog(tmp_path)
        engine = RouterEngine(config_path=config_path)

        # Pre-populate cache so it's fresh
        from dragonlight_router.catalog.cache import CatalogCache
        cache = CatalogCache(
            cache_path=Path(tmp_path / "state" / "provider_catalog.json"),
            ttl_hours=24,
        )
        live_catalog = {
            "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
            "nvidia": [CatalogEntry(model_id="nvidia_nemotron", provider="nvidia")],
        }
        cache.set(live_catalog)

        with patch.object(
            engine._refresher, "refresh", new_callable=AsyncMock
        ) as mock_refresh:
            result = engine.select_models("coding")

        mock_refresh.assert_not_called()
        assert "nvidia_retired" not in result

    def test_partial_catalog_does_not_filter_unfetched_providers(self, tmp_path: Path):
        """If a provider's catalog fails, its models still pass through.

        The filter should only exclude models whose provider WAS fetched
        but the model wasn't in the results (retired). Models from providers
        whose catalog wasn't fetched at all should pass through unfiltered.
        """
        config_path = self._setup_no_catalog(tmp_path)
        engine = RouterEngine(config_path=config_path)

        # Only groq's catalog succeeded; nvidia's endpoint was unreachable
        partial_catalog = {
            "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
        }

        with patch.object(
            engine._refresher, "refresh", new_callable=AsyncMock, return_value=partial_catalog
        ):
            result = engine.select_models("coding")

        # groq_llama70b: in catalog → passes
        assert "groq_llama70b" in result
        # nvidia_nemotron: provider not in catalog → NOT filtered (benefit of doubt)
        assert "nvidia_nemotron" in result
        # nvidia_retired: provider not in catalog → also passes (can't distinguish)
        assert "nvidia_retired" in result


class TestGetRouter:
    def test_singleton_returns_same_instance(self, tmp_path: Path):
        """Note: can't easily test singleton in pytest without resetting global state."""
        # Just test that get_router returns a RouterEngine
        import dragonlight_router.router as router_mod
        # Reset singleton for test isolation
        router_mod._router_instance = None
        config_path = _setup_config(tmp_path)
        r1 = get_router(config_path=str(config_path))
        r2 = get_router()
        assert r1 is r2
        # Clean up
        router_mod._router_instance = None
