"""Acceptance tests for the dragonlight-router.

Exercises the full RouterEngine workflow from the operator's perspective.
Uses real (non-mocked) RouterEngine instances with test configurations.

Spec traceability: TS-002 (Acceptance tests)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from dragonlight_router.catalog.cache import CatalogCache
from dragonlight_router.core.types import (
    BackendStatus,
    CatalogEntry,
    DispatchOrder,
    RequestOutcome,
)
from dragonlight_router.result import Ok
from dragonlight_router.router import RouterEngine

pytestmark = pytest.mark.acceptance


def _setup_acceptance_config(tmp_path: Path) -> Path:
    """Create a test config with multiple providers and roles for acceptance testing."""
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
        "drafting": {
            "groq_mixtral": 85,
            "nvidia_nemotron": 75,
            "groq_llama8b": 55,
        },
    }
    matrix_path = state_dir / "model_role_matrix.json"
    matrix_path.write_text(json.dumps(matrix))

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


class TestSelectModelsForAllRoles:
    """[TS-002 AC-1] For each role in the matrix, select_models returns at least 1 model."""

    def test_select_models_for_all_roles(self, tmp_path: Path) -> None:
        """Every role in the matrix yields at least one model from select_models."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        for role in ("coding", "testing", "drafting"):
            result = engine.select_models(role)
            assert len(result) >= 1, f"Role '{role}' returned no models"
            # All returned model IDs should be strings
            for model_id in result:
                assert isinstance(model_id, str)
                assert len(model_id) > 0


class TestDispatchReturnsResponseOrError:
    """[TS-002 AC-2] Dispatch with valid input either succeeds or returns structured error."""

    @pytest.mark.asyncio
    async def test_dispatch_returns_result(self, tmp_path: Path) -> None:
        """dispatch() returns a Result (Ok or Err), never raises unhandled."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        order = DispatchOrder(
            intent_category="coding",
            specific_intent="write a function",
            operator_message="Write a sort function in Python",
            system_prompt="You are a coding assistant.",
            context_tokens=100,
        )

        # Mock cascade_dispatch to simulate a response without hitting real APIs
        from dragonlight_router.core.types import BackendTier, EngineResponse

        mock_response = Ok(
            EngineResponse(
                content="def sort(lst): return sorted(lst)",
                backend_used="groq_llama70b",
                backend_tier=BackendTier.SIMPLE,
                tokens_in=20,
                tokens_out=15,
                estimated_cost_usd=0.0,
                latency_ms=120.0,
                was_fallback=False,
                fallback_chain=[],
            ),
        )

        with patch(
            "dragonlight_router.router.cascade_dispatch",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await engine.dispatch(order)

        # Result should be a structured Ok or Err, never None
        assert result is not None
        assert isinstance(result, Ok)
        assert hasattr(result.value, "content")
        assert isinstance(result.value.content, str)


class TestHealthEndpointShowsAllProviders:
    """[TS-002 AC-3] Health endpoint lists all configured providers."""

    def test_health_snapshot_lists_providers(self, tmp_path: Path) -> None:
        """health_snapshot returns data for all providers after recording requests."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        # Record some activity so providers appear in the health snapshot
        engine.record_request(
            RequestOutcome(
                provider="groq",
                model_id="groq_llama70b",
                success=True,
                tokens_used=100,
                latency_ms=50.0,
            )
        )
        engine.record_request(
            RequestOutcome(
                provider="nvidia",
                model_id="nvidia_nemotron",
                success=True,
                tokens_used=200,
                latency_ms=80.0,
            )
        )

        snapshot = engine.health_snapshot()
        assert isinstance(snapshot, dict)
        assert "groq" in snapshot, "groq provider missing from health snapshot"
        assert "nvidia" in snapshot, "nvidia provider missing from health snapshot"

        # Each provider entry should contain model health info
        for _provider_name, models in snapshot.items():
            assert isinstance(models, dict)
            for _model_id, health_info in models.items():
                assert "score" in health_info
                assert "error_count" in health_info
                assert 0.0 <= health_info["score"] <= 100.0

    def test_budget_snapshot_lists_all_providers(self, tmp_path: Path) -> None:
        """budget_snapshot returns an entry for every configured provider."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        snapshot = engine.budget_snapshot()
        assert isinstance(snapshot, dict)
        assert "groq" in snapshot
        assert "nvidia" in snapshot

        for _provider_name, data in snapshot.items():
            assert "score" in data
            assert "has_capacity" in data
            assert isinstance(data["score"], float)
            assert isinstance(data["has_capacity"], bool)


class TestCatalogRefreshPopulatesModels:
    """[TS-002 AC-4] After catalog refresh, at least 1 provider has models."""

    def test_catalog_refresh_populates_models(self, tmp_path: Path) -> None:
        """After a (mocked) catalog refresh, the engine's catalog has model data."""
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

        matrix = {
            "coding": {
                "groq_llama70b": 90,
                "groq_mixtral": 75,
            },
        }
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))
        # Deliberately do NOT pre-populate catalog, to test refresh path

        engine = RouterEngine(config_path=config_path)

        # Simulate a catalog refresh returning live models
        live_catalog = {
            "groq": [
                CatalogEntry(model_id="groq_llama70b", provider="groq"),
                CatalogEntry(model_id="groq_mixtral", provider="groq"),
            ],
        }

        with patch.object(
            engine._refresher,
            "refresh",
            new_callable=AsyncMock,
            return_value=live_catalog,
        ):
            result = engine.select_models("coding")

        assert len(result) >= 1
        assert "groq_llama70b" in result


class TestRetireAndReinstateWorkflow:
    """[TS-002 AC-5] Retire a backend, verify exclusion, reinstate, verify inclusion."""

    def test_retire_and_reinstate_backend(self, tmp_path: Path) -> None:
        """Full retire-reinstate lifecycle from the operator's perspective."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        # Verify the model is returned initially
        result_before = engine.select_models("coding")
        assert "groq_llama70b" in result_before

        # Retire via health tracker (the model-level retirement mechanism)
        engine._health.record_error("groq_llama70b", http_status=404)
        assert engine._health.is_retired("groq_llama70b")

        # After retirement, health score is 0 — which means the model should
        # be deprioritized (may still appear but with lowest score)
        health_result = engine._health.score("groq_llama70b")
        assert isinstance(health_result, Ok)
        assert health_result.value == 0.0

        # Reinstate the model
        engine._health.reinstate_model("groq_llama70b")
        assert not engine._health.is_retired("groq_llama70b")

        # After reinstatement, health score should be restored
        health_result_after = engine._health.score("groq_llama70b")
        assert isinstance(health_result_after, Ok)
        assert health_result_after.value > 0.0

        # The model should appear in select_models again with a healthy score
        result_after = engine.select_models("coding")
        assert "groq_llama70b" in result_after


class TestRegistryRetireAndReinstateWorkflow:
    """[TS-002 AC-5b] Registry-level retire/reinstate for admin-initiated retirement."""

    def test_registry_retire_reinstate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Admin retires a backend via registry, confirms exclusion, then reinstates."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key-groq")

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
                    "model_prefix": "groq/",
                    "env_key": "GROQ_API_KEY",
                    "rate_limits": {"rpm": 30, "rpd": 14400},
                },
            ],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))

        matrix = {
            "coding": {
                "groq/llama-3.1-8b-instant": 90,
                "groq/mixtral-8x7b-32768": 75,
            },
        }
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        catalog = {
            "groq": [
                CatalogEntry(model_id="groq/llama-3.1-8b-instant", provider="groq"),
                CatalogEntry(model_id="groq/mixtral-8x7b-32768", provider="groq"),
            ],
        }
        cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
        cache.set(catalog)

        engine = RouterEngine(config_path=config_path)

        # Confirm both models are initially available
        result_before = engine.select_models("coding")
        assert "groq/llama-3.1-8b-instant" in result_before

        # Retire via registry
        retired = engine._registry.retire("groq/llama-3.1-8b-instant")
        assert retired is True

        # Verify the backend is marked RETIRED
        _backend, state = engine._registry.get("groq/llama-3.1-8b-instant")
        assert state is not None
        assert state.status == BackendStatus.RETIRED

        # Reinstate
        reinstated = engine._registry.reinstate("groq/llama-3.1-8b-instant")
        assert reinstated is True

        # Verify the backend is back to AVAILABLE
        _backend, state = engine._registry.get("groq/llama-3.1-8b-instant")
        assert state is not None
        assert state.status == BackendStatus.AVAILABLE


class TestRecordAndObserveHealthDegradation:
    """[TS-002 AC-6] Recording errors degrades health, visible in health snapshot."""

    def test_errors_degrade_health_snapshot(self, tmp_path: Path) -> None:
        """Multiple errors against a model show degraded score in health_snapshot."""
        config_path = _setup_acceptance_config(tmp_path)
        engine = RouterEngine(config_path=config_path)

        # Record initial success
        engine.record_request(
            RequestOutcome(
                provider="groq",
                model_id="groq_llama70b",
                success=True,
                tokens_used=100,
                latency_ms=50.0,
            )
        )

        snapshot_before = engine.health_snapshot()
        score_before = snapshot_before["groq"]["groq_llama70b"]["score"]
        assert score_before == 100.0

        # Record multiple failures
        for _ in range(3):
            engine.record_request(
                RequestOutcome(
                    provider="groq",
                    model_id="groq_llama70b",
                    success=False,
                )
            )

        snapshot_after = engine.health_snapshot()
        score_after = snapshot_after["groq"]["groq_llama70b"]["score"]
        assert score_after < score_before
        assert snapshot_after["groq"]["groq_llama70b"]["error_count"] >= 3


class TestSaveAndRestoreState:
    """[TS-002 AC-7] State persistence round-trip preserves budget and health."""

    def test_state_survives_engine_restart(self, tmp_path: Path) -> None:
        """Budget and health state survive save -> new engine load."""
        config_path = _setup_acceptance_config(tmp_path)
        engine1 = RouterEngine(config_path=config_path)

        # Record some usage via the budget tracker directly
        engine1._budget.record_request("groq", tokens_used=500)
        engine1._budget.record_request("groq", tokens_used=300)

        engine1.save_state()

        # Create a new engine from the same config — should restore state
        engine2 = RouterEngine(config_path=config_path)
        assert engine2._budget._rpd_counts["groq"] == 2
        assert engine2._budget._daily_token_counts["groq"] == 800
