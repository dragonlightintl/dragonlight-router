"""Unit tests for provider-level cost ceiling filtering.

Spec traceability: HAZ-015 (Provider cost ceiling enforcement)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dragonlight_router.core.types import BackendTier
from dragonlight_router.dispatch.cascade import (
    DispatchContext,
    _filter_by_cost_ceiling,
    _run_cascade,
)
from dragonlight_router.result import Ok
from dragonlight_router.selection.mbr import MBRNoCandidatesError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_with_ceilings(provider_ceilings: dict[str, float | None]) -> dict:
    """Build a config dict with provider cost ceilings."""
    providers = []
    for name, ceiling in provider_ceilings.items():
        entry: dict = {"name": name}
        if ceiling is not None:
            entry["max_cost_per_mtok"] = ceiling
        providers.append(entry)
    return {"providers": providers}


def _make_order(**kwargs):
    from dragonlight_router.core.types import DispatchOrder

    defaults = {
        "intent_category": "test",
        "specific_intent": "test",
        "operator_message": "hello",
        "system_prompt": "",
        "context_tokens": 0,
        "requires_tool_use": False,
        "requires_long_context": False,
    }
    defaults.update(kwargs)
    return DispatchOrder(**defaults)


def _make_ctx(config=None):
    from dragonlight_router.core.registry import BackendRegistry

    registry = MagicMock(spec=BackendRegistry)
    budget_tracker = MagicMock()
    health_tracker = MagicMock()
    health_tracker.is_retired.return_value = False
    config = config if config is not None else {}
    return DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
    )


# ---------------------------------------------------------------------------
# _filter_by_cost_ceiling unit tests
# ---------------------------------------------------------------------------


class TestFilterByCostCeiling:
    """Provider-level cost ceiling enforcement."""

    def test_models_under_ceiling_included(self, make_backend_config):
        """Models with cost at or below the ceiling are kept."""
        cheap = make_backend_config(name="cheap", provider="openrouter", input_cost=0.0)
        config = _make_config_with_ceilings({"openrouter": 1.0})
        result = _filter_by_cost_ceiling([cheap], config)
        assert len(result) == 1
        assert result[0].name == "cheap"

    def test_models_over_ceiling_excluded(self, make_backend_config):
        """Models with cost above the ceiling are filtered out."""
        expensive = make_backend_config(name="expensive", provider="openrouter", input_cost=5.0)
        config = _make_config_with_ceilings({"openrouter": 1.0})
        result = _filter_by_cost_ceiling([expensive], config)
        assert len(result) == 0

    def test_none_ceiling_no_filtering(self, make_backend_config):
        """When max_cost_per_mtok is None (absent), all models pass."""
        expensive = make_backend_config(name="expensive", provider="openrouter", input_cost=99.0)
        config = _make_config_with_ceilings({"openrouter": None})
        result = _filter_by_cost_ceiling([expensive], config)
        assert len(result) == 1

    def test_zero_ceiling_allows_only_free(self, make_backend_config):
        """Setting max_cost_per_mtok to 0.0 allows only free (zero-cost) models."""
        free = make_backend_config(name="free", provider="openrouter", input_cost=0.0)
        paid = make_backend_config(name="paid", provider="openrouter", input_cost=0.5)
        config = _make_config_with_ceilings({"openrouter": 0.0})
        result = _filter_by_cost_ceiling([free, paid], config)
        assert len(result) == 1
        assert result[0].name == "free"

    def test_mixed_providers_filtered_independently(self, make_backend_config):
        """Each provider's ceiling applies only to its own models."""
        or_free = make_backend_config(name="or-free", provider="openrouter", input_cost=0.0)
        or_paid = make_backend_config(name="or-paid", provider="openrouter", input_cost=1.0)
        groq_model = make_backend_config(name="groq-model", provider="groq", input_cost=5.0)
        config = _make_config_with_ceilings({"openrouter": 0.0})
        result = _filter_by_cost_ceiling([or_free, or_paid, groq_model], config)
        assert len(result) == 2
        names = [c.name for c in result]
        assert "or-free" in names
        assert "groq-model" in names

    def test_empty_config_no_filtering(self, make_backend_config):
        """Empty config dict passes all candidates through."""
        model = make_backend_config(name="m1", provider="openrouter", input_cost=10.0)
        result = _filter_by_cost_ceiling([model], {})
        assert len(result) == 1

    def test_empty_candidates_returns_empty(self):
        """Empty candidate list returns empty."""
        config = _make_config_with_ceilings({"openrouter": 0.0})
        result = _filter_by_cost_ceiling([], config)
        assert result == []

    def test_exact_ceiling_value_included(self, make_backend_config):
        """A model whose cost exactly equals the ceiling is included."""
        exact = make_backend_config(name="exact", provider="openrouter", input_cost=1.5)
        config = _make_config_with_ceilings({"openrouter": 1.5})
        result = _filter_by_cost_ceiling([exact], config)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_cascade_returns_err_when_cost_ceiling_removes_all(self, make_backend_config):
        """_run_cascade returns Err when cost ceiling filters all candidates."""
        paid = make_backend_config(
            name="paid", provider="openrouter", tier=BackendTier.SIMPLE, input_cost=5.0,
        )
        config = _make_config_with_ceilings({"openrouter": 0.0})
        order = _make_order()
        ctx = _make_ctx(config=config)

        with patch(
            "dragonlight_router.dispatch.cascade._run_mbr_stage",
            return_value=Ok([paid]),
        ):
            result = await _run_cascade(order, ctx)

        assert result.is_err()
        assert isinstance(result.error, MBRNoCandidatesError)
        assert "cost ceiling" in str(result.error)
