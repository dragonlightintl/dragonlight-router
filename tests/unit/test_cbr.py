"""Unit tests for the CBR (Cost Balancing) stage."""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import BackendConfig, BackendCapabilities, BackendCostProfile, BackendRateLimits, DispatchOrder
from dragonlight_router.selection.cbr import filter_by_absolute_cost, filter_by_cost_efficiency


def make_backend_config(
    name: str,
    input_cost: float = 1.0,
    output_cost: float = 1.0,
    tpm_limit: int = 1000,
    daily_token_cap: int = 100000,
) -> BackendConfig:
    """Helper to create a BackendConfig for testing."""
    return BackendConfig(
        name=name,
        provider="test_provider",
        model=name,
        tier=None,  # type: ignore
        base_url="http://test",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=False,
            supports_json_mode=False,
            supports_system_prompts=False,
        ),
        cost=BackendCostProfile(
            input_per_mtok=input_cost,
            output_per_mtok=output_cost,
            cache_read_per_mtok=0.0,
            cache_write_per_mtok=0.0,
        ),
        rate_limits=BackendRateLimits(
            rpm=100,
            rpd=1000,
            tpm=tpm_limit,
            daily_token_cap=daily_token_cap,
        ),
        priority=0,
    )


def test_filter_by_cost_efficiency():
    """Test that filter_by_cost_efficiency keeps candidates with higher efficiency."""
    # Create three backends: low cost, medium cost, high cost
    low_cost = make_backend_config("low", input_cost=1.0, output_cost=1.0)  # avg cost 1.0
    medium_cost = make_backend_config("medium", input_cost=2.0, output_cost=2.0)  # avg cost 2.0
    high_cost = make_backend_config("high", input_cost=4.0, output_cost=4.0)  # avg cost 4.0

    candidates = [low_cost, medium_cost, high_cost]

    # Budget scores: all providers have the same budget score (say 80)
    budget_scores = {
        "test_provider": 80.0,
    }

    # Dispatch order (not used in the current implementation, but required)
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )

    # Filter
    filtered = filter_by_cost_efficiency(candidates, budget_scores, order)

    # We expect the median efficiency to be that of medium cost.
    # Efficiencies: low: 80/1 = 80, medium: 80/2 = 40, high: 80/4 = 20
    # Sorted: [20, 40, 80] -> median = 40
    # So we keep candidates with efficiency >= 40: low and medium.
    assert len(filtered) == 2
    assert {c.name for c in filtered} == {"low", "medium"}


def test_filter_by_cost_efficiency_no_budget_data():
    """Test that if no budget data, we return candidates as-is."""
    candidates = [make_backend_config("test")]
    budget_scores: dict[str, float] = {}
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    filtered = filter_by_cost_efficiency(candidates, budget_scores, order)
    assert filtered == candidates


def test_filter_by_cost_efficiency_no_candidates():
    """Test that if no candidates, we return empty list."""
    candidates: list[BackendConfig] = []
    budget_scores = {"test_provider": 80.0}
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    filtered = filter_by_cost_efficiency(candidates, budget_scores, order)
    assert filtered == []


def test_filter_by_absolute_cost():
    """Test that filter_by_absolute_cost filters by max cost."""
    low_cost = make_backend_config("low", input_cost=1.0, output_cost=1.0)  # avg 1.0
    high_cost = make_backend_config("high", input_cost=10.0, output_cost=10.0)  # avg 10.0

    candidates = [low_cost, high_cost]

    # Allow only up to 5.0 average cost per million tokens
    max_cost = 5.0

    filtered = filter_by_absolute_cost(candidates, max_cost)

    # Only low cost should pass
    assert len(filtered) == 1
    assert filtered[0].name == "low"


def test_filter_by_absolute_cost_zero_max():
    """Test that with max cost 0, only zero-cost candidates pass."""
    zero_cost = make_backend_config("zero", input_cost=0.0, output_cost=0.0)
    low_cost = make_backend_config("low", input_cost=1.0, output_cost=1.0)

    candidates = [zero_cost, low_cost]
    max_cost = 0.0

    filtered = filter_by_absolute_cost(candidates, max_cost)

    assert len(filtered) == 1
    assert filtered[0].name == "zero"