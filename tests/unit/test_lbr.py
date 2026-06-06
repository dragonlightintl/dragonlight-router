"""Unit tests for LBR (rate-limit-aware dispatch) stage."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.selection.lbr import filter_by_rate_limit


def _make_backend_config(
    name: str,
    provider: str,
    model: str = "test-model",
    tier: BackendTier = BackendTier.COMPLEX,
    base_url: str = "https://example.com",
    env_key: str | None = None,
) -> BackendConfig:
    """Helper to create a BackendConfig with sensible defaults for testing."""
    return BackendConfig(
        name=name,
        provider=provider,
        model=model,
        tier=tier,
        base_url=base_url,
        env_key=env_key,
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(rpm=10, rpd=100, tpm=1000, daily_token_cap=10000),
        priority=0,
    )


def test_filter_by_rate_limit_no_candidates():
    """When no candidates are provided, return empty list."""
    candidates = []
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = BudgetTracker(providers=[])
    result = filter_by_rate_limit(candidates, order, budget_tracker)
    assert result == []


def test_filter_by_rate_limit_no_budget_data():
    """When budget tracker returns no data, return candidates as-is."""
    # Create a candidate
    config = _make_backend_config(name="test", provider="test")
    candidates = [config]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    # Budget tracker returns Err for the provider (simulate not found)
    budget_tracker = BudgetTracker(providers=[])  # No providers configured

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Should return as-is because no score data (all zeros)
    assert result == candidates


def test_filter_by_rate_limit_median_filter():
    """Test that candidates with score >= median are kept."""
    # Create two candidates with different providers
    config_a = _make_backend_config(name="a", provider="provider_a")
    config_b = _make_backend_config(name="b", provider="provider_b")
    candidates = [config_a, config_b]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    # We need a budget tracker that returns specific scores for each provider.
    # Since BudgetTracker.score uses internal state, we'll mock it.
    budget_tracker = Mock(spec=BudgetTracker)
    # We'll make score return Ok(80.0) for provider_a and Ok(20.0) for provider_b
    def score_side_effect(provider_name):
        if provider_name == "provider_a":
            return Ok(80.0)
        else:
            return Ok(20.0)
    budget_tracker.score.side_effect = score_side_effect

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Median of [80.0, 20.0] is 50.0, so only provider_a (80.0 >= 50.0) should be kept
    assert len(result) == 1
    assert result[0].provider == "provider_a"


def test_filter_by_rate_limit_all_zero_scores():
    """When all scores are zero, return candidates as-is (median zero, all pass)."""
    config_a = _make_backend_config(name="a", provider="provider_a")
    config_b = _make_backend_config(name="b", provider="provider_b")
    candidates = [config_a, config_b]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = Mock(spec=BudgetTracker)
    budget_tracker.score.return_value = Ok(0.0)

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Median of [0.0, 0.0] is 0.0, so both candidates pass (score >= 0)
    assert len(result) == 2
    assert result == candidates


def test_filter_by_rate_limit_guard_clauses():
    """Guard clauses should raise AssertionError on invalid types."""
    # candidates not a list
    with pytest.raises(AssertionError, match="candidates must be a list"):
        filter_by_rate_limit(
            "not a list",
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            BudgetTracker(providers=[]),
        )
    # candidate not BackendConfig
    with pytest.raises(AssertionError, match="all candidates must be BackendConfig instances"):
        filter_by_rate_limit(
            ["not a config"],
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            BudgetTracker(providers=[]),
        )
    # order not DispatchOrder
    with pytest.raises(AssertionError, match="order must be DispatchOrder instance"):
        filter_by_rate_limit(
            [],
            "not order",
            BudgetTracker(providers=[]),
        )
    # budget_tracker not BudgetTracker
    with pytest.raises(AssertionError, match="budget_tracker must be BudgetTracker instance"):
        filter_by_rate_limit(
            [],
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            "not tracker",
        )