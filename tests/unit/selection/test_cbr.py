"""Tests for the CBR cost balancing stage.

Spec traceability: TM-002 (CBR cost-efficiency filtering)
"""

from __future__ import annotations

import pytest

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
    ProviderConfig,
)
from dragonlight_router.selection.cbr import filter_by_cost


def test_filter_by_cost_no_candidates():
    """[TM-002 AC-1] Empty candidate list returns empty list."""
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
    
    filtered = filter_by_cost([], order, budget_tracker)
    assert filtered == []


def test_filter_by_cost_single_candidate():
    """[TM-002 AC-1] Single candidate is always retained."""
    # Create a backend config
    capabilities = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    cost = BackendCostProfile(
        input_per_mtok=10.0,  # $10 per million input tokens
        output_per_mtok=20.0,  # $20 per million output tokens
    )
    rate_limits = BackendRateLimits(
        rpm=60,
        rpd=1000,
        tpm=10000,
        daily_token_cap=1000000,
    )
    backend = BackendConfig(
        name="test-backend",
        provider="test-provider",
        model="test-model",
        tier=BackendTier.LOCAL,
        base_url="http://test",
        env_key=None,
        capabilities=capabilities,
        cost=cost,
        rate_limits=rate_limits,
    )
    
    # Create budget tracker with the provider
    provider_config = ProviderConfig(
        name="test-provider",
        base_url="http://test",
        catalog_url=None,
        env_key=None,
        model_prefix="",
        rpm_limit=60,
        rpd_limit=1000,
        tpm_limit=10000,
        daily_token_cap=1000000,
    )
    budget_tracker = BudgetTracker(providers=[provider_config])
    
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    
    candidates = [backend]
    filtered = filter_by_cost(candidates, order, budget_tracker)
    
    # With single candidate, we should keep it (max(1, 1//2) = 1)
    assert len(filtered) == 1
    assert backend in filtered


def test_filter_by_cost_multiple_candidates():
    """[TM-002 AC-1] Multiple candidates filtered to keep more cost-effective ones."""
    # Create two backends with different costs
    capabilities = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    
    # Low cost backend
    low_cost = BackendCostProfile(
        input_per_mtok=1.0,   # $1 per million
        output_per_mtok=2.0,  # $2 per million
    )
    # High cost backend
    high_cost = BackendCostProfile(
        input_per_mtok=100.0,  # $100 per million
        output_per_mtok=200.0, # $200 per million
    )
    
    rate_limits = BackendRateLimits(
        rpm=60,
        rpd=1000,
        tpm=10000,
        daily_token_cap=1000000,
    )
    
    low_backend = BackendConfig(
        name="low-cost-backend",
        provider="test-provider",
        model="low-cost-model",
        tier=BackendTier.LOCAL,
        base_url="http://test",
        env_key=None,
        capabilities=capabilities,
        cost=low_cost,
        rate_limits=rate_limits,
    )
    
    high_backend = BackendConfig(
        name="high-cost-backend",
        provider="test-provider",
        model="high-cost-model",
        tier=BackendTier.LOCAL,
        base_url="http://test",
        env_key=None,
        capabilities=capabilities,
        cost=high_cost,
        rate_limits=rate_limits,
    )
    
    # Create budget tracker with the provider
    provider_config = ProviderConfig(
        name="test-provider",
        base_url="http://test",
        catalog_url=None,
        env_key=None,
        model_prefix="",
        rpm_limit=60,
        rpd_limit=1000,
        tpm_limit=10000,
        daily_token_cap=1000000,
    )
    budget_tracker = BudgetTracker(providers=[provider_config])
    
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    
    candidates = [low_backend, high_backend]
    filtered = filter_by_cost(candidates, order, budget_tracker)
    
    # Should keep at least one, and likely the low cost one due to higher efficiency
    assert len(filtered) >= 1
    # The low cost backend should have higher efficiency (budget score / cost)
    # Since both have same provider, same budget score, lower cost = higher efficiency
    assert low_backend in filtered


def test_filter_by_cost_assertions():
    """[TM-002 AC-2] Guard clauses raise AssertionError for invalid inputs."""
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
    
    # Test candidates not a list
    try:
        filter_by_cost("not a list", order, budget_tracker)  # type: ignore
        pytest.fail("Should have raised AssertionError")
    except AssertionError as e:
        assert "candidates must be a list" in str(e)

    # Test candidates containing non-BackendConfig
    try:
        filter_by_cost(["not a backend"], order, budget_tracker)  # type: ignore
        pytest.fail("Should have raised AssertionError")
    except AssertionError as e:
        assert "all candidates must be BackendConfig instances" in str(e)

    # Test order not a DispatchOrder
    try:
        filter_by_cost([], "not an order", budget_tracker)  # type: ignore
        pytest.fail("Should have raised AssertionError")
    except AssertionError as e:
        assert "order must be DispatchOrder instance" in str(e)

    # Test budget_tracker not a BudgetTracker
    try:
        filter_by_cost([], order, "not a tracker")  # type: ignore
        pytest.fail("Should have raised AssertionError")
    except AssertionError as e:
        assert "budget_tracker must be BudgetTracker instance" in str(e)


def test_filter_by_cost_provider_not_found():
    """[TM-002 AC-2] Unknown provider in budget tracker still returns candidates."""
    capabilities = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    cost = BackendCostProfile(
        input_per_mtok=10.0,
        output_per_mtok=20.0,
    )
    rate_limits = BackendRateLimits(
        rpm=60,
        rpd=1000,
        tpm=10000,
        daily_token_cap=1000000,
    )
    backend = BackendConfig(
        name="test-backend",
        provider="unknown-provider",  # This provider not in budget tracker
        model="test-model",
        tier=BackendTier.LOCAL,
        base_url="http://test",
        env_key=None,
        capabilities=capabilities,
        cost=cost,
        rate_limits=rate_limits,
    )
    
    # Budget tracker with no providers
    budget_tracker = BudgetTracker(providers=[])
    
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    
    candidates = [backend]
    filtered = filter_by_cost(candidates, order, budget_tracker)
    
    # Should still return the candidate (with budget score 0.0, but we keep at least 1)
    assert len(filtered) == 1
    assert backend in filtered