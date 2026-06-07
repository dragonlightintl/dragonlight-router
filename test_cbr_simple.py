#!/usr/bin/env python3
"""Simple test for CBR functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dragonlight_router.core.types import BackendConfig, BackendCapabilities, BackendCostProfile, BackendRateLimits, DispatchOrder, BackendTier
from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import ProviderConfig, Result
from dragonlight_router.selection.cbr import filter_by_cost

def make_backend_config(name: str, provider: str, input_cost: float = 1.0, output_cost: float = 1.0):
    """Helper to create a BackendConfig for testing."""
    return BackendConfig(
        name=name,
        provider=provider,
        model=name,
        tier=BackendTier.LOCAL,
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
            tpm=1000,
            daily_token_cap=100000,
        ),
        priority=0,
    )

def test_cbr_basic():
    """Test basic CBR functionality."""
    print("Testing CBR basic functionality...")
    
    # Create backends
    low_cost = make_backend_config("low-cost", "provider1", 1.0, 1.0)
    high_cost = make_backend_config("high-cost", "provider1", 10.0, 10.0)
    
    # Create provider config for budget tracker
    provider_config = ProviderConfig(
        name="provider1",
        base_url="http://test",
        catalog_url=None,
        env_key=None,
        model_prefix="",
        rpm_limit=100,
        rpd_limit=1000,
        tpm_limit=1000,
        daily_token_cap=100000,
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
    
    # Test with both candidates
    candidates = [low_cost, high_cost]
    result = filter_by_cost(candidates, order, budget_tracker)
    
    assert result.is_ok, f"Expected Ok result, got {result}"
    filtered = result.value
    print(f"Filtered candidates: {[c.name for c in filtered]}")
    
    # Should have both candidates since budget is available
    assert len(filtered) == 2, f"Expected 2 candidates, got {len(filtered)}"
    assert low_cost in filtered
    assert high_cost in filtered
    
    print("✓ Basic CBR test passed")

def test_cbr_budget_exceeded():
    """Test CBR when budget is exhausted."""
    print("Testing CBR budget exceeded...")
    
    # Create backend
    backend = make_backend_config("test", "provider1", 1.0, 1.0)
    
    # Create provider config with zero limits (simulating exhausted budget)
    provider_config = ProviderConfig(
        name="provider1",
        base_url="http://test",
        catalog_url=None,
        env_key=None,
        model_prefix="",
        rpm_limit=0,  # Zero RPM means no budget
        rpd_limit=0,
        tpm_limit=0,
        daily_token_cap=0,
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
    
    # Test with candidate that has no budget
    candidates = [backend]
    result = filter_by_cost(candidates, order, budget_tracker)
    
    assert result.is_err, f"Expected Err result when budget exceeded, got {result}"
    print("✓ Budget exceeded test passed")

def test_cbr_no_candidates():
    """Test CBR with no candidates."""
    print("Testing CBR with no candidates...")
    
    # Create empty provider list
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
    
    # Test with empty candidates
    candidates = []
    result = filter_by_cost(candidates, order, budget_tracker)
    
    assert result.is_ok, f"Expected Ok result, got {result}"
    filtered = result.value
    assert filtered == [], f"Expected empty list, got {filtered}"
    
    print("✓ No candidates test passed")

if __name__ == "__main__":
    test_cbr_basic()
    test_cbr_budget_exceeded()
    test_cbr_no_candidates()
    print("\nAll tests passed! ✓")