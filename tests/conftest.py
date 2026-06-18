"""Shared test fixtures for the dragonlight-router test suite.

These fixtures provide sensible defaults for the core frozen dataclasses
used throughout the routing system. Existing tests that construct their
own objects directly are unaffected -- these are purely additive.
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

# ---------------------------------------------------------------------------
# BackendConfig
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_backend_config() -> BackendConfig:
    """A BackendConfig with sensible test defaults.

    Represents a complex-tier backend with moderate context, no tool use,
    streaming support, and low per-token costs.
    """
    return BackendConfig(
        name="test-backend",
        provider="test-provider",
        model="test-model",
        tier=BackendTier.COMPLEX,
        base_url="https://api.test-provider.example.com/v1",
        env_key="TEST_PROVIDER_API_KEY",
        capabilities=BackendCapabilities(
            max_context_tokens=8192,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(
            input_per_mtok=1.0,
            output_per_mtok=2.0,
            cache_read_per_mtok=0.0,
            cache_write_per_mtok=0.0,
        ),
        rate_limits=BackendRateLimits(
            rpm=60,
            rpd=14400,
            tpm=100000,
            daily_token_cap=1000000,
        ),
        priority=0,
    )


# ---------------------------------------------------------------------------
# DispatchOrder
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dispatch_order() -> DispatchOrder:
    """A DispatchOrder with sensible test defaults.

    Represents a straightforward code-generation request with no special
    capability requirements.
    """
    return DispatchOrder(
        intent_category="code_generation",
        specific_intent="write_function",
        operator_message="Write a Python function to calculate fibonacci numbers",
        system_prompt="You are a helpful coding assistant",
        context_tokens=100,
        requires_tool_use=False,
        requires_long_context=False,
    )


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_provider_config() -> ProviderConfig:
    """A ProviderConfig with sensible test defaults.

    Matches the provider identity used by sample_backend_config so the
    two can be composed naturally.
    """
    return ProviderConfig(
        name="test-provider",
        base_url="https://api.test-provider.example.com/v1",
        catalog_url="https://api.test-provider.example.com/v1/models",
        env_key="TEST_PROVIDER_API_KEY",
        model_prefix="test_",
        rpm_limit=60,
        rpd_limit=14400,
        tpm_limit=100000,
        daily_token_cap=1000000,
    )


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_budget_tracker(sample_provider_config: ProviderConfig) -> BudgetTracker:
    """A BudgetTracker initialized with sample_provider_config.

    Starts with a full budget -- no requests recorded yet.
    """
    return BudgetTracker(providers=[sample_provider_config])
