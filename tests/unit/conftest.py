"""Unit-test-specific fixtures for dragonlight-router.

Provides factory fixtures that make it easy to stamp out customised
BackendConfig and DispatchOrder instances without repeating the full
dataclass construction in every test file.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
)


@pytest.fixture
def make_backend_config() -> Callable[..., BackendConfig]:
    """Factory fixture that returns a builder function for BackendConfig.

    Every parameter has a sensible default, so callers only need to
    override the fields they care about::

        def test_example(make_backend_config):
            fast = make_backend_config(name="fast", input_cost=0.5)
            slow = make_backend_config(name="slow", input_cost=10.0)
    """

    def _build(
        name: str = "test-backend",
        provider: str = "test-provider",
        model: str | None = None,
        tier: BackendTier = BackendTier.COMPLEX,
        base_url: str = "https://api.test-provider.example.com/v1",
        env_key: str | None = None,
        max_context_tokens: int = 8192,
        supports_tool_use: bool = False,
        supports_streaming: bool = True,
        supports_json_mode: bool = False,
        supports_system_prompts: bool = True,
        input_cost: float = 1.0,
        output_cost: float = 2.0,
        cache_read_cost: float = 0.0,
        cache_write_cost: float = 0.0,
        rpm: int = 60,
        rpd: int = 14400,
        tpm: int = 100000,
        daily_token_cap: int = 1000000,
        priority: int = 0,
    ) -> BackendConfig:
        return BackendConfig(
            name=name,
            provider=provider,
            model=model if model is not None else name,
            tier=tier,
            base_url=base_url,
            env_key=env_key,
            capabilities=BackendCapabilities(
                max_context_tokens=max_context_tokens,
                supports_tool_use=supports_tool_use,
                supports_streaming=supports_streaming,
                supports_json_mode=supports_json_mode,
                supports_system_prompts=supports_system_prompts,
            ),
            cost=BackendCostProfile(
                input_per_mtok=input_cost,
                output_per_mtok=output_cost,
                cache_read_per_mtok=cache_read_cost,
                cache_write_per_mtok=cache_write_cost,
            ),
            rate_limits=BackendRateLimits(
                rpm=rpm,
                rpd=rpd,
                tpm=tpm,
                daily_token_cap=daily_token_cap,
            ),
            priority=priority,
        )

    return _build


@pytest.fixture
def make_dispatch_order() -> Callable[..., DispatchOrder]:
    """Factory fixture that returns a builder function for DispatchOrder.

    Callers only override the fields relevant to their test::

        def test_tool_use(make_dispatch_order):
            order = make_dispatch_order(requires_tool_use=True)
    """

    def _build(
        intent_category: str = "test",
        specific_intent: str = "test",
        operator_message: str = "test message",
        system_prompt: str = "",
        context_tokens: int = 0,
        requires_tool_use: bool = False,
        requires_long_context: bool = False,
    ) -> DispatchOrder:
        return DispatchOrder(
            intent_category=intent_category,
            specific_intent=specific_intent,
            operator_message=operator_message,
            system_prompt=system_prompt,
            context_tokens=context_tokens,
            requires_tool_use=requires_tool_use,
            requires_long_context=requires_long_context,
        )

    return _build
