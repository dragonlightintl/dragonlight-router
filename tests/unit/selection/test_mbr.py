"""Tests for the MBR capability filtering stage."""

from __future__ import annotations

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendTier,
    DispatchOrder,
)
from dragonlight_router.selection.mbr import filter_by_capabilities


def test_filter_by_capabilities_no_requirements():
    """Test that all candidates pass when no requirements are specified."""
    # Create test backends
    caps1 = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    caps2 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )

    backend1 = BackendConfig(
        name="test1",
        provider="test",
        model="test1",
        tier=BackendTier.LOCAL,
        base_url="http://test1",
        env_key=None,
        capabilities=caps1,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend2 = BackendConfig(
        name="test2",
        provider="test",
        model="test2",
        tier=BackendTier.SIMPLE,
        base_url="http://test2",
        env_key=None,
        capabilities=caps2,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )

    candidates = [backend1, backend2]
    # Order with no requirements
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",  # Empty means no system prompt required
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )

    filtered = filter_by_capabilities(candidates, order)
    assert len(filtered) == 2
    assert backend1 in filtered
    assert backend2 in filtered


def test_filter_by_capabilities_context_tokens():
    """Test filtering by context token requirements."""
    caps1 = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    caps2 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )

    backend1 = BackendConfig(
        name="test1",
        provider="test",
        model="test1",
        tier=BackendTier.LOCAL,
        base_url="http://test1",
        env_key=None,
        capabilities=caps1,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend2 = BackendConfig(
        name="test2",
        provider="test",
        model="test2",
        tier=BackendTier.SIMPLE,
        base_url="http://test2",
        env_key=None,
        capabilities=caps2,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )

    candidates = [backend1, backend2]
    # Order requiring 6000 context tokens
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=6000,
        requires_tool_use=False,
        requires_long_context=False,
    )

    filtered = filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 has enough context tokens
    assert backend1 not in filtered  # backend1 has only 4096 tokens


def test_filter_by_capabilities_tool_use():
    """Test filtering by tool use requirement."""
    caps1 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=False,  # Does not support tools
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    caps2 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=True,  # Supports tools
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )

    backend1 = BackendConfig(
        name="test1",
        provider="test",
        model="test1",
        tier=BackendTier.LOCAL,
        base_url="http://test1",
        env_key=None,
        capabilities=caps1,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend2 = BackendConfig(
        name="test2",
        provider="test",
        model="test2",
        tier=BackendTier.SIMPLE,
        base_url="http://test2",
        env_key=None,
        capabilities=caps2,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )

    candidates = [backend1, backend2]
    # Order requiring tool use
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=True,
        requires_long_context=False,
    )

    filtered = filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports tool use
    assert backend1 not in filtered  # backend1 does not support tool use


def test_filter_by_capabilities_system_prompt():
    """Test filtering by system prompt requirement."""
    caps1 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=False,  # Does not support system prompts
    )
    caps2 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,  # Supports system prompts
    )

    backend1 = BackendConfig(
        name="test1",
        provider="test",
        model="test1",
        tier=BackendTier.LOCAL,
        base_url="http://test1",
        env_key=None,
        capabilities=caps1,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend2 = BackendConfig(
        name="test2",
        provider="test",
        model="test2",
        tier=BackendTier.SIMPLE,
        base_url="http://test2",
        env_key=None,
        capabilities=caps2,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )

    candidates = [backend1, backend2]
    # Order with system prompt (non-empty string)
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="You are a helpful assistant",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )

    filtered = filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports system prompts
    assert backend1 not in filtered  # backend1 does not support system prompts


def test_filter_by_capabilities_multiple_requirements():
    """Test filtering with multiple capability requirements."""
    caps1 = BackendCapabilities(
        max_context_tokens=4096,
        supports_tool_use=False,
        supports_streaming=True,
        supports_json_mode=False,
        supports_system_prompts=True,
    )
    caps2 = BackendCapabilities(
        max_context_tokens=8192,
        supports_tool_use=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )
    caps3 = BackendCapabilities(
        max_context_tokens=6000,
        supports_tool_use=True,
        supports_streaming=False,  # Does not support streaming
        supports_json_mode=True,
        supports_system_prompts=True,
    )

    backend1 = BackendConfig(
        name="test1",
        provider="test",
        model="test1",
        tier=BackendTier.LOCAL,
        base_url="http://test1",
        env_key=None,
        capabilities=caps1,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend2 = BackendConfig(
        name="test2",
        provider="test",
        model="test2",
        tier=BackendTier.SIMPLE,
        base_url="http://test2",
        env_key=None,
        capabilities=caps2,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )
    backend3 = BackendConfig(
        name="test3",
        provider="test",
        model="test3",
        tier=BackendTier.MODERATE,
        base_url="http://test3",
        env_key=None,
        capabilities=caps3,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )

    candidates = [backend1, backend2, backend3]
    # Order requiring: 5000+ context tokens, tool use, and system prompts
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="You are a helpful assistant",
        context_tokens=5000,
        requires_tool_use=True,
        requires_long_context=False,
    )

    filtered = filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 meets all requirements
    assert backend1 not in filtered  # Not enough context tokens, no tool use
    assert backend3 not in filtered  # No streaming (not required) but we're not testing that


def test_filter_by_capabilities_assertions():
    """Test that appropriate assertions are raised for invalid inputs."""
    # Test candidates not a list
    try:
        filter_by_capabilities(["not a list"], DispatchOrder(  # type: ignore
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        ))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "candidates must be a list" in str(e)

    # Test candidates containing non-BackendConfig
    try:
        filter_by_capabilities(["not a backend"], DispatchOrder(  # type: ignore
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        ))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "all candidates must be BackendConfig instances" in str(e)

    # Test order not a DispatchOrder
    try:
        filter_by_capabilities([], "not an order")  # type: ignore
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "order must be DispatchOrder instance" in str(e)