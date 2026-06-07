"""Tests for the MBR capability filtering stage."""

from __future__ import annotations

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendTier,
    DispatchOrder,
    Err,
    Ok,
)
from dragonlight_router.selection.mbr import filter_by_capabilities, MBRNoCandidatesError


def _make_backend(
    name: str,
    max_context_tokens: int,
    supports_tool_use: bool,
    supports_streaming: bool,
    supports_json_mode: bool,
    supports_system_prompts: bool,
) -> BackendConfig:
    """Helper to create a BackendConfig for testing."""
    caps = BackendCapabilities(
        max_context_tokens=max_context_tokens,
        supports_tool_use=supports_tool_use,
        supports_streaming=supports_streaming,
        supports_json_mode=supports_json_mode,
        supports_system_prompts=supports_system_prompts,
    )
    return BackendConfig(
        name=name,
        provider="test",
        model=name,
        tier=BackendTier.LOCAL,
        base_url=f"http://{name}",
        env_key=None,
        capabilities=caps,
        cost=None,  # type: ignore
        rate_limits=None,  # type: ignore
    )


def test_filter_by_capabilities_no_requirements() -> None:
    """Test that all candidates pass when no requirements are specified."""
    backend1 = _make_backend("test1", 4096, False, True, False, True)
    backend2 = _make_backend("test2", 8192, True, True, True, True)

    candidates = [backend1, backend2]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Ok)
    filtered = result.value
    assert len(filtered) == 2
    assert backend1 in filtered
    assert backend2 in filtered


def test_filter_by_capabilities_context_tokens() -> None:
    """Test filtering by context token requirements."""
    backend1 = _make_backend("test1", 4096, False, True, False, True)
    backend2 = _make_backend("test2", 8192, True, True, True, True)

    candidates = [backend1, backend2]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=6000,
        requires_tool_use=False,
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Ok)
    filtered = result.value
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 has enough context tokens
    assert backend1 not in filtered  # backend1 has only 4096 tokens


def test_filter_by_capabilities_tool_use() -> None:
    """Test filtering by tool use requirement."""
    backend1 = _make_backend("test1", 8192, False, True, False, True)  # No tool use
    backend2 = _make_backend("test2", 8192, True, True, True, True)  # Supports tools

    candidates = [backend1, backend2]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=True,
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Ok)
    filtered = result.value
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports tool use
    assert backend1 not in filtered  # backend1 does not support tool use


def test_filter_by_capabilities_system_prompt() -> None:
    """Test filtering by system prompt requirement."""
    backend1 = _make_backend("test1", 8192, False, True, False, False)  # No system prompt
    backend2 = _make_backend("test2", 8192, False, True, False, True)  # Supports system prompts

    candidates = [backend1, backend2]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="You are a helpful assistant",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Ok)
    filtered = result.value
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports system prompts
    assert backend1 not in filtered  # backend1 does not support system prompts


def test_filter_by_capabilities_multiple_requirements() -> None:
    """Test filtering with multiple capability requirements."""
    backend1 = _make_backend("test1", 4096, False, True, False, True)
    backend2 = _make_backend("test2", 8192, True, True, True, True)
    backend3 = _make_backend("test3", 6000, True, False, True, True)  # No streaming

    candidates = [backend1, backend2, backend3]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="You are a helpful assistant",
        context_tokens=5000,
        requires_tool_use=True,
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Ok)
    filtered = result.value
    assert len(filtered) == 2
    assert backend2 in filtered  # backend2 meets all requirements
    assert backend3 in filtered  # backend3 meets all requirements (streaming not required)
    assert backend1 not in filtered  # backend1 fails context tokens and tool use


def test_filter_by_capabilities_no_candidates() -> None:
    """Test that Err is returned when no candidates meet requirements."""
    backend1 = _make_backend("test1", 4096, False, True, False, True)
    backend2 = _make_backend("test2", 2048, False, True, False, True)  # Low context tokens

    candidates = [backend1, backend2]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="You are a helpful assistant",
        context_tokens=5000,  # Requires high context tokens
        requires_tool_use=True,  # Requires tool use
        requires_long_context=False,
    )

    result = filter_by_capabilities(candidates, order)
    assert isinstance(result, Err)
    assert isinstance(result.error, MBRNoCandidatesError)


def test_filter_by_capabilities_assertions_candidates_not_list() -> None:
    """Test that appropriate assertion is raised when candidates is not a list."""
    try:
        filter_by_capabilities("not a list", DispatchOrder(  # type: ignore
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


def test_filter_by_capabilities_assertions_candidates_not_backendconfig() -> None:
    """Test that appropriate assertion is raised when candidates contains non-BackendConfig."""
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


def test_filter_by_capabilities_assertions_order_not_dispatchorder() -> None:
    """Test that appropriate assertion is raised when order is not a DispatchOrder."""
    try:
        filter_by_capabilities([], "not an order")  # type: ignore
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "order must be DispatchOrder instance" in str(e)


def test_estimate_complexity() -> None:
    """Test the estimate_complexity function."""
    from dragonlight_router.selection.mbr import estimate_complexity

    # Base case: no requirements -> LOCAL
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    assert estimate_complexity(order) == BackendTier.LOCAL

    # Long context -> SIMPLE
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=5000,
        requires_tool_use=False,
        requires_long_context=False,
    )
    assert estimate_complexity(order) == BackendTier.SIMPLE

    # Tool use -> MODERATE
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=True,
        requires_long_context=False,
    )
    assert estimate_complexity(order) == BackendTier.MODERATE

    # Reasoning or very long context -> COMPLEX
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=9000,
        requires_tool_use=False,
        requires_long_context=False,
    )
    assert estimate_complexity(order) == BackendTier.COMPLEX

    # Multiple upgrades should still result in the highest tier
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=9000,
        requires_tool_use=True,
        requires_long_context=True,
    )
    assert estimate_complexity(order) == BackendTier.COMPLEX