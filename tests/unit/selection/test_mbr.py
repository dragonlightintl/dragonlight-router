"""Tests for the MBR capability filtering stage.

Covers AC1-AC3 (capability filtering) and AC4-AC5 (no-downgrade invariant,
LOCAL passthrough).  AC4/AC5 tests exercise the public filter_by_capabilities
function with a real BackendRegistry so they verify end-to-end behaviour.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    DispatchOrder,
    Err,
    Ok,
)
from dragonlight_router.selection.mbr import (
    _filter_by_capabilities,
    filter_by_capabilities,
    MBRNoCandidatesError,
    TIER_ORDER,
    _TIER_RANK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(
    name: str,
    max_context_tokens: int,
    supports_tool_use: bool,
    supports_streaming: bool,
    supports_json_mode: bool,
    supports_system_prompts: bool,
) -> BackendConfig:
    """Helper to create a BackendConfig for testing (legacy _filter_by_capabilities tests)."""
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


def _make_config(
    name: str,
    tier: BackendTier,
    *,
    max_context_tokens: int = 128_000,
    supports_tool_use: bool = True,
    supports_streaming: bool = True,
    supports_json_mode: bool = True,
    supports_system_prompts: bool = True,
) -> BackendConfig:
    """Full BackendConfig for registry-level tests (AC4/AC5)."""
    return BackendConfig(
        name=name,
        provider="test-provider",
        model=name,
        tier=tier,
        base_url=f"http://{name}",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=max_context_tokens,
            supports_tool_use=supports_tool_use,
            supports_streaming=supports_streaming,
            supports_json_mode=supports_json_mode,
            supports_system_prompts=supports_system_prompts,
        ),
        cost=BackendCostProfile(0.0, 0.0),
        rate_limits=BackendRateLimits(30, 1000, 6000, 0),
    )


class FakeBackend:
    """Minimal GenerativeBackend satisfying the runtime-checkable protocol."""

    def __init__(self, cfg: BackendConfig) -> None:
        self._config = cfg
        self._status = BackendStatus.AVAILABLE

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        yield "ok"
        yield ""

    async def health_check(self) -> bool:
        return True

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        pass


def _build_registry(*configs: BackendConfig) -> BackendRegistry:
    """Register FakeBackend instances for each config and return the registry."""
    registry = BackendRegistry()
    for cfg in configs:
        registry.register(FakeBackend(cfg))
    return registry


def test_filter_by_capabilities_no_requirements() -> None:
    """[TM-001 AC-1] All candidates pass when no requirements are specified."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert len(filtered) == 2
    assert backend1 in filtered
    assert backend2 in filtered


def test_filter_by_capabilities_context_tokens() -> None:
    """[TM-001 AC-1] Backends with insufficient context tokens are filtered out."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 has enough context tokens
    assert backend1 not in filtered  # backend1 has only 4096 tokens


def test_filter_by_capabilities_tool_use() -> None:
    """[TM-001 AC-2] Backends not supporting tool use are filtered when required."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports tool use
    assert backend1 not in filtered  # backend1 does not support tool use


def test_filter_by_capabilities_system_prompt() -> None:
    """[TM-001 AC-2] Backends not supporting system prompts are filtered when needed."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert len(filtered) == 1
    assert backend2 in filtered  # Only backend2 supports system prompts
    assert backend1 not in filtered  # backend1 does not support system prompts


def test_filter_by_capabilities_multiple_requirements() -> None:
    """[TM-001 AC-2] Multiple capability requirements filter correctly."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert len(filtered) == 2
    assert backend2 in filtered  # backend2 meets all requirements
    assert backend3 in filtered  # backend3 meets all requirements (streaming not required)
    assert backend1 not in filtered  # backend1 fails context tokens and tool use


def test_filter_by_capabilities_no_candidates() -> None:
    """[TM-001 AC-3] Empty list returned when no candidates meet requirements."""
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

    filtered = _filter_by_capabilities(candidates, order)
    assert filtered == []


def test_filter_by_capabilities_assertions_candidates_not_list() -> None:
    """[TM-001 AC-3] AssertionError when candidates is not a list."""
    try:
        _filter_by_capabilities("not a list", DispatchOrder(  # type: ignore
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
    """[TM-001 AC-3] AssertionError when candidates contains non-BackendConfig."""
    try:
        _filter_by_capabilities(["not a backend"], DispatchOrder(  # type: ignore
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
    """[TM-001 AC-3] AssertionError when order is not a DispatchOrder."""
    try:
        _filter_by_capabilities([], "not an order")  # type: ignore
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "order must be DispatchOrder instance" in str(e)


def test_estimate_complexity() -> None:
    """[TM-001 AC-1] estimate_complexity maps order requirements to correct tier."""
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


# ---------------------------------------------------------------------------
# AC4 — MBR never downgrades
# ---------------------------------------------------------------------------

class TestAC4NoDowngrade:
    """TM-001 AC4: MBR never returns candidates from a tier LOWER than requested.

    The invariant is enforced as a postcondition inside filter_by_capabilities.
    These tests verify the invariant holds for every tier combination and that
    an AssertionError fires if a lower-tier candidate somehow slips through
    (guarded by the invariant() helper which survives python -O).
    """

    def test_complex_request_never_returns_simple_backend(self) -> None:
        """[TM-001 AC-4] Requesting COMPLEX tier must not return SIMPLE backends."""
        simple = _make_config("simple-llm", BackendTier.SIMPLE)
        complex_ = _make_config("complex-llm", BackendTier.COMPLEX)
        registry = _build_registry(simple, complex_)

        # context_tokens > 8192 => estimate_complexity returns COMPLEX
        order = DispatchOrder(
            intent_category="code",
            specific_intent="generate",
            operator_message="Build a full application",
            system_prompt="",
            context_tokens=10_000,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok()
        candidates = result.unwrap()
        assert len(candidates) >= 1
        for c in candidates:
            assert _TIER_RANK[c.tier] >= _TIER_RANK[BackendTier.COMPLEX], (
                f"Candidate '{c.name}' at tier '{c.tier.value}' is below COMPLEX"
            )

    def test_moderate_request_never_returns_simple_or_local(self) -> None:
        """[TM-001 AC-4] Requesting MODERATE tier must not return SIMPLE or LOCAL backends."""
        local = _make_config("local-llm", BackendTier.LOCAL)
        simple = _make_config("simple-llm", BackendTier.SIMPLE)
        moderate = _make_config("moderate-llm", BackendTier.MODERATE)
        registry = _build_registry(local, simple, moderate)

        # requires_tool_use=True, low tokens => MODERATE
        order = DispatchOrder(
            intent_category="code",
            specific_intent="tool_call",
            operator_message="Use a tool",
            system_prompt="",
            context_tokens=100,
            requires_tool_use=True,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok()
        candidates = result.unwrap()
        assert len(candidates) >= 1
        for c in candidates:
            assert _TIER_RANK[c.tier] >= _TIER_RANK[BackendTier.MODERATE], (
                f"Candidate '{c.name}' at tier '{c.tier.value}' is below MODERATE"
            )

    def test_upgrade_path_never_downgrades(self) -> None:
        """[TM-001 AC-4] When requested tier has no candidates, MBR upgrades, never downgrades."""
        # Only a COMPLEX backend exists; request is for MODERATE.
        complex_ = _make_config("complex-only", BackendTier.COMPLEX)
        registry = _build_registry(complex_)

        order = DispatchOrder(
            intent_category="code",
            specific_intent="tool_call",
            operator_message="Use a tool",
            system_prompt="",
            context_tokens=100,
            requires_tool_use=True,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok()
        candidates = result.unwrap()
        assert len(candidates) == 1
        assert candidates[0].tier == BackendTier.COMPLEX

    def test_no_downgrade_when_only_lower_tier_available(self) -> None:
        """[TM-001 AC-4] When only lower-tier backends exist, MBR returns Err not downgrade."""
        simple = _make_config("simple-only", BackendTier.SIMPLE)
        registry = _build_registry(simple)

        # context_tokens > 8192 => COMPLEX requested
        order = DispatchOrder(
            intent_category="code",
            specific_intent="generate",
            operator_message="Big context request",
            system_prompt="",
            context_tokens=10_000,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_err(), "Expected Err when only lower-tier backends available"


# ---------------------------------------------------------------------------
# AC5 — LOCAL providers unlimited-rate passthrough
# ---------------------------------------------------------------------------

class TestAC5LocalPassthrough:
    """TM-001 AC5: LOCAL-tier backends bypass circuit-breaker / rate-limit checks.

    A LOCAL backend with CIRCUIT_OPEN state must still be returned as a
    candidate.  A non-LOCAL backend in the same state must be excluded.
    """

    def test_local_backend_survives_circuit_open(self) -> None:
        """[TM-001 AC-5] LOCAL backend with CIRCUIT_OPEN status is still returned."""
        local = _make_config("local-llm", BackendTier.LOCAL)
        registry = _build_registry(local)

        # Trip the circuit breaker on the local backend
        _, state = registry.get("local-llm")
        assert state is not None
        state.status = BackendStatus.CIRCUIT_OPEN

        # Simple request => LOCAL tier
        order = DispatchOrder(
            intent_category="chat",
            specific_intent="greeting",
            operator_message="Hello",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok(), "LOCAL backend should pass through despite CIRCUIT_OPEN"
        candidates = result.unwrap()
        assert any(c.name == "local-llm" for c in candidates)

    def test_non_local_backend_excluded_when_circuit_open(self) -> None:
        """[TM-001 AC-5] Non-LOCAL backend with CIRCUIT_OPEN is correctly excluded."""
        simple = _make_config("simple-llm", BackendTier.SIMPLE)
        registry = _build_registry(simple)

        _, state = registry.get("simple-llm")
        assert state is not None
        state.status = BackendStatus.CIRCUIT_OPEN

        # context_tokens > 4096 => SIMPLE tier
        order = DispatchOrder(
            intent_category="code",
            specific_intent="generate",
            operator_message="Write some code",
            system_prompt="",
            context_tokens=5000,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_err(), "CIRCUIT_OPEN SIMPLE backend should be excluded"

    def test_local_passes_while_non_local_excluded(self) -> None:
        """[TM-001 AC-5] Mixed tier: LOCAL passes through, non-LOCAL excluded."""
        local = _make_config("local-llm", BackendTier.LOCAL)
        simple = _make_config("simple-llm", BackendTier.SIMPLE)
        registry = _build_registry(local, simple)

        # Trip circuits on both
        _, local_state = registry.get("local-llm")
        assert local_state is not None
        local_state.status = BackendStatus.CIRCUIT_OPEN

        _, simple_state = registry.get("simple-llm")
        assert simple_state is not None
        simple_state.status = BackendStatus.CIRCUIT_OPEN

        # Simple request => LOCAL tier
        order = DispatchOrder(
            intent_category="chat",
            specific_intent="greeting",
            operator_message="Hello",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok()
        candidates = result.unwrap()
        # LOCAL should be present
        assert any(c.name == "local-llm" for c in candidates)
        # SIMPLE should NOT be present (circuit open, non-local)
        assert not any(c.name == "simple-llm" for c in candidates)

    def test_local_passthrough_regardless_of_rate_limited_status(self) -> None:
        """[TM-001 AC-5] LOCAL backend with RATE_LIMITED status still passes through."""
        local = _make_config("local-llm", BackendTier.LOCAL)
        registry = _build_registry(local)

        _, state = registry.get("local-llm")
        assert state is not None
        state.status = BackendStatus.RATE_LIMITED

        order = DispatchOrder(
            intent_category="chat",
            specific_intent="greeting",
            operator_message="Hello",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = filter_by_capabilities(registry, order)
        assert result.is_ok(), "LOCAL backend should pass through despite RATE_LIMITED"
        candidates = result.unwrap()
        assert any(c.name == "local-llm" for c in candidates)


# ---------------------------------------------------------------------------
# Additional gap coverage
# ---------------------------------------------------------------------------

def test_filter_by_capabilities_returns_err_on_empty_registry() -> None:
    """[TM-001 AC-3] filter_by_capabilities returns Err when registry has no backends (line 52)."""
    registry = BackendRegistry()
    order = DispatchOrder(
        intent_category="chat",
        specific_intent="greeting",
        operator_message="Hello",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    result = filter_by_capabilities(registry, order)
    assert result.is_err()


def test_resolve_tiers_unknown_tier_returns_err(monkeypatch) -> None:
    """[TM-001 AC-3] _resolve_tiers_to_try returns Err when tier absent from TIER_ORDER (lines 77-79)."""
    import dragonlight_router.selection.mbr as mbr_mod
    from dragonlight_router.selection.mbr import _resolve_tiers_to_try

    monkeypatch.setattr(mbr_mod, "TIER_ORDER", (BackendTier.LOCAL,))

    result = _resolve_tiers_to_try(BackendTier.COMPLEX)
    assert isinstance(result, Err)


def test_candidates_for_tier_no_capable_candidates() -> None:
    """[TM-001 AC-3] _candidates_for_tier returns [] when no backend meets capability reqs (lines 132-133)."""
    from dragonlight_router.selection.mbr import _candidates_for_tier

    backend = _make_config(
        "no-tools",
        BackendTier.MODERATE,
        supports_tool_use=False,
        max_context_tokens=100,
    )
    registry = _build_registry(backend)

    order = DispatchOrder(
        intent_category="code",
        specific_intent="run",
        operator_message="Do something",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=True,
        requires_long_context=False,
    )

    result = _candidates_for_tier(registry, order, BackendTier.MODERATE, BackendTier.MODERATE)
    assert result == []


def test_is_backend_healthy_missing_from_registry() -> None:
    """[TM-001 AC-5] _is_backend_healthy returns False when backend absent from registry (lines 182-188)."""
    from dragonlight_router.selection.mbr import _is_backend_healthy

    registry = BackendRegistry()
    ghost_config = _make_config("ghost", BackendTier.SIMPLE)

    result = _is_backend_healthy(registry, ghost_config, BackendTier.SIMPLE)
    assert result is False


def test_meets_requirements_long_context_passes_when_cap_sufficient() -> None:
    """[TM-001 AC-1] _meets_requirements returns True when long_context required and cap is sufficient (line 303 not taken)."""
    from dragonlight_router.selection.mbr import _meets_requirements
    from dragonlight_router.core.types import BackendCapabilities

    caps = BackendCapabilities(
        max_context_tokens=100_000,
        supports_tool_use=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )
    order = DispatchOrder(
        intent_category="chat",
        specific_intent="long",
        operator_message="big context",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=True,
    )
    assert _meets_requirements(caps, order) is True


def test_filter_by_capabilities_err_when_tier_resolve_fails(monkeypatch) -> None:
    """[TM-001 AC-3] filter_by_capabilities propagates Err from _resolve_tiers_to_try (line 52)."""
    import dragonlight_router.selection.mbr as mbr_mod

    monkeypatch.setattr(mbr_mod, "TIER_ORDER", (BackendTier.LOCAL,))

    local = _make_config("local-llm", BackendTier.LOCAL)
    registry = _build_registry(local)

    order = DispatchOrder(
        intent_category="chat",
        specific_intent="greeting",
        operator_message="Hello",
        system_prompt="",
        context_tokens=9000,
        requires_tool_use=False,
        requires_long_context=False,
    )

    result = filter_by_capabilities(registry, order)
    assert result.is_err()