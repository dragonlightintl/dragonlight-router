"""Tests for selection/complexity.py — complexity estimation.

Spec traceability: TM-019 (Complexity estimation)
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import BackendTier, ComplexityEstimate, DispatchOrder
from dragonlight_router.selection.complexity import estimate_complexity

pytestmark = pytest.mark.unit


def _order(
    intent_category: str = "general",
    specific_intent: str = "chat",
    operator_message: str = "hello",
    context_tokens: int = 500,
    requires_tool_use: bool = False,
    requires_long_context: bool = False,
) -> DispatchOrder:
    return DispatchOrder(
        intent_category=intent_category,
        specific_intent=specific_intent,
        operator_message=operator_message,
        system_prompt="You are helpful.",
        context_tokens=context_tokens,
        requires_tool_use=requires_tool_use,
        requires_long_context=requires_long_context,
    )


class TestComplexityEstimation:
    def test_short_simple_message_local(self):
        """[TM-019 AC-1] Short message, no tools, low context maps to LOCAL."""
        result = estimate_complexity(_order(operator_message="hi"))
        assert result.tier == BackendTier.LOCAL

    def test_tool_use_sonnet(self):
        """[TM-019 AC-2] Tool use required maps to MODERATE or COMPLEX."""
        result = estimate_complexity(_order(requires_tool_use=True))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_large_context_sonnet(self):
        """[TM-019 AC-2] Large context (>8k tokens) maps to MODERATE or COMPLEX."""
        result = estimate_complexity(_order(context_tokens=10000))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_long_context_flag(self):
        """[TM-019 AC-2] Explicit long context flag maps to MODERATE or COMPLEX."""
        result = estimate_complexity(_order(requires_long_context=True))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_session_lifecycle_opus(self):
        """[TM-019 AC-3] session_lifecycle intent maps to COMPLEX."""
        result = estimate_complexity(_order(intent_category="session_lifecycle"))
        assert result.tier == BackendTier.COMPLEX

    def test_engineering_build_sonnet(self):
        """[TM-019 AC-2] Engineering build maps to MODERATE."""
        result = estimate_complexity(_order(intent_category="engineering_build"))
        assert result.tier == BackendTier.MODERATE

    def test_returns_complexity_estimate(self):
        """[TM-019 AC-4] Return type is ComplexityEstimate with valid fields."""
        result = estimate_complexity(_order())
        assert isinstance(result, ComplexityEstimate)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.signals, list)

    def test_haiku_medium_message(self):
        """[TM-019 AC-1] Medium-length message with moderate context maps to SIMPLE or MODERATE."""
        result = estimate_complexity(
            _order(
                operator_message="Can you explain how the function works?",
                context_tokens=2000,
            )
        )
        assert result.tier in (BackendTier.SIMPLE, BackendTier.MODERATE)

    def test_signals_list_is_never_empty(self):
        """[TM-019 AC-4] Signals list always has at least one entry (line 52-53 ensures this)."""
        result = estimate_complexity(
            _order(
                intent_category="general",
                operator_message="hi",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            )
        )
        assert len(result.signals) >= 1

    def test_moderate_message_signal_when_long_message_low_context(self):
        """[TM-019 AC-1] Long message with low context triggers 'moderate_message' signal."""
        result = estimate_complexity(
            _order(
                operator_message="x" * 100,
                context_tokens=100,
                requires_tool_use=False,
                requires_long_context=False,
            )
        )
        assert result.tier == BackendTier.SIMPLE
        assert any("moderate_message" in s for s in result.signals)
