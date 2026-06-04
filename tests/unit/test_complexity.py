"""Tests for selection/complexity.py — complexity estimation."""
from __future__ import annotations

import pytest

from dragonlight_router.core.types import BackendTier, ComplexityEstimate, DispatchOrder
from dragonlight_router.selection.complexity import estimate_complexity


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
        """Short message, no tools, low context → LOCAL."""
        result = estimate_complexity(_order(operator_message="hi"))
        assert result.tier == BackendTier.LOCAL

    def test_tool_use_sonnet(self):
        """Tool use required → at least SONNET."""
        result = estimate_complexity(_order(requires_tool_use=True))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_large_context_sonnet(self):
        """Large context (>8k tokens) → SONNET."""
        result = estimate_complexity(_order(context_tokens=10000))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_long_context_flag(self):
        """Explicit long context flag → SONNET or above."""
        result = estimate_complexity(_order(requires_long_context=True))
        assert result.tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_session_lifecycle_opus(self):
        """session_lifecycle intent → OPUS."""
        result = estimate_complexity(_order(intent_category="session_lifecycle"))
        assert result.tier == BackendTier.COMPLEX

    def test_engineering_build_sonnet(self):
        """Engineering build → SONNET."""
        result = estimate_complexity(_order(intent_category="engineering_build"))
        assert result.tier == BackendTier.MODERATE

    def test_returns_complexity_estimate(self):
        """Return type is ComplexityEstimate."""
        result = estimate_complexity(_order())
        assert isinstance(result, ComplexityEstimate)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.signals, list)

    def test_haiku_medium_message(self):
        """Medium-length message with moderate context → HAIKU."""
        result = estimate_complexity(_order(
            operator_message="Can you explain how the function works?",
            context_tokens=2000,
        ))
        assert result.tier in (BackendTier.SIMPLE, BackendTier.MODERATE)
