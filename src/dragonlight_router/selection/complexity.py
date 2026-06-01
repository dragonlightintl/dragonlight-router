"""Complexity estimation — maps dispatch orders to required backend tiers.

Uses heuristics based on intent category, context size, tool use,
and message characteristics to estimate which tier is needed.
"""
from __future__ import annotations

from dragonlight_router.core.types import BackendTier, ComplexityEstimate, DispatchOrder

# Intent categories that require OPUS-tier reasoning
_OPUS_INTENTS = frozenset({
    "session_lifecycle",
    "strategic_planning",
    "complex_reasoning",
})

# Intent categories that require at least SONNET-tier
_SONNET_INTENTS = frozenset({
    "engineering_build",
    "code_review",
    "architecture",
    "debugging",
    "spec_writing",
})

# Context token thresholds
_LARGE_CONTEXT_THRESHOLD = 8000
_MEDIUM_CONTEXT_THRESHOLD = 2000

# Message length thresholds (characters)
_SHORT_MESSAGE_THRESHOLD = 50


def estimate_complexity(order: DispatchOrder) -> ComplexityEstimate:
    """Estimate required backend tier based on dispatch order characteristics.

    Returns ComplexityEstimate with tier, confidence, and reasoning signals.
    """
    signals: list[str] = []
    tier = BackendTier.LOCAL
    confidence = 0.8

    # OPUS gates
    if order.intent_category in _OPUS_INTENTS:
        signals.append(f"intent_category={order.intent_category} requires OPUS")
        return ComplexityEstimate(tier=BackendTier.COMPLEX, confidence=0.9, signals=signals)

    # SONNET gates
    if order.intent_category in _SONNET_INTENTS:
        signals.append(f"intent_category={order.intent_category} requires SONNET")
        tier = BackendTier.MODERATE
        confidence = 0.85

    if order.requires_tool_use:
        signals.append("requires_tool_use → SONNET minimum")
        if tier.value in ("local", "haiku"):
            tier = BackendTier.MODERATE
            confidence = 0.85

    if order.requires_long_context or order.context_tokens >= _LARGE_CONTEXT_THRESHOLD:
        signals.append(f"large_context ({order.context_tokens} tokens) → SONNET minimum")
        if tier.value in ("local", "haiku"):
            tier = BackendTier.MODERATE
            confidence = 0.8

    # If still LOCAL or HAIKU, check message characteristics
    if tier == BackendTier.LOCAL:
        msg_len = len(order.operator_message)
        if msg_len <= _SHORT_MESSAGE_THRESHOLD and order.context_tokens < _MEDIUM_CONTEXT_THRESHOLD:
            signals.append(f"short_message ({msg_len} chars) + low context → LOCAL")
        elif order.context_tokens >= _MEDIUM_CONTEXT_THRESHOLD:
            signals.append(f"medium_context ({order.context_tokens} tokens) → HAIKU")
            tier = BackendTier.SIMPLE
            confidence = 0.7
        else:
            signals.append(f"moderate_message ({msg_len} chars) → HAIKU")
            tier = BackendTier.SIMPLE
            confidence = 0.7

    if not signals:
        signals.append("default tier assignment")

    return ComplexityEstimate(tier=tier, confidence=confidence, signals=signals)
