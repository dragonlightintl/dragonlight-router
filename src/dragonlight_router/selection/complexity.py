"""Complexity estimation -- maps dispatch orders to required backend tiers.

Uses heuristics based on intent category, context size, tool use,
and message characteristics to estimate which tier is needed.
"""

from __future__ import annotations

from dragonlight_router.core.types import BackendTier, ComplexityEstimate, DispatchOrder

# Intent categories that require OPUS-tier reasoning
_OPUS_INTENTS = frozenset(
    {
        "session_lifecycle",
        "strategic_planning",
        "complex_reasoning",
    }
)

# Intent categories that require at least SONNET-tier
_SONNET_INTENTS = frozenset(
    {
        "engineering_build",
        "code_review",
        "architecture",
        "debugging",
        "spec_writing",
    }
)

# Context token thresholds
_LARGE_CONTEXT_THRESHOLD = 8000
_MEDIUM_CONTEXT_THRESHOLD = 2000

# Message length thresholds (characters)
_SHORT_MESSAGE_THRESHOLD = 50


def estimate_complexity(order: DispatchOrder) -> ComplexityEstimate:
    """Estimate required backend tier based on dispatch order characteristics.

    Returns ComplexityEstimate with tier, confidence, and reasoning signals.
    """
    assert order is not None, "order must not be None"
    assert isinstance(order, DispatchOrder), "order must be a DispatchOrder"

    signals: list[str] = []

    # Check OPUS-gate first (early return for highest tier)
    if order.intent_category in _OPUS_INTENTS:
        signals.append(f"intent_category={order.intent_category} requires OPUS")
        return _build_estimate(BackendTier.COMPLEX, 0.9, signals)

    tier, confidence = _evaluate_sonnet_gates(order, signals)
    tier, confidence = _evaluate_message_characteristics(order, tier, confidence, signals)

    assert len(signals) > 0, "signal evaluation must produce at least one signal"

    return _build_estimate(tier, confidence, signals)


def _evaluate_sonnet_gates(
    order: DispatchOrder,
    signals: list[str],
) -> tuple[BackendTier, float]:
    """Check SONNET-tier gates: intent category, tool use, large context."""
    assert isinstance(signals, list), "signals must be a list"
    assert isinstance(order, DispatchOrder), "order must be a DispatchOrder"

    tier = BackendTier.LOCAL
    confidence = 0.8

    if order.intent_category in _SONNET_INTENTS:
        signals.append(f"intent_category={order.intent_category} requires SONNET")
        tier = BackendTier.MODERATE
        confidence = 0.85

    if order.requires_tool_use:
        signals.append("requires_tool_use -> SONNET minimum")
        if tier.value in ("local", "haiku"):
            tier = BackendTier.MODERATE
            confidence = 0.85

    if order.requires_long_context or order.context_tokens >= _LARGE_CONTEXT_THRESHOLD:
        signals.append(f"large_context ({order.context_tokens} tokens) -> SONNET minimum")
        if tier.value in ("local", "haiku"):
            tier = BackendTier.MODERATE
            confidence = 0.8

    return tier, confidence


def _evaluate_message_characteristics(
    order: DispatchOrder,
    tier: BackendTier,
    confidence: float,
    signals: list[str],
) -> tuple[BackendTier, float]:
    """If still LOCAL, check message length and context size for HAIKU upgrade."""
    assert isinstance(tier, BackendTier), "tier must be a BackendTier"
    assert 0.0 <= confidence <= 1.0, "confidence must be in [0.0, 1.0]"

    if tier != BackendTier.LOCAL:
        return tier, confidence

    msg_len = len(order.operator_message)
    if msg_len <= _SHORT_MESSAGE_THRESHOLD and order.context_tokens < _MEDIUM_CONTEXT_THRESHOLD:
        signals.append(f"short_message ({msg_len} chars) + low context -> LOCAL")
        return tier, confidence

    if order.context_tokens >= _MEDIUM_CONTEXT_THRESHOLD:
        signals.append(f"medium_context ({order.context_tokens} tokens) -> HAIKU")
    else:
        signals.append(f"moderate_message ({msg_len} chars) -> HAIKU")

    return BackendTier.SIMPLE, 0.7


def _build_estimate(
    tier: BackendTier,
    confidence: float,
    signals: list[str],
) -> ComplexityEstimate:
    """Construct and validate a ComplexityEstimate."""
    result = ComplexityEstimate(tier=tier, confidence=confidence, signals=signals)

    assert isinstance(result.tier, BackendTier), "tier must be a BackendTier"
    assert 0.0 <= result.confidence <= 1.0, "confidence must be between 0.0 and 1.0"
    assert isinstance(result.signals, list), "signals must be a list"

    return result
