"""Unit tests for context_filter.py."""

from __future__ import annotations

import pytest

from src.dragonlight_router.selection.context_filter import (
    TrustTier,
    ProviderTrustTier,
    filter_by_trust_tier,
    filter_context_for_provider,
)


def test_filter_by_trust_tier_local() -> None:
    """LOCAL tier should trust all candidates."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.LOCAL) == candidates


def test_filter_by_trust_tier_haiku() -> None:
    """HAIKU tier should trust HAIKU and above."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == expected


def test_filter_by_trust_tier_sonnet() -> None:
    """SONNET tier should trust SONNET and above."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.MODERATE) == expected


def test_filter_by_trust_tier_opus() -> None:
    """OPUS tier should trust only OPUS."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == expected


def test_filter_by_trust_tier_empty() -> None:
    """Empty candidate list should return empty list."""
    assert filter_by_trust_tier([], TrustTier.LOCAL) == []
    assert filter_by_trust_tier([], TrustTier.COMPLEX) == []


def test_filter_by_trust_tier_duplicates() -> None:
    """Duplicate candidates should be preserved."""
    candidates = [TrustTier.SIMPLE, TrustTier.SIMPLE, TrustTier.MODERATE]
    expected = [TrustTier.SIMPLE, TrustTier.SIMPLE, TrustTier.MODERATE]
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == expected


def test_filter_by_trust_tier_no_matches() -> None:
    """When no candidates meet the tier, return empty list."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE]
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == []


def test_filter_by_trust_tier_all_same() -> None:
    """All candidates same tier."""
    candidates = [TrustTier.MODERATE, TrustTier.MODERATE, TrustTier.MODERATE]
    assert filter_by_trust_tier(candidates, TrustTier.MODERATE) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.LOCAL) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == []


def test_filter_context_for_provider_trusted_full_context():
    """TRUSTED providers receive full system-level context (minus PII, already absent)."""
    context = {
        "system": {"behavioral_rules": "BR1", "persona": "Assistant"},
        "history": [{"turn": 1}, {"turn": 2}, {"turn": 3}, {"turn": 4}],
        "task": "What is 2+2?",
    }
    expected = context.copy()  # Assuming PII is already absent, so no change
    result = filter_context_for_provider(context, ProviderTrustTier.TRUSTED)
    assert result == expected


def test_filter_context_for_provider_semitrusted_no_behavioral_rules():
    """SEMI_TRUSTED providers receive context without behavioral rules."""
    context = {
        "system": {"behavioral_rules": "BR1", "other_setting": "value"},
        "history": [],
        "task": "Task",
    }
    result = filter_context_for_provider(context, ProviderTrustTier.SEMI_TRUSTED)
    assert "behavioral_rules" not in result.get("system", {})
    assert result.get("system", {}).get("other_setting") == "value"


def test_filter_context_for_provider_semitrusted_persona_names():
    """SEMI_TRUSTED providers receive context with persona names replaced."""
    context = {
        "system": {"persona": "Assistant", "role": "Helper"},
        "history": [],
        "task": "Task",
    }
    result = filter_context_for_provider(context, ProviderTrustTier.SEMI_TRUSTED)
    assert result.get("system", {}).get("persona") == "[REDACTED PERSONA]"
    assert result.get("system", {}).get("role") == "Helper"


def test_filter_context_for_provider_semitrusted_limited_history():
    """SEMI_TRUSTED providers receive context with limited history (last 3 turns)."""
    context = {
        "system": {},
        "history": [
            {"turn": 1},
            {"turn": 2},
            {"turn": 3},
            {"turn": 4},
            {"turn": 5},
        ],
        "task": "Task",
    }
    result = filter_context_for_provider(context, ProviderTrustTier.SEMI_TRUSTED)
    assert len(result.get("history", [])) == 3
    assert result["history"] == [
        {"turn": 3},
        {"turn": 4},
        {"turn": 5},
    ]


def test_filter_context_for_provider_untrusted_task_only():
    """UNTRUSTED providers receive task-specific instruction only."""
    context = {
        "system": {"behavioral_rules": "BR1"},
        "history": [{"turn": 1}],
        "task": "What is the meaning of life?",
    }
    expected = {"task": "What is the meaning of life?"}
    result = filter_context_for_provider(context, ProviderTrustTier.UNTRUSTED)
    assert result == expected


def test_filter_context_for_provider_untrusted_no_task():
    """UNTRUSTED providers receive empty context if no task."""
    context = {
        "system": {},
        "history": [],
    }
    expected = {}
    result = filter_context_for_provider(context, ProviderTrustTier.UNTRUSTED)
    assert result == expected


def test_filter_context_for_provider_local_full_context():
    """LOCAL providers receive full context (no network egress risk)."""
    context = {
        "system": {"behavioral_rules": "BR1", "persona": "Assistant"},
        "history": [{"turn": 1}, {"turn": 2}],
        "task": "Task",
    }
    expected = context.copy()
    result = filter_context_for_provider(context, ProviderTrustTier.LOCAL)
    assert result == expected


def test_filter_context_for_provider_does_not_mutate_input():
    """Function should not mutate the original context dictionary."""
    context = {
        "system": {"behavioral_rules": "BR1"},
        "history": [{"turn": 1}],
        "task": "Task",
    }
    context_copy = context.copy()
    filter_context_for_provider(context, ProviderTrustTier.TRUSTED)
    assert context == context_copy


if __name__ == "__main__":
    pytest.main([__file__])