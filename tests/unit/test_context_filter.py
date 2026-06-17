"""Unit tests for context_filter.py.

Spec traceability: TM-006 (Context filtering by trust tier)
"""

from __future__ import annotations

import pytest

from src.dragonlight_router.selection.context_filter import (
    TrustTier,
    ProviderTrustTier,
    filter_by_trust_tier,
    filter_context_for_provider,
)


def test_filter_by_trust_tier_local() -> None:
    """[TM-006 AC-1] LOCAL tier should trust all candidates."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.LOCAL) == candidates


def test_filter_by_trust_tier_haiku() -> None:
    """[TM-006 AC-1] SIMPLE tier should trust SIMPLE and above."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == expected


def test_filter_by_trust_tier_sonnet() -> None:
    """[TM-006 AC-1] MODERATE tier should trust MODERATE and above."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.MODERATE, TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.MODERATE) == expected


def test_filter_by_trust_tier_opus() -> None:
    """[TM-006 AC-1] COMPLEX tier should trust only COMPLEX."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    expected = [TrustTier.COMPLEX]
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == expected


def test_filter_by_trust_tier_empty() -> None:
    """[TM-006 AC-1] Empty candidate list should return empty list."""
    assert filter_by_trust_tier([], TrustTier.LOCAL) == []
    assert filter_by_trust_tier([], TrustTier.COMPLEX) == []


def test_filter_by_trust_tier_duplicates() -> None:
    """[TM-006 AC-1] Duplicate candidates should be preserved."""
    candidates = [TrustTier.SIMPLE, TrustTier.SIMPLE, TrustTier.MODERATE]
    expected = [TrustTier.SIMPLE, TrustTier.SIMPLE, TrustTier.MODERATE]
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == expected


def test_filter_by_trust_tier_no_matches() -> None:
    """[TM-006 AC-1] When no candidates meet the tier, return empty list."""
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE]
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == []


def test_filter_by_trust_tier_all_same() -> None:
    """[TM-006 AC-1] All candidates same tier filters correctly at each level."""
    candidates = [TrustTier.MODERATE, TrustTier.MODERATE, TrustTier.MODERATE]
    assert filter_by_trust_tier(candidates, TrustTier.MODERATE) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.SIMPLE) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.LOCAL) == candidates
    assert filter_by_trust_tier(candidates, TrustTier.COMPLEX) == []


def test_filter_context_for_provider_trusted_full_context():
    """[TM-006 AC-2] TRUSTED providers receive full system-level context."""
    context = {
        "system": {"behavioral_rules": "BR1", "persona": "Assistant"},
        "history": [{"turn": 1}, {"turn": 2}, {"turn": 3}, {"turn": 4}],
        "task": "What is 2+2?",
    }
    expected = context.copy()  # Assuming PII is already absent, so no change
    result = filter_context_for_provider(context, ProviderTrustTier.TRUSTED)
    assert result == expected


def test_filter_context_for_provider_semitrusted_no_behavioral_rules():
    """[TM-006 AC-3] SEMI_TRUSTED providers receive context without behavioral rules."""
    context = {
        "system": {"behavioral_rules": "BR1", "other_setting": "value"},
        "history": [],
        "task": "Task",
    }
    result = filter_context_for_provider(context, ProviderTrustTier.SEMI_TRUSTED)
    assert "behavioral_rules" not in result.get("system", {})
    assert result.get("system", {}).get("other_setting") == "value"


def test_filter_context_for_provider_semitrusted_persona_names():
    """[TM-006 AC-3] SEMI_TRUSTED providers receive context with persona names redacted."""
    context = {
        "system": {"persona": "Assistant", "role": "Helper"},
        "history": [],
        "task": "Task",
    }
    result = filter_context_for_provider(context, ProviderTrustTier.SEMI_TRUSTED)
    assert result.get("system", {}).get("persona") == "[REDACTED PERSONA]"
    assert result.get("system", {}).get("role") == "Helper"


def test_filter_context_for_provider_semitrusted_limited_history():
    """[TM-006 AC-3] SEMI_TRUSTED providers receive limited history (last 3 turns)."""
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
    """[TM-006 AC-4] UNTRUSTED providers receive task-specific instruction only."""
    context = {
        "system": {"behavioral_rules": "BR1"},
        "history": [{"turn": 1}],
        "task": "What is the meaning of life?",
    }
    expected = {"task": "What is the meaning of life?"}
    result = filter_context_for_provider(context, ProviderTrustTier.UNTRUSTED)
    assert result == expected


def test_filter_context_for_provider_untrusted_no_task():
    """[TM-006 AC-4] UNTRUSTED providers receive empty context if no task."""
    context = {
        "system": {},
        "history": [],
    }
    expected = {}
    result = filter_context_for_provider(context, ProviderTrustTier.UNTRUSTED)
    assert result == expected


def test_filter_context_for_provider_local_full_context():
    """[TM-006 AC-2] LOCAL providers receive full context (no network egress risk)."""
    context = {
        "system": {"behavioral_rules": "BR1", "persona": "Assistant"},
        "history": [{"turn": 1}, {"turn": 2}],
        "task": "Task",
    }
    expected = context.copy()
    result = filter_context_for_provider(context, ProviderTrustTier.LOCAL)
    assert result == expected


def test_filter_context_for_provider_does_not_mutate_input():
    """[TM-006 AC-5] Context filtering does not mutate the original context dictionary."""
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