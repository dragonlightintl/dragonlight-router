"""Tests for selection/interleave.py — provider interleaving.

Spec traceability: TM-004 (Provider interleaving)
"""
from __future__ import annotations

from dragonlight_router.core.types import ModelScore
from dragonlight_router.selection.interleave import interleave_providers


def _ms(model_id: str, provider: str, composite: float) -> ModelScore:
    return ModelScore(
        model_id=model_id,
        provider=provider,
        rank=50,
        budget_score=50.0,
        health_score=50.0,
        composite=composite,
    )


class TestInterleave:
    def test_no_reorder_needed(self):
        """[TM-004 AC-1] Alternating providers already fine, no reorder needed."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "nvidia", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "nvidia", 75.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert [m.model_id for m in result] == ["m1", "m2", "m3", "m4"]

    def test_three_consecutive_reordered(self):
        """[TM-004 AC-2] Three same provider in a row gets broken up by interleaving."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "nvidia", 75.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        # m3 should not be at position 2 — nvidia should appear before 3 groqs
        providers = [m.provider for m in result]
        for i in range(len(providers) - 2):
            assert not (providers[i] == providers[i + 1] == providers[i + 2])

    def test_preserves_all_items(self):
        """[TM-004 AC-3] All items are present after interleaving (permutation invariant)."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "groq", 75.0),
            _ms("m5", "nvidia", 70.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert {m.model_id for m in result} == {"m1", "m2", "m3", "m4", "m5"}

    def test_single_provider_unchanged(self):
        """[TM-004 AC-1] If only one provider, can't interleave -- returns as-is."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert len(result) == 3

    def test_empty_list(self):
        """[TM-004 AC-1] Empty input returns empty output."""
        result = interleave_providers([], max_consecutive=2)
        assert result == []

    def test_max_consecutive_one(self):
        """[TM-004 AC-2] max_consecutive=1 forces strict alternation."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "nvidia", 80.0),
            _ms("m4", "nvidia", 75.0),
        ]
        result = interleave_providers(scored, max_consecutive=1)
        providers = [m.provider for m in result]
        for i in range(len(providers) - 1):
            assert providers[i] != providers[i + 1] or len({m.provider for m in scored}) == 1

    def test_constraint_not_satisfiable_returns_original(self):
        """[TM-004 AC-1] Returns original order when constraint cannot be satisfied (line 34)."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "groq", 75.0),
            _ms("m5", "nvidia", 70.0),
        ]
        result = interleave_providers(scored, max_consecutive=1)
        assert [m.model_id for m in result] == [m.model_id for m in scored]

    def test_is_constraint_satisfiable_returns_false_for_dominant_provider(self):
        """[TM-004 AC-2] _is_constraint_satisfiable returns False when one provider dominates."""
        from dragonlight_router.selection.interleave import _is_constraint_satisfiable

        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "groq", 75.0),
            _ms("m5", "nvidia", 70.0),
        ]
        assert _is_constraint_satisfiable(scored, max_consecutive=1) is False

    def test_build_interleaved_fallback_when_not_placed(self):
        """[TM-004 AC-3] _build_interleaved appends remaining when no placement possible."""
        from dragonlight_router.selection.interleave import _build_interleaved

        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "nvidia", 80.0),
        ]
        result = _build_interleaved(scored, max_consecutive=2)
        assert len(result) == len(scored)
        assert {m.model_id for m in result} == {"m1", "m2", "m3"}

    def test_try_place_best_returns_false_when_no_placeable(self):
        """[TM-004 AC-3] _try_place_best returns False when no candidate can be placed."""
        from dragonlight_router.selection.interleave import _try_place_best

        result = [_ms("m1", "groq", 90.0), _ms("m2", "groq", 85.0)]
        remaining = [_ms("m3", "groq", 80.0)]
        placed = _try_place_best(result, remaining, max_consecutive=2)
        assert placed is False
