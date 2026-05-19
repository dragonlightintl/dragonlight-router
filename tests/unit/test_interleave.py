"""Tests for selection/interleave.py — provider interleaving."""
from __future__ import annotations

import pytest

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
        """Alternating providers already fine."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "nvidia", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "nvidia", 75.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert [m.model_id for m in result] == ["m1", "m2", "m3", "m4"]

    def test_three_consecutive_reordered(self):
        """Three same provider in a row gets broken up."""
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
        """All items are present after interleaving."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
            _ms("m4", "groq", 75.0),
            _ms("m5", "nvidia", 70.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert set(m.model_id for m in result) == {"m1", "m2", "m3", "m4", "m5"}

    def test_single_provider_unchanged(self):
        """If only one provider, can't interleave — returns as-is."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "groq", 80.0),
        ]
        result = interleave_providers(scored, max_consecutive=2)
        assert len(result) == 3

    def test_empty_list(self):
        result = interleave_providers([], max_consecutive=2)
        assert result == []

    def test_max_consecutive_one(self):
        """max_consecutive=1 forces strict alternation."""
        scored = [
            _ms("m1", "groq", 90.0),
            _ms("m2", "groq", 85.0),
            _ms("m3", "nvidia", 80.0),
            _ms("m4", "nvidia", 75.0),
        ]
        result = interleave_providers(scored, max_consecutive=1)
        providers = [m.provider for m in result]
        for i in range(len(providers) - 1):
            assert providers[i] != providers[i + 1] or len(set(m.provider for m in scored)) == 1
