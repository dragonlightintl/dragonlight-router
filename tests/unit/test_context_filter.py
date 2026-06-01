"""Unit tests for context_filter.py."""

from __future__ import annotations

import pytest

from src.dragonlight_router.selection.context_filter import TrustTier, filter_by_trust_tier


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


if __name__ == "__main__":
    pytest.main([__file__])