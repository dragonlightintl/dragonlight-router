"""Context trust tier filtering for DIAN CECHT."""

from __future__ import annotations

from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class TrustTier(Enum):
    """Trust tiers for context filtering."""

    LOCAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4


def filter_by_trust_tier(candidates: list[TrustTier], required_tier: TrustTier) -> list[TrustTier]:
    """Return candidates allowed for the required trust tier.

    Trust hierarchy:
        LOCAL trusts all tiers (LOCAL, HAIKU, SONNET, OPUS)
        HAIKU trusts HAIKU and above (HAIKU, SONNET, OPUS)
        SONNET trusts SONNET and above (SONNET, OPUS)
        OPUS trusts only OPUS

    Args:
        candidates: List of trust tier candidates to filter.
        required_tier: Minimum required trust tier.

    Returns:
        List of candidates that meet or exceed the required trust tier.
    """
    # Guard clauses
    assert isinstance(candidates, list), "candidates must be a list"
    assert isinstance(required_tier, TrustTier), "required_tier must be TrustTier enum"

    logger.debug(
        "filtering candidates by trust tier",
        candidate_count=len(candidates),
        required_tier=required_tier.name,
    )

    # LOCAL tier trusts all
    if required_tier == TrustTier.LOCAL:
        return candidates

    # Define minimum allowed tier index (higher number = higher trust)
    min_index = required_tier.value

    # Filter candidates where candidate.value >= min_index
    filtered = [c for c in candidates if c.value >= min_index]

    logger.debug(
        "filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    return filtered