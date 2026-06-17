"""Rate-Limit Balancing (LBR) stage for the cascade router.

Filters model candidates based on rate-limit budget availability.
"""
from __future__ import annotations

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, BackendRateLimits, BackendTier, DispatchOrder
from dragonlight_router.result import Err, Ok, Result

logger = structlog.get_logger(__name__)


def _hard_capacity_gate(
    candidates: list[BackendConfig],
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """HAZ-005 mitigation: Hard has_capacity() gate before median filtering.

    Removes candidates whose provider has zero remaining capacity (RPM, RPD,
    TPM, or daily token cap exhausted). LOCAL tier backends bypass this gate.
    This prevents routing to providers that are definitively over their limits,
    regardless of the softer median-threshold filter that follows.
    """
    filtered = []
    for candidate in candidates:
        if candidate.tier == BackendTier.LOCAL:
            filtered.append(candidate)
            continue
        if budget_tracker.has_capacity(candidate.provider):
            filtered.append(candidate)
        else:
            logger.debug(
                "candidate_removed_no_capacity",
                backend=candidate.name,
                provider=candidate.provider,
            )
    assert len(filtered) <= len(candidates), "capacity gate must not add candidates"
    return filtered


def filter_by_rate_limit(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """Filter candidates by rate-limit budget using hard gate + median-score threshold.

    First applies a hard has_capacity() gate (HAZ-005) to remove providers
    with zero remaining capacity. Then retains candidates whose provider
    score is >= the median. LOCAL tier backends bypass both filters.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(isinstance(c, BackendConfig) for c in candidates), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"

    _log_rate_limit_entry(candidates, order)

    if not candidates:
        logger.debug("no candidates, returning as-is")
        return candidates

    # HAZ-005: Hard gate — remove providers with zero capacity
    capacity_filtered = _hard_capacity_gate(candidates, budget_tracker)
    if not capacity_filtered:
        logger.debug("all candidates removed by hard capacity gate")
        return []

    provider_scores = _collect_provider_scores(capacity_filtered, budget_tracker)
    median = _compute_median(capacity_filtered, provider_scores)
    logger.debug("rate-limit score median computed", median=median, provider_scores=provider_scores)

    filtered = _apply_median_threshold(capacity_filtered, provider_scores, median)
    logger.debug("rate-limit filtering complete", original_count=len(candidates), filtered_count=len(filtered))

    assert len(filtered) <= len(candidates), "filtered count must not exceed original"
    return filtered


def _log_rate_limit_entry(candidates: list[BackendConfig], order: DispatchOrder) -> None:
    """Log entry parameters for rate-limit filtering."""
    logger.debug(
        "filtering candidates by rate-limit budget",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )


def _collect_provider_scores(
    candidates: list[BackendConfig],
    budget_tracker: BudgetTracker,
) -> dict[str, float]:
    """Collect per-provider budget scores, caching by provider name."""
    assert len(candidates) > 0, "candidates must not be empty"

    provider_scores: dict[str, float] = {}
    for candidate in candidates:
        provider = candidate.provider
        if provider in provider_scores:
            continue
        provider_scores[provider] = _extract_score(budget_tracker, provider)

    assert len(provider_scores) > 0, "must have at least one provider score"
    return provider_scores


def _extract_score(budget_tracker: BudgetTracker, provider: str) -> float:
    """Extract a numeric score from the budget tracker for a provider."""
    assert isinstance(provider, str), "provider must be a string"
    assert len(provider) > 0, "provider must be non-empty"

    score_result = budget_tracker.score(provider)
    if isinstance(score_result, Ok):
        return score_result.value
    if hasattr(score_result, 'value'):
        return float(score_result.value)
    return 0.0


def _compute_median(
    candidates: list[BackendConfig],
    provider_scores: dict[str, float],
) -> float:
    """Compute median score across all candidate providers."""
    assert len(candidates) > 0, "candidates must not be empty"
    assert len(provider_scores) > 0, "provider_scores must not be empty"

    all_scores = sorted(provider_scores[c.provider] for c in candidates)
    n = len(all_scores)
    if n % 2 == 1:
        return all_scores[n // 2]
    return (all_scores[n // 2 - 1] + all_scores[n // 2]) / 2.0


def _apply_median_threshold(
    candidates: list[BackendConfig],
    provider_scores: dict[str, float],
    median: float,
) -> list[BackendConfig]:
    """Retain candidates whose provider score >= median, with LOCAL bypass (TM-003 AC4)."""
    assert isinstance(median, (int, float)), "median must be numeric"
    assert len(candidates) > 0, "candidates must not be empty"

    return [
        c for c in candidates
        if c.tier == BackendTier.LOCAL or provider_scores[c.provider] >= median
    ]


# Placeholder for select_final_candidate function that would be used by the cascade
def select_final_candidate(candidates: list[BackendConfig]) -> BackendConfig:
    """Select the final candidate from the list, breaking ties.

    In a full implementation, this would use additional scoring or heuristics
    to break ties between equally qualified candidates.

    Args:
        candidates: List of backend configurations to choose from.

    Returns:
        The selected BackendConfig.

    Raises:
        ValueError: If candidates list is empty.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(isinstance(c, BackendConfig) for c in candidates), "all candidates must be BackendConfig"

    if not candidates:
        raise ValueError("Cannot select from empty candidate list")

    return candidates[0]
