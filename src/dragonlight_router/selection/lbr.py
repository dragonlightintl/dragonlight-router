"""Rate-Limit Balancing (LBR) stage for the cascade router.

Filters model candidates based on rate-limit budget availability.
"""

from __future__ import annotations

import secrets

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    BackendConfig,
    BackendTier,
    DispatchOrder,
    ScoredCandidate,
)
from dragonlight_router.result import Ok

logger = structlog.get_logger(__name__)

_srng = secrets.SystemRandom()


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


def _filter_by_median_score(
    candidates: list[BackendConfig],
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """Score providers, compute median threshold, and return candidates above it."""
    assert len(candidates) > 0, "candidates must not be empty for median filtering"
    provider_scores = _collect_provider_scores(candidates, budget_tracker)
    median = _compute_median(candidates, provider_scores)
    logger.debug("rate-limit score median computed", median=median, provider_scores=provider_scores)
    return _apply_median_threshold(candidates, provider_scores, median)


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
    assert all(isinstance(c, BackendConfig) for c in candidates), (
        "all candidates must be BackendConfig instances"
    )
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), (
        "budget_tracker must be BudgetTracker instance"
    )

    _log_rate_limit_entry(candidates, order)
    if not candidates:
        return candidates

    # HAZ-005: Hard gate — remove providers with zero capacity
    capacity_filtered = _hard_capacity_gate(candidates, budget_tracker)
    if not capacity_filtered:
        return []

    filtered = _filter_by_median_score(capacity_filtered, budget_tracker)
    logger.debug("rate-limit filtering complete", original=len(candidates), filtered=len(filtered))

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
    if hasattr(score_result, "value"):
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
        c
        for c in candidates
        if c.tier == BackendTier.LOCAL or provider_scores[c.provider] >= median
    ]


# Minimum weight floor — prevents zero-score candidates from being
# completely unreachable while keeping their selection probability low.
_SCORE_FLOOR = 0.01


def select_final_candidate(candidates: list[ScoredCandidate]) -> BackendConfig:
    """Select a candidate using weighted random selection based on composite scores.

    Higher-scored candidates are selected proportionally more often, but
    lower-scored candidates still receive some traffic. This distributes
    load across similarly-scored providers and allows recovery from
    provider degradation before scores fully diverge.

    Args:
        candidates: Scored candidates from the cascade pipeline, sorted
            best-first by composite score.

    Returns:
        The selected BackendConfig.

    Raises:
        ValueError: If candidates list is empty.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(isinstance(c, ScoredCandidate) for c in candidates), (
        "all candidates must be ScoredCandidate"
    )

    if not candidates:
        raise ValueError("Cannot select from empty candidate list")

    if len(candidates) == 1:
        return candidates[0].config

    weights = [max(c.score, _SCORE_FLOOR) for c in candidates]
    selected = _srng.choices(candidates, weights=weights, k=1)[0]

    logger.debug(
        "weighted_random_selection",
        selected=selected.config.name,
        selected_score=round(selected.score, 4),
        candidate_count=len(candidates),
    )

    return selected.config
