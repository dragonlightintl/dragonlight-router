"""Rate-Limit Balancing (LBR) stage for the cascade router.

Filters model candidates based on rate-limit budget availability.
"""

from __future__ import annotations

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, DispatchOrder
from dragonlight_router.result import Err, Ok

logger = structlog.get_logger(__name__)


def filter_by_rate_limit(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """Filter candidates based on rate-limit budget availability.

    Args:
        candidates: List of backend configurations from previous stage.
        order: Dispatch order (may contain context for rate-limit adjustment).
        budget_tracker: Budget tracker to get rate-limit scores for providers.

    Returns:
        List of candidates that pass rate-limit checks.
    """
    # Guard clauses
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"

    logger.debug(
        "filtering candidates by rate-limit budget",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )

    # If no candidates, return as-is
    if not candidates:
        logger.debug("no candidates, returning as-is")
        return candidates

    # Get rate-limit scores for each candidate's provider
    scores: dict[str, float] = {}
    for candidate in candidates:
        provider = candidate.provider
        # If we haven't computed the score for this provider yet, do so
        if provider not in scores:
            score_result = budget_tracker.score(provider)
            if isinstance(score_result, Ok):
                scores[provider] = score_result.value
            else:
                # If provider not found, treat as 0 score (will be filtered out if median > 0)
                scores[provider] = 0.0

    logger.debug(
        "rate-limit scores retrieved",
        provider_count=len(scores),
        scores=scores,
    )

    # If no score data, return as-is (let later stages handle)
    if not any(scores.values()):
        logger.debug("no rate-limit data, returning as-is")
        return candidates

    # For each candidate, get the score
    candidate_scores: list[float] = []
    provider_to_score: dict[str, float] = {}

    for candidate in candidates:
        provider = candidate.provider
        score = scores.get(provider, 0.0)
        candidate_scores.append(score)
        provider_to_score[provider] = score

    if not candidate_scores:
        return candidates

    # Compute median score as threshold
    sorted_scores = sorted(candidate_scores)
    n = len(sorted_scores)
    if n % 2 == 1:
        median_score = sorted_scores[n // 2]
    else:
        median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0

    # Filter: keep candidates with score >= median
    filtered: list[BackendConfig] = []
    for candidate in candidates:
        provider = candidate.provider
        score = provider_to_score.get(provider, 0.0)
        if score >= median_score:
            filtered.append(candidate)

    logger.debug(
        "rate-limit filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
        median_score=median_score,
    )

    return filtered