"""Cost Balancing (CBR) stage for the cascade router.

Filters model candidates based on cost-effectiveness and budget constraints.
"""

from __future__ import annotations

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, BackendCostProfile, DispatchOrder
from dragonlight_router.result import Err, Ok, Result

logger = structlog.get_logger(__name__)


def filter_by_cost(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """Filter candidates based on cost effectiveness given available budget.

    Args:
        candidates: List of backend configurations from MBR stage.
        order: Dispatch order (may contain context for cost adjustment).
        budget_tracker: Budget tracker to get budget scores for providers.

    Returns:
        List of candidates that pass cost balancing checks.
    """
    # Guard clauses
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"

    logger.debug(
        "filtering candidates by cost",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )

    # If no candidates, return as-is
    if not candidates:
        logger.debug("no candidates, returning as-is")
        return candidates

    # Get budget scores for each candidate's provider
    budget_scores: dict[str, float] = {}
    for candidate in candidates:
        provider = candidate.provider
        # If we haven't computed the score for this provider yet, do so
        if provider not in budget_scores:
            score_result = budget_tracker.score(provider)
            if isinstance(score_result, Ok):
                budget_scores[provider] = score_result.value
            else:
                # If provider not found, treat as 0 score (will be filtered out if median > 0)
                budget_scores[provider] = 0.0

    logger.debug(
        "budget scores retrieved",
        provider_count=len(budget_scores),
        scores=budget_scores,
    )

    # If no budget data, return as-is (let later stages handle)
    if not any(budget_scores.values()):
        logger.debug("no budget data, returning as-is")
        return candidates

    # For each candidate, compute a cost efficiency score
    # We define cost efficiency as: budget_score / (normalized_cost + epsilon)
    # where normalized_cost is the average cost per token relative to a reference.
    # Since we don't have a reference, we'll use the inverse of cost: higher cost -> lower efficiency.
    # We'll then keep candidates with efficiency above a threshold (e.g., median).

    efficiencies: list[float] = []
    provider_to_efficiency: dict[str, float] = {}

    for candidate in candidates:
        provider = candidate.provider
        budget_score = budget_scores.get(provider, 0.0)
        cost_profile: BackendCostProfile = candidate.cost

        # Use average of input and output cost per million tokens
        avg_cost_per_mtok = (cost_profile.input_per_mtok + cost_profile.output_per_mtok) / 2.0
        # Avoid division by zero; if cost is zero, treat as high efficiency
        if avg_cost_per_mtok <= 0:
            efficiency = float('inf')
        else:
            # Higher budget score and lower cost -> higher efficiency
            efficiency = budget_score / (avg_cost_per_mtok + 1e-9)

        efficiencies.append(efficiency)
        provider_to_efficiency[provider] = efficiency

    if not efficiencies:
        return candidates

    # Compute median efficiency as threshold
    sorted_eff = sorted(efficiencies)
    n = len(sorted_eff)
    if n % 2 == 1:
        median_efficiency = sorted_eff[n // 2]
    else:
        median_efficiency = (sorted_eff[n // 2 - 1] + sorted_eff[n // 2]) / 2.0

    # Filter: keep candidates with efficiency >= median
    filtered: list[BackendConfig] = []
    for candidate in candidates:
        provider = candidate.provider
        efficiency = provider_to_efficiency.get(provider, 0.0)
        if efficiency >= median_efficiency:
            filtered.append(candidate)

    logger.debug(
        "cost filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
        median_efficiency=median_efficiency,
    )

    return filtered


def filter_by_absolute_cost(
    candidates: list[BackendConfig],
    max_cost_per_mtok: float,
) -> list[BackendConfig]:
    """Filter candidates that exceed an absolute cost threshold.

    Args:
        candidates: List of backend configurations.
        max_cost_per_mtok: Maximum allowed average cost per million tokens.

    Returns:
        List of candidates with cost at or below the threshold.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(max_cost_per_mtok, (int, float)) and max_cost_per_mtok >= 0, \
        "max_cost_per_mtok must be a non-negative number"

    logger.debug(
        "filtering by absolute cost",
        candidate_count=len(candidates),
        max_cost_per_mtok=max_cost_per_mtok,
    )

    filtered: list[BackendConfig] = []
    for candidate in candidates:
        cost_profile: BackendCostProfile = candidate.cost
        avg_cost = (cost_profile.input_per_mtok + cost_profile.output_per_mtok) / 2.0
        if avg_cost <= max_cost_per_mtok:
            filtered.append(candidate)

    logger.debug(
        "absolute cost filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    return filtered