"""Cost Balancing (CBR) stage for the cascade router.

Filters model candidates based on cost-effectiveness and budget constraints.
"""

from __future__ import annotations

import structlog
from typing import Dict, List

from dragonlight_router.core.types import BackendConfig, BackendCostProfile, DispatchOrder

logger = structlog.get_logger(__name__)


def filter_by_cost_efficiency(
    candidates: List[BackendConfig],
    budget_scores: Dict[str, float],
    order: DispatchOrder,
) -> List[BackendConfig]:
    """Filter candidates based on cost efficiency given available budget.

    Args:
        candidates: List of backend configurations from MBR stage.
        budget_scores: Mapping of provider name to budget score (0-100).
        order: Dispatch order (may contain context for cost adjustment).

    Returns:
        List of candidates that pass cost balancing checks.
    """
    # Guard clauses
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(budget_scores, dict), "budget_scores must be a dict"
    assert all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in budget_scores.items()
    ), "budget_scores must map str to float/int"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    logger.debug(
        "filtering candidates by cost efficiency",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )

    # If no candidates or no budget data, return as-is (let later stages handle)
    if not candidates or not budget_scores:
        logger.debug("no candidates or budget data, returning as-is")
        return candidates

    # For each candidate, compute a cost efficiency score
    # We define cost efficiency as: budget_score / (normalized_cost + epsilon)
    # where normalized_cost is the average cost per token relative to a reference.
    # Since we don't have a reference, we'll use the inverse of cost: higher cost -> lower efficiency.
    # We'll then keep candidates with efficiency above a threshold (e.g., median).

    efficiencies: List[float] = []
    provider_to_efficiency: Dict[str, float] = {}

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
    filtered: List[BackendConfig] = []
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
    candidates: List[BackendConfig],
    max_cost_per_mtok: float,
) -> List[BackendConfig]:
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

    filtered: List[BackendConfig] = []
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