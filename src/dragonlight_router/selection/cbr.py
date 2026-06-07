"""Cost Balancing (CBR) stage for the cascade router.

Filters model candidates based on cost-effectiveness and budget constraints.
"""
from __future__ import annotations

import structlog
from typing import List

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, BackendCostProfile, DispatchOrder
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    cost_governor_active,
    cost_adjusted_weights,
    score_candidate,
)

logger = structlog.get_logger(__name__)


def filter_by_cost(
    candidates: List[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
) -> Result[List[BackendConfig], Exception]:
    """Filter candidates based on cost effectiveness given available budget.
    
    Implements hard budget filtering and scoring with cost governor support.
    
    Args:
        candidates: List of backend configurations from MBR stage.
        order: Dispatch order (may contain context for cost adjustment).
        budget_tracker: Budget tracker to get budget scores for providers.
        
    Returns:
        Ok(list of candidates that pass cost balancing checks) or
        Err(Exception) if all providers exceed budget.
    """
    # Precondition assertions
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
        return Ok(candidates)

    # STEP 1: Hard filter - exclude providers with spent_usd >= budget_usd
    # A budget score of 0.0 indicates no budget remaining (spent_usd >= budget_usd)
    filtered_candidates = []
    budget_scores = {}
    
    for candidate in candidates:
        provider = candidate.provider
        # Get budget score (0-100 where 0 = no budget remaining)
        budget_result = budget_tracker.score(provider)
        budget_score = budget_result.value if hasattr(budget_result, 'value') else 50.0
        budget_scores[provider] = budget_score
        
        # Hard filter: exclude if no budget remaining (score == 0.0)
        if budget_score > 0.0:
            filtered_candidates.append(candidate)
    
    logger.debug(
        "budget filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered_candidates),
        budget_scores=budget_scores,
    )

    # If all providers exceed budget, return BudgetExceededError
    if not filtered_candidates:
        from dragonlight_router.core.errors import BudgetExceededError
        return Err(BudgetExceededError("All providers exceed budget"))
    
    # STEP 2: Score candidates using ScoringWeights with cost governor logic
    # Extract config from order for cost governor thresholds
    config = getattr(order, 'config', {}) if hasattr(order, 'config') else {}
    
    # For now, we'll estimate spend from budget tracker logic
    # In a full implementation, these would come from actual spending tracking
    daily_spend = 0.0  # Placeholder - would be calculated from actual costs
    monthly_spend = 0.0  # Placeholder - would be calculated from actual costs
    
    weights = ScoringWeightsConfig()  # Default canonical weights (cost=0.35, latency=0.25, priority=0.20, queue=0.10, health=0.10)
    if cost_governor_active(daily_spend, monthly_spend, config):
        weights = cost_adjusted_weights(weights)  # Shifts to cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05
        logger.debug("cost governor active", adjusted_weights=weights.__dict__)
    
    # Score each candidate
    scored_candidates = []
    for candidate in filtered_candidates:
        score = score_candidate(
            config=candidate,
            order=order,
            weights=weights,
            budget_tracker=budget_tracker,
            health_tracker=None,  # Would be passed in full implementation
        )
        scored_candidates.append((score, candidate))
    
    # Sort by score descending (highest score = best candidate)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return just the candidates in score order
    result_candidates = [candidate for score, candidate in scored_candidates]
    
    logger.debug(
        "cost scoring complete",
        candidate_count=len(result_candidates),
        weights_used=weights.__dict__,
    )
    
    return Ok(result_candidates)


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