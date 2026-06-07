"""Cascade dispatch — MBR → CBR → LBR composition."""

from __future__ import annotations

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import BackendConfig, BackendTier, DispatchOrder
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.result import Err, Ok
from dragonlight_router.selection.cbr import (
    filter_by_absolute_cost,
    filter_by_cost,
)
from dragonlight_router.selection.lbr import filter_by_rate_limit
from dragonlight_router.selection.mbr import (
    estimate_complexity,
    filter_by_capabilities,
    MBRNoCandidatesError,
)
from dragonlight_router.selection.scoring import compute_composite_score


def route(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict,
) -> BackendConfig:
    """Run the MBR → CBR → LBR cascade and return the selected BackendConfig.

    Args:
        order: The dispatch order containing capability requirements.
        registry: The backend registry to fetch candidates from.
        budget_tracker: Tracks per-provider budget availability.
        health_tracker: Tracks per-model health and circuit breaker state.
        config: Router configuration dictionary.

    Returns:
        The BackendConfig selected by the cascade.

    Raises:
        MBRNoCandidatesError: If no candidates pass the MBR stage.
    """
    # MBR: Filter by capability tier and health, with graceful upgrade
    mbr_result = filter_by_capabilities(registry, order)
    if mbr_result.is_err:
        raise mbr_result.error
    candidates = mbr_result.value

    # CBR: Filter by budget and score with cost-effectiveness
    candidates = filter_by_cost(candidates, order, budget_tracker)
    
    # Score candidates for final selection
    # Convert candidates to format expected by scoring: (model_id, rank, provider)
    # For now, we'll use a simple ranking based on cost efficiency
    scored_candidates = []
    for candidate in candidates:
        # Simple ranking: lower cost = better rank (1 is best)
        avg_cost = (candidate.cost.input_per_mtok + candidate.cost.output_per_mtok) / 2.0
        # Invert cost for ranking (higher score = better)
        rank_score = 100.0 / (avg_cost + 1e-9) if avg_cost > 0 else 100.0
        rank_score = min(rank_score, 100.0)  # Cap at 100
        
        budget_result = budget_tracker.score(candidate.provider)
        budget_score = budget_result.value if isinstance(budget_result, Ok) else 50.0
        
        health_result = health_tracker.score(candidate.model)
        health_score = health_result.value if isinstance(health_result, Ok) else 50.0
        
        composite = compute_composite_score(
            rank=int(rank_score),
            budget_score=budget_score,
            health_score=health_score,
        )
        
        scored_candidates.append((candidate.model, int(rank_score), candidate.provider, composite, candidate))
    
    # Sort by composite score descending
    scored_candidates.sort(key=lambda x: x[3], reverse=True)
    
    # LBR: Filter by rate limit and select final candidate
    if scored_candidates:
        # Extract just the candidates for LBR filtering
        lbr_candidates = [item[4] for item in scored_candidates]
        lbr_result = filter_by_rate_limit(lbr_candidates, order, budget_tracker)
        if isinstance(lbr_result, list):  # filter_by_rate_limit returns List[BackendConfig]
            final_candidates = lbr_result
        else:
            # Fallback if it ever returns something else
            final_candidates = [item[4] for item in scored_candidates]
        
        # Select the highest scoring candidate that passed LBR
        if final_candidates:
            # Find the candidate with highest composite score
            best_candidate = None
            best_score = -1.0
            
            for model_id, rank, provider, composite, candidate in scored_candidates:
                if candidate in final_candidates and composite > best_score:
                    best_score = composite
                    best_candidate = candidate
            
            if best_candidate is not None:
                return best_candidate
    
    # Fallback: return first candidate if any exist, otherwise raise
    if candidates:
        return candidates[0]
    
    raise MBRNoCandidatesError("No candidates available after cascade filtering")