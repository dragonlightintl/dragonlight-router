"""Cascade dispatch — MBR → CBR → LBR composition."""
from __future__ import annotations

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import BackendConfig, BackendTier, DispatchOrder, EngineResponse
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.cbr import filter_by_cost, score_candidate
from dragonlight_router.selection.lbr import filter_by_rate_limit, select_final_candidate
from dragonlight_router.selection.mbr import (
    estimate_complexity,
    filter_by_capabilities,
    MBRNoCandidatesError,
)
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    cost_governor_active,
    cost_adjusted_weights,
)

logger = structlog.get_logger(__name__)


def route(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict,
) -> Result[BackendConfig, Exception]:
    """Run the MBR → CBR → LBR cascade and return the selected BackendConfig.
    
    Args:
        order: The dispatch order containing capability requirements.
        registry: The backend registry to fetch candidates from.
        budget_tracker: Tracks per-provider budget availability.
        health_tracker: Tracks per-model health and circuit breaker state.
        config: Router configuration dictionary.
        
    Returns:
        Ok(BackendConfig) selected by the cascade or
        Err(Exception) if no candidates remain after filtering.
    """
    # Precondition assertions
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"
    assert isinstance(health_tracker, HealthTracker), "health_tracker must be HealthTracker instance"
    assert isinstance(config, dict), "config must be a dict"

    logger.debug(
        "running cascade dispatch",
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
    )

    # MBR: Filter by capability tier and health, with graceful upgrade
    mbr_result = filter_by_capabilities(registry, order)
    if mbr_result.is_err:
        logger.debug("MBR stage failed", error=str(mbr_result.error))
        return mbr_result  # Propagate the Err
    
    candidates = mbr_result.value
    logger.debug("MBR stage complete", candidate_count=len(candidates))

    # CBR: Filter by budget and score with cost-effectiveness
    # First apply hard budget filter
    cbr_result = filter_by_cost(candidates, order, budget_tracker)
    if cbr_result.is_err:
        logger.debug("CBR stage failed", error=str(cbr_result.error))
        return cbr_result  # Propagate the Err (e.g., BudgetExceededError)
    
    cbr_candidates = cbr_result.value
    logger.debug("CBR filtering complete", candidate_count=len(cbr_candidates))

    if not cbr_candidates:
        # This should be handled by filter_by_cost returning Err, but just in case
        from dragonlight_router.core.errors import BudgetExceededError
        return Err(BudgetExceededError("No candidates remain after budget filtering"))

    # Score candidates using canonical ScoringWeights with cost governor logic
    # Determine if cost governor should be active
    daily_spend = 0.0  # Would come from actual spending tracking in full implementation
    monthly_spend = 0.0  # Would come from actual spending tracking in full implementation
    
    weights = ScoringWeightsConfig()  # Default canonical weights
    if cost_governor_active(daily_spend, monthly_spend, config):
        weights = cost_adjusted_weights(weights)
        logger.debug("cost governor active", adjusted_weights=weights.__dict__)
    
    # Score each candidate
    scored_candidates = []
    for candidate in cbr_candidates:
        score = score_candidate(
            config=candidate,
            order=order,
            weights=weights,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )
        scored_candidates.append((score, candidate))
    
    # Sort by score descending (highest score = best candidate)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Extract just the candidates for LBR filtering
    lbr_candidates = [candidate for score, candidate in scored_candidates]
    logger.debug("CBR scoring complete", candidate_count=len(lbr_candidates))

    # LBR: Filter by rate limit and select final candidate
    lbr_result = filter_by_rate_limit(lbr_candidates, order, budget_tracker)
    if lbr_result.is_err:
        logger.debug("LBR stage failed", error=str(lbr_result.error))
        return lbr_result  # Propagate the Err (e.g., LBRNoCapacityError)
    
    lbr_candidates = lbr_result.value
    logger.debug("LBR filtering complete", candidate_count=len(lbr_candidates))

    if not lbr_candidates:
        # This should be handled by filter_by_rate_limit returning Err, but just in case
        from dragonlight_router.core.errors import LBRNoCapacityError
        return Err(LBRNoCapacityError("No candidates remain after rate limit filtering"))

    # Select the final candidate using tie-breaking logic
    final_candidate = select_final_candidate(lbr_candidates)
    logger.debug(
        "cascade dispatch complete",
        selected_provider=final_candidate.provider,
        selected_model=final_candidate.model,
    )

    return Ok(final_candidate)


def dispatch(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict,
) -> Result[EngineResponse, Exception]:
    """Execute the full dispatch pipeline and return an EngineResponse or DispatchFailure.
    
    This is the main entry point for engine-style consumers.
    
    Args:
        order: The dispatch order containing capability requirements.
        registry: The backend registry to fetch candidates from.
        budget_tracker: Tracks per-provider budget availability.
        health_tracker: Tracks per-model health and circuit breaker state.
        config: Router configuration dictionary.
        
    Returns:
        Ok(EngineResponse) if dispatch successful or
        Err(Exception) if all backends exhausted or error occurred.
    """
    # Precondition assertions
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"
    assert isinstance(health_tracker, HealthTracker), "health_tracker must be HealthTracker instance"
    assert isinstance(config, dict), "config must be a dict"

    logger.debug("starting dispatch pipeline")

    # Run the cascade routing
    route_result = route(order, registry, budget_tracker, health_tracker, config)
    if route_result.is_err:
        # Convert routing error to Exception (in full implementation would map to specific dispatch errors)
        error = route_result.error
        logger.debug("dispatch failed", error=str(error))
        return Err(error)
    
    # Successful routing - get the selected backend
    backend_config = route_result.value
    
    # In a full implementation, we would now:
    # 1. Apply context filtering based on provider trust tier
    # 2. Select the appropriate PAL adapter
    # 3. Execute the generation via the adapter
    # 4. Record usage and update budget/health trackers
    # 5. Construct and return EngineResponse
    
    # For now, we'll return a placeholder EngineResponse indicating successful routing
    from dragonlight_router.core.types import EngineResponse
    
    # Placeholder values - in reality these would come from actual generation
    engine_response = EngineResponse(
        content="[Generation would happen here via PAL adapter]",
        backend_used=backend_config.name,
        backend_tier=backend_config.tier,
        tokens_in=0,  # Would be actual input tokens
        tokens_out=0,  # Would be actual output tokens
        estimated_cost_usd=0.0,  # Would be calculated from actual usage
        latency_ms=0.0,  # Would be actual latency
        was_fallback=False,  # Would be True if we had to fallback
        fallback_chain=[],  # Would list attempted backends
    )
    
    logger.debug(
        "dispatch successful",
        backend_used=engine_response.backend_used,
        backend_tier=engine_response.backend_tier.value,
    )
    
    return Ok(engine_response)