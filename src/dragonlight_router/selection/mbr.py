"""Model-Based Ranking (MBR) capability filtering stage."""

from __future__ import annotations

import structlog
from typing import List

from dragonlight_router.core.errors import MBRNoCandidatesError
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendStatus,
    BackendTier,
    DispatchOrder,
    Err,
    Ok,
    Result,
)

logger = structlog.get_logger(__name__)


def filter_by_capabilities(
    registry: BackendRegistry,
    order: DispatchOrder,
) -> Result[List[BackendConfig], MBRNoCandidatesError]:
    """Filter candidates by capability tier and health, with graceful upgrade.

    Args:
        registry: The backend registry to fetch candidates from.
        order: The dispatch order containing capability requirements.

    Returns:
        Ok(list of candidates that meet the capability tier (or one above) and are healthy) or
        Err(MBRNoCandidatesError) if no candidates meet the requirements.
    """
    # Precondition assertion
    assert isinstance(registry, BackendRegistry), "registry must be a BackendRegistry instance"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    logger.debug(
        "filtering candidates by capability tier and health",
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
        has_system_prompt=bool(order.system_prompt),
    )

    # Estimate the complexity tier
    requested_tier = estimate_complexity(order)
    logger.debug("estimated complexity tier", tier=requested_tier.value)

    # Determine the tiers to try: requested tier and the next tier (if exists and we are not at the highest)
    tier_order = [BackendTier.LOCAL, BackendTier.SIMPLE, BackendTier.MODERATE, BackendTier.COMPLEX]
    try:
        requested_index = tier_order.index(requested_tier)
    except ValueError:
        # This should not happen because requested_tier is a BackendTier
        logger.error("unknown tier", tier=requested_tier)
        return Err(MBRNoCandidatesError(f"Unknown tier: {requested_tier}"))

    tiers_to_try = [requested_tier]
    if requested_index < len(tier_order) - 1:
        next_tier = tier_order[requested_index + 1]
        tiers_to_try.append(next_tier)

    # Try each tier in order
    for tier in tiers_to_try:
        logger.debug("trying tier", tier=tier.value)
        # Get candidates for this tier
        tier_candidates = registry.get_by_tier(tier)
        if not tier_candidates:
            logger.debug("no candidates in tier", tier=tier.value)
            continue

        # Filter by capabilities
        capable_candidates = _filter_by_capabilities(tier_candidates, order)
        if not capable_candidates:
            logger.debug("no capable candidates in tier", tier=tier.value)
            continue

        # Filter by health: exclude circuit_open
        healthy_candidates = []
        for candidate in capable_candidates:
            backend, state = registry.get(candidate.name)
            if backend is None or state is None:
                # This should not happen if the registry is consistent, but skip if missing
                logger.warning(
                    "backend or state missing",
                    backend_name=candidate.name,
                    backend_found=backend is not None,
                    state_found=state is not None,
                )
                continue
            if state.status != BackendStatus.CIRCUIT_OPEN:
                healthy_candidates.append(candidate)
            else:
                logger.debug(
                    "skipping circuit open backend",
                    backend_name=candidate.name,
                    tier=tier.value,
                )

        if healthy_candidates:
            logger.debug(
                "found healthy candidates",
                tier=tier.value,
                count=len(healthy_candidates),
            )
            return Ok(healthy_candidates)
        else:
            logger.debug(
                "no healthy candidates in tier after health filter",
                tier=tier.value,
            )

    # If we tried all tiers and got zero candidates
    logger.warning(
        "no candidates found after trying tiers",
        tried_tiers=[t.value for t in tiers_to_try],
    )
    return Err(MBRNoCandidatesError(
        f"No candidates meet the required capability tier (requested: {requested_tier.value}) "
        f"and health after trying tiers: {[t.value for t in tiers_to_try]}"
    ))


def estimate_complexity(order: DispatchOrder) -> BackendTier:
    """Estimate the required backend tier based on dispatch order complexity.

    Args:
        order: Dispatch order to analyze.

    Returns:
        BackendTier estimate for the order's complexity requirements.
    """
    # Precondition assertion
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    # Start with BASE tier (which is LOCAL in our BackendTier enum)
    tier = BackendTier.LOCAL

    # Upgrade based on requirements
    if order.requires_long_context or order.context_tokens > 4096:
        tier = BackendTier.SIMPLE
    if order.requires_tool_use:
        tier = BackendTier.MODERATE
    if order.context_tokens > 8192:
        tier = BackendTier.COMPLEX

    # Postcondition assertion
    assert tier in BackendTier, "tier must be a valid BackendTier"

    logger.debug(
        "estimated complexity",
        order_intent=order.intent_category,
        estimated_tier=tier.value,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
    )

    return tier


def _filter_by_capabilities(
    candidates: List[BackendConfig],
    order: DispatchOrder,
) -> List[BackendConfig]:
    """Filter candidates based on capability requirements from the dispatch order.

    Args:
        candidates: List of backend configurations to filter.
        order: Dispatch order containing capability requirements.

    Returns:
        List of candidates that meet all required capabilities.
    """
    # Precondition assertions
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    logger.debug(
        "filtering candidates by capabilities",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
        has_system_prompt=bool(order.system_prompt),
    )

    # If no requirements, return all candidates
    if not _has_any_requirements(order):
        logger.debug("no capability requirements, returning all candidates")
        return candidates

    filtered = []
    for candidate in candidates:
        if _meets_requirements(candidate.capabilities, order):
            filtered.append(candidate)

    logger.debug(
        "filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    return filtered


def _has_any_requirements(order: DispatchOrder) -> bool:
    """Check if the dispatch order has any capability requirements."""
    return (
        order.context_tokens > 0
        or order.requires_tool_use
        or order.requires_long_context
        or bool(order.system_prompt)
        # Note: JSON mode and streaming are harder to infer from order,
        # but could be added if needed in the future
    )


def _meets_requirements(caps: BackendCapabilities, order: DispatchOrder) -> bool:
    """Check if a backend's capabilities meet the order's requirements."""
    # Context tokens requirement
    if order.context_tokens > 0 and caps.max_context_tokens < order.context_tokens:
        return False

    # Tool use requirement
    if order.requires_tool_use and not caps.supports_tool_use:
        return False

    # Long context requirement (same as context tokens for now)
    if order.requires_long_context and caps.max_context_tokens < order.context_tokens:
        return False

    # System prompt requirement
    if bool(order.system_prompt) and not caps.supports_system_prompts:
        return False

    # TODO: Could add streaming and JSON mode requirements if we can infer them from order
    # For now, we don't have explicit flags for these in DispatchOrder

    return True