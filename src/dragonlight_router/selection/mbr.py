"""Model-Based Ranking (MBR) capability filtering stage."""

from __future__ import annotations

import structlog

from dragonlight_router.core.errors import MBRNoCandidatesError
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.state import invariant
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

__all__ = [
    "MBRNoCandidatesError",
    "filter_by_capabilities",
    "estimate_complexity",
    "TIER_ORDER",
]

# Canonical tier ordering — index position defines rank (0 = lowest).
TIER_ORDER: tuple[BackendTier, ...] = (
    BackendTier.LOCAL,
    BackendTier.SIMPLE,
    BackendTier.MODERATE,
    BackendTier.COMPLEX,
)
_TIER_RANK = {tier: idx for idx, tier in enumerate(TIER_ORDER)}

# HAZ-013 mitigation: Intent categories that require higher-tier backends.
# Maps intent_category values to their minimum required BackendTier.
# Intent categories not listed here default to heuristic-based estimation.
_INTENT_TIER_FLOOR: dict[str, BackendTier] = {
    # Complex reasoning / strategic work requires COMPLEX tier
    "complex_reasoning": BackendTier.COMPLEX,
    "strategic_planning": BackendTier.COMPLEX,
    "architecture": BackendTier.COMPLEX,
    # Engineering tasks require at least MODERATE tier
    "engineering_build": BackendTier.MODERATE,
    "code_review": BackendTier.MODERATE,
    "debugging": BackendTier.MODERATE,
    "spec_writing": BackendTier.MODERATE,
    "code_generation": BackendTier.MODERATE,
    # Analytical tasks benefit from SIMPLE at minimum
    "data_analysis": BackendTier.SIMPLE,
    "summarization": BackendTier.SIMPLE,
}


def filter_by_capabilities(
    registry: BackendRegistry,
    order: DispatchOrder,
) -> Result[list[BackendConfig], MBRNoCandidatesError]:
    """Filter candidates by capability tier and health, with graceful upgrade.

    Args:
        registry: The backend registry to fetch candidates from.
        order: The dispatch order containing capability requirements.

    Returns:
        Ok(list of candidates that meet the capability tier (or one above) and are healthy) or
        Err(MBRNoCandidatesError) if no candidates meet the requirements.
    """
    assert isinstance(registry, BackendRegistry), "registry must be a BackendRegistry instance"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    _log_capability_filter_entry(order)

    requested_tier = estimate_complexity(order)
    logger.debug("estimated complexity tier", tier=requested_tier.value)

    tiers_to_try = _resolve_tiers_to_try(requested_tier)
    if isinstance(tiers_to_try, Err):
        return tiers_to_try

    return _try_tiers(registry, order, tiers_to_try.value, requested_tier)


def _log_capability_filter_entry(order: DispatchOrder) -> None:
    """Log entry point for capability filtering."""
    logger.debug(
        "filtering candidates by capability tier and health",
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
        has_system_prompt=bool(order.system_prompt),
    )


def _resolve_tiers_to_try(
    requested_tier: BackendTier,
) -> Result[list[BackendTier], MBRNoCandidatesError]:
    """Determine the tiers to try: requested tier and the next tier (if exists)."""
    assert isinstance(requested_tier, BackendTier), "requested_tier must be a BackendTier"

    try:
        requested_index = TIER_ORDER.index(requested_tier)
    except ValueError:
        logger.error("unknown tier", tier=requested_tier)
        return Err(MBRNoCandidatesError(f"Unknown tier: {requested_tier}"))

    tiers = [requested_tier]
    if requested_index < len(TIER_ORDER) - 1:
        tiers.append(TIER_ORDER[requested_index + 1])

    assert len(tiers) >= 1, "must have at least one tier to try"
    return Ok(tiers)


def _try_tiers(
    registry: BackendRegistry,
    order: DispatchOrder,
    tiers_to_try: list[BackendTier],
    requested_tier: BackendTier,
) -> Result[list[BackendConfig], MBRNoCandidatesError]:
    """Try each tier in order, returning the first with healthy candidates."""
    assert len(tiers_to_try) >= 1, "must have at least one tier to try"
    assert all(isinstance(t, BackendTier) for t in tiers_to_try), "all tiers must be BackendTier"

    for tier in tiers_to_try:
        logger.debug("trying tier", tier=tier.value)
        healthy = _candidates_for_tier(registry, order, tier, requested_tier)
        if healthy:
            return Ok(healthy)

    logger.warning(
        "no candidates found after trying tiers",
        tried_tiers=[t.value for t in tiers_to_try],
    )
    return Err(
        MBRNoCandidatesError(
            f"No candidates meet the required capability tier (requested: {requested_tier.value}) "
            f"and health after trying tiers: {[t.value for t in tiers_to_try]}"
        )
    )


def _candidates_for_tier(
    registry: BackendRegistry,
    order: DispatchOrder,
    tier: BackendTier,
    requested_tier: BackendTier,
) -> list[BackendConfig]:
    """Get healthy, capable candidates for a single tier."""
    assert isinstance(tier, BackendTier), "tier must be a BackendTier"
    assert isinstance(requested_tier, BackendTier), "requested_tier must be a BackendTier"

    tier_candidates = registry.get_by_tier(tier)
    if not tier_candidates:
        logger.debug("no candidates in tier", tier=tier.value)
        return []

    capable_candidates = _filter_by_capabilities(tier_candidates, order)
    if not capable_candidates:
        logger.debug("no capable candidates in tier", tier=tier.value)
        return []

    healthy = _filter_healthy(registry, capable_candidates, tier)
    if not healthy:
        logger.debug("no healthy candidates in tier after health filter", tier=tier.value)
        return []

    _enforce_no_downgrade(healthy, requested_tier)
    logger.debug("found healthy candidates", tier=tier.value, count=len(healthy))
    return healthy


def _filter_healthy(
    registry: BackendRegistry,
    candidates: list[BackendConfig],
    tier: BackendTier,
) -> list[BackendConfig]:
    """Filter candidates by health: exclude circuit_open / rate-limited.

    AC5: LOCAL tier backends bypass all rate-limit and circuit-breaker
    checks -- they run on-box so capacity is unlimited.
    """
    assert isinstance(candidates, list), "candidates must be a list"

    healthy: list[BackendConfig] = []
    for candidate in candidates:
        if candidate.tier == BackendTier.LOCAL:
            healthy.append(candidate)
            logger.debug(
                "local backend passthrough",
                backend_name=candidate.name,
                tier=tier.value,
            )
            continue

        if _is_backend_healthy(registry, candidate, tier):
            healthy.append(candidate)

    assert all(isinstance(c, BackendConfig) for c in healthy), (
        "all healthy candidates must be BackendConfig"
    )
    return healthy


def _is_backend_healthy(
    registry: BackendRegistry,
    candidate: BackendConfig,
    tier: BackendTier,
) -> bool:
    """Check if a non-LOCAL backend is healthy (not circuit-open)."""
    assert isinstance(candidate, BackendConfig), "candidate must be BackendConfig"
    assert candidate.tier != BackendTier.LOCAL, "LOCAL backends bypass health checks"

    backend, state = registry.get(candidate.name)
    if backend is None or state is None:
        logger.warning(
            "backend or state missing",
            backend_name=candidate.name,
            backend_found=backend is not None,
            state_found=state is not None,
        )
        return False

    if state.status == BackendStatus.CIRCUIT_OPEN:
        logger.debug("skipping circuit open backend", backend_name=candidate.name, tier=tier.value)
        return False

    if state.status == BackendStatus.KEY_INVALID:
        logger.debug("skipping key_invalid backend", backend_name=candidate.name, tier=tier.value)
        return False

    return True


def _enforce_no_downgrade(
    candidates: list[BackendConfig],
    requested_tier: BackendTier,
) -> None:
    """AC4 postcondition: MBR NEVER downgrades.

    Every candidate must be at the requested tier or above.
    """
    assert len(candidates) > 0, "cannot enforce downgrade check on empty list"
    assert requested_tier in _TIER_RANK, "requested_tier must be a known tier"

    requested_rank = _TIER_RANK[requested_tier]
    for c in candidates:
        invariant(
            _TIER_RANK[c.tier] >= requested_rank,
            f"MBR invariant violated: candidate '{c.name}' tier "
            f"'{c.tier.value}' is below requested tier "
            f"'{requested_tier.value}'",
        )


def _apply_intent_floor(tier: BackendTier, intent_category: str) -> BackendTier:
    """HAZ-013: Apply intent-based tier floor — never lower, only raise.

    If the intent has a mapped floor tier that is higher than the
    heuristic-estimated tier, upgrade to the floor. Otherwise return
    the tier unchanged.
    """
    intent_floor = _INTENT_TIER_FLOOR.get(intent_category)
    if intent_floor is None:
        return tier

    floor_rank = _TIER_RANK.get(intent_floor, 0)
    current_rank = _TIER_RANK.get(tier, 0)
    if floor_rank > current_rank:
        logger.debug(
            "intent_floor_applied",
            intent=intent_category,
            previous_tier=tier.value,
            floor_tier=intent_floor.value,
        )
        return intent_floor
    return tier


def estimate_complexity(order: DispatchOrder) -> BackendTier:
    """Estimate the required backend tier based on dispatch order complexity.

    HAZ-013 mitigation: Uses intent_category to set a minimum tier floor
    via _apply_intent_floor(), ensuring tasks like code_review or architecture
    are routed to appropriately capable backends.

    Args:
        order: Dispatch order to analyze.

    Returns:
        BackendTier estimate for the order's complexity requirements.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    tier = BackendTier.LOCAL

    if order.requires_long_context or order.context_tokens > 4096:
        tier = BackendTier.SIMPLE
    if order.requires_tool_use:
        tier = BackendTier.MODERATE
    if order.context_tokens > 8192:
        tier = BackendTier.COMPLEX

    tier = _apply_intent_floor(tier, order.intent_category)

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
    candidates: list[BackendConfig],
    order: DispatchOrder,
) -> list[BackendConfig]:
    """Filter candidates based on capability requirements from the dispatch order."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(isinstance(c, BackendConfig) for c in candidates), (
        "all candidates must be BackendConfig instances"
    )
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    _log_capability_filter_details(len(candidates), order)

    if not _has_any_requirements(order):
        logger.debug("no capability requirements, returning all candidates")
        return candidates

    filtered = [c for c in candidates if _meets_requirements(c.capabilities, order)]
    logger.debug("filtering complete", original_count=len(candidates), filtered_count=len(filtered))
    return filtered


def _log_capability_filter_details(candidate_count: int, order: DispatchOrder) -> None:
    """Log capability filtering parameters."""
    logger.debug(
        "filtering candidates by capabilities",
        candidate_count=candidate_count,
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        requires_tool_use=order.requires_tool_use,
        requires_long_context=order.requires_long_context,
        has_system_prompt=bool(order.system_prompt),
    )


def _has_any_requirements(order: DispatchOrder) -> bool:
    """Check if the dispatch order has any capability requirements."""
    return (
        order.context_tokens > 0
        or order.requires_tool_use
        or order.requires_long_context
        or bool(order.system_prompt)
    )


def _meets_requirements(caps: BackendCapabilities, order: DispatchOrder) -> bool:
    """Check if a backend's capabilities meet the order's requirements."""
    assert isinstance(caps, BackendCapabilities), "caps must be BackendCapabilities"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder"

    if order.context_tokens > 0 and caps.max_context_tokens < order.context_tokens:
        return False

    if order.requires_tool_use and not caps.supports_tool_use:
        return False

    return not (bool(order.system_prompt) and not caps.supports_system_prompts)
