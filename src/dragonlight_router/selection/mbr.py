"""Model-Based Ranking (MBR) capability filtering stage."""

from __future__ import annotations

import structlog

from dragonlight_router.core.types import BackendCapabilities, BackendConfig, DispatchOrder

logger = structlog.get_logger(__name__)


def filter_by_capabilities(
    candidates: list[BackendConfig],
    order: DispatchOrder,
) -> list[BackendConfig]:
    """Filter candidates based on capability requirements from the dispatch order.

    Args:
        candidates: List of backend configurations to filter.
        order: Dispatch order containing capability requirements.

    Returns:
        List of candidates that meet all required capabilities.
    """
    # Guard clauses
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