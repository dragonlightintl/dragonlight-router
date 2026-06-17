"""Cascade dispatch — MBR → CBR → LBR composition."""
from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

import dragonlight_router.adapters as _adapters_mod
from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.errors import BudgetExceededError, LBRNoCapacityError
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import BackendConfig, BackendStatus, BackendTier, DispatchFailure, DispatchOrder, EngineResponse
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.cbr import filter_by_cost, score_candidate
from dragonlight_router.selection.context_filter import ProviderTrustTier, filter_context_for_provider
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

# Penalty multiplier applied to scores of DEGRADED backends (0.5 = halve the score).
_DEGRADED_SCORE_PENALTY = 0.5

# Mapping from caller-specified trust tier strings to ProviderTrustTier ordering.
# Higher numeric value = more restrictive trust requirement.
_TRUST_TIER_RANK: dict[str, int] = {
    "untrusted": 0,
    "semi_trusted": 1,
    "trusted": 2,
    "local": 3,
}

# Reverse mapping from ProviderTrustTier enum to rank for comparison.
_PROVIDER_TRUST_RANK: dict[ProviderTrustTier, int] = {
    ProviderTrustTier.UNTRUSTED: 0,
    ProviderTrustTier.SEMI_TRUSTED: 1,
    ProviderTrustTier.TRUSTED: 2,
    ProviderTrustTier.LOCAL: 3,
}


@dataclass(frozen=True)
class DispatchContext:
    """Grouped context for dispatch pipeline operations (QA-004: parameter count > 4)."""

    registry: BackendRegistry
    budget_tracker: BudgetTracker
    health_tracker: HealthTracker
    config: dict


def _tier_to_provider_trust(tier: BackendTier) -> ProviderTrustTier:
    """Map a BackendTier to a ProviderTrustTier for context filtering.

    LOCAL backends get full context (no egress risk).
    SIMPLE/MODERATE are semi-trusted (limited context).
    COMPLEX backends are trusted (full context).
    """
    mapping = {
        BackendTier.LOCAL: ProviderTrustTier.LOCAL,
        BackendTier.SIMPLE: ProviderTrustTier.SEMI_TRUSTED,
        BackendTier.MODERATE: ProviderTrustTier.SEMI_TRUSTED,
        BackendTier.COMPLEX: ProviderTrustTier.TRUSTED,
    }
    return mapping.get(tier, ProviderTrustTier.UNTRUSTED)


def _compute_aggregate_spend(
    budget_tracker: BudgetTracker,
    candidates: list[BackendConfig],
) -> tuple[float, float]:
    """Compute aggregate daily and monthly spend across all candidate providers.

    Uses each candidate's cost profile to derive avg_cost_per_token,
    then queries the budget tracker for spend estimates.

    Returns:
        (daily_spend_usd, monthly_spend_usd) aggregated across providers.
    """
    seen_providers: set[str] = set()
    total_daily = 0.0
    total_monthly = 0.0

    for candidate in candidates:
        if candidate.provider in seen_providers:
            continue
        seen_providers.add(candidate.provider)

        # Average cost per token in USD (cost profile is per million tokens)
        avg_cost_per_token = (
            (candidate.cost.input_per_mtok + candidate.cost.output_per_mtok) / 2.0 / 1_000_000.0
        )
        total_daily += budget_tracker.daily_spend_usd(candidate.provider, avg_cost_per_token)
        total_monthly += budget_tracker.monthly_spend_usd(candidate.provider, avg_cost_per_token)

    return total_daily, total_monthly


def _run_mbr_stage(
    order: DispatchOrder,
    ctx: DispatchContext,
) -> Result[list[BackendConfig], Exception]:
    """MBR stage: filter by capability tier and health with graceful upgrade."""
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(ctx, DispatchContext), "ctx must be DispatchContext instance"

    mbr_result = filter_by_capabilities(ctx.registry, order)
    if mbr_result.is_err():
        logger.debug("MBR stage failed", error=str(mbr_result.error))
        return mbr_result

    logger.debug("MBR stage complete", candidate_count=len(mbr_result.value))
    return mbr_result


def _run_cbr_stage(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ctx: DispatchContext,
) -> Result[list[BackendConfig], Exception]:
    """CBR stage: filter by budget, score with cost-effectiveness, rank candidates."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    daily_spend, monthly_spend = _compute_aggregate_spend(ctx.budget_tracker, candidates)

    cbr_candidates = filter_by_cost(
        candidates, order, ctx.budget_tracker,
        daily_spend=daily_spend, monthly_spend=monthly_spend,
        config=ctx.config, health_tracker=ctx.health_tracker,
    )
    logger.debug("CBR filtering complete", candidate_count=len(cbr_candidates))

    if not cbr_candidates:
        return Err(BudgetExceededError("No candidates remain after budget filtering"))

    weights = ScoringWeightsConfig()
    if cost_governor_active(daily_spend, monthly_spend, ctx.config):
        weights = cost_adjusted_weights(weights)

    scored = _score_and_rank_candidates(cbr_candidates, order, weights, ctx)
    logger.debug("CBR scoring complete", candidate_count=len(scored))
    return Ok(scored)


def _score_and_rank_candidates(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    weights: ScoringWeightsConfig,
    ctx: DispatchContext,
) -> list[BackendConfig]:
    """Score each candidate and return them sorted best-first."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    scored_candidates: list[tuple[float, BackendConfig]] = []
    for candidate in candidates:
        score = score_candidate(
            config=candidate, order=order, weights=weights,
            budget_tracker=ctx.budget_tracker, health_tracker=ctx.health_tracker,
        )
        score = _apply_degraded_penalty(score, candidate, ctx.registry)
        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [candidate for _score, candidate in scored_candidates]


def _apply_degraded_penalty(
    score: float,
    candidate: BackendConfig,
    registry: BackendRegistry,
) -> float:
    """Halve the score of DEGRADED backends to deprioritize them."""
    _backend, state = registry.get(candidate.name)
    if state is not None and state.status == BackendStatus.DEGRADED:
        penalized = score * _DEGRADED_SCORE_PENALTY
        logger.debug(
            "degraded backend deprioritized",
            backend_name=candidate.name,
            original_score=round(score, 4),
            penalized_score=round(penalized, 4),
        )
        return penalized
    return score


def _run_lbr_stage(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ctx: DispatchContext,
) -> Result[list[BackendConfig], Exception]:
    """LBR stage: filter by rate limit capacity."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    lbr_candidates = filter_by_rate_limit(candidates, order, ctx.budget_tracker)
    logger.debug("LBR filtering complete", candidate_count=len(lbr_candidates))

    if not lbr_candidates:
        return Err(LBRNoCapacityError("No candidates remain after rate limit filtering"))
    return Ok(lbr_candidates)


def _filter_by_trust_floor(
    candidates: list[BackendConfig],
    context_trust_tier: str | None,
) -> list[BackendConfig]:
    """Filter candidates whose provider trust is below the caller-specified floor.

    HAZ-001 mitigation: If the DispatchOrder specifies a context_trust_tier,
    only backends whose provider trust rank meets or exceeds the requested
    floor are retained. When no trust tier is specified, all candidates pass.
    """
    if context_trust_tier is None:
        return candidates

    floor_rank = _TRUST_TIER_RANK.get(context_trust_tier.lower())
    if floor_rank is None:
        logger.warning(
            "unknown_context_trust_tier",
            context_trust_tier=context_trust_tier,
        )
        return candidates

    filtered = []
    for candidate in candidates:
        provider_trust = _tier_to_provider_trust(candidate.tier)
        provider_rank = _PROVIDER_TRUST_RANK.get(provider_trust, 0)
        if provider_rank >= floor_rank:
            filtered.append(candidate)
        else:
            logger.debug(
                "candidate_filtered_by_trust_floor",
                backend=candidate.name,
                provider_trust=provider_trust.name,
                required_floor=context_trust_tier,
            )

    assert len(filtered) <= len(candidates), "trust filter must not add candidates"
    return filtered


def _run_cascade(
    order: DispatchOrder,
    ctx: DispatchContext,
) -> Result[list[BackendConfig], Exception]:
    """Run MBR -> trust floor -> CBR -> LBR cascade and return the full ranked candidate list.

    Unlike route(), this returns ALL surviving candidates so dispatch() can
    implement fallback across the ranked list.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(ctx, DispatchContext), "ctx must be DispatchContext instance"

    logger.debug(
        "running cascade pipeline",
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )

    mbr_result = _run_mbr_stage(order, ctx)
    if mbr_result.is_err():
        return mbr_result

    # HAZ-001: Enforce caller-specified trust floor before cost/rate scoring
    trust_filtered = _filter_by_trust_floor(
        mbr_result.value, order.context_trust_tier,
    )
    if not trust_filtered:
        return Err(MBRNoCandidatesError(
            "No candidates meet the requested context_trust_tier floor"
        ))

    cbr_result = _run_cbr_stage(order, trust_filtered, ctx)
    if cbr_result.is_err():
        return cbr_result

    return _run_lbr_stage(order, cbr_result.value, ctx)


def route(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict,
) -> Result[BackendConfig, Exception]:
    """Run the MBR -> CBR -> LBR cascade and return the selected BackendConfig.

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
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"

    ctx = DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
    )
    cascade_result = _run_cascade(order, ctx)
    if cascade_result.is_err():
        return cascade_result

    candidates = cascade_result.value
    final_candidate = select_final_candidate(candidates)
    logger.debug(
        "cascade dispatch complete",
        selected_provider=final_candidate.provider,
        selected_model=final_candidate.model,
    )
    return Ok(final_candidate)


def _build_dispatch_context(
    order: DispatchOrder,
) -> dict:
    """Build the base context dict for context filtering from a DispatchOrder."""
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    base_context: dict = {}
    if order.system_prompt:
        base_context["system"] = {"prompt": order.system_prompt}
    base_context["task"] = order.operator_message
    return base_context


def _build_messages(
    filtered_context: dict,
    fallback_message: str,
) -> list[dict[str, str]]:
    """Build the messages list from filtered context for adapter generation."""
    messages: list[dict[str, str]] = []
    system_content = filtered_context.get("system", {})
    if isinstance(system_content, dict):
        prompt_text = system_content.get("prompt", "")
        if prompt_text:
            messages.append({"role": "system", "content": prompt_text})
    elif isinstance(system_content, str) and system_content:
        messages.append({"role": "system", "content": system_content})

    task_content = filtered_context.get("task", fallback_message)
    messages.append({"role": "user", "content": task_content if task_content else fallback_message})
    return messages


async def _try_adapter_dispatch(
    backend_config: BackendConfig,
    base_context: dict,
    order: DispatchOrder,
    ctx: DispatchContext,
    fallback_chain: list[str],
) -> Result[EngineResponse, RuntimeError]:
    """Attempt a single adapter call and return Ok(EngineResponse) or Err on failure."""
    assert isinstance(backend_config, BackendConfig), "backend_config must be BackendConfig"

    adapter = _adapters_mod.create_adapter(backend_config)

    provider_trust = _tier_to_provider_trust(backend_config.tier)
    filtered_context = filter_context_for_provider(base_context, provider_trust)
    messages = _build_messages(filtered_context, order.operator_message)

    t0 = time.monotonic()
    content_parts: list[str] = []
    async for chunk in adapter.generate(
        messages, max_tokens=4096, temperature=0.7, stream=True,
    ):
        content_parts.append(chunk)
    latency_ms = (time.monotonic() - t0) * 1000.0

    content = "".join(content_parts)
    tokens_in = sum(len(m.get("content", "")) for m in messages) // 4
    tokens_out = len(content) // 4
    cost_usd = (
        (tokens_in / 1_000_000) * backend_config.cost.input_per_mtok
        + (tokens_out / 1_000_000) * backend_config.cost.output_per_mtok
    )

    ctx.health_tracker.record_success(backend_config.model, latency_ms)
    ctx.budget_tracker.record_request(backend_config.provider, tokens_in + tokens_out)
    adapter.record_usage(tokens_in, tokens_out)

    return Ok(_build_engine_response(
        backend_config, content, tokens_in, tokens_out, cost_usd, latency_ms, fallback_chain,
    ))


def _build_engine_response(
    backend_config: BackendConfig,
    content: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float,
    fallback_chain: list[str],
) -> EngineResponse:
    """Assemble the final EngineResponse from generation results."""
    return EngineResponse(
        content=content,
        backend_used=backend_config.name,
        backend_tier=backend_config.tier,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=cost_usd,
        latency_ms=latency_ms,
        was_fallback=len(fallback_chain) > 0,
        fallback_chain=list(fallback_chain),
    )


def _record_adapter_failure(
    exc: RuntimeError | ValueError | ConnectionError | OSError | TypeError,
    backend_config: BackendConfig,
    ctx: DispatchContext,
) -> None:
    """Record a backend generation failure in the health tracker."""
    http_status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "__cause__", None), "status_code", None
    )
    ctx.health_tracker.record_error(backend_config.model, http_status=http_status)


async def _handle_fallback_chain(
    candidates: list[BackendConfig],
    base_context: dict,
    order: DispatchOrder,
    ctx: DispatchContext,
) -> Result[EngineResponse, Exception]:
    """Iterate through candidates attempting generation, falling back on failure."""
    assert len(candidates) > 0, "candidates must not be empty"

    fallback_chain: list[str] = []
    last_error: RuntimeError | ValueError | ConnectionError | OSError | TypeError | None = None

    for backend_config in candidates:
        logger.debug("attempting generation", backend=backend_config.name, attempt=len(fallback_chain) + 1)
        try:
            result = await _try_adapter_dispatch(backend_config, base_context, order, ctx, fallback_chain)
        except (RuntimeError, ValueError, ConnectionError, OSError, TypeError) as exc:
            last_error = exc
            fallback_chain.append(backend_config.name)
            _record_adapter_failure(exc, backend_config, ctx)
            logger.warning("backend generation failed", backend=backend_config.name, error=str(exc))
            continue

        if isinstance(result, Ok):
            logger.debug("dispatch successful", backend_used=result.value.backend_used)
            return result

    exhaustion_msg = (
        f"All {len(fallback_chain)} backends exhausted. "
        f"Fallback chain: {' → '.join(fallback_chain)}. "
        f"Last error: {last_error}"
    )
    logger.error("dispatch exhausted all backends", fallback_chain=fallback_chain)
    return Err(DispatchFailure(
        message=exhaustion_msg,
        attempted_backends=list(fallback_chain),
        error_details={"error_type": type(last_error).__name__ if last_error else "unknown"},
    ))


async def dispatch(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict,
) -> Result[EngineResponse, Exception]:
    """Execute the full dispatch pipeline with fallback and return an EngineResponse.

    This is the main entry point for engine-style consumers.

    Runs the MBR -> CBR -> LBR cascade to get a ranked candidate list, then
    attempts generation on each candidate in order until one succeeds.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"

    ctx = DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
    )
    logger.debug("starting dispatch pipeline")

    cascade_result = _run_cascade(order, ctx)
    if cascade_result.is_err():
        logger.debug("dispatch failed at cascade stage", error=str(cascade_result.error))
        return Err(cascade_result.error)

    candidates = cascade_result.value
    assert len(candidates) > 0, "cascade must return at least one candidate"

    base_context = _build_dispatch_context(order)
    return await _handle_fallback_chain(candidates, base_context, order, ctx)