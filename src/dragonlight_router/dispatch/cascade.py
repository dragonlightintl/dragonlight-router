"""Cascade dispatch — MBR → IBR → CBR → LBR composition."""

from __future__ import annotations

import json
import time
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

import dragonlight_router.adapters as _adapters_mod
from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.caching.simple import SimpleCache
from dragonlight_router.config.schema import IntentClassificationConfig, PinnedDispatchConfig
from dragonlight_router.core.errors import BudgetExceededError, LBRNoCapacityError
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendConfig,
    BackendCostProfile,
    BackendStatus,
    BackendTier,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    GenerativeBackend,
    ScoredCandidate,
    StreamChunk,
)

# Re-export pinned dispatch functions so existing imports from cascade still work.
from dragonlight_router.dispatch.pinned import (  # noqa: F401
    _pinned_dispatch_full,
    _pinned_dispatch_stream,
    _pinned_preflight,
    _pinned_route,
)
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.cbr import filter_by_cost, score_candidate
from dragonlight_router.selection.context_filter import (
    ProviderTrustTier,
    filter_context_for_provider,
)
from dragonlight_router.selection.ibr import IBRResult, run_ibr_stage
from dragonlight_router.selection.lbr import filter_by_rate_limit, select_final_candidate
from dragonlight_router.selection.mbr import (
    MBRNoCandidatesError,
    filter_by_capabilities,
)
from dragonlight_router.selection.feedback import FeedbackStore
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    _STAKES_TO_WEIGHTS,
    classify_request_stakes,
    cost_adjusted_weights,
    cost_governor_active,
    intent_weights_for_category,
)
from dragonlight_router.selection.spectrograph import SpectrographProfileLoader

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SEC: Error sanitization — prevent provider internals from leaking to clients
# ---------------------------------------------------------------------------


def _sanitize_dispatch_error(exc: BaseException) -> tuple[str, str]:
    """Map a provider exception to an opaque error code and message.

    Returns (error_code, sanitized_message) suitable for HTTP response payloads.
    The raw exception should still be logged via structlog for debugging, but
    must never be included in response payloads sent to HTTP clients because
    provider error messages can contain internal URLs, auth header patterns,
    or model identifiers not meant for external consumption.
    """
    if isinstance(exc, ConnectionError):
        return "PROVIDER_UNAVAILABLE", "Provider connection failed"
    if isinstance(exc, TimeoutError):
        return "PROVIDER_TIMEOUT", "Provider request timed out"
    if isinstance(exc, ValueError):
        return "DISPATCH_INVALID", "Invalid dispatch parameters"
    if isinstance(exc, TypeError):
        return "DISPATCH_TYPE_ERROR", "Dispatch type mismatch"
    if isinstance(exc, OSError):
        return "PROVIDER_IO_ERROR", "Provider I/O error"
    if isinstance(exc, RuntimeError):
        return "DISPATCH_FAILED", "Provider request failed"

    return "DISPATCH_FAILED", f"Dispatch failed ({type(exc).__name__})"


# Penalty multiplier applied to scores of DEGRADED backends (0.5 = halve the score).
_DEGRADED_SCORE_PENALTY = 0.5

# HAZ-010: Default chars-per-token ratio for estimation when provider
# does not report usage. The heuristic is ~4 chars/token for English text.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Module-level response cache — set via configure_cache(), None = caching disabled.
# DEVIATION CS-MUTABLE-002: intentionally mutable — runtime cache/singleton.
_dispatch_cache: SimpleCache | None = None


def configure_cache(
    db_path: Path,
    max_entries: int = 1000,
    ttl_s: int = 3600,
) -> SimpleCache:
    """Initialize and configure the dispatch response cache.

    Called during router startup to enable caching. Returns the cache
    instance for lifecycle management (e.g. close on shutdown).
    """
    global _dispatch_cache  # noqa: PLW0603
    assert isinstance(db_path, Path), "db_path must be a Path instance"
    assert max_entries > 0, "max_entries must be positive"
    _dispatch_cache = SimpleCache(db_path=db_path, max_entries=max_entries, ttl_s=ttl_s)
    logger.info(
        "dispatch_cache_configured",
        db_path=str(db_path),
        max_entries=max_entries,
        ttl_s=ttl_s,
    )
    return _dispatch_cache


def get_cache() -> SimpleCache | None:
    """Return the active dispatch cache, or None if caching is disabled."""
    return _dispatch_cache


def _reset_cache() -> None:
    """Reset the dispatch cache (for test isolation)."""
    global _dispatch_cache  # noqa: PLW0603
    _dispatch_cache = None


# Mapping from caller-specified trust tier strings to ProviderTrustTier ordering.
# Higher numeric value = more restrictive trust requirement.
_TRUST_TIER_RANK: types.MappingProxyType[str, int] = types.MappingProxyType(
    {
        "untrusted": 0,
        "semi_trusted": 1,
        "trusted": 2,
        "local": 3,
    }
)

# Reverse mapping from ProviderTrustTier enum to rank for comparison.
_PROVIDER_TRUST_RANK: types.MappingProxyType[ProviderTrustTier, int] = types.MappingProxyType(
    {
        ProviderTrustTier.UNTRUSTED: 0,
        ProviderTrustTier.SEMI_TRUSTED: 1,
        ProviderTrustTier.TRUSTED: 2,
        ProviderTrustTier.LOCAL: 3,
    }
)


@dataclass(frozen=True)
class DispatchContext:
    """Grouped context for dispatch pipeline operations (QA-004: parameter count > 4)."""

    registry: BackendRegistry
    budget_tracker: BudgetTracker
    health_tracker: HealthTracker
    config: dict[str, Any]
    ibr_config: IntentClassificationConfig | None = None
    spectrograph_loader: SpectrographProfileLoader | None = None
    classification_adapter: GenerativeBackend | None = None
    pinned_dispatch_config: PinnedDispatchConfig = PinnedDispatchConfig()
    feedback_store: FeedbackStore | None = None


def _estimate_token_count(char_count: int) -> int:
    """HAZ-010: Estimate token count from character count.

    Uses a simple chars/4 heuristic. This function centralizes the
    estimation so it can be replaced with a tokenizer library or
    provider-reported usage in the future.
    """
    assert char_count >= 0, "char_count must be non-negative"
    return max(1, char_count // _CHARS_PER_TOKEN_ESTIMATE)


def _log_token_estimation(
    estimated: int,
    actual_chars: int,
    backend_name: str,
    direction: str,
) -> None:
    """HAZ-010: Log token estimation details for observability.

    Logs the estimation ratio so operators can monitor estimation
    accuracy and adjust if needed.
    """
    logger.debug(
        "token_estimation",
        backend=backend_name,
        direction=direction,
        estimated_tokens=estimated,
        char_count=actual_chars,
        chars_per_token=_CHARS_PER_TOKEN_ESTIMATE,
    )


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
    if isinstance(mbr_result, Err):
        logger.debug("MBR stage failed", error=str(mbr_result.error))
        return Err(mbr_result.error)

    logger.debug("MBR stage complete", candidate_count=len(mbr_result.value))
    return Ok(mbr_result.value)


def _run_cbr_stage(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ctx: DispatchContext,
    ibr_result: IBRResult | None = None,
) -> Result[list[ScoredCandidate], Exception]:
    """CBR stage: filter by budget, score with cost-effectiveness, rank candidates."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    daily_spend, monthly_spend = _compute_aggregate_spend(ctx.budget_tracker, candidates)

    cbr_candidates = filter_by_cost(
        candidates,
        order,
        ctx.budget_tracker,
        daily_spend=daily_spend,
        monthly_spend=monthly_spend,
        config=ctx.config,
        health_tracker=ctx.health_tracker,
    )
    logger.debug("CBR filtering complete", candidate_count=len(cbr_candidates))

    if not cbr_candidates:
        return Err(BudgetExceededError("No candidates remain after budget filtering"))

    weights = _resolve_cbr_weights(ibr_result, ctx, order)
    if cost_governor_active(daily_spend, monthly_spend, ctx.config):
        weights = cost_adjusted_weights(weights)

    scored = _score_and_rank_candidates(
        cbr_candidates,
        order,
        weights,
        ctx,
        ibr_result=ibr_result,
    )
    logger.debug("CBR scoring complete", candidate_count=len(scored))
    return Ok(scored)


async def _run_ibr_stage(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ctx: DispatchContext,
) -> IBRResult | None:
    """Run the IBR stage if configured. Returns None when IBR is not wired.

    IBR must NEVER block or fail the cascade (IBR-SYS-03).
    All errors are caught and degraded to inactive result.
    """
    if ctx.ibr_config is None or ctx.spectrograph_loader is None:
        return None

    try:
        result = await run_ibr_stage(
            order=order,
            candidates=candidates,
            ibr_config=ctx.ibr_config,
            spectrograph_loader=ctx.spectrograph_loader,
            classification_adapter=ctx.classification_adapter,
            feedback_store=ctx.feedback_store,
        )
        logger.debug("IBR stage complete", ibr_active=result.ibr_active)
        return result
    except (KeyError, ValueError, TypeError, RuntimeError, OSError, TimeoutError):
        logger.warning("ibr_stage_cascade_error", exc_info=True)
        return None


def _resolve_cbr_weights(
    ibr_result: IBRResult | None,
    ctx: DispatchContext,
    order: DispatchOrder | None = None,
) -> ScoringWeightsConfig:
    """Determine CBR scoring weights based on IBR result, intent category, and context signals.

    Weight resolution order:
    1. Compute both intent-based and context-escalation weights.
       If both are active, use whichever is higher-stakes (higher
       spectrograph_match weight = higher stakes).
    2. If IBR is active with valid spectrograph scores, use 6-dimension
       weights with the configured spectrograph_match_weight (IBR-SCORE-02).
    3. Otherwise fall back to default ScoringWeightsConfig.
    """
    if order is not None:
        # Try intent-based classification first
        intent_weights = intent_weights_for_category(order.intent_category)
        default_weights = ScoringWeightsConfig()

        # Context-based escalation
        stakes = classify_request_stakes(order)
        stakes_weights = _STAKES_TO_WEIGHTS.get(stakes)

        if intent_weights != default_weights and stakes_weights is not None:
            # Both signals active: use whichever is higher-stakes
            # (higher spectrograph_match weight = higher stakes)
            if stakes_weights.spectrograph_match > intent_weights.spectrograph_match:
                logger.debug(
                    "cbr_weights_escalated_by_context",
                    intent_category=order.intent_category,
                    stakes=stakes,
                    cost=stakes_weights.cost,
                    spectrograph_match=stakes_weights.spectrograph_match,
                )
                return stakes_weights
            logger.debug(
                "cbr_weights_from_intent_category",
                intent_category=order.intent_category,
                stakes=stakes,
                cost=intent_weights.cost,
                spectrograph_match=intent_weights.spectrograph_match,
            )
            return intent_weights
        elif intent_weights != default_weights:
            logger.debug(
                "cbr_weights_from_intent_category",
                intent_category=order.intent_category,
                stakes=stakes,
                cost=intent_weights.cost,
                spectrograph_match=intent_weights.spectrograph_match,
            )
            return intent_weights
        elif stakes_weights is not None:
            logger.debug(
                "cbr_weights_from_context_stakes",
                intent_category=order.intent_category,
                stakes=stakes,
                cost=stakes_weights.cost,
                spectrograph_match=stakes_weights.spectrograph_match,
            )
            return stakes_weights

    if ibr_result is not None and ibr_result.ibr_active:
        ibr_cfg = ctx.ibr_config
        weight = ibr_cfg.spectrograph_match_weight if ibr_cfg is not None else 0.15
        return ScoringWeightsConfig(
            cost=0.30,
            latency=0.20,
            priority=0.15,
            queue=0.10,
            health=0.10,
            spectrograph_match=weight,
        )
    return ScoringWeightsConfig()


# DEVIATION CS-PARAM-001: _score_and_rank_candidates takes 5 params — dataclass would break API.
def _score_and_rank_candidates(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    weights: ScoringWeightsConfig,
    ctx: DispatchContext,
    ibr_result: IBRResult | None = None,
) -> list[ScoredCandidate]:
    """Score each candidate and return them sorted best-first with scores preserved."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    spectrograph_scores = ibr_result.spectrograph_scores if ibr_result is not None else {}

    scored_candidates: list[ScoredCandidate] = []
    for candidate in candidates:
        spectrograph_match = spectrograph_scores.get(candidate.name, 0.0)
        score = score_candidate(
            config=candidate,
            order=order,
            weights=weights,
            budget_tracker=ctx.budget_tracker,
            health_tracker=ctx.health_tracker,
            spectrograph_match=spectrograph_match,
        )
        score = _apply_degraded_penalty(score, candidate, ctx.registry)
        scored_candidates.append(ScoredCandidate(config=candidate, score=score))

    scored_candidates.sort(key=lambda sc: sc.score, reverse=True)
    return scored_candidates


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
    candidates: list[ScoredCandidate],
    ctx: DispatchContext,
) -> Result[list[ScoredCandidate], Exception]:
    """LBR stage: filter by rate limit capacity, preserving scores."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert len(candidates) > 0, "candidates must not be empty"

    # Build a lookup from BackendConfig name -> ScoredCandidate for re-wrapping
    scored_by_name: dict[str, ScoredCandidate] = {sc.config.name: sc for sc in candidates}

    # LBR filters on BackendConfig — unwrap, filter, re-wrap
    configs = [sc.config for sc in candidates]
    lbr_configs = filter_by_rate_limit(configs, order, ctx.budget_tracker)
    logger.debug("LBR filtering complete", candidate_count=len(lbr_configs))

    if not lbr_configs:
        return Err(LBRNoCapacityError("No candidates remain after rate limit filtering"))

    lbr_scored = [scored_by_name[cfg.name] for cfg in lbr_configs]
    return Ok(lbr_scored)


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


async def _run_cascade(
    order: DispatchOrder,
    ctx: DispatchContext,
) -> Result[list[ScoredCandidate], Exception]:
    """Run MBR -> trust floor -> IBR -> CBR -> LBR cascade and return the full ranked list.

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
    if isinstance(mbr_result, Err):
        return mbr_result

    # Filter out models retired by the health tracker (404/403 at inference).
    # MBR checks registry state but not the health tracker's retirement list,
    # so retired models can slip through MBR on subsequent dispatches.
    non_retired = [
        c for c in mbr_result.value if not ctx.health_tracker.is_retired(c.model)
    ]
    if not non_retired:
        return Err(
            MBRNoCandidatesError("No candidates remain after filtering retired models")
        )
    logger.debug(
        "retired_model_filter",
        before=len(mbr_result.value),
        after=len(non_retired),
        retired=[c.name for c in mbr_result.value if ctx.health_tracker.is_retired(c.model)],
    )

    # HAZ-001: Enforce caller-specified trust floor before cost/rate scoring
    trust_filtered = _filter_by_trust_floor(
        non_retired,
        order.context_trust_tier,
    )
    if not trust_filtered:
        return Err(
            MBRNoCandidatesError("No candidates meet the requested context_trust_tier floor")
        )

    # IBR stage: classify intent and compute flavor scores (IBR-PIPE-01)
    ibr_result = await _run_ibr_stage(order, trust_filtered, ctx)

    cbr_result = _run_cbr_stage(order, trust_filtered, ctx, ibr_result=ibr_result)
    if isinstance(cbr_result, Err):
        return cbr_result

    return _run_lbr_stage(order, cbr_result.value, ctx)


# DEVIATION CS-004: route is 41 lines.
# Justification: Public API entry point with cascade orchestration, context construction,
# and result handling. Extraction would fragment the dispatch contract.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
# DEVIATION CS-PARAM-001: route takes 9 params — dataclass would break API.
async def route(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict[str, Any],
    *,
    ibr_config: IntentClassificationConfig | None = None,
    spectrograph_loader: SpectrographProfileLoader | None = None,
    classification_adapter: GenerativeBackend | None = None,
    pinned_dispatch_config: PinnedDispatchConfig | None = None,
    feedback_store: FeedbackStore | None = None,
) -> Result[BackendConfig, Exception]:
    """Run the MBR -> IBR -> CBR -> LBR cascade and return the selected BackendConfig.

    When ``order.model`` is set, bypasses the cascade entirely and resolves
    the backend directly via the registry (pinned dispatch path).

    Args:
        order: The dispatch order containing capability requirements.
        registry: The backend registry to fetch candidates from.
        budget_tracker: Tracks per-provider budget availability.
        health_tracker: Tracks per-model health and circuit breaker state.
        config: Router configuration dictionary.

    Returns:
        Ok(BackendConfig) selected by the cascade (or pinned lookup) or
        Err(Exception) if no candidates remain after filtering.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"

    ctx = DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
        ibr_config=ibr_config,
        spectrograph_loader=spectrograph_loader,
        classification_adapter=classification_adapter,
        pinned_dispatch_config=pinned_dispatch_config or PinnedDispatchConfig(),
        feedback_store=feedback_store,
    )

    # Pinned dispatch early branch — bypass cascade entirely.
    if order.model is not None:
        return await _pinned_route(order, ctx)

    cascade_result = await _run_cascade(order, ctx)
    if isinstance(cascade_result, Err):
        return cascade_result

    scored_candidates = cascade_result.value
    selected = select_final_candidate(scored_candidates)
    logger.debug(
        "cascade dispatch complete",
        selected_provider=selected.provider,
        selected_model=selected.model,
    )
    return Ok(selected)


def _build_dispatch_context(
    order: DispatchOrder,
) -> dict[str, Any]:
    """Build the base context dict for context filtering from a DispatchOrder."""
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

    base_context: dict[str, Any] = {}
    if order.system_prompt:
        base_context["system"] = {"prompt": order.system_prompt}
    base_context["task"] = order.operator_message
    return base_context


def _build_messages(
    filtered_context: dict[str, Any],
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


def _resolve_messages(
    order: DispatchOrder,
    base_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve the message list for an adapter call.

    When the order carries structured messages (tool-use conversations),
    returns those directly. Otherwise falls back to _build_messages()
    for the legacy system+user text path.
    """
    if order.messages is not None:
        return list(order.messages)
    return _build_messages(base_context, order.operator_message)


def _estimate_and_log_tokens(
    messages: list[dict],
    content: str,
    backend_name: str,
) -> tuple[int, int]:
    """HAZ-010: Estimate input/output token counts and log for observability."""
    input_chars = 0
    for m in messages:
        msg_content = m.get("content", "")
        if isinstance(msg_content, str):
            input_chars += len(msg_content)
        elif msg_content is not None:
            input_chars += len(str(msg_content))
    tokens_in = _estimate_token_count(input_chars)
    tokens_out = _estimate_token_count(len(content))
    _log_token_estimation(tokens_in, input_chars, backend_name, "input")
    _log_token_estimation(tokens_out, len(content), backend_name, "output")
    return tokens_in, tokens_out


def _compute_cost_usd(
    tokens_in: int,
    tokens_out: int,
    cost: BackendCostProfile,
) -> float:
    """Compute estimated cost in USD from token counts and cost profile."""
    return (tokens_in / 1_000_000) * cost.input_per_mtok + (
        tokens_out / 1_000_000
    ) * cost.output_per_mtok


# DEVIATION CS-PARAM-001: _record_dispatch_success takes 8 params — dataclass would break API.
def _record_dispatch_success(
    backend_config: BackendConfig,
    ctx: DispatchContext,
    adapter: Any,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float,
    fallback_chain: list[str],
) -> None:
    """Record a successful dispatch in health/budget trackers and log it."""
    ctx.health_tracker.record_success(backend_config.model, latency_ms)
    ctx.budget_tracker.record_request(backend_config.provider, tokens_in + tokens_out)
    adapter.record_usage(tokens_in, tokens_out)

    logger.info(
        "dispatch_result",
        provider=backend_config.provider,
        model=backend_config.model,
        latency_ms=round(latency_ms, 2),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        success=True,
        cache_hit=False,
        estimated_cost_usd=round(cost_usd, 8),
        was_fallback=len(fallback_chain) > 0,
    )


# DEVIATION CS-004: _try_adapter_dispatch is 43 lines.
# Justification: Single adapter dispatch attempt with context filtering, generation,
# token estimation, and tracking. Core pipeline step; further extraction already done.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
# DEVIATION CS-PARAM-001: _try_adapter_dispatch takes 5 params — dataclass would break API.
async def _try_adapter_dispatch(
    backend_config: BackendConfig,
    base_context: dict[str, Any],
    order: DispatchOrder,
    ctx: DispatchContext,
    fallback_chain: list[str],
) -> Result[EngineResponse, RuntimeError]:
    """Attempt a single adapter call and return Ok(EngineResponse) or Err on failure.

    HAZ-014 mitigation: Creates a fresh adapter per dispatch attempt so
    concurrent requests do not share mutable adapter state (_status).

    When the order carries tool definitions, uses the adapter's
    generate_with_tools() method for non-streaming tool-use dispatch.
    """
    assert isinstance(backend_config, BackendConfig), "backend_config must be BackendConfig"

    # HAZ-014: Fresh adapter instance prevents concurrent status mutation
    adapter = _adapters_mod.create_adapter(backend_config)
    assert adapter.status == BackendStatus.AVAILABLE, (
        f"Fresh adapter must start AVAILABLE, got {adapter.status}"
    )

    provider_trust = _tier_to_provider_trust(backend_config.tier)
    filtered_context = filter_context_for_provider(base_context, provider_trust)
    messages = _resolve_messages(order, filtered_context)

    t0 = time.monotonic()

    # Tool-use path: non-streaming, returns full message with tool_calls
    if order.tools and hasattr(adapter, "generate_with_tools"):
        tools_list = list(order.tools)
        result_msg = await adapter.generate_with_tools(
            messages,
            max_tokens=4096,
            temperature=0.7,
            tools=tools_list,
            tool_choice=order.tool_choice,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        content = result_msg.get("content", "") or ""
        tool_calls_resp = result_msg.get("tool_calls")
        finish_reason = result_msg.get("finish_reason")
        tokens_in, tokens_out = _estimate_and_log_tokens(messages, content, backend_config.name)
        cost_usd = _compute_cost_usd(tokens_in, tokens_out, backend_config.cost)

        _record_dispatch_success(
            backend_config,
            ctx,
            adapter,
            tokens_in,
            tokens_out,
            cost_usd,
            latency_ms,
            fallback_chain,
        )

        return Ok(
            _build_engine_response(
                backend_config,
                content,
                tokens_in,
                tokens_out,
                cost_usd,
                latency_ms,
                fallback_chain,
                tool_calls=tool_calls_resp,
                finish_reason=finish_reason,
            )
        )

    # Standard text-only path: streaming generation
    content_parts: list[str] = []
    async for chunk in adapter.generate(
        messages,
        max_tokens=4096,
        temperature=0.7,
        stream=True,
    ):
        content_parts.append(chunk)
    latency_ms = (time.monotonic() - t0) * 1000.0

    content = "".join(content_parts)
    tokens_in, tokens_out = _estimate_and_log_tokens(messages, content, backend_config.name)
    cost_usd = _compute_cost_usd(tokens_in, tokens_out, backend_config.cost)

    _record_dispatch_success(
        backend_config,
        ctx,
        adapter,
        tokens_in,
        tokens_out,
        cost_usd,
        latency_ms,
        fallback_chain,
    )

    return Ok(
        _build_engine_response(
            backend_config,
            content,
            tokens_in,
            tokens_out,
            cost_usd,
            latency_ms,
            fallback_chain,
        )
    )


# DEVIATION CS-PARAM-001: _build_engine_response takes 9 params — dataclass would break API.
def _build_engine_response(
    backend_config: BackendConfig,
    content: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float,
    fallback_chain: list[str],
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
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
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def _record_adapter_failure(
    exc: RuntimeError | ValueError | ConnectionError | OSError | TypeError,
    backend_config: BackendConfig,
    ctx: DispatchContext,
) -> None:
    """Record a backend generation failure in the health tracker."""
    cause = getattr(exc, "__cause__", None)
    http_status = (
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(cause, "status_code", None)
        or getattr(getattr(cause, "response", None), "status_code", None)
    )
    ctx.health_tracker.record_error(backend_config.model, http_status=http_status)

    # Structured dispatch failure logging
    logger.info(
        "dispatch_result",
        provider=backend_config.provider,
        model=backend_config.model,
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        success=False,
        cache_hit=False,
        error_type=type(exc).__name__,
        http_status=http_status,
    )


def _apply_fallback_policy(
    candidates: list[BackendConfig],
    order: DispatchOrder,
) -> list[BackendConfig]:
    """HAZ-004 mitigation: Filter candidates based on fallback_policy.

    - "allow" (default): all candidates eligible for fallback
    - "deny": only the first (primary) candidate is tried
    - "same_tier": only candidates at the same tier as the primary
    """
    policy = order.fallback_policy
    assert policy in ("allow", "deny", "same_tier"), f"invalid fallback_policy: {policy}"

    if policy == "allow" or len(candidates) <= 1:
        return candidates

    if policy == "deny":
        logger.debug(
            "fallback_policy_deny",
            primary=candidates[0].name,
            filtered_count=len(candidates) - 1,
        )
        return candidates[:1]

    # same_tier: keep only candidates matching the primary's tier
    primary_tier = candidates[0].tier
    filtered = [c for c in candidates if c.tier == primary_tier]
    logger.debug(
        "fallback_policy_same_tier",
        primary_tier=primary_tier.value,
        original_count=len(candidates),
        filtered_count=len(filtered),
    )
    assert len(filtered) >= 1, "same_tier filter must keep at least the primary"
    return filtered


# DEVIATION CS-004: _handle_fallback_chain is 51 lines.
# Justification: The try/except fallback iteration, policy application, and exhaustion
# error construction form a single linear flow; splitting would scatter the fallback logic.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
async def _handle_fallback_chain(
    candidates: list[BackendConfig],
    base_context: dict[str, Any],
    order: DispatchOrder,
    ctx: DispatchContext,
) -> Result[EngineResponse, Exception]:
    """Iterate through candidates attempting generation, falling back on failure.

    HAZ-004: Applies fallback_policy before iterating to restrict which
    candidates are eligible for fallback dispatch.
    """
    assert len(candidates) > 0, "candidates must not be empty"

    # HAZ-004: Apply fallback policy to restrict candidate pool
    eligible = _apply_fallback_policy(candidates, order)

    fallback_chain: list[str] = []
    last_error: RuntimeError | ValueError | ConnectionError | OSError | TypeError | None = None

    for backend_config in eligible:
        # Skip candidates retired mid-cascade (e.g. a prior candidate in this
        # dispatch returned 404/403, triggering retirement — subsequent
        # candidates with the same model should be skipped).
        if ctx.health_tracker.is_retired(backend_config.model):
            logger.debug(
                "skipping_retired_model_in_fallback",
                backend=backend_config.name,
                model=backend_config.model,
            )
            fallback_chain.append(backend_config.name)
            continue

        logger.debug(
            "attempting generation",
            backend=backend_config.name,
            attempt=len(fallback_chain) + 1,
        )
        try:
            result = await _try_adapter_dispatch(
                backend_config,
                base_context,
                order,
                ctx,
                fallback_chain,
            )
        except (RuntimeError, ValueError, ConnectionError, OSError, TypeError) as exc:
            last_error = exc
            fallback_chain.append(backend_config.name)
            _record_adapter_failure(exc, backend_config, ctx)
            logger.warning("backend generation failed", backend=backend_config.name, error=str(exc))
            continue

        if isinstance(result, Ok):
            # Minimum output token check (non-streaming only): if the model
            # returned fewer tokens than the threshold, treat as insufficient
            # output and continue the fallback chain to the next candidate.
            min_tokens = order.min_output_tokens
            if min_tokens > 0 and result.value.tokens_out < min_tokens:
                logger.warning(
                    "insufficient_output_tokens",
                    backend=backend_config.name,
                    output_tokens=result.value.tokens_out,
                    min_output_tokens=min_tokens,
                )
                ctx.health_tracker.record_error(
                    backend_config.model, http_status=None,
                )
                fallback_chain.append(backend_config.name)
                continue

            logger.debug("dispatch successful", backend_used=result.value.backend_used)
            return result

    if last_error is not None:
        error_code, sanitized_msg = _sanitize_dispatch_error(last_error)
    else:
        error_code, sanitized_msg = "DISPATCH_FAILED", "Unknown error"
    exhaustion_msg = (
        f"All {len(fallback_chain)} backends exhausted. "
        f"Fallback chain: {' → '.join(fallback_chain)}."
    )
    logger.error(
        "dispatch exhausted all backends",
        fallback_chain=fallback_chain,
        last_error=str(last_error),
    )
    return Err(
        DispatchFailure(  # type: ignore[arg-type]
            message=exhaustion_msg,
            attempted_backends=list(fallback_chain),
            error_details={"error_code": error_code, "error_message": sanitized_msg},
        )
    )


# DEVIATION CS-004: _try_cache_lookup is 44 lines.
# Justification: Cache key construction, lookup, and deserialization are a single
# linear flow; extracting pieces would duplicate the key-building logic.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def _try_cache_lookup(order: DispatchOrder) -> EngineResponse | None:
    """Attempt to retrieve a cached response for the given order.

    Returns an EngineResponse if cache hit, None on miss or if caching is disabled.
    Only caches deterministic requests (temperature == 0 or very low).
    """
    cache = _dispatch_cache
    if cache is None:
        return None

    # Build cache key from order fields
    messages = [{"role": "user", "content": order.operator_message}]
    if order.system_prompt:
        messages.insert(0, {"role": "system", "content": order.system_prompt})

    cache_key = SimpleCache.make_key(
        model_id=order.intent_category,
        system_prompt=order.system_prompt,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
    )

    cached_value = cache.get(cache_key)
    if cached_value is None:
        return None

    try:
        data = json.loads(cached_value)
        assert isinstance(data, dict), "cached value must be a JSON object"
        return EngineResponse(
            content=data["content"],
            backend_used=data["backend_used"],
            backend_tier=BackendTier(data["backend_tier"]),
            tokens_in=data["tokens_in"],
            tokens_out=data["tokens_out"],
            estimated_cost_usd=0.0,
            latency_ms=0.0,
            was_fallback=False,
            fallback_chain=[],
        )
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        logger.warning("cache_deserialize_failed", cache_key=cache_key[:16])
        return None


def _store_cache_response(order: DispatchOrder, response: EngineResponse) -> None:
    """Store a successful dispatch response in the cache."""
    cache = _dispatch_cache
    if cache is None:
        return

    messages = [{"role": "user", "content": order.operator_message}]
    if order.system_prompt:
        messages.insert(0, {"role": "system", "content": order.system_prompt})

    cache_key = SimpleCache.make_key(
        model_id=order.intent_category,
        system_prompt=order.system_prompt,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
    )

    cache_value = json.dumps(
        {
            "content": response.content,
            "backend_used": response.backend_used,
            "backend_tier": response.backend_tier.value,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
        },
        separators=(",", ":"),
    )

    cache.put(cache_key, cache_value)
    logger.debug("response_cached", cache_key=cache_key[:16])


def _log_cache_hit(cached: EngineResponse) -> None:
    """Log a cache hit dispatch result."""
    logger.info(
        "dispatch_result",
        provider=cached.backend_used,
        model="",
        latency_ms=0.0,
        input_tokens=cached.tokens_in,
        output_tokens=cached.tokens_out,
        success=True,
        cache_hit=True,
    )


# DEVIATION CS-004: dispatch is 48 lines.
# Justification: Main dispatch entry point with cache check, cascade, fallback chain,
# and cache storage. Public API contract; cache hit path already extracted.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
# DEVIATION CS-PARAM-001: dispatch takes 9 params — dataclass would break API.
async def dispatch(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict[str, Any],
    *,
    ibr_config: IntentClassificationConfig | None = None,
    spectrograph_loader: SpectrographProfileLoader | None = None,
    classification_adapter: GenerativeBackend | None = None,
    pinned_dispatch_config: PinnedDispatchConfig | None = None,
    feedback_store: FeedbackStore | None = None,
) -> Result[EngineResponse, Exception]:
    """Execute the full dispatch pipeline with fallback and return an EngineResponse.

    This is the main entry point for engine-style consumers.

    When ``order.model`` is set, bypasses the cascade entirely and dispatches
    directly to the pinned backend (no fallback chain).

    Otherwise checks the response cache first. On miss, runs the
    MBR -> IBR -> CBR -> LBR cascade to get a ranked candidate list,
    then attempts generation on each candidate in order until one succeeds.
    Successful responses are cached for future hits.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"

    ctx = DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
        ibr_config=ibr_config,
        spectrograph_loader=spectrograph_loader,
        classification_adapter=classification_adapter,
        pinned_dispatch_config=pinned_dispatch_config or PinnedDispatchConfig(),
        feedback_store=feedback_store,
    )

    # Pinned dispatch early branch — bypass cascade entirely.
    if order.model is not None:
        return await _pinned_dispatch_full(order, ctx)

    cached = _try_cache_lookup(order)
    if cached is not None:
        _log_cache_hit(cached)
        return Ok(cached)

    logger.debug("starting dispatch pipeline")

    cascade_result = await _run_cascade(order, ctx)
    if isinstance(cascade_result, Err):
        logger.debug("dispatch failed at cascade stage", error=str(cascade_result.error))
        return Err(cascade_result.error)

    scored_candidates = cascade_result.value
    assert len(scored_candidates) > 0, "cascade must return at least one candidate"

    candidates = [sc.config for sc in scored_candidates]
    base_context = _build_dispatch_context(order)
    result = await _handle_fallback_chain(candidates, base_context, order, ctx)

    if isinstance(result, Ok):
        _store_cache_response(order, result.value)

    return result


# DEVIATION CS-PARAM-001: _build_stream_metadata_chunk takes 6 params — dataclass would break API.
def _build_stream_metadata_chunk(
    backend_config: BackendConfig,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float,
    fallback_chain: list[str],
) -> StreamChunk:
    """Build the final metadata StreamChunk after streaming completes."""
    return StreamChunk(
        event_type="metadata",
        backend_used=backend_config.name,
        backend_tier=backend_config.tier.value,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=cost_usd,
        latency_ms=latency_ms,
        was_fallback=len(fallback_chain) > 0,
        fallback_chain=list(fallback_chain),
    )


# DEVIATION CS-004: _try_streaming_dispatch is 48 lines.
# Justification: Async generator that yields token chunks during generation; the yield
# prevents extracting the loop body. Token estimation and metadata construction already
# extracted into helpers.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
# DEVIATION CS-PARAM-001: _try_streaming_dispatch takes 5 params — dataclass would break API.
async def _try_streaming_dispatch(
    backend_config: BackendConfig,
    base_context: dict[str, Any],
    order: DispatchOrder,
    ctx: DispatchContext,
    fallback_chain: list[str],
) -> AsyncIterator[StreamChunk]:
    """Stream tokens from a single adapter, yielding StreamChunk events.

    Yields token chunks as they arrive, then a final metadata chunk
    with cost/latency/fallback info. On adapter failure, yields an
    error chunk and returns (caller handles fallback).

    HAZ-014 mitigation: Creates a fresh adapter per dispatch attempt.
    """
    assert isinstance(backend_config, BackendConfig), "backend_config must be BackendConfig"

    # HAZ-014: Fresh adapter instance prevents concurrent status mutation
    adapter = _adapters_mod.create_adapter(backend_config)
    assert adapter.status == BackendStatus.AVAILABLE, (
        f"Fresh adapter must start AVAILABLE, got {adapter.status}"
    )
    provider_trust = _tier_to_provider_trust(backend_config.tier)
    filtered_context = filter_context_for_provider(base_context, provider_trust)
    messages = _resolve_messages(order, filtered_context)

    t0 = time.monotonic()
    tokens_out_chars = 0

    async for chunk in adapter.generate(
        messages,
        max_tokens=4096,
        temperature=0.7,
        stream=True,
    ):
        tokens_out_chars += len(chunk)
        yield StreamChunk(event_type="token", content=chunk)

    latency_ms = (time.monotonic() - t0) * 1000.0
    tokens_in, tokens_out = _estimate_and_log_tokens(
        messages,
        "x" * tokens_out_chars,
        backend_config.name,
    )
    cost_usd = _compute_cost_usd(tokens_in, tokens_out, backend_config.cost)

    ctx.health_tracker.record_success(backend_config.model, latency_ms)
    ctx.budget_tracker.record_request(backend_config.provider, tokens_in + tokens_out)
    adapter.record_usage(tokens_in, tokens_out)

    yield _build_stream_metadata_chunk(
        backend_config,
        tokens_in,
        tokens_out,
        cost_usd,
        latency_ms,
        fallback_chain,
    )


async def _stream_with_fallback(
    eligible: list[BackendConfig],
    base_context: dict[str, Any],
    order: DispatchOrder,
    ctx: DispatchContext,
) -> AsyncIterator[StreamChunk]:
    """Iterate through eligible backends, streaming from the first that succeeds.

    Yields token/metadata chunks on success, or an error chunk if all backends fail.
    """
    fallback_chain: list[str] = []

    for backend_config in eligible:
        # Skip candidates retired mid-cascade (404/403 from prior candidate).
        if ctx.health_tracker.is_retired(backend_config.model):
            logger.debug(
                "skipping_retired_model_in_stream_fallback",
                backend=backend_config.name,
                model=backend_config.model,
            )
            fallback_chain.append(backend_config.name)
            continue

        try:
            async for chunk in _try_streaming_dispatch(
                backend_config,
                base_context,
                order,
                ctx,
                fallback_chain,
            ):
                yield chunk
            return  # Success — metadata chunk already yielded
        except (RuntimeError, ValueError, ConnectionError, OSError, TypeError) as exc:
            fallback_chain.append(backend_config.name)
            _record_adapter_failure(exc, backend_config, ctx)
            logger.warning(
                "streaming_backend_failed",
                backend=backend_config.name,
                error=str(exc),
            )

    yield StreamChunk(
        event_type="error",
        error_message=(
            f"All {len(fallback_chain)} backends exhausted. "
            f"Fallback chain: {' -> '.join(fallback_chain)}"
        ),
    )


# DEVIATION CS-004: dispatch_stream is 42 lines.
# Justification: Async generator public API that must yield from both error and success
# paths. Fallback iteration already extracted into _stream_with_fallback.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
# DEVIATION CS-PARAM-001: dispatch_stream takes 9 params — dataclass would break API.
async def dispatch_stream(
    order: DispatchOrder,
    registry: BackendRegistry,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker,
    config: dict[str, Any],
    *,
    ibr_config: IntentClassificationConfig | None = None,
    spectrograph_loader: SpectrographProfileLoader | None = None,
    classification_adapter: GenerativeBackend | None = None,
    pinned_dispatch_config: PinnedDispatchConfig | None = None,
    feedback_store: FeedbackStore | None = None,
) -> AsyncIterator[StreamChunk]:
    """Execute the cascade and stream tokens as they arrive from the LLM.

    When ``order.model`` is set, bypasses the cascade entirely and streams
    directly from the pinned backend (no fallback).

    Yields StreamChunk objects with event_type "token" for content,
    "metadata" for final response metadata, and "error" for failures.
    Implements fallback: if a backend fails mid-stream, logs the error
    and tries the next candidate.
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(registry, BackendRegistry), "registry must be BackendRegistry instance"

    ctx = DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
        ibr_config=ibr_config,
        spectrograph_loader=spectrograph_loader,
        classification_adapter=classification_adapter,
        pinned_dispatch_config=pinned_dispatch_config or PinnedDispatchConfig(),
        feedback_store=feedback_store,
    )

    # Pinned dispatch early branch — bypass cascade entirely.
    if order.model is not None:
        async for chunk in _pinned_dispatch_stream(order, ctx):
            yield chunk
        return

    logger.debug("starting streaming dispatch pipeline")

    cascade_result = await _run_cascade(order, ctx)
    if isinstance(cascade_result, Err):
        yield StreamChunk(
            event_type="error",
            error_message=str(cascade_result.error),
        )
        return

    scored_candidates = cascade_result.value
    assert len(scored_candidates) > 0, "cascade must return at least one candidate"

    candidates = [sc.config for sc in scored_candidates]
    eligible = _apply_fallback_policy(candidates, order)
    base_context = _build_dispatch_context(order)

    async for chunk in _stream_with_fallback(eligible, base_context, order, ctx):
        yield chunk
