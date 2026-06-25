"""Pinned dispatch — model-pinning v0.1.0 spec section 2.3.

Extracted from cascade.py to reduce file size. All pinned dispatch paths
(_pinned_preflight, _pinned_route, _pinned_dispatch_full, _pinned_dispatch_stream)
live here while sharing helpers from cascade.py.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from dragonlight_router.core.types import (
    BackendConfig,
    BackendStatus,
    BudgetExhaustedError,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    ModelNotFoundError,
    ModelUnhealthyError,
    StreamChunk,
)
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.context_filter import filter_context_for_provider

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports from cascade to avoid circular dependencies.
# These are helpers that remain in cascade.py as they are shared with
# the cascade dispatch path.
# ---------------------------------------------------------------------------


def _cascade():
    """Lazy import of cascade module to avoid circular import at module level."""
    from dragonlight_router.dispatch import cascade

    return cascade


# ---------------------------------------------------------------------------
# Pinned dispatch — model-pinning v0.1.0 spec section 2.3.
# ---------------------------------------------------------------------------


# DEVIATION DCS-FUNC-LEN — _pinned_preflight is 86 lines.
# Justification: linear preflight pipeline with sequential guard clauses; splitting
# would scatter the pre-flight contract across multiple functions without clarity gain.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def _pinned_preflight(
    order: DispatchOrder,
    ctx: Any,
) -> Result[BackendConfig, ModelNotFoundError | ModelUnhealthyError | BudgetExhaustedError]:
    """Run pinned dispatch pre-flight checks (registry, health, budget).

    Returns Ok(BackendConfig) when all checks pass, or an Err with the
    specific rejection reason. Shared by all pinned dispatch entry points.
    """
    assert order.model is not None, "pinned preflight requires order.model to be set"

    model_name = order.model
    backend, state = ctx.registry.get(model_name)

    # Step 1-2: Registry lookup — not found
    if backend is None or state is None:
        logger.info(
            "pinned_dispatch_rejected",
            model=model_name,
            reason="not_found",
            request_id=order.request_id,
        )
        return Err(
            ModelNotFoundError(
                model=model_name,
                message=f"pinned model not found in registry: {model_name}",
            )
        )

    # Step 3: Retired check
    if state.status == BackendStatus.RETIRED:
        logger.info(
            "pinned_dispatch_rejected",
            model=model_name,
            reason="retired",
            request_id=order.request_id,
        )
        return Err(
            ModelUnhealthyError(
                model=model_name,
                status="retired",
                message=f"pinned model is retired: {model_name}",
            )
        )

    # Step 3b: KEY_INVALID treated as retired (HAZ-PIN-003)
    if state.status == BackendStatus.KEY_INVALID:
        logger.info(
            "pinned_dispatch_rejected",
            model=model_name,
            reason="retired",
            request_id=order.request_id,
        )
        return Err(
            ModelUnhealthyError(
                model=model_name,
                status="retired",
                message=f"pinned model has invalid key: {model_name}",
            )
        )

    # Step 4: Circuit breaker check (when honor_health is true)
    if state.is_circuit_open() and ctx.pinned_dispatch_config.honor_health:
        logger.info(
            "pinned_dispatch_rejected",
            model=model_name,
            reason="circuit_open",
            request_id=order.request_id,
        )
        return Err(
            ModelUnhealthyError(
                model=model_name,
                status="circuit_open",
                message=f"pinned model is unhealthy (circuit open): {model_name}",
            )
        )

    # Step 5: Budget capacity check
    if not ctx.budget_tracker.has_capacity(backend.config.provider):
        logger.info(
            "pinned_dispatch_rejected",
            model=model_name,
            reason="budget_exhausted",
            provider=backend.config.provider,
            request_id=order.request_id,
        )
        return Err(
            BudgetExhaustedError(
                model=model_name,
                provider=backend.config.provider,
                message=f"pinned model's provider budget exhausted: {backend.config.provider}",
            )
        )

    return Ok(backend.config)


async def _pinned_route(
    order: DispatchOrder,
    ctx: Any,
) -> Result[BackendConfig, Exception]:
    """Pinned dispatch path for route() — resolve backend, skip cascade.

    Returns Ok(BackendConfig) on success or Err with the rejection reason.
    """
    assert order.model is not None, "_pinned_route requires order.model"

    logger.info(
        "pinned_dispatch_start",
        model=order.model,
        request_id=order.request_id,
        dispatch_mode="pinned",
    )

    return _pinned_preflight(order, ctx)  # type: ignore[return-value]


# DEVIATION DCS-FUNC-LEN — _pinned_dispatch_full is 124 lines.
# Justification: end-to-end pinned dispatch pipeline including preflight, adapter call,
# response construction, and cache storage. Splitting would fragment the dispatch contract.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
async def _pinned_dispatch_full(
    order: DispatchOrder,
    ctx: Any,
) -> Result[EngineResponse, Exception]:
    """Pinned dispatch path for dispatch() — resolve, call adapter, return EngineResponse.

    No fallback chain. On adapter failure, returns Err immediately.
    """
    assert order.model is not None, "_pinned_dispatch_full requires order.model"

    cascade = _cascade()

    logger.info(
        "pinned_dispatch_start",
        model=order.model,
        request_id=order.request_id,
        dispatch_mode="pinned",
    )

    # Cache check (AC-PIN-018)
    cached = cascade._try_cache_lookup(order)
    if cached is not None:
        cascade._log_cache_hit(cached)
        return Ok(cached)

    preflight = _pinned_preflight(order, ctx)
    if isinstance(preflight, Err):
        return preflight  # type: ignore[return-value]

    backend_config = preflight.value
    assert isinstance(backend_config, BackendConfig), "preflight must return BackendConfig"

    # Rate limit check via check_and_reserve (AC-PIN-008)
    reserved = await ctx.budget_tracker.check_and_reserve(backend_config.provider)
    if not reserved:
        logger.info(
            "pinned_dispatch_rejected",
            model=order.model,
            reason="rate_limited",
            provider=backend_config.provider,
            request_id=order.request_id,
        )
        return Err(
            BudgetExhaustedError(  # type: ignore[arg-type]
                model=order.model,
                provider=backend_config.provider,
                message=f"pinned model's provider rate limit exhausted: {backend_config.provider}",
            )
        )

    # Build context and apply trust tier filtering (AC-PIN-011)
    base_context = cascade._build_dispatch_context(order)

    # HAZ-014: Fresh adapter per dispatch attempt
    adapter = cascade._adapters_mod.create_adapter(backend_config)
    assert adapter.status == BackendStatus.AVAILABLE, (
        f"Fresh adapter must start AVAILABLE, got {adapter.status}"
    )

    provider_trust = cascade._tier_to_provider_trust(backend_config.tier)
    filtered_context = filter_context_for_provider(base_context, provider_trust)
    messages = cascade._resolve_messages(order, filtered_context)

    # Dispatch to adapter
    t0 = time.monotonic()
    tool_calls_resp: list[dict] | None = None
    finish_reason: str | None = None
    try:
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
            content = result_msg.get("content", "") or ""
            tool_calls_resp = result_msg.get("tool_calls")
            finish_reason = result_msg.get("finish_reason")
        else:
            content_parts: list[str] = []
            async for chunk in adapter.generate(
                messages,
                max_tokens=4096,
                temperature=0.7,
                stream=True,
            ):
                content_parts.append(chunk)
            content = "".join(content_parts)
    except (RuntimeError, ValueError, ConnectionError, OSError, TypeError) as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        cascade._record_adapter_failure(exc, backend_config, ctx)
        error_code, sanitized_msg = cascade._sanitize_dispatch_error(exc)
        logger.info(
            "pinned_dispatch_failed",
            model=order.model,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_ms=round(latency_ms, 2),
            request_id=order.request_id,
            dispatch_mode="pinned",
        )
        return Err(
            DispatchFailure(  # type: ignore[arg-type]
                message=f"pinned model dispatch failed: {order.model}",
                attempted_backends=[backend_config.name],
                error_details={"error_code": error_code, "error_message": sanitized_msg},
            )
        )

    latency_ms = (time.monotonic() - t0) * 1000.0
    tokens_in, tokens_out = cascade._estimate_and_log_tokens(messages, content, backend_config.name)
    cost_usd = cascade._compute_cost_usd(tokens_in, tokens_out, backend_config.cost)

    # Record success in health/budget trackers (AC-PIN-009, AC-PIN-010)
    cascade._record_dispatch_success(
        backend_config,
        ctx,
        adapter,
        tokens_in,
        tokens_out,
        cost_usd,
        latency_ms,
        fallback_chain=[],
    )

    response = EngineResponse(
        content=content,
        backend_used=backend_config.name,
        backend_tier=backend_config.tier,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=cost_usd,
        latency_ms=latency_ms,
        was_fallback=False,
        fallback_chain=[],
        dispatch_mode="pinned",
        tool_calls=tool_calls_resp,
        finish_reason=finish_reason,
    )

    logger.info(
        "pinned_dispatch_complete",
        model=order.model,
        latency_ms=round(latency_ms, 2),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=round(cost_usd, 8),
        request_id=order.request_id,
        dispatch_mode="pinned",
    )

    # Cache store (AC-PIN-019)
    cascade._store_cache_response(order, response)

    return Ok(response)


# DEVIATION DCS-FUNC-LEN — _pinned_dispatch_stream is 124 lines.
# Justification: end-to-end pinned streaming dispatch pipeline mirroring _pinned_dispatch_full.
# Splitting would fragment the streaming contract. Approved by: architect.
# Scope: this function. Expiration: revisit 2026-09-01.
async def _pinned_dispatch_stream(
    order: DispatchOrder,
    ctx: Any,
) -> AsyncIterator[StreamChunk]:
    """Pinned dispatch path for dispatch_stream() — stream from a single backend.

    No fallback. On preflight rejection, yields an error chunk.
    On adapter failure, yields an error chunk.
    """
    assert order.model is not None, "_pinned_dispatch_stream requires order.model"

    cascade = _cascade()

    logger.info(
        "pinned_dispatch_start",
        model=order.model,
        request_id=order.request_id,
        dispatch_mode="pinned",
    )

    preflight = _pinned_preflight(order, ctx)
    if isinstance(preflight, Err):
        error = preflight.error
        yield StreamChunk(
            event_type="error",
            error_message=getattr(error, "message", str(error)),
            dispatch_mode="pinned",
        )
        return

    backend_config = preflight.value
    assert isinstance(backend_config, BackendConfig), "preflight must return BackendConfig"

    # Rate limit check via check_and_reserve (AC-PIN-008)
    reserved = await ctx.budget_tracker.check_and_reserve(backend_config.provider)
    if not reserved:
        logger.info(
            "pinned_dispatch_rejected",
            model=order.model,
            reason="rate_limited",
            provider=backend_config.provider,
            request_id=order.request_id,
        )
        yield StreamChunk(
            event_type="error",
            error_message=(
                f"pinned model's provider rate limit exhausted: {backend_config.provider}"
            ),
            dispatch_mode="pinned",
        )
        return

    # Build context and apply trust tier filtering (AC-PIN-011)
    base_context = cascade._build_dispatch_context(order)

    # HAZ-014: Fresh adapter per dispatch attempt
    adapter = cascade._adapters_mod.create_adapter(backend_config)
    assert adapter.status == BackendStatus.AVAILABLE, (
        f"Fresh adapter must start AVAILABLE, got {adapter.status}"
    )
    provider_trust = cascade._tier_to_provider_trust(backend_config.tier)
    filtered_context = filter_context_for_provider(base_context, provider_trust)
    messages = cascade._resolve_messages(order, filtered_context)

    t0 = time.monotonic()
    tokens_out_chars = 0

    try:
        async for chunk in adapter.generate(
            messages,
            max_tokens=4096,
            temperature=0.7,
            stream=True,
        ):
            tokens_out_chars += len(chunk)
            yield StreamChunk(event_type="token", content=chunk, dispatch_mode="pinned")
    except (RuntimeError, ValueError, ConnectionError, OSError, TypeError) as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        cascade._record_adapter_failure(exc, backend_config, ctx)
        _error_code, sanitized_msg = cascade._sanitize_dispatch_error(exc)
        logger.info(
            "pinned_dispatch_failed",
            model=order.model,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_ms=round(latency_ms, 2),
            request_id=order.request_id,
            dispatch_mode="pinned",
        )
        yield StreamChunk(
            event_type="error",
            error_message=f"pinned model dispatch failed: {sanitized_msg}",
            dispatch_mode="pinned",
        )
        return

    latency_ms = (time.monotonic() - t0) * 1000.0
    tokens_in, tokens_out = cascade._estimate_and_log_tokens(
        messages,
        "x" * tokens_out_chars,
        backend_config.name,
    )
    cost_usd = cascade._compute_cost_usd(tokens_in, tokens_out, backend_config.cost)

    # Record success (AC-PIN-009, AC-PIN-010)
    ctx.health_tracker.record_success(backend_config.model, latency_ms)
    ctx.budget_tracker.record_request(backend_config.provider, tokens_in + tokens_out)
    adapter.record_usage(tokens_in, tokens_out)

    logger.info(
        "pinned_dispatch_complete",
        model=order.model,
        latency_ms=round(latency_ms, 2),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=round(cost_usd, 8),
        request_id=order.request_id,
        dispatch_mode="pinned",
    )

    yield StreamChunk(
        event_type="metadata",
        backend_used=backend_config.name,
        backend_tier=backend_config.tier.value,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost_usd=cost_usd,
        latency_ms=latency_ms,
        was_fallback=False,
        fallback_chain=[],
        dispatch_mode="pinned",
    )
