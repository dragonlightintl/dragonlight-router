"""HTTP route handlers for the router API.

All routes operate on a shared RouterEngine instance.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from dragonlight_router.catalog import refresher as _refresher_mod
from dragonlight_router.core.types import BackendTier, RequestOutcome, DispatchOrder, EngineResponse, DispatchFailure, StreamChunk
from dragonlight_router.result import Ok, Err
from dragonlight_router.router import RouterEngine

logger = structlog.get_logger()

# --- Input validation constants ---

_MAX_STRING_LENGTH = 100_000
_MAX_RESPONSE_LENGTH = 500_000
_SELECT_MAX_TOP_N = 500
_DISPATCH_REQUIRED_FIELDS = ("intent_category", "specific_intent", "operator_message", "context_tokens")

# Matches control characters EXCEPT newline (\n), carriage return (\r), and tab (\t)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --- Admin endpoint paths requiring auth (HAZ-011) ---

_ADMIN_PATHS = frozenset({"/v1/retire", "/v1/reinstate", "/v1/catalog/refresh"})


# --- Shared helpers ---


def _format_error_response(message: str, status_code: int) -> JSONResponse:
    """Build a standardized error JSONResponse."""
    return JSONResponse({"error": message}, status_code=status_code)


def _check_admin_auth(request: Request) -> JSONResponse | None:
    """Verify admin bearer token for protected endpoints.

    HAZ-011 mitigation: Admin endpoints (retire, reinstate, catalog/refresh)
    require a valid Authorization header when admin_api_key is configured.
    Returns a 401 JSONResponse if auth fails, or None if auth passes.
    """
    engine: RouterEngine = request.app.state.engine
    admin_key = engine._config.admin_api_key

    # No admin key configured — open access (backward compatible)
    if not admin_key:
        return None

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return _format_error_response("Missing or invalid Authorization header", 401)

    provided_token = auth_header[7:]  # Strip "Bearer "
    if provided_token != admin_key:
        logger.warning("admin_auth_failed", path=request.url.path)
        return _format_error_response("Invalid admin API key", 401)

    return None


def _sanitize_prompt(text: str) -> str:
    """Sanitize operator prompt text before LLM dispatch.

    - Strips null bytes and control characters (preserves newlines, carriage
      returns, and tabs).
    - Truncates to ``_MAX_STRING_LENGTH`` characters.
    - Logs a warning when the input is modified.
    """
    sanitized = _CONTROL_CHAR_RE.sub("", text)

    if len(sanitized) > _MAX_STRING_LENGTH:
        sanitized = sanitized[:_MAX_STRING_LENGTH]

    if sanitized != text:
        logger.warning(
            "prompt_sanitized",
            original_length=len(text),
            sanitized_length=len(sanitized),
        )

    return sanitized


def _validate_llm_response(content: str) -> str:
    """Validate and clean LLM response content before returning to client.

    - Verifies the response is a non-empty string.
    - Strips null bytes.
    - Truncates excessively long responses (>500K chars) with a warning.
    """
    if not isinstance(content, str) or not content:
        logger.warning("llm_response_empty")
        return ""

    # Strip null bytes
    cleaned = content.replace("\x00", "")

    if len(cleaned) > _MAX_RESPONSE_LENGTH:
        logger.warning(
            "llm_response_truncated",
            original_length=len(cleaned),
            max_length=_MAX_RESPONSE_LENGTH,
        )
        cleaned = cleaned[:_MAX_RESPONSE_LENGTH]

    return cleaned


# --- Select endpoint ---


def _validate_select_request(body: dict) -> str | None:
    """Validate /v1/select request body. Returns error message or None."""
    role = body.get("role")
    if not role:
        return "missing required field: role"
    if not isinstance(role, str) or len(role) > _MAX_STRING_LENGTH:
        return "invalid role: must be string under 100K chars"
    top_n = body.get("top_n", 12)
    if not isinstance(top_n, int) or top_n < 1 or top_n > _SELECT_MAX_TOP_N:
        return f"invalid top_n: must be integer between 1 and {_SELECT_MAX_TOP_N}"
    return None


def _build_select_scores(
    models: list[str],
    engine: RouterEngine,
    tier_lookup: dict[str, BackendTier],
) -> list[dict]:
    """Compute scored entries for each selected model."""
    scores = []
    for model_id in models:
        health_result = engine._health.score(model_id)
        health_score = health_result.value if isinstance(health_result, Ok) else 100.0
        provider = engine._resolve_provider(model_id)
        budget_result = engine._budget.score(provider) if provider else Ok(100.0)
        budget_score = budget_result.value if isinstance(budget_result, Ok) else 100.0

        backend_tier = tier_lookup.get(model_id, BackendTier.SIMPLE)
        scores.append({
            "model_id": model_id,
            "health_score": round(health_score, 1),
            "budget_score": round(budget_score, 1),
            "complexity_tier": backend_tier.value,
            "trust_tier": _backend_tier_to_trust(backend_tier),
        })
    return scores


def _backend_tier_to_trust(tier: BackendTier) -> str:
    """Map BackendTier to trust tier string.

    Uses the same mapping as context_filter:
        LOCAL    → "local"
        COMPLEX  → "trusted"
        MODERATE → "semi_trusted"
        SIMPLE   → "semi_trusted"
    """
    if tier == BackendTier.LOCAL:
        return "local"
    if tier == BackendTier.COMPLEX:
        return "trusted"
    return "semi_trusted"


async def select_handler(request: Request) -> JSONResponse:
    """POST /v1/select — return ranked model IDs for a role."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _format_error_response("invalid JSON body", 400)

    validation_error = _validate_select_request(body)
    if validation_error:
        return _format_error_response(validation_error, 400)

    role = body["role"]
    top_n = body.get("top_n", 12)
    exclude_providers = body.get("exclude_providers")
    exclude = frozenset(exclude_providers) if exclude_providers else None

    models = engine.select_models(role, top_n=top_n, exclude_providers=exclude)

    # Build model_id → BackendTier lookup from registry
    tier_lookup: dict[str, BackendTier] = {}
    for _name, backend, _state in engine._registry.all_backends():
        cfg = backend.config
        tier_lookup[cfg.name] = cfg.tier
        tier_lookup[cfg.model] = cfg.tier

    scores = _build_select_scores(models, engine, tier_lookup)
    return JSONResponse({"models": models, "scores": scores})


async def record_handler(request: Request) -> JSONResponse:
    """POST /v1/record — record a request outcome."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _format_error_response("invalid JSON body", 400)

    provider = body.get("provider")
    model_id = body.get("model_id")
    success = body.get("success")

    if not provider or not model_id or success is None:
        return _format_error_response(
            "missing required fields: provider, model_id, success", 400,
        )

    tokens_used = body.get("tokens_used", 0)
    latency_ms = body.get("latency_ms", 0.0)

    engine.record_request(
        RequestOutcome(
            provider=provider,
            model_id=model_id,
            success=success,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
    )

    return JSONResponse({"status": "ok"})


async def health_handler(request: Request) -> JSONResponse:
    """GET /v1/health — full health and budget snapshot."""
    engine: RouterEngine = request.app.state.engine

    return JSONResponse({
        "budget": engine.budget_snapshot(),
        "health": engine.health_snapshot(),
    })


async def catalog_handler(request: Request) -> JSONResponse:
    """GET /v1/catalog — catalog status."""
    engine: RouterEngine = request.app.state.engine

    catalog_result = engine._catalog.get()
    stale = engine._catalog.is_stale()

    model_count = 0
    providers: list[str] = []
    if isinstance(catalog_result, Ok):
        catalog = catalog_result.value
        providers = list(catalog.keys())
        model_count = sum(len(entries) for entries in catalog.values())

    return JSONResponse({
        "stale": stale,
        "providers": providers,
        "model_count": model_count,
    })


async def _execute_catalog_refresh(engine: RouterEngine) -> JSONResponse:
    """Run the catalog refresh and return an appropriate JSONResponse."""
    refresher = _refresher_mod.CatalogRefresher()
    result = await refresher.refresh(engine._config.providers)
    if isinstance(result, Ok):
        catalog = result.value
        engine._catalog.set(catalog)
        model_count = sum(len(entries) for entries in catalog.values())
        return JSONResponse({
            "status": "ok",
            "providers_refreshed": list(catalog.keys()),
            "model_count": model_count,
        })
    return JSONResponse({
        "status": "error",
        "error": "Catalog refresh failed",
    }, status_code=500)


async def catalog_refresh_handler(request: Request) -> JSONResponse:
    """POST /v1/catalog/refresh — trigger a catalog refresh.

    HAZ-011: Requires admin auth when admin_api_key is configured.
    """
    auth_error = _check_admin_auth(request)
    if auth_error is not None:
        return auth_error

    engine: RouterEngine = request.app.state.engine

    try:
        return await _execute_catalog_refresh(engine)
    except (OSError, ValueError, LookupError) as exc:
        logger.error("catalog_refresh_failed", error=str(exc), exc_info=True)
        return JSONResponse({
            "status": "error",
            "error": "Catalog refresh failed",
        }, status_code=500)


# --- Dispatch endpoint ---


def _validate_dispatch_request(body: dict) -> str | None:
    """Validate /v1/dispatch request body. Returns error message or None."""
    for field in _DISPATCH_REQUIRED_FIELDS:
        if field not in body:
            return f"missing required field: {field}"

    # Type and length validation for string fields
    for field in ("intent_category", "specific_intent", "operator_message"):
        value = body[field]
        if not isinstance(value, str) or len(value) > _MAX_STRING_LENGTH:
            return f"invalid {field}: must be string under 100K chars"

    if "system_prompt" in body:
        sp = body["system_prompt"]
        if not isinstance(sp, str) or len(sp) > _MAX_STRING_LENGTH:
            return "invalid system_prompt: must be string under 100K chars"

    # context_tokens must be a non-negative integer
    ct = body["context_tokens"]
    if not isinstance(ct, int) or ct < 0:
        return "invalid context_tokens: must be a non-negative integer"

    return None


def _build_dispatch_order(body: dict) -> DispatchOrder:
    """Construct a DispatchOrder from a validated request body."""
    return DispatchOrder(
        intent_category=body["intent_category"],
        specific_intent=body["specific_intent"],
        operator_message=_sanitize_prompt(body["operator_message"]),
        system_prompt=_sanitize_prompt(body.get("system_prompt", "")),
        context_tokens=int(body["context_tokens"]),
        requires_tool_use=body.get("requires_tool_use", False),
        requires_long_context=body.get("requires_long_context", False),
        persona=body.get("persona"),
        request_id=body.get("request_id"),
        stream_id=body.get("stream_id"),
        context_trust_tier=body.get("context_trust_tier"),
    )


def _format_dispatch_response(engine_response: EngineResponse) -> JSONResponse:
    """Serialize an EngineResponse to a JSON HTTP response."""
    return JSONResponse({
        "content": engine_response.content,
        "backend_used": engine_response.backend_used,
        "backend_tier": engine_response.backend_tier.value,
        "tokens_in": engine_response.tokens_in,
        "tokens_out": engine_response.tokens_out,
        "estimated_cost_usd": engine_response.estimated_cost_usd,
        "latency_ms": engine_response.latency_ms,
        "was_fallback": engine_response.was_fallback,
        "fallback_chain": engine_response.fallback_chain,
    })


def _format_dispatch_failure(error: object) -> JSONResponse:
    """Serialize a dispatch failure (Err branch) to a JSON HTTP response."""
    if isinstance(error, DispatchFailure):
        return JSONResponse({
            "message": error.message,
            "attempted_backends": error.attempted_backends,
            "error_details": error.error_details,
        }, status_code=500)
    return JSONResponse({
        "message": "Dispatch failed",
        "attempted_backends": [],
        "error_details": {"error_type": type(error).__name__},
    }, status_code=500)


def _format_stream_chunk(chunk: StreamChunk) -> str:
    """Serialize a StreamChunk to an SSE data line.

    Each event is a JSON-encoded object on a ``data:`` line, followed by
    two newlines (SSE protocol). The event_type field is used as the SSE
    event name for client-side routing.
    """
    payload: dict = {"event": chunk.event_type}

    if chunk.event_type == "token":
        payload["content"] = chunk.content
    elif chunk.event_type == "metadata":
        payload.update({
            "backend_used": chunk.backend_used,
            "backend_tier": chunk.backend_tier,
            "tokens_in": chunk.tokens_in,
            "tokens_out": chunk.tokens_out,
            "estimated_cost_usd": chunk.estimated_cost_usd,
            "latency_ms": chunk.latency_ms,
            "was_fallback": chunk.was_fallback,
            "fallback_chain": chunk.fallback_chain or [],
        })
    elif chunk.event_type == "error":
        payload["error_message"] = chunk.error_message

    return f"data: {json.dumps(payload)}\n\n"


async def _stream_dispatch_generator(
    engine: RouterEngine,
    order: DispatchOrder,
) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted lines from dispatch_stream.

    Wraps the engine's streaming dispatch in SSE formatting and catches
    unexpected exceptions to emit a final error event.
    """
    try:
        async for chunk in engine.dispatch_stream(order):
            if chunk.event_type == "token":
                chunk = StreamChunk(
                    event_type="token",
                    content=_validate_llm_response(chunk.content),
                )
            yield _format_stream_chunk(chunk)
    except (RuntimeError, ConnectionError, ValueError, TypeError, OSError) as exc:
        logger.error("streaming_dispatch_failed", error=str(exc), exc_info=True)
        error_chunk = StreamChunk(
            event_type="error",
            error_message="Internal server error",
        )
        yield _format_stream_chunk(error_chunk)


async def dispatch_handler(request: Request) -> JSONResponse | StreamingResponse:
    """POST /v1/dispatch — execute the MBR->CBR->LBR cascade and return EngineResponse.

    When ``stream`` is true in the request body, returns an SSE stream
    (``text/event-stream``) of token chunks followed by a metadata event.
    Otherwise returns a standard JSON response.
    """
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _format_error_response("invalid JSON body", 400)

    validation_error = _validate_dispatch_request(body)
    if validation_error:
        return _format_error_response(validation_error, 400)

    order = _build_dispatch_order(body)

    # Streaming path — return SSE StreamingResponse
    if body.get("stream", False):
        return StreamingResponse(
            _stream_dispatch_generator(engine, order),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path — accumulate and return JSON
    try:
        dispatch_result = await engine.dispatch(order)
    except (RuntimeError, ConnectionError, ValueError, TypeError, OSError) as exc:
        logger.error("dispatch_failed", error=str(exc), exc_info=True)
        return _format_error_response("Internal server error", 500)

    if isinstance(dispatch_result, Ok):
        engine_response = dispatch_result.value
        engine_response = EngineResponse(
            content=_validate_llm_response(engine_response.content),
            backend_used=engine_response.backend_used,
            backend_tier=engine_response.backend_tier,
            tokens_in=engine_response.tokens_in,
            tokens_out=engine_response.tokens_out,
            estimated_cost_usd=engine_response.estimated_cost_usd,
            latency_ms=engine_response.latency_ms,
            was_fallback=engine_response.was_fallback,
            fallback_chain=engine_response.fallback_chain,
        )
        return _format_dispatch_response(engine_response)
    return _format_dispatch_failure(dispatch_result.error)


async def retire_handler(request: Request) -> JSONResponse:
    """POST /v1/retire — retire a backend from the active pool.

    Accepts a backend name and marks it as retired so the cascade
    skips it until explicitly reinstated.

    HAZ-011: Requires admin auth when admin_api_key is configured.
    """
    auth_error = _check_admin_auth(request)
    if auth_error is not None:
        return auth_error

    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _format_error_response("invalid JSON body", 400)

    backend_name = body.get("backend")
    if not backend_name:
        return _format_error_response("missing required field: backend", 400)

    found = engine._registry.retire(backend_name)
    if not found:
        return _format_error_response("backend not found", 404)

    return JSONResponse({"retired": True, "backend": backend_name})


async def reinstate_handler(request: Request) -> JSONResponse:
    """POST /v1/reinstate — reinstate a previously retired backend.

    Accepts a backend name and returns it to the active pool so
    the cascade considers it again.

    HAZ-011: Requires admin auth when admin_api_key is configured.
    """
    auth_error = _check_admin_auth(request)
    if auth_error is not None:
        return auth_error

    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _format_error_response("invalid JSON body", 400)

    backend_name = body.get("backend")
    if not backend_name:
        return _format_error_response("missing required field: backend", 400)

    found = engine._registry.reinstate(backend_name)
    if not found:
        return _format_error_response("backend not found", 404)

    return JSONResponse({"reinstated": True, "backend": backend_name})
