"""HTTP route handlers for the router API.

All routes operate on a shared RouterEngine instance.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from dragonlight_router import __version__
from dragonlight_router.catalog import refresher as _refresher_mod
from dragonlight_router.core.types import (
    BackendTier,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    RequestOutcome,
    StreamChunk,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.router import RouterEngine
from dragonlight_router.server.metrics import MetricsCollector

logger = structlog.get_logger()

# --- Input validation constants ---

_MAX_STRING_LENGTH = 100_000
_MAX_RESPONSE_LENGTH = 500_000
_SELECT_MAX_TOP_N = 500
_DISPATCH_REQUIRED_FIELDS = ("intent_category", "specific_intent", "operator_message", "context_tokens")

# HAZ-007: Allowed intent_category values — rejects unknown values to prevent
# adversarial intent injection affecting routing decisions.
_ALLOWED_INTENT_CATEGORIES: frozenset[str] = frozenset({
    "code_generation", "code_review", "debugging", "architecture",
    "engineering_build", "spec_writing", "documentation",
    "session_lifecycle", "strategic_planning", "complex_reasoning",
    "casual_chat", "creative_writing", "data_analysis",
    "summarization", "translation", "search", "general",
    "test",  # For test/development usage
})

# HAZ-004: Allowed fallback_policy values.
_ALLOWED_FALLBACK_POLICIES: frozenset[str] = frozenset({
    "allow", "deny", "same_tier",
})

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


def _validate_select_request(body: dict[str, Any]) -> str | None:
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
) -> list[dict[str, object]]:
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
    """GET /v1/health — full health and budget snapshot.

    HAZ-003 mitigation: Includes router-level availability status so
    callers can detect degraded/unavailable state before dispatching.
    """
    engine: RouterEngine = request.app.state.engine

    return JSONResponse({
        "status": engine._health.availability_status(),
        "version": __version__,
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


def _validate_dispatch_request(body: dict[str, Any]) -> str | None:
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

    # HAZ-007: Validate intent_category against allowed set
    intent = body["intent_category"]
    if intent not in _ALLOWED_INTENT_CATEGORIES:
        return f"invalid intent_category: '{intent}' not in allowed set"

    # HAZ-004: Validate fallback_policy if provided
    fp = body.get("fallback_policy", "allow")
    if fp not in _ALLOWED_FALLBACK_POLICIES:
        return f"invalid fallback_policy: must be one of {sorted(_ALLOWED_FALLBACK_POLICIES)}"

    return None


def _build_dispatch_order(body: dict[str, Any]) -> DispatchOrder:
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
        fallback_policy=body.get("fallback_policy", "allow"),
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
    payload: dict[str, object] = {"event": chunk.event_type}

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


# --- Readiness probe ---


async def ready_handler(request: Request) -> JSONResponse:
    """GET /v1/ready — readiness probe.

    Returns 200 only when:
    - The catalog has been refreshed at least once (CatalogCache has data).
    - The RouterEngine is initialized (present on app.state).

    Returns 503 with a reason when not ready. Separate from /v1/health
    which is a liveness probe that always returns 200.
    """
    engine: RouterEngine | None = getattr(request.app.state, "engine", None)

    if engine is None:
        return JSONResponse(
            {"ready": False, "reason": "RouterEngine not initialized"},
            status_code=503,
        )

    # Check catalog has been populated at least once
    catalog_result = engine._catalog.get()
    if isinstance(catalog_result, Err):
        # Catalog might be stale but still readable — check if the file exists
        # and has data at all (separate from TTL staleness).
        if not engine._catalog._path.exists():
            return JSONResponse(
                {"ready": False, "reason": "Catalog has not been refreshed yet"},
                status_code=503,
            )

    return JSONResponse({"ready": True})


# --- Metrics endpoint ---


async def metrics_handler(request: Request) -> JSONResponse:
    """GET /metrics — return JSON metrics summary.

    Returns per-endpoint request counts, error counts, latency percentiles,
    router-level dispatch stats, uptime, and memory usage.
    """
    metrics: MetricsCollector = request.app.state.metrics
    return JSONResponse(metrics.snapshot())


# --- OpenAPI schema endpoint ---


def _build_openapi_schema() -> dict[str, Any]:
    """Build the OpenAPI 3.0.3 schema for the Dragonlight Router API.

    Returns a static dict describing all endpoints, request/response bodies,
    status codes, and headers.
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Dragonlight Router API",
            "description": "Multi-provider intelligent LLM routing — model selection + cascade dispatch.",
            "version": "0.2.6",
        },
        "paths": {
            "/v1/select": {
                "post": {
                    "summary": "Select ranked models for a role",
                    "operationId": "selectModels",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["role"],
                                    "properties": {
                                        "role": {"type": "string", "description": "Role to select models for."},
                                        "top_n": {"type": "integer", "default": 12, "minimum": 1, "maximum": 500},
                                        "exclude_providers": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Provider names to exclude.",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Ranked model list with scores.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "models": {"type": "array", "items": {"type": "string"}},
                                            "scores": {"type": "array", "items": {"type": "object"}},
                                        },
                                    },
                                },
                            },
                        },
                        "400": {"description": "Validation error."},
                    },
                },
            },
            "/v1/dispatch": {
                "post": {
                    "summary": "Dispatch a request through the MBR-CBR-LBR cascade",
                    "operationId": "dispatchRequest",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["intent_category", "specific_intent", "operator_message", "context_tokens"],
                                    "properties": {
                                        "intent_category": {"type": "string"},
                                        "specific_intent": {"type": "string"},
                                        "operator_message": {"type": "string"},
                                        "context_tokens": {"type": "integer", "minimum": 0},
                                        "system_prompt": {"type": "string"},
                                        "requires_tool_use": {"type": "boolean", "default": False},
                                        "requires_long_context": {"type": "boolean", "default": False},
                                        "persona": {"type": "string"},
                                        "stream": {"type": "boolean", "default": False},
                                        "fallback_policy": {
                                            "type": "string",
                                            "enum": ["allow", "deny", "same_tier"],
                                            "default": "allow",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Dispatch result (JSON or SSE stream).",
                            "headers": {
                                "X-Request-ID": {"schema": {"type": "string"}, "description": "Request correlation ID."},
                            },
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "backend_used": {"type": "string"},
                                            "backend_tier": {"type": "string"},
                                            "tokens_in": {"type": "integer"},
                                            "tokens_out": {"type": "integer"},
                                            "estimated_cost_usd": {"type": "number"},
                                            "latency_ms": {"type": "number"},
                                            "was_fallback": {"type": "boolean"},
                                            "fallback_chain": {"type": "array", "items": {"type": "string"}},
                                        },
                                    },
                                },
                            },
                        },
                        "400": {"description": "Validation error."},
                        "500": {"description": "Dispatch failure."},
                    },
                },
            },
            "/v1/record": {
                "post": {
                    "summary": "Record a request outcome",
                    "operationId": "recordOutcome",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["provider", "model_id", "success"],
                                    "properties": {
                                        "provider": {"type": "string"},
                                        "model_id": {"type": "string"},
                                        "success": {"type": "boolean"},
                                        "tokens_used": {"type": "integer", "default": 0},
                                        "latency_ms": {"type": "number", "default": 0.0},
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {"description": "Outcome recorded."},
                        "400": {"description": "Validation error."},
                    },
                },
            },
            "/v1/health": {
                "get": {
                    "summary": "Liveness probe with health and budget snapshot",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Always returns 200 with status, budget, and health data.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["healthy", "degraded", "unavailable"]},
                                            "budget": {"type": "object"},
                                            "health": {"type": "object"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/ready": {
                "get": {
                    "summary": "Readiness probe",
                    "operationId": "readinessCheck",
                    "responses": {
                        "200": {
                            "description": "Router is ready to serve traffic.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ready": {"type": "boolean", "enum": [True]},
                                        },
                                    },
                                },
                            },
                        },
                        "503": {
                            "description": "Router is not yet ready.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ready": {"type": "boolean", "enum": [False]},
                                            "reason": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/catalog": {
                "get": {
                    "summary": "Catalog status",
                    "operationId": "catalogStatus",
                    "responses": {
                        "200": {
                            "description": "Catalog status with provider list and model count.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stale": {"type": "boolean"},
                                            "providers": {"type": "array", "items": {"type": "string"}},
                                            "model_count": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/catalog/refresh": {
                "post": {
                    "summary": "Trigger catalog refresh",
                    "operationId": "catalogRefresh",
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {"description": "Catalog refreshed successfully."},
                        "401": {"description": "Unauthorized."},
                        "500": {"description": "Refresh failed."},
                    },
                },
            },
            "/v1/retire": {
                "post": {
                    "summary": "Retire a backend from the active pool",
                    "operationId": "retireBackend",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["backend"],
                                    "properties": {
                                        "backend": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {"description": "Backend retired."},
                        "400": {"description": "Missing backend field."},
                        "401": {"description": "Unauthorized."},
                        "404": {"description": "Backend not found."},
                    },
                },
            },
            "/v1/reinstate": {
                "post": {
                    "summary": "Reinstate a retired backend",
                    "operationId": "reinstateBackend",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["backend"],
                                    "properties": {
                                        "backend": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {"description": "Backend reinstated."},
                        "400": {"description": "Missing backend field."},
                        "401": {"description": "Unauthorized."},
                        "404": {"description": "Backend not found."},
                    },
                },
            },
            "/metrics": {
                "get": {
                    "summary": "Operational metrics",
                    "operationId": "getMetrics",
                    "responses": {
                        "200": {
                            "description": "JSON metrics summary.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "uptime_seconds": {"type": "number"},
                                            "memory_mb": {"type": "number"},
                                            "endpoints": {"type": "object"},
                                            "router": {
                                                "type": "object",
                                                "properties": {
                                                    "total_dispatches": {"type": "integer"},
                                                    "fallback_count": {"type": "integer"},
                                                    "circuit_breaker_trips": {"type": "integer"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/openapi.json": {
                "get": {
                    "summary": "OpenAPI schema",
                    "operationId": "getOpenApiSchema",
                    "responses": {
                        "200": {
                            "description": "OpenAPI 3.0.3 schema document.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        },
                    },
                },
            },
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                },
            },
        },
    }


# Cache the schema dict at module level (built once, served many times).
_OPENAPI_SCHEMA: dict[str, Any] | None = None


async def openapi_handler(request: Request) -> JSONResponse:
    """GET /openapi.json — serve the OpenAPI 3.0.3 schema."""
    global _OPENAPI_SCHEMA  # noqa: PLW0603
    if _OPENAPI_SCHEMA is None:
        _OPENAPI_SCHEMA = _build_openapi_schema()
    return JSONResponse(_OPENAPI_SCHEMA)
