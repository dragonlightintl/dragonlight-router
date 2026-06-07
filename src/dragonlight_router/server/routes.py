"""HTTP route handlers for the router API.

All routes operate on a shared RouterEngine instance.
"""
from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import JSONResponse

from dragonlight_router.router import RouterEngine
from dragonlight_router.core.types import RequestOutcome, DispatchOrder, EngineResponse, DispatchFailure
from dragonlight_router.result import Ok, Err


async def select_handler(request: Request) -> JSONResponse:
    """POST /v1/select — return ranked model IDs for a role."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    role = body.get("role")
    if not role:
        return JSONResponse({"error": "missing required field: role"}, status_code=400)

    top_n = body.get("top_n", 12)
    exclude_providers = body.get("exclude_providers")
    exclude = frozenset(exclude_providers) if exclude_providers else None

    models = engine.select_models(role, top_n=top_n, exclude_providers=exclude)

    # Compute scores for response
    scores = []
    for model_id in models:
        health_result = engine._health.score(model_id)
        health_score = health_result.value if isinstance(health_result, Ok) else 100.0
        provider = engine._resolve_provider(model_id)
        budget_result = engine._budget.score(provider) if provider else Ok(100.0)
        budget_score = budget_result.value if isinstance(budget_result, Ok) else 100.0
        scores.append({
            "model_id": model_id,
            "health_score": round(health_score, 1),
            "budget_score": round(budget_score, 1),
        })

    return JSONResponse({"models": models, "scores": scores})


async def record_handler(request: Request) -> JSONResponse:
    """POST /v1/record — record a request outcome."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    provider = body.get("provider")
    model_id = body.get("model_id")
    success = body.get("success")

    if not provider or not model_id or success is None:
        return JSONResponse(
            {"error": "missing required fields: provider, model_id, success"},
            status_code=400,
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


async def catalog_refresh_handler(request: Request) -> JSONResponse:
    """POST /v1/catalog/refresh — trigger a catalog refresh."""
    engine: RouterEngine = request.app.state.engine

    # Import refresher
    from dragonlight_router.catalog.refresher import CatalogRefresher
    from dragonlight_router.result import Ok

    refresher = CatalogRefresher()
    try:
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
        else:
            # Refresh failed
            return JSONResponse({
                "status": "error",
                "error": result.error.message
            }, status_code=500)
    except (OSError, ValueError, LookupError) as exc:
        return JSONResponse({
            "status": "error",
            "error": str(exc)
        }, status_code=500)


async def dispatch_handler(request: Request) -> JSONResponse:
    """POST /v1/dispatch — execute the MBR→CBR→LBR cascade and return EngineResponse."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    # Validate required fields for DispatchOrder
    required_fields = ["intent_category", "specific_intent", "operator_message", "context_tokens"]
    for field in required_fields:
        if field not in body:
            return JSONResponse(
                {"error": f"missing required field: {field}"},
                status_code=400,
            )

    # Construct DispatchOrder from body
    order = DispatchOrder(
        intent_category=body["intent_category"],
        specific_intent=body["specific_intent"],
        operator_message=body["operator_message"],
        system_prompt=body.get("system_prompt", ""),
        context_tokens=int(body["context_tokens"]),
        requires_tool_use=body.get("requires_tool_use", False),
        requires_long_context=body.get("requires_long_context", False),
        persona=body.get("persona"),
        request_id=body.get("request_id"),
        stream_id=body.get("stream_id"),
        context_trust_tier=body.get("context_trust_tier"),
    )

    # Execute dispatch via RouterEngine.dispatch (implemented in TM-010)
    try:
        dispatch_result = engine.dispatch(order)
        if isinstance(dispatch_result, Ok):
            engine_response = dispatch_result.value
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
        else:
            # Dispatch failed - return DispatchFailure as JSON
            error = dispatch_result.error
            if isinstance(error, DispatchFailure):
                return JSONResponse({
                    "message": error.message,
                    "attempted_backends": error.attempted_backends,
                    "error_details": error.error_details,
                }, status_code=500)
            else:
                # Other exception
                return JSONResponse({
                    "message": str(error),
                    "attempted_backends": [],
                    "error_details": {"error_type": type(error).__name__}
                }, status_code=500)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)