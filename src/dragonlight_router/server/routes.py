"""HTTP route handlers for the router API.

All routes operate on a shared RouterEngine instance.
"""
from __future__ import annotations

from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from dragonlight_router.router import RouterEngine


async def select_handler(request: Request) -> JSONResponse:
    """POST /v1/select — return ranked model IDs for a role."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except Exception:
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
        health_score = engine._health.score(model_id)
        provider = engine._resolve_provider(model_id)
        budget_score = engine._budget.score(provider) if provider else 100.0
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
    except Exception:
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
        provider,
        model_id,
        success=success,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
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

    catalog = engine._catalog.get()
    stale = engine._catalog.is_stale()

    model_count = 0
    providers: list[str] = []
    if catalog:
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

    refresher = CatalogRefresher()
    try:
        catalog = await refresher.refresh(engine._config.providers)
        engine._catalog.set(catalog)
        model_count = sum(len(entries) for entries in catalog.values())
        return JSONResponse({
            "status": "ok",
            "providers_refreshed": list(catalog.keys()),
            "model_count": model_count,
        })
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": str(exc)},
            status_code=500,
        )
