"""Starlette application — HTTP server for the router API.

Provides factory function create_app() for testing and a main()
entrypoint for the CLI script.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

from dragonlight_router.router import RouterEngine
from dragonlight_router.server.middleware import RateLimitMiddleware
from dragonlight_router.server.routes import (
    catalog_handler,
    catalog_refresh_handler,
    dispatch_handler,
    health_handler,
    record_handler,
    reinstate_handler,
    retire_handler,
    select_handler,
)


def create_app(config_path: Path | None = None, **overrides: Any) -> Starlette:
    """Create and configure the Starlette application.

    Accepts a config_path for testing; uses default resolution otherwise.
    """
    engine = RouterEngine(config_path=config_path, **overrides)

    import structlog as _structlog
    _lifespan_logger = _structlog.get_logger()

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        # Bootstrap: refresh catalog at startup (concurrent, one fetch per provider).
        try:
            await engine._async_refresh_catalog()
        except Exception:  # noqa: BLE001
            _lifespan_logger.warning("startup_catalog_refresh_failed")

        # Start health check loop
        task = asyncio.create_task(engine.start_health_check_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    routes = [
        Route("/v1/select", select_handler, methods=["POST"]),
        Route("/v1/dispatch", dispatch_handler, methods=["POST"]),
        Route("/v1/record", record_handler, methods=["POST"]),
        Route("/v1/health", health_handler, methods=["GET"]),
        Route("/v1/catalog", catalog_handler, methods=["GET"]),
        Route("/v1/catalog/refresh", catalog_refresh_handler, methods=["POST"]),
        Route("/v1/retire", retire_handler, methods=["POST"]),
        Route("/v1/reinstate", reinstate_handler, methods=["POST"]),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.engine = engine
    app.add_middleware(RateLimitMiddleware)
    return app


def main() -> None:
    """CLI entrypoint — run the server with uvicorn."""
    config_env = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
    config_path = Path(config_env) if config_env else None
    app = create_app(config_path=config_path)
    host = os.environ.get("DRAGONLIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("DRAGONLIGHT_PORT", "8100"))
    uvicorn.run(app, host=host, port=port)
