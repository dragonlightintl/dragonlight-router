"""Starlette application — HTTP server for the router API.

Provides factory function create_app() for testing and a main()
entrypoint for the CLI script.

Graceful shutdown: uvicorn handles SIGTERM/SIGINT natively and triggers
the Starlette lifespan exit. The lifespan cancels the health check loop,
persists budget+health state, and closes the shared httpx session.
``timeout_graceful_shutdown`` gives in-flight requests time to drain.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

from dragonlight_router.router import RouterEngine
from dragonlight_router.server.logging import configure_logging
from dragonlight_router.server.metrics import MetricsCollector
from dragonlight_router.server.middleware import (
    CORSMiddleware,
    RateLimitMiddleware,
    RequestCorrelationMiddleware,
    get_cors_config,
)
from dragonlight_router.server.routes import (
    catalog_handler,
    catalog_refresh_handler,
    dispatch_handler,
    health_handler,
    metrics_handler,
    openapi_handler,
    ready_handler,
    record_handler,
    reinstate_handler,
    retire_handler,
    select_handler,
)

# Default graceful shutdown timeout: 10 seconds for in-flight request draining.
_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = 10


def create_app(config_path: Path | None = None, **overrides: Any) -> Starlette:
    """Create and configure the Starlette application.

    Accepts a config_path for testing; uses default resolution otherwise.
    HAZ-006: Configures structlog with secret-scrubbing before any logging.
    """
    configure_logging()
    engine = RouterEngine(config_path=config_path, **overrides)

    _lifespan_logger = structlog.get_logger()

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        _lifespan_logger.info("server_starting")

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
            _lifespan_logger.info("server_shutting_down")

            # Cancel the health check background task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            # HAZ-012: Persist budget + health state on shutdown
            try:
                engine.save_state()
            except (OSError, ValueError, TypeError) as exc:
                _lifespan_logger.warning("shutdown_state_save_failed", error=str(exc))

            _lifespan_logger.info("server_shutdown_complete")

    routes = [
        Route("/v1/select", select_handler, methods=["POST"]),
        Route("/v1/dispatch", dispatch_handler, methods=["POST"]),
        Route("/v1/record", record_handler, methods=["POST"]),
        Route("/v1/health", health_handler, methods=["GET"]),
        Route("/v1/ready", ready_handler, methods=["GET"]),
        Route("/v1/catalog", catalog_handler, methods=["GET"]),
        Route("/v1/catalog/refresh", catalog_refresh_handler, methods=["POST"]),
        Route("/v1/retire", retire_handler, methods=["POST"]),
        Route("/v1/reinstate", reinstate_handler, methods=["POST"]),
        Route("/metrics", metrics_handler, methods=["GET"]),
        Route("/openapi.json", openapi_handler, methods=["GET"]),
    ]

    metrics = MetricsCollector()

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.engine = engine
    app.state.metrics = metrics
    # Middleware is applied in reverse order (last added = outermost).
    # Order: CORS (outermost) → Correlation → RateLimit (innermost)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestCorrelationMiddleware, metrics=metrics)
    app.add_middleware(CORSMiddleware, **get_cors_config())
    return app


def main() -> None:
    """CLI entrypoint — run the server with uvicorn.

    Configures graceful shutdown so in-flight requests have time to drain
    before the process exits. SIGTERM/SIGINT are handled by uvicorn which
    triggers the Starlette lifespan exit path.
    """
    config_env = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
    config_path = Path(config_env) if config_env else None
    app = create_app(config_path=config_path)
    host = os.environ.get("DRAGONLIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("DRAGONLIGHT_PORT", "8100"))
    graceful_timeout = int(
        os.environ.get(
            "DRAGONLIGHT_GRACEFUL_SHUTDOWN_TIMEOUT",
            str(_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT),
        )
    )

    # Ignore SIGTERM in the parent — let uvicorn's signal handling drive
    # the shutdown sequence. This prevents duplicate signal handling when
    # running under container orchestrators (Docker, Kubernetes) that send
    # SIGTERM followed by SIGKILL.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=graceful_timeout,
    )
