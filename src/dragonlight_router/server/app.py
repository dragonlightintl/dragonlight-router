"""Starlette application — HTTP server for the router API.

Provides factory function create_app() for testing and a main()
entrypoint for the CLI script.
"""
from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Route

from dragonlight_router.router import RouterEngine
from dragonlight_router.server.routes import (
    catalog_handler,
    catalog_refresh_handler,
    health_handler,
    record_handler,
    select_handler,
)


def create_app(config_path: Path | None = None, **overrides) -> Starlette:
    """Create and configure the Starlette application.

    Accepts a config_path for testing; uses default resolution otherwise.
    """
    engine = RouterEngine(config_path=config_path, **overrides)

    routes = [
        Route("/v1/select", select_handler, methods=["POST"]),
        Route("/v1/record", record_handler, methods=["POST"]),
        Route("/v1/health", health_handler, methods=["GET"]),
        Route("/v1/catalog", catalog_handler, methods=["GET"]),
        Route("/v1/catalog/refresh", catalog_refresh_handler, methods=["POST"]),
    ]

    app = Starlette(routes=routes)
    app.state.engine = engine
    return app


def main() -> None:
    """CLI entrypoint — run the server with uvicorn."""
    import os
    import uvicorn

    app = create_app()
    host = os.environ.get("DRAGONLIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("DRAGONLIGHT_PORT", "8100"))
    uvicorn.run(app, host=host, port=port)
