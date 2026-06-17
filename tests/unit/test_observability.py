"""Tests for observability features: readiness probe, metrics endpoint,
OpenAPI schema, and correlation ID + metrics middleware integration.

Covers: /v1/ready, /metrics, /openapi.json, RequestCorrelationMiddleware
with MetricsCollector integration.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.server.app import create_app
from dragonlight_router.server.metrics import MetricsCollector
from dragonlight_router.server.middleware import RequestCorrelationMiddleware
from dragonlight_router.server.routes import _build_openapi_schema


def _setup_test_env(tmp_path: Path) -> Path:
    """Create a full test config + state for server testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [
            {
                "name": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "model_prefix": "groq_",
                "rate_limits": {"rpm": 30, "rpd": 14400},
            },
        ],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    # Role matrix
    matrix = {"coding": {"groq_llama70b": 90}}
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    # Catalog cache
    from dragonlight_router.catalog.cache import CatalogCache
    from dragonlight_router.core.types import CatalogEntry

    catalog = {
        "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    return config_path


# ---------------------------------------------------------------------------
# Readiness probe: /v1/ready
# ---------------------------------------------------------------------------


class TestReadyEndpoint:
    def test_ready_returns_200_when_catalog_populated(self, tmp_path: Path):
        """GET /v1/ready returns 200 when engine exists and catalog has data."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    def test_ready_returns_503_when_no_catalog_file(self, tmp_path: Path):
        """GET /v1/ready returns 503 when catalog file does not exist."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Remove the catalog file to simulate never-refreshed state
        catalog_path = engine._catalog._path
        if catalog_path.exists():
            catalog_path.unlink()

        client = TestClient(app)
        response = client.get("/v1/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "not been refreshed" in data["reason"]

    def test_ready_returns_503_when_engine_missing(self, tmp_path: Path):
        """GET /v1/ready returns 503 when engine is not on app.state."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)

        # Remove engine from state
        del app.state.engine

        client = TestClient(app)
        response = client.get("/v1/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "not initialized" in data["reason"]

    def test_ready_returns_200_even_with_stale_catalog(self, tmp_path: Path):
        """GET /v1/ready returns 200 if catalog file exists but is stale.

        The readiness probe checks that catalog has been populated at least
        once, not whether it is within TTL. Staleness is a liveness concern.
        """
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        engine = app.state.engine

        # Make catalog stale by writing with old timestamp
        import time
        catalog_path = engine._catalog._path
        data = json.loads(catalog_path.read_text())
        data["timestamp"] = time.time() - 200_000  # Very old
        catalog_path.write_text(json.dumps(data))

        client = TestClient(app)
        response = client.get("/v1/ready")
        # Even though catalog.get() returns Err (stale), the file exists
        # so readiness passes
        assert response.status_code == 200
        assert response.json()["ready"] is True


# ---------------------------------------------------------------------------
# Metrics endpoint: /metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, tmp_path: Path):
        """GET /metrics returns 200 with expected structure."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "memory_mb" in data
        assert "endpoints" in data
        assert "router" in data

    def test_metrics_records_prior_requests(self, tmp_path: Path):
        """GET /metrics reflects requests made before the /metrics call."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)

        # Make a few requests first
        client.get("/v1/health")
        client.get("/v1/health")
        client.get("/v1/catalog")

        response = client.get("/metrics")
        data = response.json()
        endpoints = data["endpoints"]

        # Health endpoint should have 2 requests (the /metrics request
        # itself may or may not be counted depending on ordering — it's
        # tracked by middleware which runs after the response)
        assert "GET /v1/health" in endpoints
        assert endpoints["GET /v1/health"]["request_count"] >= 2

    def test_metrics_includes_router_counters(self, tmp_path: Path):
        """GET /metrics includes router-level dispatch stats."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/metrics")
        data = response.json()
        router = data["router"]
        assert "total_dispatches" in router
        assert "fallback_count" in router
        assert "circuit_breaker_trips" in router


# ---------------------------------------------------------------------------
# OpenAPI schema: /openapi.json
# ---------------------------------------------------------------------------


class TestOpenApiEndpoint:
    def test_openapi_returns_200(self, tmp_path: Path):
        """GET /openapi.json returns 200."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_openapi_valid_structure(self, tmp_path: Path):
        """GET /openapi.json returns valid OpenAPI 3.0.3 structure."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/openapi.json")
        data = response.json()
        assert data["openapi"] == "3.0.3"
        assert "info" in data
        assert data["info"]["title"] == "Dragonlight Router API"
        assert "paths" in data

    def test_openapi_contains_all_endpoints(self, tmp_path: Path):
        """OpenAPI schema describes all router endpoints."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/openapi.json")
        paths = response.json()["paths"]
        expected_paths = [
            "/v1/select",
            "/v1/dispatch",
            "/v1/record",
            "/v1/health",
            "/v1/ready",
            "/v1/catalog",
            "/v1/catalog/refresh",
            "/v1/retire",
            "/v1/reinstate",
            "/metrics",
            "/openapi.json",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing path: {path}"

    def test_openapi_idempotent(self, tmp_path: Path):
        """Multiple calls to /openapi.json return the same schema."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        r1 = client.get("/openapi.json").json()
        r2 = client.get("/openapi.json").json()
        assert r1 == r2


class TestBuildOpenApiSchema:
    def test_schema_has_security_schemes(self):
        """Schema includes bearer auth security scheme."""
        schema = _build_openapi_schema()
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
        assert "bearerAuth" in schema["components"]["securitySchemes"]

    def test_schema_dispatch_has_request_body(self):
        """Dispatch endpoint has required request body with all fields."""
        schema = _build_openapi_schema()
        dispatch = schema["paths"]["/v1/dispatch"]["post"]
        assert dispatch["requestBody"]["required"] is True
        props = dispatch["requestBody"]["content"]["application/json"]["schema"]["properties"]
        assert "intent_category" in props
        assert "operator_message" in props
        assert "stream" in props

    def test_schema_ready_has_503_response(self):
        """Ready endpoint documents 503 response."""
        schema = _build_openapi_schema()
        ready = schema["paths"]["/v1/ready"]["get"]
        assert "503" in ready["responses"]

    def test_schema_health_has_status_enum(self):
        """Health endpoint documents status enum values."""
        schema = _build_openapi_schema()
        health = schema["paths"]["/v1/health"]["get"]
        resp_schema = health["responses"]["200"]["content"]["application/json"]["schema"]
        assert "healthy" in resp_schema["properties"]["status"]["enum"]


# ---------------------------------------------------------------------------
# Correlation ID middleware — test with metrics integration
# ---------------------------------------------------------------------------


class TestCorrelationIdMiddleware:
    def test_generates_request_id(self, tmp_path: Path):
        """Middleware generates X-Request-ID when client does not provide one."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/health")
        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None
        # Should be a valid UUID4
        parsed = uuid.UUID(request_id, version=4)
        assert str(parsed) == request_id

    def test_preserves_client_request_id(self, tmp_path: Path):
        """Middleware uses client-provided X-Request-ID."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        custom_id = "my-custom-correlation-id-12345"
        response = client.get("/v1/health", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id

    def test_request_id_present_on_error_response(self, tmp_path: Path):
        """X-Request-ID is present even on error responses."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.post(
            "/v1/select",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "X-Request-ID" in response.headers


class TestMiddlewareMetricsIntegration:
    def test_middleware_records_to_metrics_collector(self, tmp_path: Path):
        """Middleware passes request stats to MetricsCollector."""
        config_path = _setup_test_env(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)

        # Make requests
        client.get("/v1/health")
        client.get("/v1/catalog")

        # Check metrics were recorded
        metrics: MetricsCollector = app.state.metrics
        snap = metrics.snapshot()
        assert "GET /v1/health" in snap["endpoints"]
        assert snap["endpoints"]["GET /v1/health"]["request_count"] >= 1

    def test_correlation_middleware_without_metrics(self):
        """Middleware works correctly when no MetricsCollector is provided."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse

        async def _dummy(request):
            return JSONResponse({"ok": True})

        inner = Starlette(routes=[Route("/ping", _dummy, methods=["GET"])])
        # Create middleware without metrics (default None)
        inner.add_middleware(RequestCorrelationMiddleware)
        client = TestClient(inner)
        response = client.get("/ping")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
