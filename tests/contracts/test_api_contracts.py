"""Contract tests for the HTTP API.

Verifies that every endpoint returns the documented response shape,
handles invalid input correctly, and maintains consistent error formats.

Spec traceability: TM-009 (HTTP API endpoints), OpenAPI 3.0.3 schema
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.core.types import (
    BackendTier,
    CatalogEntry,
    DispatchFailure,
    EngineResponse,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.server.app import create_app

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Shared test environment setup
# ---------------------------------------------------------------------------


def _setup_test_env(tmp_path: Path) -> Path:
    """Create a minimal router config + state directory for testing."""
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

    catalog = {
        "groq": [CatalogEntry(model_id="groq_llama70b", provider="groq")],
    }
    cache = CatalogCache(cache_path=state_dir / "provider_catalog.json", ttl_hours=24)
    cache.set(catalog)

    return config_path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create a TestClient backed by a real RouterEngine."""
    config_path = _setup_test_env(tmp_path)
    app = create_app(config_path=config_path)
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /v1/select
# ---------------------------------------------------------------------------


class TestSelectContract:
    """POST /v1/select must return {models: list[str], scores: list[dict]}."""

    def test_select_returns_models_and_scores(self, client: TestClient) -> None:
        """Response must have 'models' (list of strings) and 'scores' (list of dicts)."""
        resp = client.post("/v1/select", json={"role": "coding", "top_n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data, "Response must contain 'models'"
        assert "scores" in data, "Response must contain 'scores'"
        assert isinstance(data["models"], list), "'models' must be a list"
        assert isinstance(data["scores"], list), "'scores' must be a list"
        for model in data["models"]:
            assert isinstance(model, str), f"Each model must be a string, got {type(model)}"
        for score in data["scores"]:
            assert isinstance(score, dict), f"Each score must be a dict, got {type(score)}"

    def test_select_score_shape(self, client: TestClient) -> None:
        """Each score entry must have model_id, health_score, budget_score."""
        resp = client.post("/v1/select", json={"role": "coding"})
        assert resp.status_code == 200
        scores = resp.json()["scores"]
        for score in scores:
            assert "model_id" in score, "Score must have 'model_id'"
            assert "health_score" in score, "Score must have 'health_score'"
            assert "budget_score" in score, "Score must have 'budget_score'"

    def test_select_rejects_missing_role(self, client: TestClient) -> None:
        """Missing 'role' must return 400 with error."""
        resp = client.post("/v1/select", json={})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data, "Error response must have 'error' field"

    def test_select_rejects_invalid_json(self, client: TestClient) -> None:
        """Non-JSON body must return 400."""
        resp = client.post(
            "/v1/select",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data


# ---------------------------------------------------------------------------
# POST /v1/dispatch
# ---------------------------------------------------------------------------


class TestDispatchContract:
    """POST /v1/dispatch response shape contracts."""

    def _valid_dispatch_body(self) -> dict:
        return {
            "intent_category": "code_generation",
            "specific_intent": "write_function",
            "operator_message": "Write hello world",
            "system_prompt": "You are helpful",
            "context_tokens": 100,
        }

    def test_dispatch_rejects_missing_required_fields(self, client: TestClient) -> None:
        """Missing required fields must return 400 with error."""
        resp = client.post("/v1/dispatch", json={"context_tokens": 100})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data, "Validation error must contain 'error'"

    def test_dispatch_rejects_invalid_json(self, client: TestClient) -> None:
        """Non-JSON body must return 400."""
        resp = client.post(
            "/v1/dispatch",
            content=b"{{bad json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_dispatch_rejects_negative_context_tokens(self, client: TestClient) -> None:
        """Negative context_tokens must return 400."""
        body = self._valid_dispatch_body()
        body["context_tokens"] = -1
        resp = client.post("/v1/dispatch", json=body)
        assert resp.status_code == 400

    def test_dispatch_rejects_invalid_intent_category(self, client: TestClient) -> None:
        """Unknown intent_category must return 400."""
        body = self._valid_dispatch_body()
        body["intent_category"] = "definitely_not_a_valid_category"
        resp = client.post("/v1/dispatch", json=body)
        assert resp.status_code == 400

    def test_dispatch_rejects_invalid_fallback_policy(self, client: TestClient) -> None:
        """Unknown fallback_policy must return 400."""
        body = self._valid_dispatch_body()
        body["fallback_policy"] = "invalid_policy"
        resp = client.post("/v1/dispatch", json=body)
        assert resp.status_code == 400

    def test_dispatch_success_response_shape(self, client: TestClient) -> None:
        """Successful dispatch must return the documented response fields."""
        mock_response = EngineResponse(
            content="Hello world",
            backend_used="groq/llama-70b",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            latency_ms=100.0,
            was_fallback=False,
            fallback_chain=[],
        )
        body = self._valid_dispatch_body()
        with patch.object(
            client.app.state.engine,
            "dispatch",
            new_callable=AsyncMock,
            return_value=Ok(mock_response),
        ):
            resp = client.post("/v1/dispatch", json=body)
        assert resp.status_code == 200
        data = resp.json()
        required_fields = {
            "content",
            "backend_used",
            "backend_tier",
            "tokens_in",
            "tokens_out",
            "estimated_cost_usd",
            "latency_ms",
            "was_fallback",
            "fallback_chain",
            "dispatch_mode",
        }
        for field in required_fields:
            assert field in data, f"Dispatch response missing '{field}'"

    def test_dispatch_failure_response_shape(self, client: TestClient) -> None:
        """Dispatch failure must return {message, attempted_backends, error_details}."""
        failure = DispatchFailure(
            message="All backends exhausted",
            attempted_backends=["groq/llama-70b"],
            error_details={"groq/llama-70b": "timeout"},
        )
        body = self._valid_dispatch_body()
        with patch.object(
            client.app.state.engine,
            "dispatch",
            new_callable=AsyncMock,
            return_value=Err(failure),
        ):
            resp = client.post("/v1/dispatch", json=body)
        assert resp.status_code == 500
        data = resp.json()
        assert "message" in data, "Failure response must contain 'message'"
        assert "attempted_backends" in data, "Failure response must contain 'attempted_backends'"
        assert "error_details" in data, "Failure response must contain 'error_details'"


# ---------------------------------------------------------------------------
# POST /v1/record
# ---------------------------------------------------------------------------


class TestRecordContract:
    """POST /v1/record must return {status: "ok"} on valid input."""

    def test_record_returns_status_ok(self, client: TestClient) -> None:
        """Valid record request returns {status: "ok"}."""
        body = {
            "provider": "groq",
            "model_id": "groq_llama70b",
            "success": True,
            "tokens_used": 100,
            "latency_ms": 50.0,
        }
        resp = client.post("/v1/record", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}, f"Expected {{status: ok}}, got {data}"

    def test_record_rejects_missing_fields(self, client: TestClient) -> None:
        """Missing required fields must return 400."""
        resp = client.post("/v1/record", json={"provider": "groq"})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_record_rejects_invalid_json(self, client: TestClient) -> None:
        """Non-JSON body must return 400."""
        resp = client.post(
            "/v1/record",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_record_with_quality_rating(self, client: TestClient) -> None:
        """Record with valid quality_rating still returns {status: "ok"}."""
        body = {
            "provider": "groq",
            "model_id": "groq_llama70b",
            "success": True,
            "quality_rating": 4,
        }
        resp = client.post("/v1/record", json=body)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_record_rejects_invalid_quality_rating(self, client: TestClient) -> None:
        """quality_rating outside 1-5 must return 400."""
        body = {
            "provider": "groq",
            "model_id": "groq_llama70b",
            "success": True,
            "quality_rating": 10,
        }
        resp = client.post("/v1/record", json=body)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------


class TestHealthContract:
    """GET /v1/health must return {status, version, budget, health}."""

    def test_health_returns_expected_shape(self, client: TestClient) -> None:
        """Health response must have status, version, budget, health."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data, "Must have 'status'"
        assert "version" in data, "Must have 'version'"
        assert "budget" in data, "Must have 'budget'"
        assert "health" in data, "Must have 'health'"
        assert isinstance(data["status"], str), "'status' must be a string"
        assert isinstance(data["version"], str), "'version' must be a string"
        assert isinstance(data["budget"], dict), "'budget' must be a dict"
        assert isinstance(data["health"], dict), "'health' must be a dict"

    def test_health_always_returns_200(self, client: TestClient) -> None:
        """Health endpoint is a liveness probe -- always 200."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /v1/catalog
# ---------------------------------------------------------------------------


class TestCatalogContract:
    """GET /v1/catalog must return {stale, providers, model_count}."""

    def test_catalog_returns_expected_shape(self, client: TestClient) -> None:
        """Catalog response must have stale, providers, model_count."""
        resp = client.get("/v1/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert "stale" in data, "Must have 'stale'"
        assert "providers" in data, "Must have 'providers'"
        assert "model_count" in data, "Must have 'model_count'"
        assert isinstance(data["stale"], bool), "'stale' must be a bool"
        assert isinstance(data["providers"], list), "'providers' must be a list"
        assert isinstance(data["model_count"], int), "'model_count' must be an int"


# ---------------------------------------------------------------------------
# GET /v1/ready
# ---------------------------------------------------------------------------


class TestReadyContract:
    """GET /v1/ready must return {ready: bool} or {ready: bool, reason: str}."""

    def test_ready_returns_ready_field(self, client: TestClient) -> None:
        """Response must always contain 'ready' as a boolean."""
        resp = client.get("/v1/ready")
        data = resp.json()
        assert "ready" in data, "Must have 'ready' field"
        assert isinstance(data["ready"], bool), "'ready' must be a bool"

    def test_ready_returns_200_or_503(self, client: TestClient) -> None:
        """Ready endpoint returns 200 (ready) or 503 (not ready)."""
        resp = client.get("/v1/ready")
        assert resp.status_code in (200, 503), f"Expected 200 or 503, got {resp.status_code}"

    def test_not_ready_includes_reason(self, client: TestClient) -> None:
        """When not ready (503), response must include 'reason'."""
        # Remove engine to force not-ready
        original_engine = client.app.state.engine
        client.app.state.engine = None
        try:
            resp = client.get("/v1/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["ready"] is False
            assert "reason" in data, "Not-ready response must include 'reason'"
            assert isinstance(data["reason"], str)
        finally:
            client.app.state.engine = original_engine


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetricsContract:
    """GET /metrics must return {uptime_seconds, memory_mb, endpoints, router}."""

    def test_metrics_returns_expected_shape(self, client: TestClient) -> None:
        """Metrics response must have the documented top-level fields."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_seconds" in data, "Must have 'uptime_seconds'"
        assert "memory_mb" in data, "Must have 'memory_mb'"
        assert "endpoints" in data, "Must have 'endpoints'"
        assert "router" in data, "Must have 'router'"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["memory_mb"], (int, float))
        assert isinstance(data["endpoints"], dict)
        assert isinstance(data["router"], dict)

    def test_metrics_router_shape(self, client: TestClient) -> None:
        """router sub-object must have dispatch counters."""
        resp = client.get("/metrics")
        router = resp.json()["router"]
        assert "total_dispatches" in router
        assert "fallback_count" in router
        assert "circuit_breaker_trips" in router


# ---------------------------------------------------------------------------
# GET /openapi.json
# ---------------------------------------------------------------------------


class TestOpenApiContract:
    """GET /openapi.json must return a valid OpenAPI 3.0.3 schema."""

    def test_openapi_returns_valid_schema(self, client: TestClient) -> None:
        """Response must be valid JSON with OpenAPI 3.0.3 structure."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["openapi"] == "3.0.3", "Must be OpenAPI 3.0.3"
        assert "info" in data, "Must have 'info'"
        assert "paths" in data, "Must have 'paths'"
        assert "title" in data["info"]
        assert "version" in data["info"]

    def test_openapi_documents_all_endpoints(self, client: TestClient) -> None:
        """Schema must document all public endpoints."""
        data = client.get("/openapi.json").json()
        paths = data["paths"]
        expected_paths = [
            "/v1/select",
            "/v1/dispatch",
            "/v1/record",
            "/v1/health",
            "/v1/ready",
            "/v1/catalog",
            "/metrics",
            "/openapi.json",
        ]
        for path in expected_paths:
            assert path in paths, f"OpenAPI schema missing path: {path}"


# ---------------------------------------------------------------------------
# Cross-cutting: error response format
# ---------------------------------------------------------------------------


class TestErrorResponseFormat:
    """Error responses must always contain 'error' or 'message' field."""

    def test_validation_errors_have_error_field(self, client: TestClient) -> None:
        """400 validation errors use {error: str}."""
        resp = client.post("/v1/select", json={})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert isinstance(data["error"], str)

    def test_invalid_json_errors_have_error_field(self, client: TestClient) -> None:
        """Invalid JSON returns 400 with {error: str}."""
        for endpoint in ["/v1/select", "/v1/dispatch", "/v1/record"]:
            resp = client.post(
                endpoint,
                content=b"not valid json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 400, f"{endpoint} did not return 400"
            data = resp.json()
            assert "error" in data, f"{endpoint} error response missing 'error' field"
