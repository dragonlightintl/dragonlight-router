"""OpenAPI 3.0.3 schema definition for the Dragonlight Router API.

Extracted from routes.py to reduce file size. Contains the static schema
builder and the cached schema handler.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse


# DEVIATION CS-004: _build_openapi_schema is 383 lines.
# Justification: Static OpenAPI schema definition as a single dict literal. Splitting
# into per-endpoint functions would scatter the schema across ~12 functions with no
# reuse, making the spec harder to read and maintain.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def _build_openapi_schema() -> dict[str, Any]:
    """Build the OpenAPI 3.0.3 schema for the Dragonlight Router API.

    Returns a static dict describing all endpoints, request/response bodies,
    status codes, and headers.
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Dragonlight Router API",
            "description": (
                "Multi-provider intelligent LLM routing — model selection + cascade dispatch."
            ),
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
                                        "role": {
                                            "type": "string",
                                            "description": "Role to select models for.",
                                        },
                                        "top_n": {
                                            "type": "integer",
                                            "default": 12,
                                            "minimum": 1,
                                            "maximum": 500,
                                        },
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
                                            "models": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "scores": {
                                                "type": "array",
                                                "items": {"type": "object"},
                                            },
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
                    "summary": "Dispatch a request through the cascade or pinned model",
                    "operationId": "dispatchRequest",
                    "description": (
                        "When ``model`` is set, bypasses the MBR-IBR-CBR-LBR cascade "
                        "and dispatches directly to the specified backend (pinned dispatch). "
                        "When ``model`` is absent, runs the full cascade (default behavior)."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "operator_message",
                                        "context_tokens",
                                    ],
                                    "properties": {
                                        "model": {
                                            "type": "string",
                                            "description": (
                                                "Backend name to pin (e.g. "
                                                "'anthropic/claude-sonnet-4-20250514'). "
                                                "When set, bypasses cascade. When absent, "
                                                "intent_category and specific_intent are "
                                                "required."
                                            ),
                                        },
                                        "intent_category": {"type": "string"},
                                        "specific_intent": {"type": "string"},
                                        "operator_message": {"type": "string"},
                                        "context_tokens": {"type": "integer", "minimum": 0},
                                        "system_prompt": {"type": "string"},
                                        "requires_tool_use": {"type": "boolean", "default": False},
                                        "requires_long_context": {
                                            "type": "boolean",
                                            "default": False,
                                        },
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
                                "X-Request-ID": {
                                    "schema": {"type": "string"},
                                    "description": "Request correlation ID.",
                                },
                            },
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "backend_used": {"type": "string"},
                                            "backend_tier": {"type": "string"},
                                            "dispatch_mode": {
                                                "type": "string",
                                                "enum": ["cascade", "pinned"],
                                                "description": (
                                                    "Whether the request was routed via "
                                                    "cascade or pinned dispatch."
                                                ),
                                            },
                                            "tokens_in": {"type": "integer"},
                                            "tokens_out": {"type": "integer"},
                                            "estimated_cost_usd": {"type": "number"},
                                            "latency_ms": {"type": "number"},
                                            "was_fallback": {"type": "boolean"},
                                            "fallback_chain": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "400": {
                            "description": (
                                "Validation error, or pinned model not found in registry."
                            ),
                        },
                        "429": {
                            "description": (
                                "Pinned model's provider budget or rate limit exhausted."
                            ),
                        },
                        "500": {"description": "Dispatch failure."},
                        "503": {
                            "description": ("Pinned model is unhealthy (circuit open)."),
                        },
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
                            "description": (
                                "Always returns 200 with status, budget, and health data."
                            ),
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string",
                                                "enum": ["healthy", "degraded", "unavailable"],
                                            },
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
                                            "providers": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
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
# DEVIATION CS-MUTABLE-002: intentionally mutable — runtime cache/singleton.
_OPENAPI_SCHEMA: dict[str, Any] | None = None


async def openapi_handler(request: Request) -> JSONResponse:
    """GET /openapi.json — serve the OpenAPI 3.0.3 schema."""
    global _OPENAPI_SCHEMA  # noqa: PLW0603
    if _OPENAPI_SCHEMA is None:
        _OPENAPI_SCHEMA = _build_openapi_schema()
    return JSONResponse(_OPENAPI_SCHEMA)
