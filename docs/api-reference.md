# API Reference

The router exposes an HTTP API via Starlette/Uvicorn on port 8100 by default. All request and response bodies are JSON.

Every response includes an `X-Request-ID` header for correlation. The router either reads this from the incoming request or generates a UUID4.

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/select` | none | Return ranked model IDs for a role |
| `POST` | `/v1/dispatch` | none | Full cascade dispatch (MBR/CBR/LBR + adapter call) |
| `POST` | `/v1/record` | none | Record request outcome (budget + health) |
| `GET` | `/v1/health` | none | Liveness probe with health and budget snapshot |
| `GET` | `/v1/ready` | none | Readiness probe (503 until catalog is loaded) |
| `GET` | `/v1/catalog` | none | Current catalog status |
| `POST` | `/v1/catalog/refresh` | admin | Trigger immediate catalog refresh |
| `POST` | `/v1/retire` | admin | Retire a backend from the active pool |
| `POST` | `/v1/reinstate` | admin | Reinstate a previously retired backend |
| `GET` | `/metrics` | none | Operational metrics (request counts, latency, dispatch stats) |
| `GET` | `/openapi.json` | none | OpenAPI 3.0.3 schema document |

Admin endpoints require a Bearer token when `DRAGONLIGHT_ADMIN_API_KEY` is set. Without it, they are open.

---

## POST /v1/select

Return a ranked list of model IDs for a given role.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `role` | string | yes | | Role to select models for (e.g., `"code_review"`) |
| `top_n` | integer | no | 12 | Maximum number of models to return (1-500) |
| `exclude_providers` | string[] | no | | Provider names to exclude from results |

```json
{
  "role": "code_review",
  "top_n": 5,
  "exclude_providers": ["openrouter"]
}
```

### Response (200)

```json
{
  "models": [
    "groq/llama-3.3-70b-versatile",
    "cerebras/llama3.1-70b",
    "gemini/gemini-2.0-flash"
  ],
  "scores": [
    {
      "model_id": "groq/llama-3.3-70b-versatile",
      "health_score": 98.5,
      "budget_score": 72.0,
      "complexity_tier": "SIMPLE",
      "trust_tier": "semi_trusted"
    }
  ]
}
```

### Errors

| Status | Condition |
|---|---|
| 400 | Missing `role`, invalid `top_n`, or malformed JSON |

---

## POST /v1/dispatch

Execute the full MBR/CBR/LBR cascade, call the selected provider adapter, and return the response. Falls back through the ranked list on failure.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `intent_category` | string | yes | | Intent type. Must be one of the allowed categories (see below) |
| `specific_intent` | string | yes | | Freeform description of the specific task |
| `operator_message` | string | yes | | The prompt to send to the LLM |
| `context_tokens` | integer | yes | | Estimated token count of the context |
| `system_prompt` | string | no | `""` | System prompt for the LLM |
| `requires_tool_use` | boolean | no | `false` | Filter to models with tool-use capability |
| `requires_long_context` | boolean | no | `false` | Filter to long-context models |
| `persona` | string | no | | Persona identifier for context filtering |
| `stream` | boolean | no | `false` | Return SSE stream instead of JSON |
| `fallback_policy` | string | no | `"allow"` | One of `allow`, `deny`, `same_tier` |
| `request_id` | string | no | | Client-provided request correlation ID |
| `context_trust_tier` | string | no | | Trust tier for context filtering |

**Allowed intent categories:** `architecture`, `casual_chat`, `code_generation`, `code_review`, `complex_reasoning`, `creative_writing`, `data_analysis`, `debugging`, `documentation`, `engineering_build`, `general`, `search`, `session_lifecycle`, `spec_writing`, `strategic_planning`, `summarization`, `test`, `translation`

### Response (200, non-streaming)

```json
{
  "content": "The generated response text...",
  "backend_used": "groq/llama-3.3-70b-versatile",
  "backend_tier": "SIMPLE",
  "tokens_in": 256,
  "tokens_out": 1024,
  "estimated_cost_usd": 0.0012,
  "latency_ms": 620,
  "was_fallback": false,
  "fallback_chain": []
}
```

### Response (200, streaming)

When `stream` is `true`, the response is an SSE stream (`text/event-stream`). Each event is a JSON object on a `data:` line:

**Token event:**
```
data: {"event": "token", "content": "Hello"}
```

**Metadata event (final):**
```
data: {"event": "metadata", "backend_used": "groq/llama-3.3-70b-versatile", "backend_tier": "SIMPLE", "tokens_in": 256, "tokens_out": 1024, "estimated_cost_usd": 0.0012, "latency_ms": 620, "was_fallback": false, "fallback_chain": []}
```

**Error event:**
```
data: {"event": "error", "error_message": "Internal server error"}
```

### Errors

| Status | Condition |
|---|---|
| 400 | Missing required field, invalid `intent_category`, or malformed JSON |
| 500 | All backends in the cascade failed |

The 500 response includes `attempted_backends` and `error_details` for diagnostics.

---

## POST /v1/record

Record the outcome of a request for budget and health tracking.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `provider` | string | yes | | Provider name (e.g., `"groq"`) |
| `model_id` | string | yes | | Full model ID (e.g., `"groq/llama-3.3-70b-versatile"`) |
| `success` | boolean | yes | | Whether the request succeeded |
| `tokens_used` | integer | no | `0` | Total tokens consumed |
| `latency_ms` | number | no | `0.0` | Request latency in milliseconds |

### Response (200)

```json
{"status": "ok"}
```

---

## GET /v1/health

Liveness probe. Always returns 200. The `status` field indicates overall router health.

### Response (200)

```json
{
  "status": "healthy",
  "version": "0.2.6",
  "key_invalid_count": 0,
  "budget": { "groq": { "rpm_used": 5, "rpm_limit": 30 } },
  "health": { "groq/llama-3.3-70b-versatile": { "score": 98.5, "circuit": "CLOSED" } }
}
```

The `status` field is one of `healthy`, `degraded`, or `unavailable`.

A non-zero `key_invalid_count` indicates providers with credential issues detected during the last catalog refresh.

---

## GET /v1/ready

Readiness probe. Returns 200 when the router is initialized and the catalog has been loaded at least once. Returns 503 otherwise.

### Response (200)

```json
{"ready": true}
```

### Response (503)

```json
{"ready": false, "reason": "Catalog has not been refreshed yet"}
```

---

## GET /v1/catalog

Catalog status showing which providers are loaded and the total model count.

### Response (200)

```json
{
  "stale": false,
  "providers": ["groq", "cerebras", "gemini", "mistral"],
  "model_count": 47
}
```

---

## POST /v1/catalog/refresh

Trigger an immediate catalog refresh from all configured providers. Requires admin auth when `DRAGONLIGHT_ADMIN_API_KEY` is set.

```bash
curl -s -X POST http://127.0.0.1:8100/v1/catalog/refresh \
  -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY"
```

### Response (200)

```json
{
  "status": "ok",
  "providers_refreshed": ["groq", "cerebras", "gemini"],
  "model_count": 47,
  "auth_failures": {"anthropic": 401}
}
```

The `auth_failures` field appears only when one or more providers returned authentication errors during the refresh.

---

## POST /v1/retire

Retire a backend from the active pool. The cascade skips retired backends until they are explicitly reinstated. Requires admin auth.

### Request body

```json
{"backend": "groq/llama-3.3-70b-versatile"}
```

### Response (200)

```json
{"retired": true, "backend": "groq/llama-3.3-70b-versatile"}
```

---

## POST /v1/reinstate

Reinstate a previously retired backend. The backend returns to the AVAILABLE pool immediately. Requires admin auth.

### Request body

```json
{"backend": "groq/llama-3.3-70b-versatile"}
```

### Response (200)

```json
{"reinstated": true, "backend": "groq/llama-3.3-70b-versatile"}
```

---

## GET /metrics

Operational metrics: per-endpoint request counts, error counts, latency percentiles, dispatch stats, uptime, and memory usage.

### Response (200)

```json
{
  "uptime_seconds": 3600.5,
  "memory_mb": 128.3,
  "endpoints": {
    "/v1/select": {"count": 1200, "error_count": 2, "p50_ms": 12.5, "p99_ms": 45.0}
  },
  "router": {
    "total_dispatches": 500,
    "fallback_count": 23,
    "circuit_breaker_trips": 1
  }
}
```

---

## OpenAPI spec

The full OpenAPI 3.0.3 schema is served at `GET /openapi.json` and also available as a static file at `docs/openapi.yaml` in the repository.
