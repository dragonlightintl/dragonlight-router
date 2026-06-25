# Factory-Router Service Boundary Design

**Status:** Proposed
**Authors:** GOIBNIU (engineering), LUGH (interface design)
**Date:** 2026-06-24

## Problem Statement

The Dragonlight Software Factory currently integrates with the Router in three
conflicting ways simultaneously, creating a fragmented architecture where state
is split between process memory and shared storage, rate limits are per-process,
and the router's cascade/scoring logic is mostly bypassed.

### Current Integration Paths

**Path 1 — Direct model construction (`_create_model()` in `coding_agent.py:1267`).**
The factory constructs `OpenAIChatModel` instances directly against provider APIs
(NIM, Groq, OpenRouter, Mistral, Cerebras, Gemini, Anthropic) with per-provider
`httpx.AsyncClient` throttle transports. The router is completely absent. This is
the path used by the static chain (`_FALLBACK_CHAIN` + `_select_next_model()`).

**Path 2 — In-process `RouterEngine` for selection only (`_get_factory_router()` in
`factory.py:2652`).** When `ROUTER_ENABLED=1`, the factory creates a `RouterEngine`
singleton in its own process, calls `router.select_models(role)` for candidate
ranking, then constructs models via `_create_model()` anyway. Outcomes are recorded
back via `router.record_request()`. Health and budget state is in-process only.

**Path 3 — Full router dispatch (`_run_via_router()` in `factory.py:4042`).** When
`ROUTER_DISPATCH=1`, the factory creates a `RouterModel` (pydantic-ai `Model`
adapter in `router_model.py`) that wraps `router.dispatch()` for end-to-end
dispatch including model selection, API calls, fallback, and health tracking. This
path delegates everything to the in-process `RouterEngine`.

### What Breaks

| Problem | Root Cause |
|---------|-----------|
| Health tracking is per-process | `RouterEngine` stores health in memory; concurrent factory runs each have their own circuit breakers |
| Rate limits are per-process | `_create_model()` creates per-process `httpx.AsyncClient` with throttle transports; 3 concurrent runs = 3x the rate limit |
| Budget tracking partially shared | SQLite WAL allows concurrent reads, but health state is not persisted to DB |
| Router cascade logic unused | Path 1 (static chain) and Path 2 (select-only) both bypass the cascade, scoring, and fallback logic |
| `_FALLBACK_CHAIN` duplicates router's model-role matrix | Hardcoded list in `factory.py:257` drifts from the router's curated rankings |
| `_resolve_via_router()` reads matrix JSON directly | Bypasses the router's catalog-aware filtering and health scoring (factory.py:2739) |

## Recommendation: Option C — HTTP Select + Direct Execute + HTTP Record

Option C is the correct boundary because of an irreducible constraint: **pydantic-ai
requires a `Model` object with a `.request()` method for multi-turn tool-use agent
loops.** The router's `/v1/dispatch` endpoint is single-turn — it takes a prompt and
returns a completion. Multi-turn agentic conversations (tool calls, retries, context
accumulation) happen inside pydantic-ai's `Agent.run()` loop, which needs a live
connection to the model API.

This means the factory must call provider APIs directly for execution. But model
**selection** and **outcome recording** should be centralized in the router to get
shared health, budget, and rate-limit state across all concurrent factory processes.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Factory Process (1..N concurrent)                      │
│                                                         │
│  1. POST /v1/select-for-factory                         │
│     ← ranked candidates + provider configs              │
│                                                         │
│  2. Build pydantic-ai Model from candidate[0] config    │
│     Run Agent.run() with multi-turn tool-use loop       │
│     (direct API calls to provider)                      │
│                                                         │
│  3. POST /v1/record                                     │
│     → success/failure, tokens, latency, model_id        │
│                                                         │
│  4. If failed + retryable: try candidate[1], goto 2     │
│                                                         │
│  5. POST /v1/acquire-rate-slot (before each API call)   │
│     ← granted / denied + retry-after                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│  Router HTTP Service (single process, port 8100)        │
│                                                         │
│  Shared state:                                          │
│  - Health tracking (circuit breakers, latency scores)   │
│  - Budget tracking (per-provider token spend)           │
│  - Rate limiting (per-provider RPM with sliding window) │
│  - Catalog (live model availability)                    │
│  - Model-role matrix (curated rankings)                 │
│  - Spectrograph profiles (IBR flavor scoring)           │
│                                                         │
│  Endpoints:                                             │
│  - POST /v1/select-for-factory (new)                    │
│  - POST /v1/record (existing)                           │
│  - POST /v1/acquire-rate-slot (new)                     │
│  - POST /v1/release-rate-slot (new)                     │
│  - GET  /v1/health (existing)                           │
│  - GET  /v1/ready (existing)                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## New Endpoints

### `POST /v1/select-for-factory`

A factory-specific selection endpoint that returns not just ranked model IDs but
the **provider configuration** needed to construct pydantic-ai `Model` objects
client-side. This is the key difference from the existing `/v1/select` which returns
only model IDs.

**Request:**

```json
{
  "role": "coding",
  "intent_category": "test_generation",
  "complexity": "standard",
  "top_n": 5,
  "exclude_models": ["nvidia_nim/qwen3-coder-480b"],
  "requires_tool_use": true,
  "context_tokens": 12000
}
```

**Response:**

```json
{
  "candidates": [
    {
      "model_id": "nvidia_nim/moonshotai/kimi-k2.6",
      "provider": "nvidia_nim",
      "provider_config": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_NIM_API_KEY",
        "protocol": "openai",
        "rpm_limit": 60,
        "max_tokens": 16384
      },
      "health_score": 95.2,
      "budget_score": 88.0,
      "composite_score": 91.6,
      "tier": "complex",
      "trust_tier": "trusted"
    },
    {
      "model_id": "mistral/codestral-latest",
      "provider": "mistral",
      "provider_config": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "protocol": "openai",
        "rpm_limit": null,
        "max_tokens": 8192
      },
      "health_score": 90.0,
      "budget_score": 92.5,
      "composite_score": 91.2,
      "tier": "moderate",
      "trust_tier": "semi_trusted"
    }
  ],
  "request_id": "sel_abc123"
}
```

The `provider_config` block gives the factory everything it needs to call
`OpenAIProvider(base_url=..., api_key=os.environ[api_key_env])` and construct a
pydantic-ai `OpenAIChatModel` or `AnthropicModel` directly. The factory no longer
needs its own provider-routing `if/elif` chain in `_create_model()`.

**Key design choice:** The `api_key_env` field names the environment variable, not
the key value itself. API keys never transit the HTTP boundary. The factory process
already has the keys in its environment.

### `POST /v1/acquire-rate-slot`

Centralized rate limiting. Before each LLM API call, the factory asks the router
for permission. The router maintains a sliding-window counter per provider.

**Request:**

```json
{
  "provider": "nvidia_nim",
  "model_id": "nvidia_nim/moonshotai/kimi-k2.6",
  "estimated_tokens": 4000
}
```

**Response (granted):**

```json
{
  "granted": true,
  "slot_id": "slot_xyz789",
  "expires_at": "2026-06-24T15:30:45Z"
}
```

**Response (denied):**

```json
{
  "granted": false,
  "retry_after_ms": 2300,
  "reason": "rpm_limit_exceeded",
  "current_rpm": 60,
  "limit_rpm": 60
}
```

The `slot_id` is used for release/timeout. If the factory crashes without releasing,
the slot expires at `expires_at` (30s default).

### `POST /v1/release-rate-slot`

Called after each LLM API call completes (success or failure) to release the rate
slot and report actual token usage.

**Request:**

```json
{
  "slot_id": "slot_xyz789",
  "actual_tokens": 3842,
  "success": true,
  "latency_ms": 4521.0
}
```

This merges rate-slot release with outcome recording for the common case. The
existing `/v1/record` endpoint remains for recording outcomes without a rate slot
(e.g., for non-factory consumers).

### `POST /v1/record` (existing, unchanged)

Continues to accept outcome data as-is. The factory uses this for recording
aggregate build outcomes or for cases where rate-slot acquisition was skipped.

```json
{
  "provider": "nvidia_nim",
  "model_id": "nvidia_nim/moonshotai/kimi-k2.6",
  "success": true,
  "tokens_used": 3842,
  "latency_ms": 4521.0,
  "quality_rating": 4
}
```

## Factory Client Design

### New module: `factory/scripts/router_client.py`

Replaces both `router_model.py` (the pydantic-ai `RouterModel` adapter) and the
in-process `_get_factory_router()` singleton. The factory communicates with the
router exclusively over HTTP.

```python
"""HTTP client for the Dragonlight Router service.

Replaces in-process RouterEngine usage. All model selection, rate limiting,
and outcome recording go through the router's HTTP API. The factory constructs
pydantic-ai Model objects locally using provider configs from the router.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicProvider
from pydantic_ai.settings import ModelSettings


ROUTER_BASE_URL = os.environ.get("DRAGONLIGHT_ROUTER_URL", "http://127.0.0.1:8100")

# Shared HTTP client for router API calls (not LLM calls).
_router_http: httpx.AsyncClient | None = None


def _get_router_http() -> httpx.AsyncClient:
    global _router_http
    if _router_http is None:
        _router_http = httpx.AsyncClient(
            base_url=ROUTER_BASE_URL,
            timeout=httpx.Timeout(10.0),
        )
    return _router_http


@dataclass(frozen=True)
class ModelCandidate:
    """A ranked model candidate from the router."""
    model_id: str
    provider: str
    base_url: str
    api_key_env: str
    protocol: str  # "openai" | "anthropic"
    rpm_limit: int | None
    max_tokens: int | None
    health_score: float
    budget_score: float
    composite_score: float
    tier: str
    trust_tier: str


@dataclass(frozen=True)
class RateSlot:
    """An acquired rate-limit slot."""
    slot_id: str
    expires_at: str
    granted: bool
    retry_after_ms: int | None = None


async def select_models(
    role: str,
    *,
    intent_category: str = "coding",
    complexity: str = "standard",
    top_n: int = 5,
    exclude_models: list[str] | None = None,
    requires_tool_use: bool = True,
    context_tokens: int = 0,
) -> list[ModelCandidate]:
    """Ask the router for ranked model candidates."""
    client = _get_router_http()
    body = {
        "role": role,
        "intent_category": intent_category,
        "complexity": complexity,
        "top_n": top_n,
        "exclude_models": exclude_models or [],
        "requires_tool_use": requires_tool_use,
        "context_tokens": context_tokens,
    }
    resp = await client.post("/v1/select-for-factory", json=body)
    resp.raise_for_status()
    data = resp.json()

    return [
        ModelCandidate(
            model_id=c["model_id"],
            provider=c["provider"],
            base_url=c["provider_config"]["base_url"],
            api_key_env=c["provider_config"]["api_key_env"],
            protocol=c["provider_config"]["protocol"],
            rpm_limit=c["provider_config"].get("rpm_limit"),
            max_tokens=c["provider_config"].get("max_tokens"),
            health_score=c["health_score"],
            budget_score=c["budget_score"],
            composite_score=c["composite_score"],
            tier=c["tier"],
            trust_tier=c["trust_tier"],
        )
        for c in data["candidates"]
    ]


def build_pydantic_model(candidate: ModelCandidate) -> Model:
    """Construct a pydantic-ai Model from a router-provided candidate.

    This replaces _create_model() in coding_agent.py. The provider config
    comes from the router, so the factory no longer needs provider-specific
    if/elif chains.
    """
    api_key = os.environ.get(candidate.api_key_env, "")
    max_tokens = candidate.max_tokens
    settings = ModelSettings(max_tokens=max_tokens) if max_tokens else None

    # Strip provider prefix from model_id for the API call.
    # e.g. "nvidia_nim/moonshotai/kimi-k2.6" -> "moonshotai/kimi-k2.6"
    name = candidate.model_id
    prefix = candidate.provider + "/"
    if name.startswith(prefix):
        name = name[len(prefix):]

    if candidate.protocol == "anthropic":
        return AnthropicModel(
            name,
            provider=AnthropicProvider(api_key=api_key),
            settings=settings,
        )

    # Default: OpenAI-compatible protocol
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(
            base_url=candidate.base_url,
            api_key=api_key,
            http_client=httpx.AsyncClient(timeout=httpx.Timeout(300.0)),
        ),
        settings=settings,
    )


async def acquire_rate_slot(
    provider: str,
    model_id: str,
    estimated_tokens: int = 4000,
) -> RateSlot:
    """Acquire a rate-limit slot from the router before calling the LLM API."""
    client = _get_router_http()
    resp = await client.post("/v1/acquire-rate-slot", json={
        "provider": provider,
        "model_id": model_id,
        "estimated_tokens": estimated_tokens,
    })
    resp.raise_for_status()
    data = resp.json()
    return RateSlot(
        slot_id=data.get("slot_id", ""),
        expires_at=data.get("expires_at", ""),
        granted=data["granted"],
        retry_after_ms=data.get("retry_after_ms"),
    )


async def release_rate_slot(
    slot_id: str,
    *,
    actual_tokens: int,
    success: bool,
    latency_ms: float,
) -> None:
    """Release a rate-limit slot and report outcome."""
    client = _get_router_http()
    await client.post("/v1/release-rate-slot", json={
        "slot_id": slot_id,
        "actual_tokens": actual_tokens,
        "success": success,
        "latency_ms": latency_ms,
    })


async def record_outcome(
    provider: str,
    model_id: str,
    *,
    success: bool,
    tokens_used: int = 0,
    latency_ms: float = 0.0,
    quality_rating: int | None = None,
) -> None:
    """Record a model outcome with the router (non-rate-slot path)."""
    client = _get_router_http()
    body: dict[str, Any] = {
        "provider": provider,
        "model_id": model_id,
        "success": success,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
    }
    if quality_rating is not None:
        body["quality_rating"] = quality_rating
    await client.post("/v1/record", json=body)
```

### How Multi-Turn pydantic-ai Conversations Work

The factory's `run_coding_agent()` function (coding_agent.py:2155) creates a
pydantic-ai `Agent` and calls `agent.run()`. This internally loops:

1. Send messages to model via `model.request(messages, settings, params)`
2. Parse response for tool calls
3. Execute tools locally (file writes, shell commands)
4. Append tool results to messages
5. Send updated messages back to model
6. Repeat until model returns final text (no tool calls)

Under Option C, the `Model` object passed to `Agent(model=...)` is a standard
pydantic-ai `OpenAIChatModel` or `AnthropicModel` constructed from the router's
provider config. There is no `RouterModel` adapter in the loop. The model talks
directly to the provider API.

The router's role is bookending: **select before, record after.**

```
Router: select candidates
  ↓
Factory: build_pydantic_model(candidates[0])
Factory: agent = Agent(model=model, tools=[...])
Factory: result = await agent.run(prompt)
  │  ↑↓ multi-turn tool-use loop (direct to provider API)
  ↓
Router: record outcome (success/fail, tokens, latency)
  ↓
If failed + retryable:
  Factory: build_pydantic_model(candidates[1])
  (repeat)
```

### How Rate Limiting Works Across Concurrent Runs

Current state: Each factory process creates its own `httpx.AsyncClient` with a
`ThrottleTransport` that rate-limits RPM per-process. Three concurrent runs hitting
NIM at 60 RPM each = 180 RPM actual, exceeding the provider limit.

New state: The router maintains a single sliding-window RPM counter per provider.
Each factory process must `acquire_rate_slot()` before each LLM API call. The router
either grants the slot (under the RPM limit) or denies it with a `retry_after_ms`
hint.

**Implementation in the router:**

```python
# New file: dragonlight_router/rate_limiter.py

@dataclass
class RateLimiter:
    """Sliding-window rate limiter with slot tracking.

    Shared across all consumers via the router process. Slots expire
    automatically if not released (crash safety).
    """

    # Provider -> list of (timestamp, slot_id) for active window
    _windows: dict[str, list[tuple[float, str]]]
    # slot_id -> (provider, expires_at) for outstanding slots
    _active_slots: dict[str, tuple[str, float]]
    # Provider -> RPM limit (from config)
    _limits: dict[str, int]

    def acquire(self, provider: str, model_id: str) -> RateSlot: ...
    def release(self, slot_id: str) -> None: ...
    def _prune_expired(self, provider: str) -> None: ...
```

The factory integration point is in the agent execution path. Since pydantic-ai's
`Model.request()` is async, the rate-slot acquisition wraps each request:

**Phase 1 (pragmatic):** Rate-slot acquisition happens at the factory level, once
per `run_coding_agent()` call. This is coarse-grained (one slot per entire agent
run, not per turn) but simple and catches the main concurrency problem.

**Phase 2 (precise):** A thin pydantic-ai `Model` wrapper acquires a slot before
each `.request()` call and releases it after, giving per-turn rate limiting. This
is a `RateLimitedModel` wrapper:

```python
class RateLimitedModel(Model):
    """Wraps a pydantic-ai Model with router-based rate limiting."""

    def __init__(self, inner: Model, provider: str, model_id: str):
        self._inner = inner
        self._provider = provider
        self._model_id = model_id

    async def request(self, messages, model_settings, model_request_parameters):
        slot = await acquire_rate_slot(self._provider, self._model_id)
        if not slot.granted:
            await asyncio.sleep(slot.retry_after_ms / 1000)
            slot = await acquire_rate_slot(self._provider, self._model_id)
            if not slot.granted:
                raise RuntimeError(f"Rate limited by router: retry after {slot.retry_after_ms}ms")
        try:
            result = await self._inner.request(messages, model_settings, model_request_parameters)
            await release_rate_slot(slot.slot_id, actual_tokens=..., success=True, latency_ms=...)
            return result
        except Exception:
            await release_rate_slot(slot.slot_id, actual_tokens=0, success=False, latency_ms=...)
            raise
```

## Router-Side Changes

### New files

| File | Purpose |
|------|---------|
| `dragonlight_router/rate_limiter.py` | `RateLimiter` class with sliding-window RPM tracking and slot management |
| `dragonlight_router/server/factory_routes.py` | Route handlers for `/v1/select-for-factory`, `/v1/acquire-rate-slot`, `/v1/release-rate-slot` |

### Modified files

| File | Change |
|------|--------|
| `server/app.py` | Import and register new routes from `factory_routes.py` |
| `server/routes.py` | Add `_ALLOWED_INTENT_CATEGORIES` entries for factory intent categories (`"coding"`, `"test_generation"`, `"test_property"`, `"implementation"`, `"implementation_complex"`, `"coherence_merge"`, `"audit"`) |
| `router.py` | Add `get_provider_config(model_id) -> dict` method that returns base_url, api_key_env, protocol for a model. Add `RateLimiter` as a component on `RouterEngine` |
| `core/types.py` | Add `ProviderConnectionInfo` dataclass |

### New type: `ProviderConnectionInfo`

```python
@dataclass(frozen=True)
class ProviderConnectionInfo:
    """Connection details for a provider, returned by select-for-factory.

    api_key_env is the environment variable name, never the key value.
    """
    base_url: str
    api_key_env: str
    protocol: str  # "openai" | "anthropic"
    rpm_limit: int | None
    max_tokens: int | None
```

### Provider config mapping

The router already knows provider details from its config (`router.yaml` provider
entries). The mapping from provider name to `ProviderConnectionInfo` is:

```yaml
# Already in router.yaml under providers:
- name: nvidia_nim
  base_url: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_NIM_API_KEY
  # New fields to add:
  protocol: openai
  rpm_limit: 60

- name: groq
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY
  protocol: openai
  rpm_limit: 28
```

The `api_key_env` and `rpm_limit` fields are additions to the existing provider
config schema. Currently the router knows base_url and api_key from config, but
doesn't expose them via HTTP. The new endpoint surfaces this information (minus
the actual key value) so the factory can construct clients.

## Factory-Side Changes

### Files to create

| File | Purpose |
|------|---------|
| `scripts/router_client.py` | HTTP client for router API (replaces in-process `RouterEngine` usage) |

### Files to modify

| File | Change |
|------|--------|
| `scripts/factory.py` | Replace `_get_factory_router()`, `_select_next_model()`, `_resolve_via_router()`, `_record_router_outcome()` with calls to `router_client` |
| `scripts/factory.py` | Remove `_FALLBACK_CHAIN`, `_MODEL_ROUTING` (router owns model rankings) |
| `scripts/factory.py` | Replace `ROUTER_ENABLED` / `ROUTER_DISPATCH` env vars with `DRAGONLIGHT_ROUTER_URL` |
| `scripts/coding_agent.py` | Replace `_create_model()` with `router_client.build_pydantic_model()` |
| `scripts/coding_agent.py` | Remove direct `from dragonlight_router.health.circuit_breaker import CircuitBreaker` import |
| `scripts/coding_agent.py` | Remove all provider-specific throttle transport construction |

### Files to delete

| File | Reason |
|------|--------|
| `scripts/router_model.py` | The `RouterModel` pydantic-ai adapter is replaced by direct model construction from router-provided configs. The router no longer dispatches LLM calls for the factory. |

### Environment variable changes

| Old | New | Notes |
|-----|-----|-------|
| `ROUTER_ENABLED=1` | Removed | Router is always used when `DRAGONLIGHT_ROUTER_URL` is set |
| `ROUTER_DISPATCH=1` | Removed | Full dispatch mode eliminated; factory always does select+execute+record |
| (none) | `DRAGONLIGHT_ROUTER_URL=http://127.0.0.1:8100` | Router service address |

## Migration Path

### Phase 0: Router service readiness (router repo)

1. Add `api_key_env`, `protocol`, `rpm_limit` fields to provider config schema in
   `router.yaml` and the config model
2. Implement `RateLimiter` class in `dragonlight_router/rate_limiter.py`
3. Add `get_provider_config(model_id)` method to `RouterEngine`
4. Implement `POST /v1/select-for-factory` handler in `server/factory_routes.py`
5. Implement `POST /v1/acquire-rate-slot` and `POST /v1/release-rate-slot` handlers
6. Register new routes in `server/app.py`
7. Add factory intent categories to `_ALLOWED_INTENT_CATEGORIES`
8. Test: router serves `select-for-factory` with correct provider configs
9. Test: rate-slot acquire/release works under concurrency
10. Deploy router as a persistent service (systemd unit or launchd plist)

### Phase 1: Factory client introduction (factory repo)

1. Create `scripts/router_client.py` with `select_models()`, `build_pydantic_model()`,
   `acquire_rate_slot()`, `release_rate_slot()`, `record_outcome()`
2. Add `DRAGONLIGHT_ROUTER_URL` to factory environment
3. Modify `_select_next_model()` to call `router_client.select_models()` when
   `DRAGONLIGHT_ROUTER_URL` is set, falling back to static chain otherwise
4. Modify `_create_model()` to use `router_client.build_pydantic_model()` when
   a `ModelCandidate` is available
5. Modify `_record_router_outcome()` to call `router_client.record_outcome()`
6. Test: factory works with router service running
7. Test: factory still works without router (static chain fallback)

### Phase 2: Remove legacy paths (factory repo)

1. Remove `_get_factory_router()` (in-process `RouterEngine` singleton)
2. Remove `from dragonlight_router` imports from `factory.py` and `coding_agent.py`
3. Remove `_FALLBACK_CHAIN` and `_MODEL_ROUTING` dicts
4. Remove `ROUTER_ENABLED` and `ROUTER_DISPATCH` env var checks
5. Delete `scripts/router_model.py`
6. Remove `dragonlight-router` from factory's Python dependencies (only HTTP client
   remains as the integration point)
7. Remove per-provider throttle transports from `_create_model()` (replaced by
   router-managed rate limiting)
8. Test: factory operates correctly with router as sole model selection authority

### Phase 3: Per-turn rate limiting (factory repo, optional)

1. Implement `RateLimitedModel` wrapper in `router_client.py`
2. Wrap models from `build_pydantic_model()` in `RateLimitedModel`
3. This gives per-API-call rate limiting instead of per-agent-run

## What This Does Not Change

- **`select_model_for_ticket()`** — This complexity-to-model routing function moves
  its logic to the router's `select-for-factory` handler, which receives complexity
  as a parameter and uses it alongside health/budget/catalog to rank candidates.
  The function itself is removed from the factory.

- **`_run_via_router()` / full dispatch path** — Eliminated entirely. The factory
  never sends prompts through the router. The router never calls LLM APIs on behalf
  of the factory.

- **pydantic-ai agent loop** — Unchanged. `run_coding_agent()` still creates an
  `Agent` and calls `agent.run()`. The only difference is how the `Model` object is
  constructed (from router-provided config instead of hardcoded provider mapping).

- **Tool definitions and execution** — Unchanged. Tools are defined in the factory,
  executed in the factory, and never transit the router.

- **Non-factory router consumers** — The existing `/v1/select`, `/v1/dispatch`, and
  `/v1/record` endpoints are unchanged. DAOS sessions, ad-hoc queries, and other
  router consumers continue to use the existing API.

## Concurrency Model

With the router as a single HTTP service:

- **Health tracking:** Single process, single set of circuit breakers. All factory
  runs see the same health state. A model that fails in run A is immediately
  circuit-broken for run B.

- **Budget tracking:** Single process, single budget counter. Token spend from all
  concurrent runs is aggregated in real-time.

- **Rate limiting:** Single sliding-window counter per provider. Slot acquisition
  serializes access across all factory processes. If NIM allows 60 RPM, 3 concurrent
  factory runs collectively cannot exceed 60 RPM.

- **Catalog:** Single catalog refresh cycle. All factory runs share the same live
  model availability data.

## Failure Modes

| Scenario | Behavior |
|----------|----------|
| Router unreachable | Factory falls back to static chain (Phase 1 only; Phase 2 removes fallback, factory fails fast) |
| Router returns 5xx | Factory retries with exponential backoff (3 attempts, 1s/3s/9s) |
| Rate slot denied | Factory sleeps `retry_after_ms` and retries |
| Rate slot not released (factory crash) | Slot expires after 30s; router prunes on next acquire |
| Router restarts | Health/budget state loaded from disk on startup; rate-slot state is lost (slots expire naturally) |

## Open Questions

1. **Should the factory fall back to static chain permanently, or only during
   migration?** Post-Phase 2, the factory has no hardcoded model list. If the router
   is down, the factory cannot select models. This is arguably correct (the router
   IS the model authority) but makes the factory dependent on the router being up.

2. **Should rate-slot acquisition be blocking or async-with-retry?** The current
   design has the factory sleep on denial. An alternative is a queue-based approach
   where the factory submits work to the router and gets called back, but this adds
   significant complexity.

3. **Should the router persist rate-slot state to disk?** Currently in-memory only.
   On router restart, all slots are lost and the window resets. This means a brief
   burst above RPM limits after restart. Acceptable for current scale.
