# Factory-Router Service Boundary Design

**Status:** Revised — SQLite WAL + Library Import
**Authors:** GOIBNIU (engineering), LUGH (interface design)
**Date:** 2026-06-24
**Revised:** 2026-06-25

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

---

## Chosen Architecture: SQLite WAL + Library Import

The operator chose a simpler architecture than the originally proposed HTTP service
model. Instead of running the router as a persistent HTTP server for state
coordination, all shared state lives in **SQLite WAL databases** that multiple
processes access directly via filesystem. Model selection uses **library import**
of `RouterEngine` rather than HTTP calls.

### Why This Over HTTP

- **No server process to manage.** No systemd/launchd unit, no port binding, no
  health checks for the router itself. One fewer process that can go down.
- **SQLite WAL already solves the multi-process problem.** WAL mode gives concurrent
  readers with serialized writers. `BEGIN IMMEDIATE` transactions provide atomicity
  for check-then-act sequences (rate limiting, budget reservation). This is the
  same mechanism a single-process HTTP server would use internally, minus the
  HTTP layer.
- **Library import is simpler than HTTP for in-process selection.** The factory
  already has `dragonlight-router` as a dependency. Calling
  `router.select_for_task()` is a direct Python function call with zero
  serialization overhead, no network round-trips, and full type safety via the
  `ModelCandidate` dataclass.
- **State survives process restarts by default.** SQLite files persist on disk.
  No need for explicit state-save hooks on shutdown.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Factory Process (1..N concurrent)                      │
│                                                         │
│  1. from dragonlight_router.router import RouterEngine  │
│     router = RouterEngine(config)                       │
│     candidates = router.select_for_task(                │
│         intent_category="implementation",               │
│         complexity="standard",                          │
│     )                                                   │
│                                                         │
│  2. Build pydantic-ai Model from candidates[0]          │
│     Run Agent.run() with multi-turn tool-use loop       │
│     (direct API calls to provider)                      │
│                                                         │
│  3. router.record_request(outcome)                      │
│     → success/failure, tokens, latency, model_id        │
│                                                         │
│  4. If failed + retryable: try candidates[1], goto 2    │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│  SQLite WAL Databases (~/.dragonlight/router/)          │
│                                                         │
│  health.db   — retirement, suspension, error counts,    │
│                circuit breaker states                    │
│  budget.db   — request_log (RPM/RPD/TPM/daily tokens)   │
│  rate.db     — rate_slots (sliding-window RPM per key)  │
│                                                         │
│  All three use:                                         │
│  - PRAGMA journal_mode=WAL (concurrent readers)         │
│  - PRAGMA busy_timeout=5000 (writer contention)         │
│  - BEGIN IMMEDIATE (atomic check-then-act)              │
│  - Short-lived connections (open/close per operation)   │
│  - os.chmod(0o600) (owner-only file permissions)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### What's Already Built

Three SQLite WAL modules implement the shared-state layer. All follow the same
pattern and are operational:

**`health/health_db.py` — `HealthDB` class.**
Persistent retirement/suspension decisions, error counts, and circuit breaker
states. Replaces the in-memory health tracking that was lost on process restart
and invisible across concurrent factory runs.

| Table | Purpose |
|-------|---------|
| `retired_models` | Permanently retired models (e.g. 404 at inference) |
| `suspended_models` | Temporarily suspended models with TTL (e.g. 403 auth/budget) |
| `error_counts` | Per-model error counters with last-error timestamps |
| `breaker_states` | Circuit breaker state, opened_at, error_timestamps (JSON) |

Key operations: `retire_model()`, `is_retired()`, `reinstate_model()`,
`suspend_model()`, `is_suspended()`, `is_unavailable()`,
`save_breaker_state()`, `load_breaker_state()`, `prune_expired_suspensions()`.

**`rate/limiter.py` — `RateLimiter` class.**
Cross-process RPM coordination via sliding-window slot tracking. Rate limits are
enforced per API key (not per provider name), because providers rate-limit by key.
Solves the N-process multiplier problem: 3 concurrent factory runs sharing one
NIM key collectively cannot exceed the 60 RPM limit.

| Table | Purpose |
|-------|---------|
| `rate_slots` | Active rate slots with rate_key, provider, model_id, process_id, acquired_at, expires_at |

Key operations: `acquire(provider, model_id, api_key_env=)` returns a `RateSlot`
(granted/denied with retry_after_ms hint), `release(slot_id)`,
`get_usage(provider)` returns a `RateUsage` snapshot.

Crash safety: slots have a 30-second TTL. If a process crashes without calling
`release()`, the slot expires and is pruned on the next `acquire()` call. The
`BEGIN IMMEDIATE` transaction in `acquire()` prevents concurrent writers from
inserting between the count check and the insert.

**`budget/tracker.py` — `BudgetTracker` class (pre-existing, enhanced).**
Budget tracking with RPM, RPD, TPM, and daily token cap dimensions. The original
in-memory implementation was enhanced with a SQLite WAL backend (`db_path` param).
In shared mode, all instances coordinate through a single `budget.db`. The
`check_and_reserve()` method uses `BEGIN IMMEDIATE` for atomic check-then-record.

| Table | Purpose |
|-------|---------|
| `request_log` | All requests with provider, timestamp, tokens_used |

Key operations: `score(provider)` returns 0-100 budget availability,
`record_request(provider, tokens)`, `check_and_reserve(provider, tokens)` for
atomic budget reservation, `has_capacity(provider)`, `daily_spend_usd()`.

### Model Selection via Library Import

The factory imports `RouterEngine` and calls `select_for_task()` directly. This
method runs the full selection pipeline: health check, budget scoring,
CBR/IBR/MBR composite scoring, cascade ordering, tool-use filtering. It returns
a list of `ModelCandidate` dataclasses with connection configs.

```python
from dragonlight_router.router import RouterEngine
from dragonlight_router.core.types import ModelCandidate

router = RouterEngine(config)

candidates: list[ModelCandidate] = router.select_for_task(
    intent_category="implementation",
    complexity="standard",
    stakes="mid",
    requires_tool_use=True,
    context_tokens=12000,
    exclude_models=["nvidia_nim/qwen3-coder-480b"],
)

# Each ModelCandidate contains:
#   model_id: str          — e.g. "nvidia_nim/moonshotai/kimi-k2.6"
#   provider: str          — e.g. "nvidia_nim"
#   base_url: str          — e.g. "https://integrate.api.nvidia.com/v1"
#   api_key_env: str       — e.g. "NVIDIA_NIM_API_KEY" (env var name, not the key)
#   protocol: str          — "openai" | "anthropic"
#   health_score: float
#   composite_score: float
```

The `api_key_env` field references the environment variable name. API keys never
leave the process boundary. The factory reads `os.environ[candidate.api_key_env]`
to construct the pydantic-ai `Model` object.

### How Multi-Turn pydantic-ai Conversations Work

The factory's `run_coding_agent()` function creates a pydantic-ai `Agent` and
calls `agent.run()`. This internally loops:

1. Send messages to model via `model.request(messages, settings, params)`
2. Parse response for tool calls
3. Execute tools locally (file writes, shell commands)
4. Append tool results to messages
5. Send updated messages back to model
6. Repeat until model returns final text (no tool calls)

The `Model` object passed to `Agent(model=...)` is a standard pydantic-ai
`OpenAIChatModel` or `AnthropicModel` constructed from the `ModelCandidate`'s
connection config. There is no `RouterModel` adapter in the loop. The model
talks directly to the provider API.

The router's role is bookending: **select before, record after.**

```
Router: select_for_task() → candidates
  ↓
Factory: build pydantic-ai Model from candidates[0]
Factory: agent = Agent(model=model, tools=[...])
Factory: result = await agent.run(prompt)
  │  ↑↓ multi-turn tool-use loop (direct to provider API)
  ↓
Router: record_request(outcome)
  ↓
If failed + retryable:
  Factory: build pydantic-ai Model from candidates[1]
  (repeat)
```

### How Rate Limiting Works Across Concurrent Runs

Previous state: Each factory process creates its own `httpx.AsyncClient` with a
`ThrottleTransport` that rate-limits RPM per-process. Three concurrent runs hitting
NIM at 60 RPM each = 180 RPM actual, exceeding the provider limit.

Current state: All factory processes share `rate.db` via SQLite WAL. Before each
LLM API call, the factory calls `rate_limiter.acquire(provider, model_id,
api_key_env=key_env)`. The `RateLimiter` uses a `BEGIN IMMEDIATE` transaction to
atomically check the sliding-window count against the configured RPM limit and
insert a slot if under limit, or deny with a `retry_after_ms` hint if at/over
limit. After the API call completes, the factory calls `rate_limiter.release(slot_id)`.

If a process crashes without releasing, the slot auto-expires after 30 seconds
and is pruned on the next `acquire()` call.

### Concurrency Model

With SQLite WAL as the coordination layer:

- **Health tracking:** Shared `health.db`. A model retired by factory run A is
  immediately visible to factory run B on its next `is_unavailable()` check. Circuit
  breaker states persist across process restarts.

- **Budget tracking:** Shared `budget.db`. Token spend from all concurrent runs is
  aggregated via `request_log` table. `check_and_reserve()` uses `BEGIN IMMEDIATE`
  for cross-process atomicity.

- **Rate limiting:** Shared `rate.db`. Slot acquisition serializes via
  `BEGIN IMMEDIATE`. If NIM allows 60 RPM, 3 concurrent factory runs collectively
  cannot exceed 60 RPM.

- **Catalog:** Each `RouterEngine` instance has its own catalog state (model
  availability). Catalog refresh is per-process but reads from the same config.

### Migration Path (Actual)

**Completed:**

1. `health/health_db.py` — `HealthDB` class with retirement, suspension, error
   count, and breaker state tables. Follows WAL pattern.
2. `rate/limiter.py` — `RateLimiter` class with sliding-window RPM, per-key
   rate limiting, slot TTL crash safety. Follows WAL pattern.
3. `budget/tracker.py` — `BudgetTracker` enhanced with SQLite WAL backend
   (original in-memory mode preserved for tests/benchmarks).
4. `router.py` — `select_for_task()` method on `RouterEngine` returns
   `ModelCandidate` objects with full connection config.
5. `core/types.py` — `ModelCandidate` dataclass with model_id, provider,
   base_url, api_key_env, protocol, health_score, composite_score.

**Remaining (factory-side):**

1. Replace `_get_factory_router()` singleton with `RouterEngine` configured to
   use shared SQLite WAL databases at a well-known path (e.g. `~/.dragonlight/router/`).
2. Replace `_select_next_model()` and `_resolve_via_router()` with
   `router.select_for_task()`.
3. Replace `_create_model()` with model construction from `ModelCandidate` config.
4. Add `rate_limiter.acquire()`/`release()` around LLM API calls.
5. Remove `_FALLBACK_CHAIN`, `_MODEL_ROUTING`, `ROUTER_ENABLED`, `ROUTER_DISPATCH`.
6. Delete `scripts/router_model.py`.

---

## Future Direction: Transparent LLM Proxy

The emerging preferred pattern moves beyond library import to make the router a
**transparent LLM proxy** that exposes a standard OpenAI-compatible
`/v1/chat/completions` endpoint. Under this model, pydantic-ai connects to the
router as if it were any OpenAI-compatible provider. The router transparently
handles model selection, health tracking, fallback, and rate limiting behind the
scenes.

### Why This Is the End State

- **Zero factory-side routing logic.** The factory sets one URL
  (`DRAGONLIGHT_ROUTER_URL`) and one API key. No `select_for_task()`, no
  `ModelCandidate`, no model construction. pydantic-ai's `OpenAIChatModel` points
  at the router and the router figures out the rest.
- **Works with any pydantic-ai consumer.** Any tool that speaks the OpenAI protocol
  can use the router without importing `dragonlight-router` as a library. DAOS
  sessions, ad-hoc scripts, notebook agents all get smart routing for free.
- **Multi-turn conversations work naturally.** The router proxies each
  `.request()` call independently. Model selection can adapt mid-conversation
  (e.g., fall back to a different model on 429/500 without the consumer knowing).
- **Rate limiting is invisible.** The router queues or delays requests internally
  rather than returning deny/retry-after to the caller. The factory never needs
  to implement retry-on-rate-limit logic.

### Architecture (Transparent Proxy)

```
┌─────────────────────────────────────────────────────────┐
│  Factory Process (1..N concurrent)                      │
│                                                         │
│  model = OpenAIChatModel(                               │
│      "auto",  # or a role hint like "coding"            │
│      provider=OpenAIProvider(                           │
│          base_url="http://127.0.0.1:8100/v1",           │
│          api_key="local",                               │
│      ),                                                 │
│  )                                                      │
│  agent = Agent(model=model, tools=[...])                │
│  result = await agent.run(prompt)                       │
│  # That's it. No select, no record, no rate-slot.       │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Router Proxy (port 8100)                               │
│                                                         │
│  POST /v1/chat/completions                              │
│  1. Parse model field for role/intent hints             │
│  2. select_for_task() → ranked candidates               │
│  3. acquire rate slot for top candidate                 │
│  4. Forward request to actual provider API              │
│  5. On success: stream/return response, record outcome  │
│  6. On failure: try next candidate (transparent retry)  │
│  7. Release rate slot                                   │
│                                                         │
│  All backed by the same SQLite WAL databases:           │
│  health.db, budget.db, rate.db                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### What Needs to Be Built

| Component | Description |
|-----------|-------------|
| Proxy endpoint | `POST /v1/chat/completions` handler that accepts OpenAI-format requests, selects a backend, forwards the request, streams the response back |
| Model field routing | Parse the `model` field for routing hints (e.g. `"coding"`, `"testing"`, `"reasoning"`) and map to `select_for_task()` intent categories |
| Streaming passthrough | SSE streaming from provider to caller, with buffering for retry on mid-stream failure |
| Transparent fallback | On provider error (429, 500, timeout), retry with next candidate without the caller seeing the failure |
| Token counting | Extract token usage from provider response to feed into budget tracking |

### Open Questions (Proxy)

1. **How does the factory communicate intent/complexity to the proxy?** The
   OpenAI `/v1/chat/completions` spec has no field for intent category or task
   complexity. Options: encode in the `model` field (e.g. `"coding/complex"`),
   use a custom header (`X-Dragonlight-Intent`), or use extra_body parameters.

2. **How does the factory record outcome quality?** Quality ratings (pass/fail of
   generated code) come after the agent run completes, not during the API call.
   The proxy would need a separate outcome-recording endpoint or the factory
   would need to call `record_request()` via library import alongside the proxy.

3. **Streaming retry on mid-stream failure.** If the provider fails partway through
   a streamed response, the proxy has already sent partial data to the caller.
   Options: buffer the entire response before sending (defeats streaming latency),
   or accept that mid-stream failures surface to the caller (acceptable for
   current scale).

---

## Superseded: Option C — HTTP Select + Direct Execute + HTTP Record

> **Note:** This section documents the originally proposed HTTP service architecture.
> It was superseded by the SQLite WAL + library import approach described above.
> Retained as reference for the design reasoning that led to the current direction.

Option C proposed running the router as a persistent HTTP service with dedicated
endpoints for model selection, rate-slot management, and outcome recording. The
factory would communicate with the router exclusively over HTTP.

The core insight was correct: **pydantic-ai requires a `Model` object with a
`.request()` method for multi-turn tool-use agent loops**, so the factory must
call provider APIs directly for execution. The question was how to coordinate
selection and state across concurrent processes.

The HTTP approach would have required:
- A persistent router server process (systemd/launchd managed)
- New endpoints: `POST /v1/select-for-factory`, `POST /v1/acquire-rate-slot`,
  `POST /v1/release-rate-slot`
- An HTTP client module in the factory (`router_client.py`)
- Server availability as a hard dependency for the factory

The SQLite WAL approach achieves the same coordination guarantees (shared health,
budget, rate limiting across processes) without the operational overhead of running
and monitoring a persistent HTTP service. The tradeoff is that the factory must
import `dragonlight-router` as a Python library dependency, which the transparent
proxy future direction would eventually eliminate.

### Original HTTP Architecture (for reference)

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

### Original HTTP Endpoint Specs (for reference)

<details>
<summary>POST /v1/select-for-factory</summary>

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
    }
  ],
  "request_id": "sel_abc123"
}
```
</details>

<details>
<summary>POST /v1/acquire-rate-slot</summary>

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
</details>

<details>
<summary>POST /v1/release-rate-slot</summary>

**Request:**

```json
{
  "slot_id": "slot_xyz789",
  "actual_tokens": 3842,
  "success": true,
  "latency_ms": 4521.0
}
```
</details>

## What This Does Not Change

- **pydantic-ai agent loop** — Unchanged. `run_coding_agent()` still creates an
  `Agent` and calls `agent.run()`. The only difference is how the `Model` object is
  constructed (from router-provided config instead of hardcoded provider mapping).

- **Tool definitions and execution** — Unchanged. Tools are defined in the factory,
  executed in the factory, and never transit the router.

- **Non-factory router consumers** — The existing `/v1/select`, `/v1/dispatch`, and
  `/v1/record` HTTP endpoints are unchanged. DAOS sessions, ad-hoc queries, and
  other router consumers continue to use the existing HTTP API.

## Failure Modes

| Scenario | Behavior |
|----------|----------|
| SQLite busy (writer contention) | `busy_timeout=5000` — SQLite retries internally for up to 5 seconds before raising `OperationalError` |
| Rate slot denied | `acquire()` returns `RateSlot(granted=False, retry_after_ms=N)` — caller sleeps and retries |
| Rate slot not released (process crash) | Slot expires after 30s; pruned on next `acquire()` call |
| Database file corrupted | SQLite WAL has built-in crash recovery; WAL checkpoint restores consistent state |
| Process restarts | All state is on disk in SQLite; no warm-up needed. Rate slot window may briefly allow a burst (expired slots from crashed processes haven't been pruned yet) |

## Resolved Questions

1. **Should the factory fall back to static chain?** No. The router library is
   always available as a Python import. There is no "router down" scenario because
   there is no server — the SQLite databases are always accessible via filesystem.
   The static chain can be removed.

2. **Should rate-slot acquisition be blocking or async-with-retry?** The
   `RateLimiter.acquire()` method is synchronous and returns immediately with
   either a granted slot or a denied result with `retry_after_ms`. The caller
   decides whether to sleep-and-retry or try a different provider. This keeps the
   rate limiter simple and the retry policy in the factory's control.

3. **Should the router persist rate-slot state to disk?** Yes — it does. Rate
   slots are in SQLite WAL, which persists to disk. On process restart, the
   existing window is preserved. Expired slots are pruned lazily.
