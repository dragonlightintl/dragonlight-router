# Architecture

The Dragonlight Router is a multi-provider LLM routing engine that selects the best available model for each request, dispatches through provider adapters, and manages fallback cascading across 11 providers. It exposes a dual interface: import `RouterEngine` as a Python library, or run it as an HTTP sidecar via Starlette/Uvicorn.

## Subsystems

The router is composed of 11 subsystems, each with a single responsibility.

| Subsystem | Package | Responsibility |
|---|---|---|
| **RouterEngine** | `dragonlight_router.router` | Central orchestrator — wires all subsystems, exposes `select_models()` and `record_request()` |
| **RoleMatrix** | `dragonlight_router.roles` | Hot-reloadable JSON mapping from roles to ranked model ID lists |
| **BudgetTracker** | `dragonlight_router.budget` | Sliding-window RPM, RPD, and TPM tracking per provider; emits a 0-100 budget score |
| **HealthTracker** | `dragonlight_router.health` | Per-model error counts and EWMA latency; emits a 0-100 health score |
| **CircuitBreaker** | `dragonlight_router.health` | CLOSED/OPEN/HALF_OPEN state machine; prevents requests to consistently failing models |
| **CatalogCache** | `dragonlight_router.catalog` | File-backed TTL cache of live provider model lists |
| **CatalogRefresher** | `dragonlight_router.catalog` | Concurrent async fetch from each provider's `/v1/models` endpoint |
| **Server** | `dragonlight_router.server` | Starlette HTTP API with middleware stack (rate limiting, CORS, correlation IDs) |
| **SimpleCache** | `dragonlight_router.cache` | SHA-256 exact-match response cache backed by SQLite (WAL mode) |
| **SemanticCache** | `dragonlight_router.cache` | Character n-gram Jaccard similarity cache for near-duplicate prompt detection |
| **ComplexityEstimator** | `dragonlight_router.complexity` | Heuristic mapping from intent + context size to tier (LOCAL/HAIKU/SONNET/OPUS) |

## Cascade pipeline

Every dispatch request flows through a three-stage selection cascade before adapter dispatch:

```
                     DispatchOrder
                          |
                +---------v----------+
                |   MBR (Model-Based |
                |   Ranking)         |
                |   - Capability     |
                |     filtering      |
                |   - Tier matching  |
                |   - Status gating  |
                +---------+----------+
                          |
                 capable candidates
                          |
                +---------v----------+
                |   CBR (Cost-Based  |
                |   Ranking)         |
                |   - Budget score   |
                |   - Health score   |
                |   - Weighted       |
                |     composite      |
                +---------+----------+
                          |
                  scored candidates
                          |
                +---------v----------+
                |   LBR (Limit-Based |
                |   Ranking)         |
                |   - RPM/RPD/TPM    |
                |     capacity gate  |
                |   - Provider       |
                |     interleave     |
                |   - Weighted       |
                |     random select  |
                +---------+----------+
                          |
                   final candidate
                          |
                +---------v----------+
                |   Adapter Dispatch |
                |   - Context filter |
                |   - Provider call  |
                |   - Fallback on    |
                |     failure        |
                +---------+----------+
                          |
                    EngineResponse
```

**MBR** eliminates models that lack required capabilities (tool use, long context), are below the estimated complexity tier, or are retired/circuit-broken.

**CBR** scores surviving candidates on budget headroom and recent health using configurable weights. A cost governor dynamically shifts weights when budget pressure is high.

**LBR** enforces hard rate-limit capacity gates (RPM, RPD, TPM), interleaves across providers to prevent thundering-herd concentration, and uses weighted random selection for the final pick.

**Dispatch** calls the selected adapter, and on failure walks the fallback chain (remaining candidates in ranked order) until one succeeds or all are exhausted.

## Package structure

```
src/dragonlight_router/
  adapters/       Provider adapter implementations (11 providers + OpenAI-compat base)
  budget/         Sliding-window RPM/RPD/TPM tracking, budget scoring, disk persistence
  caching/        SHA-256 exact-match cache + n-gram Jaccard semantic cache
  catalog/        File-backed TTL provider model catalog + async catalog refresher
  config/         YAML config loader, provider/role config models
  core/           Shared types (frozen dataclasses, Result, BackendConfig), registry, errors
  dispatch/       Cascade composition — MBR+CBR+LBR pipeline + adapter call + fallback
  health/         Per-model health tracking (EWMA latency, error counts), circuit breaker
  roles/          Hot-reloadable JSON role-to-model matrix
  selection/      MBR, CBR, LBR stages, scoring functions, complexity estimator, context filter
  server/         Starlette HTTP app, route handlers, metrics collector, middleware
```

## Design decisions

**Result type pattern.** All fallible operations return `Result[T, E]` (an `Ok | Err` union) instead of raising exceptions. This makes error paths explicit in function signatures and prevents silent exception swallowing in the cascade. See [ADR-001](adr/001-result-type-pattern.md).

**Provider adapter pattern.** Every provider implements a `GenerativeBackend` protocol. Most inherit from an OpenAI-compatible base class that handles the common chat-completion wire format, with provider-specific overrides only where the API diverges. See [ADR-002](adr/002-provider-adapter-pattern.md).

**Three-stage cascade.** MBR/CBR/LBR separate capability filtering, cost scoring, and rate-limit enforcement into independent stages with clear contracts. Each stage reduces the candidate set, and each can be tested and tuned independently. See [ADR-003](adr/003-cascade-dispatch-design.md).

**Frozen dataclasses.** Core data types (`DispatchOrder`, `EngineResponse`, `BackendConfig`, `ScoredCandidate`) are frozen dataclasses — immutable after construction. This prevents accidental mutation as objects flow through the pipeline.

**Dual interface.** The `RouterEngine` class is a pure library with no HTTP dependency. The Starlette server is an optional thin wrapper. Applications can embed the engine directly or run it as a sidecar — same budget, health, and circuit-breaking logic either way.

## State management

| State | Storage | Update frequency |
|---|---|---|
| Budget counters (RPM/RPD/TPM per provider) | In-memory + periodic flush to `router_state/budget.json` | Every request |
| Health scores (error count, EWMA latency per model) | In-memory | Every recorded outcome |
| Circuit breaker states (CLOSED/OPEN/HALF_OPEN per model) | In-memory | On error threshold breach |
| Provider catalog (model lists per provider) | File-backed TTL cache at `router_state/catalog.json` | On refresh (default 24h TTL) |
| Role matrix (role-to-model mapping) | Hot-reloaded JSON at `router_state/model_role_matrix.json` | On file change |

All in-memory state is reconstructible from disk on restart. Budget counters flush at a configurable interval (default 5s). The catalog refresher runs on startup and on demand via the admin API.
