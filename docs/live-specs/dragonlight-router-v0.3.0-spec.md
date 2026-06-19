# Dragonlight Router v0.3.0 Live Spec

**Version:** 0.3.0  
**Package version:** 0.3.0  
**Effective:** 2026-06-18  
**Status:** Canonical (supersedes v0.2.0 spec + delta-spec cycle)

## 1. System Overview

The Dragonlight Router is a multi-provider LLM routing engine that intelligently selects and dispatches requests across 11 provider backends. It provides two interfaces:

- **select_models(role)** -- Factory-style: returns ranked model IDs for a role from the model-role matrix, scored by health, budget, and rank.
- **dispatch(order)** -- Engine-style: runs the full MBR->CBR->LBR cascade, selects a backend, calls the LLM adapter, and returns an EngineResponse with content, cost, and metadata.

**Key properties:** Thread-safe singleton RouterEngine, frozen configuration via Pydantic, Result[T, E] return types for all fallible operations, structured logging via structlog, async dispatch with streaming support.

### AC-SYS

- SYS-01: RouterEngine MUST provide both `select_models()` and `dispatch()` interfaces.
- SYS-02: RouterEngine MUST be a thread-safe singleton via `get_router()` with `_router_lock`.
- SYS-03: All configuration models MUST be frozen (`ConfigDict(frozen=True)`).

## 2. Architecture

### 2.1 Cascade Pipeline

```
DispatchOrder
    |
    v
[MBR] Capability filter + tier upgrade + health gate
    |
    v
[Trust Floor] Filter by context_trust_tier (HAZ-001)
    |
    v
[CBR] Budget filter + cost-aware scoring + cost governor
    |
    v
[LBR] Rate-limit filter + weighted random selection
    |
    v
[Dispatch] Fallback chain with context filtering + cache integration
    |
    v
EngineResponse / StreamChunk[]
```

### 2.2 Subsystem Table

| Subsystem | Package | Purpose |
|-----------|---------|---------|
| RouterEngine | `router.py` | Orchestrator, singleton, dual interface |
| Cascade Dispatch | `dispatch/cascade.py` | MBR->CBR->LBR composition, fallback chain |
| MBR | `selection/mbr.py` | Capability tier filtering with upgrade |
| CBR | `selection/cbr.py` | Budget filtering and cost-aware scoring |
| LBR | `selection/lbr.py` | Rate-limit filtering, weighted random selection |
| Scoring | `selection/scoring.py` | 5-dimension composite scoring with cost governor |
| Context Filter | `selection/context_filter.py` | Trust-tier-based context redaction |
| Health Tracker | `health/tracker.py` | Per-model health scoring, retirement |
| Circuit Breaker | `health/circuit_breaker.py` | CLOSED->OPEN->HALF_OPEN state machine |
| Health Check Loop | `health/check_loop.py` | 30s background probe, SLO enforcement |
| Budget Tracker | `budget/tracker.py` | Sliding-window RPM/TPM, daily RPD/token cap |
| Provider Adapters | `adapters/` | 11 provider backends, OpenAI-compatible base |
| Caching | `caching/` | SimpleCache (exact), SemanticCache (Jaccard) |
| Catalog | `catalog/` | Live model catalog with TTL refresh |
| Role Matrix | `roles/matrix.py` | Model-role mapping from JSON |
| Config | `config/` | YAML loader, Pydantic schema |
| Registry | `core/registry.py` | BackendConfig + BackendState storage |
| SSRF Validation | `core/validation.py` | URL validation for SSRF prevention |
| HTTP Server | `server/` | Starlette routes, middleware, metrics |

## 3. Provider Adapters

### 3.1 Supported Providers (11)

| Provider | Adapter Key | Base Class | Notes |
|----------|-------------|------------|-------|
| Anthropic | `anthropic` | Custom | Native API, x-api-key header |
| Cerebras | `cerebras` | OpenAICompatibleBackend | |
| Cohere | `cohere` | Custom | v2 API, Bearer auth |
| Google/Gemini | `google` | Custom | x-goog-api-key header |
| Groq | `groq` | OpenAICompatibleBackend | |
| Local/Ollama | `local` | Custom | http://localhost, no auth |
| Mistral | `mistral` | OpenAICompatibleBackend | |
| NVIDIA NIM | `nvidia` | OpenAICompatibleBackend | |
| OpenAI | `openai` | OpenAICompatibleBackend | |
| OpenRouter | `openrouter` | OpenAICompatibleBackend | |
| Together | `together` | OpenAICompatibleBackend | |

### 3.2 OpenAI-Compatible Base Class

`_openai_compat.py` provides `OpenAICompatibleBackend` -- 7 of 11 adapters inherit from it. Subclasses override only `_completions_path`, `_auth_header_key`, or streaming parse behavior. Base class handles: httpx SSE streaming, auth header injection, retry with exponential backoff + jitter, status tracking, usage recording.

### 3.3 GenerativeBackend Protocol

All adapters implement the `GenerativeBackend` protocol:
- `config: BackendConfig` (property)
- `status: BackendStatus` (property)
- `generate(messages, *, max_tokens, temperature, stream) -> AsyncIterator[str]`
- `health_check() -> bool`
- `record_usage(tokens_in, tokens_out) -> None`

### AC-ADAPT

- ADAPT-01: All 11 providers MUST have a registered adapter in `_PROVIDER_MAP`.
- ADAPT-02: `create_adapter(config)` MUST return a `GenerativeBackend`-conforming instance.
- ADAPT-03: Adapters MUST yield string chunks via async iteration in streaming mode.
- ADAPT-04: Fresh adapter instances MUST start with `BackendStatus.AVAILABLE` (HAZ-014).

## 4. Selection Pipeline

### 4.1 MBR -- Model-Based Ranking

**Module:** `selection/mbr.py`

1. **Estimate complexity** from DispatchOrder: context_tokens, requires_tool_use, requires_long_context, and intent_category floor (HAZ-013).
2. **Tier assignment:** LOCAL < SIMPLE < MODERATE < COMPLEX.
3. **Filter by tier:** Get candidates from registry at estimated tier.
4. **Capability filter:** Check max_context_tokens, supports_tool_use, supports_system_prompts.
5. **Health filter:** Exclude CIRCUIT_OPEN and KEY_INVALID backends. LOCAL backends bypass health checks.
6. **Tier upgrade:** If no candidates at estimated tier, try one tier above (never downgrade -- enforced via `invariant()`).

**Intent tier floors (HAZ-013):** complex_reasoning/strategic_planning/architecture -> COMPLEX; engineering_build/code_review/debugging/spec_writing/code_generation -> MODERATE; data_analysis/summarization -> SIMPLE.

### AC-MBR

- MBR-01: MBR MUST exclude backends with status `CIRCUIT_OPEN` from candidate list.
- MBR-02: MBR MUST exclude backends with status `KEY_INVALID` from candidate list.
- MBR-03: MBR MUST upgrade one tier (never downgrade) when no candidates at estimated tier.
- MBR-04: LOCAL-tier backends MUST bypass all health and rate-limit checks.
- MBR-05: The no-downgrade invariant MUST be enforced via `invariant()` on every return path.
- MBR-06: Intent category floors (HAZ-013) MUST be applied, only raising tier, never lowering.

### 4.2 CBR -- Cost-Based Ranking

**Module:** `selection/cbr.py`, `selection/scoring.py`

1. **Hard budget filter:** Exclude providers with budget score of 0.0 (fully exhausted).
2. **Scoring:** Each candidate scored via `score_candidate()` using `ScoringWeightsConfig`:
   - Default weights: cost=0.35, latency=0.25, priority=0.20, queue=0.10, health=0.10 (sum=1.0).
   - All dimensions normalized to [0.0, 1.0] before weighting.
3. **Cost governor:** When `daily_spend >= cost_down_threshold_daily` (default 100 USD) or `monthly_spend >= cost_down_threshold_monthly` (default 1000 USD), weights shift to: cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05.
4. **DEGRADED penalty:** Backends with `BackendStatus.DEGRADED` receive a 0.5x score multiplier.
5. **Sort:** Candidates ranked by composite score descending.

### AC-CBR

- CBR-01: ScoringWeightsConfig weights MUST sum to 1.0 (enforced via `__post_init__` assertion).
- CBR-02: Cost governor MUST activate when daily OR monthly spend exceeds thresholds.
- CBR-03: Cost governor MUST shift cost weight to 0.70.
- CBR-04: DEGRADED backends MUST receive 0.5x score penalty in cascade scoring.
- CBR-05: All normalized scores MUST be in [0.0, 1.0] range (assertion-enforced).

### 4.3 LBR -- Load-Based Ranking

**Module:** `selection/lbr.py`

1. **Hard capacity gate (HAZ-005):** Remove providers with zero remaining capacity (RPM, RPD, TPM, or daily token cap exhausted). LOCAL backends bypass.
2. **Median-score filter:** Compute per-provider budget scores, retain candidates at or above median.
3. **Weighted random selection:** `select_final_candidate()` uses `random.choices()` with scores as weights. Minimum score floor of 0.01 prevents zero-weight candidates.

### AC-LBR

- LBR-01: Providers with zero capacity MUST be excluded by the hard gate.
- LBR-02: LOCAL backends MUST bypass both capacity gate and median filter.
- LBR-03: Final selection MUST use weighted random (not deterministic top-1).
- LBR-04: Score floor of 0.01 MUST prevent complete exclusion of low-score candidates.

## 5. Dispatch Cascade

**Module:** `dispatch/cascade.py`

### 5.1 Pipeline

1. **Cache check:** Look up response in SimpleCache (SHA-256 key from model_id + system_prompt + messages + temperature + max_tokens). Return cached EngineResponse on hit.
2. **Run cascade:** MBR -> trust floor -> CBR -> LBR. Returns ranked `list[ScoredCandidate]`.
3. **Apply fallback policy (HAZ-004):** `"allow"` (default) = all candidates eligible; `"deny"` = primary only; `"same_tier"` = same BackendTier as primary.
4. **Context filtering:** Build base context from DispatchOrder. Per-candidate: map BackendTier to ProviderTrustTier, filter context fields accordingly.
5. **Adapter dispatch:** Create fresh adapter per attempt (HAZ-014). Generate via async streaming. Estimate tokens (chars/4 heuristic, HAZ-010). Record success/failure in health + budget trackers.
6. **Fallback:** On adapter exception (RuntimeError, ValueError, ConnectionError, OSError, TypeError), record failure, add to fallback chain, try next candidate. On exhaustion, return DispatchFailure.
7. **Cache store:** On success, store response in SimpleCache.

### 5.2 Streaming Dispatch

`dispatch_stream()` yields `StreamChunk` objects with event_type:
- `"token"` -- content token from the LLM
- `"metadata"` -- final metadata (backend_used, tokens, cost, latency, fallback info)
- `"error"` -- generation failure

SSE formatting via `_format_stream_chunk()` produces `data: {json}\n\n` lines.

### 5.3 Context Trust Tiers

| BackendTier | ProviderTrustTier | Context Filtering |
|-------------|-------------------|-------------------|
| LOCAL | LOCAL | Full context, no egress risk |
| COMPLEX | TRUSTED | Full context |
| MODERATE | SEMI_TRUSTED | Remove behavioral_rules, redact persona, limit history to 3 turns |
| SIMPLE | SEMI_TRUSTED | Same as MODERATE |

DispatchOrder.context_trust_tier sets a minimum floor -- candidates below are excluded (HAZ-001).

### AC-DISPATCH

- DISP-01: Cached responses MUST be returned without running the cascade.
- DISP-02: Fresh adapter MUST be created per dispatch attempt (HAZ-014).
- DISP-03: Fallback policy MUST restrict eligible candidates before iteration.
- DISP-04: Token estimation MUST use chars/4 heuristic until tokenizer is available (HAZ-010).
- DISP-05: SEMI_TRUSTED providers MUST NOT receive behavioral_rules or persona details.
- DISP-06: UNTRUSTED providers MUST receive task instruction only.

## 6. Health System

### 6.1 Health Tracker

**Module:** `health/tracker.py`

Per-model scoring (0-100): retired=0, circuit_open=0, 3+ errors=30, 1-2 errors=70, 0 errors=100.

- `record_success()` -- resets errors, updates EMA latency (alpha=0.2).
- `record_error()` -- increments error count, feeds circuit breaker. HTTP 404 triggers immediate retirement.
- `availability_status()` -- "healthy" / "degraded" / "unavailable" based on available model ratio.
- State persistence via `get_state()` / `restore_state()` (HAZ-003/HAZ-012).

### 6.2 Circuit Breaker

**Module:** `health/circuit_breaker.py`

State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED (or back to OPEN).

- **Trip:** 3 errors within 120s window -> OPEN.
- **Cooldown:** 60s base + jittered offset (0 to 25% of cooldown). HAZ-009 mitigation: prevents synchronized recovery across breakers.
- **Probe:** After cooldown, transition to HALF_OPEN (allow 1 request). Success -> CLOSED. Failure -> re-OPEN with fresh jitter.
- State persistence via `get_state()` / `restore_state()` (HAZ-012).

### 6.3 Health Check Loop

**Module:** `health/check_loop.py`

- 30s interval background probe of all backends.
- SLO enforcement: 3 consecutive SLO violations (>5000ms latency) -> DEGRADED status.
- DEGRADED backends receive 0.5x score penalty in cascade (CBR stage).
- Failures do not crash the loop.
- Optional `on_cycle` callback (HAZ-008): fires every N cycles for catalog refresh (~1 hour).

### AC-HEALTH

- HEALTH-01: Circuit breaker MUST trip after 3 errors within 120s.
- HEALTH-02: Cooldown MUST include jitter (HAZ-009) to prevent synchronized recovery.
- HEALTH-03: HTTP 404 at inference time MUST trigger immediate model retirement.
- HEALTH-04: Health and circuit breaker state MUST survive process restart (HAZ-003/HAZ-012).
- HEALTH-05: Health check loop MUST NOT crash on individual probe failures.
- HEALTH-06: 3 consecutive SLO violations MUST transition backend to DEGRADED.

## 7. Budget System

**Module:** `budget/tracker.py`

### 7.1 Tracking Dimensions

| Dimension | Window | Mechanism |
|-----------|--------|-----------|
| RPM | 60s sliding | deque of timestamps, prune on read |
| RPD | UTC day | Counter with midnight reset |
| TPM | 60s sliding | deque of (timestamp, token_count) tuples |
| Daily token cap | UTC day | Counter with midnight reset |

### 7.2 Scoring

`score()` returns 0-100: minimum of all four dimension ratios * 100. Unknown providers return 100.0 (open access).

### 7.3 Spend Tracking

- `daily_spend_usd(provider, avg_cost_per_token)` -- tokens_today * cost.
- `monthly_spend_usd(provider, avg_cost_per_token)` -- daily * 30 approximation.
- Cost profiles (`BackendCostProfile`) specify USD per million tokens for input/output.
- Per-model exact match -> provider-level default -> zero fallback.

### 7.4 Concurrency Safety

`check_and_reserve()` uses `asyncio.Lock` for atomic check-then-record (HAZ-002).

### 7.5 State Persistence

Daily counters (RPD + daily token) persisted via `get_state()` / `restore_state()` (HAZ-012). Sliding windows (RPM/TPM) are NOT restored (stale on restart).

### AC-BUDGET

- BUDGET-01: RPM sliding window MUST prune entries older than 60s.
- BUDGET-02: Daily counters MUST reset at UTC midnight.
- BUDGET-03: `has_capacity()` MUST check all four dimensions.
- BUDGET-04: Budget state MUST survive process restart for daily counters (HAZ-012).
- BUDGET-05: `check_and_reserve()` MUST be atomic under asyncio.Lock (HAZ-002).

## 8. Caching

### 8.1 SimpleCache

**Module:** `caching/simple.py`

SQLite-backed, SHA-256 keyed, TTL-based (default 3600s). Max 1000 entries with LRU eviction. Cache key = SHA-256 of (model_id, system_prompt, messages, temperature, max_tokens). Integrated into dispatch pipeline: check before cascade, store on success.

### 8.2 SemanticCache

**Module:** `caching/semantic.py`

Character n-gram (size=3) similarity using Jaccard coefficient. Default threshold 0.95. Max 500 entries. Stores precomputed n-gram sets as JSON. Available but not wired into the dispatch pipeline (opt-in).

### AC-CACHE

- CACHE-01: SimpleCache key MUST be deterministic SHA-256 of request parameters.
- CACHE-02: Expired entries MUST be deleted on read.
- CACHE-03: Entry count MUST NOT exceed max_entries (evict oldest).
- CACHE-04: SemanticCache similarity MUST use Jaccard coefficient on character n-grams.

## 9. Security

### 9.1 SSRF Prevention

**Module:** `core/validation.py`

`validate_provider_url()` rejects: private IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x), cloud metadata endpoints (169.254.169.254, metadata.google.internal), non-https schemes (http allowed only for localhost/Ollama). Wired into catalog refresh path. [KNOWN GAP: not yet wired into adapter dispatch path -- SEC-003.]

### 9.2 CORS

**Module:** `server/middleware.py`

Controlled via environment variables:
- `DRAGONLIGHT_CORS_ORIGINS` -- comma-separated origins (empty = CORS disabled).
- `DRAGONLIGHT_CORS_METHODS` -- default `GET,POST,OPTIONS`.
- `DRAGONLIGHT_CORS_HEADERS` -- default `*`.
- `DRAGONLIGHT_CORS_CREDENTIALS` -- default `false`.

No wildcard default. CORS disabled entirely when no origins configured.

### 9.3 Rate Limiting

Token-bucket middleware: 60 requests/minute per client IP. Returns 429 on exceeded.

### 9.4 Admin Auth

Bearer token auth on admin endpoints (`/v1/retire`, `/v1/reinstate`, `/v1/catalog/refresh`). Configured via `admin_api_key` in RouterConfig. Rate limiting on auth failures: 5 failures in 60s from same IP -> 429. [KNOWN GAP: defaults to None/open access -- SEC-006.]

### 9.5 Input/Output Validation

- **Prompt sanitization:** Strips null bytes and control chars (preserves \n, \r, \t), truncates at 100K chars.
- **Output validation:** Verifies non-empty string, strips null bytes, truncates at 500K chars.
- **Intent category whitelist (HAZ-007):** 17 allowed values; rejects unknown intent categories.
- **Fallback policy whitelist (HAZ-004):** `allow`, `deny`, `same_tier`.

### 9.6 Container Hardening

Dockerfile defaults to `127.0.0.1`. docker-compose.yml: `cap_drop: ALL`, `no-new-privileges`, `read_only` root filesystem, `tmpfs /tmp`.

### AC-SEC

- SEC-01: CORS MUST be disabled by default (no wildcard).
- SEC-02: SSRF validation MUST reject private IPs and cloud metadata endpoints.
- SEC-03: Admin endpoints MUST require Bearer token when `admin_api_key` is configured.
- SEC-04: Admin auth failures MUST be rate-limited (5 failures / 60s -> 429).
- SEC-05: Prompt input MUST be sanitized before LLM dispatch.
- SEC-06: LLM output MUST be validated before returning to client.
- SEC-07: Intent category MUST be validated against allowed set (HAZ-007).

## 10. HTTP API

**Framework:** Starlette + Uvicorn

### 10.1 Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/select` | None | Ranked model selection by role |
| POST | `/v1/dispatch` | None | Cascade dispatch (JSON or SSE stream) |
| POST | `/v1/record` | None | Record request outcome |
| GET | `/v1/health` | None | Liveness probe + health/budget snapshot |
| GET | `/v1/ready` | None | Readiness probe (catalog populated?) |
| GET | `/v1/catalog` | None | Catalog status |
| POST | `/v1/catalog/refresh` | Admin | Trigger catalog refresh |
| POST | `/v1/retire` | Admin | Retire a backend |
| POST | `/v1/reinstate` | Admin | Reinstate a retired backend |
| GET | `/metrics` | None | JSON metrics (uptime, per-endpoint stats) |
| GET | `/openapi.json` | None | OpenAPI 3.0.3 schema |

### 10.2 Middleware Stack

1. **RateLimitMiddleware** -- 60 req/min per IP, token bucket.
2. **RequestCorrelationMiddleware** -- X-Request-ID injection, structured request logging, optional metrics recording.
3. **CORSMiddleware** (conditional) -- only added when `DRAGONLIGHT_CORS_ORIGINS` is set.

### AC-API

- API-01: All POST endpoints MUST validate request body fields.
- API-02: `/v1/dispatch` with `stream: true` MUST return `text/event-stream` SSE.
- API-03: `/v1/health` MUST include `status`, `version`, `key_invalid_count`, `budget`, `health`.
- API-04: `/v1/ready` MUST return 503 when catalog has not been refreshed.
- API-05: Admin endpoints MUST return 401 when auth fails, 429 when rate-limited.
- API-06: All responses MUST include `X-Request-ID` header.

## 11. Configuration

### 11.1 router.yaml Schema

```yaml
state_dir: ./router_state           # State persistence directory
catalog_ttl_hours: 24               # Catalog cache TTL
budget_flush_interval_s: 5          # Budget state flush interval
default_top_n: 12                   # Default model count for select
max_consecutive_same_provider: 2    # Provider interleaving limit
admin_api_key: null                 # Bearer token for admin endpoints

providers:
  - name: nvidia_nim
    base_url: https://integrate.api.nvidia.com/v1
    catalog_url: https://integrate.api.nvidia.com/v1/models
    env_key: NVIDIA_NIM_API_KEY
    model_prefix: "nvidia_nim/"
    rate_limits:
      rpm: 40
      rpd: null         # null = unlimited
      tpm: null
      daily_token_cap: null
```

### 11.2 model_role_matrix.json

Maps roles (e.g., "code_generation", "architecture") to ranked lists of model IDs with provider prefix (e.g., "groq/llama-3.3-70b-versatile"). Loaded at boot, reloaded on file change via `RoleMatrix.reload_if_changed()`.

### 11.3 Environment Variables

| Variable | Purpose |
|----------|---------|
| `{PROVIDER}_API_KEY` | Provider auth (per env_key in config) |
| `DRAGONLIGHT_CORS_ORIGINS` | CORS allowed origins |
| `DRAGONLIGHT_CORS_METHODS` | CORS allowed methods |
| `DRAGONLIGHT_CORS_HEADERS` | CORS allowed headers |
| `DRAGONLIGHT_CORS_CREDENTIALS` | CORS credentials flag |
| `DRAGONLIGHT_HOST` | Server bind address (default 127.0.0.1) |

## 12. Testing

### 12.1 Test Pyramid

| Layer | Location | Count | Description |
|-------|----------|-------|-------------|
| Unit | `tests/unit/` | ~800+ | Per-module isolation |
| Integration | `tests/integration/` | ~100+ | Cross-module cascade E2E |
| Acceptance | `tests/acceptance/` | ~50+ | User-story-level validation |
| Property-based | `tests/unit/test_properties.py` | 23 | Hypothesis @given tests |
| Smoke | `tests/smoke/` | Stub | Live provider connectivity |
| Contracts | `tests/contracts/` | Stub | Consumer-driven contracts |

**Total:** 1079 tests passing. Coverage: 99.17%.

### 12.2 Tooling

- **pytest** with pytest-asyncio, pytest-cov, pytest-timeout (60s signal).
- **Hypothesis** for property-based testing of scoring, budget, MBR, LBR, interleave invariants.
- **ruff** for linting (E, F, I, UP, B, C4, SIM, D100, D101).
- **mypy** strict mode.
- **bandit** SAST (B101 skipped -- assertions are preconditions).

### AC-TEST

- TEST-01: All tests MUST pass (`pytest --timeout=60`).
- TEST-02: Coverage MUST be >= 80% (currently 99.17%).
- TEST-03: Property-based tests MUST cover scoring, budget, MBR, and LBR invariants.
- TEST-04: Ruff MUST report zero violations.

## 13. Deployment

### 13.1 Docker

- `Dockerfile` with `DRAGONLIGHT_HOST=127.0.0.1` default.
- `docker-compose.yml` with security hardening: `cap_drop: ALL`, `no-new-privileges`, `read_only`, `tmpfs /tmp`.
- `requirements-hashed.txt` with `--require-hashes` verification.

### 13.2 Dependencies

Pinned to exact versions in `pyproject.toml`:
- Core: pydantic==2.12.5, pyyaml==6.0.3, httpx==0.28.1, structlog==25.5.0
- Server: starlette==1.0.0, uvicorn==0.44.0
- Adapters: openai==2.32.0
- Cache: aiosqlite==0.22.1
- Dev: pytest==9.0.3, hypothesis==6.152.1, mypy==1.20.2, ruff==0.15.12, bandit==1.9.4

### 13.3 Makefile

Standard targets: `make test`, `make lint`, `make typecheck`, `make security`, `make all`.

### AC-DEPLOY

- DEPLOY-01: All dependencies MUST be pinned to exact versions.
- DEPLOY-02: Dockerfile MUST bind to 127.0.0.1 by default.
- DEPLOY-03: docker-compose MUST drop all capabilities and enable read-only filesystem.

## 14. Quality Standards

### 14.1 Coding Standards Compliance

- **Function length:** 40-line hard limit. DEVIATION records with 2026-09-01 expiry for 22 functions that exceed the limit due to async generators, linear flow, or public API contracts.
- **Assertion density:** >= 2 assertions per non-trivial function.
- **Nesting depth:** <= 3 levels.
- **Parameter count:** <= 4 (frozen dataclasses used for grouping: DispatchContext, CostFilterParams, ScoringContext, HealthCheckConfig, CacheKeyParams).
- **Exception handling:** Specific exception types only. `except Exception` banned (1 deviation at I/O boundary with DEVIATION record).
- **Type annotations:** Complete on all function signatures.
- **Configuration:** All config models frozen.
- **Logging:** structlog throughout (no stdlib logging).
- **Imports:** Top-level only (module-reference pattern where mock patching requires it).

### 14.2 DEVIATION Records

Functions exceeding standards limits carry inline DEVIATION comments with:
- Deviation ID (e.g., CS-004)
- Justification
- Architect approval
- Scope limitation
- Expiration date (2026-09-01)

### 14.3 FMEA Hazard Register

14 hazards tracked in `docs/hazard-register.md`:
- HAZ-001: Data exposure via context leakage (trust tiers)
- HAZ-002: Budget race condition (asyncio.Lock)
- HAZ-003: Health state loss on restart (persistence)
- HAZ-004: Uncontrolled fallback (fallback_policy)
- HAZ-005: Rate limit breach (hard capacity gate)
- HAZ-007: Intent injection (whitelist validation)
- HAZ-008: Stale catalog (periodic refresh)
- HAZ-009: Synchronized circuit breaker recovery (jittered cooldown)
- HAZ-010: Token count inaccuracy (chars/4 estimation + logging)
- HAZ-011: Unauthorized admin access (bearer auth)
- HAZ-012: Budget state loss on restart (persistence)
- HAZ-013: Under-capable model routing (intent tier floors)
- HAZ-014: Concurrent adapter state mutation (fresh instances)

## 15. Known Gaps and Carried-Forward Items

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| SEC-006 | High | Admin API key defaults to None (open access) | Open -- intentional for local dev |
| TS-001 | High | PBT coverage gaps in cascade dispatch and health modules | Partially resolved |
| TS-002 | High | Consumer-driven contract tests missing (contracts/ stub) | Partially resolved |
| SEC-003 | High | SSRF validation not wired into adapter dispatch path | Partially resolved |

These items are tracked in the active delta-spec (`docs/live-specs/delta-spec.jsonl`).
