# Dragonlight Router — Multi-Provider LLM Routing Engine

**Spec ID:** `dragonlight-router-v0.2.0`  
**Version:** 0.2.0  
**Status:** review  
**Completeness:** 0.92  
**Aliases:** Cascade Router, cascade router, dragonlight router, load balancing router, cost based router, multi-provider router  
**Created:** 2026-05-30T21:05:37.332174Z  

> Intelligent model selection across 8 providers. Given a role (logical task type), the router consults a hot-reloadable role-to-model matrix, filters against live provider catalogs, scores candidates on budget headroom and recent health, interleaves across providers, and returns a ranked list of model IDs. Also implements the canonical MBR→CBR→LBR three-function cascade for engine-style dispatch consumers.

---

## 1. Behavioral Contract

### 1.1 Operations

#### `select_models`

Return ranked model IDs for a role. Factory's primary entry point. Reads role matrix, refreshes catalog if stale, filters by live catalog, scores by budget/health/rank, interleaves providers, returns top_n.

**Preconditions:**
- role is non-empty string
- role matrix file exists (empty dict if missing, returns [])
- budget tracker initialized with provider configs

**Postconditions:**
- result is sorted by composite score descending
- no provider appears more than max_consecutive_same_provider times consecutively
- result length <= top_n
- models from fetched providers are present in live catalog
- excluded providers do not appear in result
- composite score = rank*0.6 + budget_score*0.25 + health_score*0.15

**Side Effects:**
- triggers catalog refresh if stale (async HTTP calls to providers)
- triggers role matrix hot-reload if file mtime changed

#### `record_request`

Record request outcome for budget and health tracking. Feeds the feedback loop that keeps scores accurate.

**Preconditions:**
- provider name is valid (unknown providers tracked with default-unlimited budget)
- model_id is non-empty string

**Postconditions:**
- budget RPM window updated with current timestamp
- budget RPD counter incremented
- health tracker: success resets error count, failure increments it
- health tracker: success updates EMA latency (alpha=0.2)
- health tracker: failure with http_status=404 triggers model retirement
- health tracker: 3+ consecutive errors trips circuit breaker (60s cooldown)

**Side Effects:**
- modifies in-memory budget windows (deque append)
- modifies in-memory health state (error counts, circuit breaker state)

#### `estimate_complexity`

Map a DispatchOrder to a required backend tier using heuristic signals. This is the MBR (Model Based Router) input — it determines the minimum capability tier.

**Preconditions:**
- dispatch_order has valid intent_category
- dispatch_order.context_tokens >= 0

**Postconditions:**
- returns ComplexityEstimate with tier in {LOCAL, HAIKU, SONNET, OPUS}
- OPUS intents: session_lifecycle, strategic_planning, complex_reasoning
- SONNET intents: engineering_build, code_review, architecture, debugging, spec_writing
- tool use requires minimum SONNET
- large context (>=8000 tokens) requires minimum SONNET
- short message + low context stays LOCAL
- confidence is 0.7-0.9 depending on signal strength
- signals list explains the reasoning chain

#### `interleave_providers`

Reorder scored models so no provider appears max_consecutive+1 times in a row. Preserves score ordering where possible.

**Preconditions:**
- scored_models is sorted by composite score descending
- max_consecutive >= 1

**Postconditions:**
- no provider appears more than max_consecutive times consecutively (unless only one provider exists)
- relative score ordering preserved where constraint allows
- total model count unchanged

#### `dispatch`

Full cascade dispatch for engine consumers. Takes a DispatchOrder, runs MBR→CBR→LBR, executes dispatch via selected backend adapter, returns EngineResponse or DispatchFailure.

**Preconditions:**
- dispatch_order is a valid frozen DispatchOrder
- at least one backend is registered in BackendRegistry
- circuit breaker is not open for all candidates

**Postconditions:**
- returns EngineResponse on success with cost and latency telemetry
- returns DispatchFailure only when ALL backends exhausted
- fallback chain tracks attempted backends
- budget and health state updated for the backend used
- dispatch log and budget update are transactional

**Side Effects:**
- HTTP dispatch to selected provider
- circuit breaker state transitions
- budget consumption recording
- dispatch log write

### 1.2 Inputs

| Name | Type | Constraints | Source |
|------|------|-------------|--------|
| `role` | `str` | non-empty, must exist in role matrix or returns empty list | operator or application request |
| `top_n` | `int` | 1-100, default from config (12) | application request |
| `exclude_providers` | `frozenset[str] | None` | provider names to skip, None = no exclusions | application request |
| `dispatch_order` | `DispatchOrder` | frozen dataclass with intent_category, context_tokens, etc. | engine pipeline (Core → Engine boundary) |

### 1.3 Edge Cases

| Scenario | Expected Behavior | Narrative |
|----------|-------------------|-----------|
| role not found in role matrix | return empty list (not error) | the application asks for a role that hasn't been configured yet |
| all models for a role are in circuit-open state | return empty list (circuit-open models score 0 and may still appear if no alternatives exist) | all providers are having a bad day simultaneously |
| catalog is stale and refresh fails | proceed with last-known-good catalog; log warning; models from unfetched providers pass through unfiltered | network partition during catalog refresh — degrade gracefully |
| only one provider configured | interleaving returns models as-is (no constraint possible); all traffic goes to that provider | minimal deployment with a single provider |
| budget exhausted for all providers | all providers score budget=0; ranked lower but not excluded (composite score still has rank and health components); if dispatch path: returns BudgetExceededError | the operator has spent their daily budget |
| model returns 404 at inference time | HealthTracker retires the model immediately; model is excluded from future selections until reinstated or catalog refresh removes it | provider decommissions a model without warning |
| role matrix file deleted at runtime | hot-reload reads missing file as empty dict; select_models returns empty list; no crash | someone accidentally deletes the matrix file |
| concurrent select_models calls while catalog is refreshing | select_models uses last-known-good catalog; refresh runs in background; no blocking or deadlock | high-concurrency scenario with overlapping refreshes |

### 1.4 Error States

| Trigger | Behavior | Severity | Recovery |
|---------|----------|----------|----------|
| invalid YAML configuration | raise exception from config loader with path and error details; fail fast at boot | fatal | fix config file and restart |
| provider catalog endpoint unreachable | log warning, skip provider, continue with remaining providers; partial results returned | degraded | automatic — next refresh cycle retries |
| budget state file corrupt on disk | load_budget_state returns None; fresh start with zero counters | minor | automatic — starts fresh, no crash |
| all backends fail during dispatch | return DispatchFailure with attempted_backends list and per-backend error details | critical | operator intervention required; circuit breakers will open for failing backends |
| circuit breaker opens for a backend | backend excluded from routing for 60s; after cooldown, HALF_OPEN allows 1 probe; success → CLOSED, failure → re-OPEN | degraded | automatic after cooldown period |

### 1.5 Invariants (11)

**INV-01:** select_models never raises an exception for valid input — it returns empty list instead

**INV-02:** composite score is deterministic given same inputs (rank, budget_score, health_score)

**INV-03:** no provider appears more than max_consecutive_same_provider times consecutively (unless single-provider)

**INV-04:** circuit breaker cooldown is always respected — no request sent to circuit-open backend within cooldown window

**INV-05:** PII never crosses the Core→Engine boundary (pre-redacted by CAL); router never sees unredacted content

**INV-06:** dispatch log write and budget update are transactional — both succeed or both roll back

**INV-07:** role matrix hot-reload is atomic — no partial reads

**INV-08:** catalog cache writes are atomic (.tmp → rename pattern)

**INV-09:** budget persistence writes are atomic (.tmp → rename pattern)

**INV-10:** model retirement via 404 is immediate and irreversible until explicit reinstatement or catalog refresh

**INV-11:** composition order MBR → CBR → LBR is load-bearing and not configurable

### 1.6 Signatures (25)

#### `RouterEngine.__init__`

```python
def __init__(self, config_path: Path | None = None, **overrides: Any) -> None
```

**Returns:** None

**Raises:** `Exception` when config YAML is invalid or missing required fields

#### `RouterEngine.select_models`

```python
def select_models(self, role: str, *, top_n: int = 12, exclude_providers: frozenset[str] | None = None) -> list[str]
```

**Returns:** list[str] — ranked provider-prefixed model IDs, sorted by composite score descending, interleaved

#### `RouterEngine.record_request`

```python
def record_request(self, provider: str, model_id: str, *, success: bool, tokens_used: int = 0, latency_ms: float = 0.0) -> None
```

**Returns:** None

#### `RouterEngine.health_snapshot`

```python
def health_snapshot(self) -> dict[str, Any]
```

**Returns:** dict[str, Any] — health state of all tracked models/backends

#### `RouterEngine.budget_snapshot`

```python
def budget_snapshot(self) -> dict[str, Any]
```

**Returns:** dict[str, Any] — budget score and capacity per provider

#### `estimate_complexity`

```python
def estimate_complexity(order: DispatchOrder) -> ComplexityEstimate
```

**Returns:** ComplexityEstimate — tier (LOCAL/HAIKU/SONNET/OPUS), confidence (0.7-0.9), signals list

#### `interleave_providers`

```python
def interleave_providers(scored_models: list[ModelScore], max_consecutive: int = 2) -> list[ModelScore]
```

**Returns:** list[ModelScore] — reordered to avoid consecutive same-provider runs

#### `compute_composite_score`

```python
def compute_composite_score(rank: int, budget_score: float, health_score: float) -> float
```

**Returns:** float — weighted composite: rank*0.6 + budget*0.25 + health*0.15 (0-100 scale)

#### `BudgetTracker.score`

```python
def score(self, provider_name: str) -> float
```

**Returns:** float — budget availability 0-100; unknown providers return 100.0

#### `BudgetTracker.has_capacity`

```python
def has_capacity(self, provider_name: str) -> bool
```

**Returns:** bool — True if provider has RPM and RPD headroom

#### `HealthTracker.score`

```python
def score(self, model_id: str) -> float
```

**Returns:** float — health score 0-100 (retired=0, circuit_open=0, 3+errors=30, 1-2=70, 0=100)

#### `CircuitBreaker.allow_request`

```python
def allow_request(self) -> bool
```

**Returns:** bool — True if CLOSED or HALF_OPEN (after cooldown elapsed)

#### `CircuitBreaker.record_success`

```python
def record_success(self) -> None
```

**Returns:** None — resets circuit to CLOSED, clears error timestamps

#### `CircuitBreaker.record_error`

```python
def record_error(self) -> None
```

**Returns:** None — may trip circuit to OPEN if threshold reached

#### `RoleMatrix.get_ranked_models`

```python
def get_ranked_models(self, role: str) -> list[tuple[str, int]]
```

**Returns:** list[tuple[str, int]] — [(model_id, rank), ...] sorted by rank descending; empty list for unknown roles

#### `RoleMatrix.reload_if_changed`

```python
def reload_if_changed(self) -> None
```

**Returns:** None — reloads matrix from file if mtime changed

#### `CatalogCache.get`

```python
def get(self) -> dict[str, list[CatalogEntry]] | None
```

**Returns:** dict[str, list[CatalogEntry]] | None — None if stale or missing

#### `CatalogRefresher.refresh`

```python
async def refresh(self, providers: list[ProviderSchema]) -> dict[str, list[CatalogEntry]]
```

**Returns:** dict[str, list[CatalogEntry]] — partial results; failed providers skipped

#### `SimpleCache.get`

```python
def get(self, key: str) -> str | None
```

**Returns:** str | None — cached response or None if missing/expired

#### `SimpleCache.make_key`

```python
@staticmethod def make_key(model_id: str, system_prompt: str, messages: list[dict], temperature: float, max_tokens: int) -> str
```

**Returns:** str — deterministic SHA-256 hex digest

#### `SemanticCache.get_similar`

```python
def get_similar(self, text: str) -> str | None
```

**Returns:** str | None — cached response if similarity >= threshold, else None

#### `get_router`

```python
def get_router(config_path: str | Path | None = None, **overrides: Any) -> RouterEngine
```

**Returns:** RouterEngine — thread-safe singleton instance

#### `load_config`

```python
def load_config(config_path: Path | None = None) -> RouterConfig
```

**Returns:** RouterConfig — validated configuration model

**Raises:** `Exception` when YAML is invalid or fails Pydantic validation

#### `save_budget_state`

```python
def save_budget_state(state: dict, path: Path) -> None
```

**Returns:** None

**Raises:** `OSError` when file write fails after temp file cleanup

#### `load_budget_state`

```python
def load_budget_state(path: Path) -> dict | None
```

**Returns:** dict | None — loaded state or None if missing/corrupt

### 1.7 Types (19)

#### `BackendTier` — `Enum`

| Field | Type | Semantics |
|-------|------|-----------|
| `LOCAL` | `str` | zero cost, offline, 3-8 tok/s, limited quality |
| `HAIKU` | `str` | fast, cheap/free, simple tasks |
| `SONNET` | `str` | moderate reasoning, most generative work |
| `OPUS` | `str` | deep analysis, multi-step planning, ambiguous |

#### `BackendStatus` — `Enum`

| Field | Type | Semantics |
|-------|------|-----------|
| `AVAILABLE` | `str` | healthy and accepting requests |
| `RATE_LIMITED` | `str` | rate limit hit |
| `DAILY_CAP_HIT` | `str` | daily request cap reached |
| `ERROR` | `str` | experiencing errors |
| `CIRCUIT_OPEN` | `str` | circuit breaker tripped, excluded from routing |
| `OFFLINE` | `str` | unreachable |

#### `CircuitState` — `Enum`

| Field | Type | Semantics |
|-------|------|-----------|
| `CLOSED` | `str` | normal operation |
| `OPEN` | `str` | tripped, requests blocked |
| `HALF_OPEN` | `str` | probing after cooldown, 1 request allowed |

#### `DispatchOrder` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `intent_category` | `str` | classified intent from Core |
| `specific_intent` | `str` | specific intent within category |
| `operator_message` | `str` | pre-redacted operator message |
| `system_prompt` | `str` | assembled system prompt |
| `context_tokens` | `int` | estimated context size |
| `requires_tool_use` | `bool` | whether tool use needed |
| `requires_long_context` | `bool` | whether long context needed |
| `persona` | `str | None` | active persona name |
| `request_id` | `int | None` | unique request identifier |
| `stream_id` | `str | None` | stream identifier |

#### `EngineResponse` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `content` | `str` | generated text content |
| `backend_used` | `str` | name of backend that served the response |
| `backend_tier` | `BackendTier` | tier of the backend used |
| `tokens_in` | `int` | input tokens consumed |
| `tokens_out` | `int` | output tokens produced |
| `estimated_cost_usd` | `float` | estimated cost in USD |
| `latency_ms` | `float` | end-to-end latency in ms |
| `was_fallback` | `bool` | True if primary failed and cascade advanced |
| `fallback_chain` | `list[str]` | list of attempted backends before success |

#### `DispatchFailure` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `message` | `str` | human-readable failure description |
| `attempted_backends` | `list[str]` | backends tried before giving up |
| `error_details` | `dict[str, str]` | per-backend error messages |

#### `ComplexityEstimate` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `tier` | `BackendTier` | estimated minimum tier |
| `confidence` | `float` | 0.7-0.9 confidence level |
| `signals` | `list[str]` | reasoning chain explaining the estimate |

#### `BackendError` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `backend_name` | `str` | name of the failing backend |
| `error_type` | `str` | category of error |
| `message` | `str` | error detail |
| `http_status` | `int | None` | HTTP status code if applicable |
| `retryable` | `bool` | whether retry might succeed |

#### `ModelScore` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `model_id` | `str` | provider-prefixed model identifier |
| `provider` | `str` | provider name |
| `rank` | `int` | role matrix rank (0-100) |
| `budget_score` | `float` | budget availability (0-100) |
| `health_score` | `float` | model health (0-100) |
| `composite` | `float` | weighted composite score (0-100) |

#### `CatalogEntry` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `model_id` | `str` | provider-prefixed model ID |
| `provider` | `str` | provider name |
| `created` | `int | None` | unix timestamp of model creation |

#### `ProviderConfig` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `name` | `str` | unique provider identifier |
| `base_url` | `str` | OpenAI-compatible base URL |
| `catalog_url` | `str | None` | catalog endpoint URL |
| `env_key` | `str | None` | env var name for API key |
| `model_prefix` | `str` | prefix for model IDs (e.g. 'groq/') |
| `rpm_limit` | `int` | requests per minute limit |
| `rpd_limit` | `int | None` | requests per day limit |
| `tpm_limit` | `int | None` | tokens per minute limit |

#### `BackendConfig` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `name` | `str` | unique backend identifier |
| `provider` | `str` | provider name |
| `model` | `str` | provider-specific model identifier |
| `tier` | `BackendTier` | capability tier |
| `base_url` | `str` | OpenAI-compatible endpoint |
| `env_key` | `str | None` | env var for API key |
| `capabilities` | `BackendCapabilities` | immutable capability declaration |
| `cost` | `BackendCostProfile` | per-token cost structure |
| `rate_limits` | `BackendRateLimits` | provider-imposed rate limits |
| `priority` | `int` | lower = higher priority within same tier |

#### `BackendCapabilities` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `max_context_tokens` | `int` | max context window |
| `supports_tool_use` | `bool` | tool/function calling |
| `supports_streaming` | `bool` | streaming responses |
| `supports_json_mode` | `bool` | JSON output mode |
| `supports_system_prompts` | `bool` | system prompt support |

#### `BackendCostProfile` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `input_per_mtok` | `float` | USD per million input tokens |
| `output_per_mtok` | `float` | USD per million output tokens |
| `cache_read_per_mtok` | `float` | USD per million cache read tokens |
| `cache_write_per_mtok` | `float` | USD per million cache write tokens |

#### `BackendRateLimits` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `rpm` | `int` | requests per minute |
| `rpd` | `int` | requests per day |
| `tpm` | `int` | tokens per minute |
| `daily_token_cap` | `int` | tokens per day (0 = unlimited) |

#### `BackendState` — `dataclass`

| Field | Type | Semantics |
|-------|------|-----------|
| `status` | `BackendStatus` | current runtime status |
| `request_timestamps` | `deque[float]` | rolling 60s window |
| `requests_today` | `int` | daily request counter |
| `tokens_today` | `int` | daily token counter |
| `consecutive_errors` | `int` | current error streak |
| `circuit_open_until` | `float` | timestamp when circuit reopens |
| `avg_latency_ms` | `float` | EMA latency (alpha=0.1) |

#### `RouterConfigError` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `message` | `str` | error description |
| `config_path` | `str | None` | path to invalid config |

#### `CatalogRefreshError` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `provider` | `str` | provider that failed |
| `message` | `str` | error detail |
| `http_status` | `int | None` | HTTP status if applicable |

#### `StatePersistenceError` — `dataclass(frozen=True)`

| Field | Type | Semantics |
|-------|------|-----------|
| `path` | `str` | file path that failed |
| `message` | `str` | error detail |
| `operation` | `str` | 'read' or 'write' |

---

## 2. Constitution References

| Type | Reference | Constraint |
|------|-----------|------------|
| architecture | `canonical/daos-engine-canonical-spec.md#8` | MBR→CBR→LBR three-function composition order is load-bearing and not configurable; cascade router is ALWAYS engaged |
| architecture | `canonical/daos-engine-core-contract-spec.md#8.2` | Core produces DispatchOrder with pre-redacted IR; Engine never sees unredacted PII; router consumes DispatchOrder + real-time backend state |
| architecture | `archive/engine/cascade-router-spec.md#3` | All backends implement GenerativeBackend protocol; OpenAI-compatible wire format; provider-agnostic abstraction |
| architecture | `archive/engine/multi-provider-router-spec.md#1.1` | Cascade router is always engaged, not bypassed when all backends online; routing is complexity-driven; Per Session 49 decision (locked) |
| coding_standard | `pyproject.toml [tool.mypy]` | mypy strict mode; no untyped defs; all public APIs fully typed |
| coding_standard | `pyproject.toml [tool.ruff]` | ruff lint: E, F, I, UP, B, C4, SIM; line-length 100; target py311 |
| coding_standard | `pyproject.toml [tool.pytest]` | pytest with asyncio_mode=auto; coverage fail-under=80; cov-report=term-missing |
| security | `canonical/dragonlight-security-implementation-spec.md#8.7` | PII never crosses Core→Engine boundary; context filtering by trust tier (DIAN CECHT); principle of least information |
| security | `canonical/dragonlight-security-implementation-spec.md#circuit-breaker` | Per-backend circuit breaker pattern; 3 consecutive errors → 60s cooldown; separate from engine-level breaker |
| performance | `config/router.yaml` | catalog TTL 24h; budget flush interval 5s; default top_n=12; max_consecutive_same_provider=2 |
| compatibility | `README.md#v0.1-scope` | v0.1 SELECTS models only — does NOT dispatch; dispatch/ and adapters/ packages are reserved stubs for future release |

---

## 3. User Stories

### US-001: As a application developer, I want to get the best available LLM for a task without managing 8 different provider SDKs

**Outcome:** one ranked list of model IDs replaces eight provider-specific integrations

**Narrative:** the developer has an LLM application that needs to call different providers based on task type Then they keep adding if/else blocks for provider selection, rate limits, health checks, and budget tracking Then they install dragonlight-router, call select_models('code_review'), and get a ranked list instantly Finally they use their existing OpenAI SDK with the top model, call record_request() after, and the router handles the rest

**5 Whys:**
1. Q: Why do you need a router? → A: because managing 8 provider SDKs manually is error-prone and unmaintainable
2. Q: Why is it unmaintainable? → A: because each provider has different rate limits, health states, and cost profiles that change independently
3. Q: Why does that matter? → A: because hardcoding provider logic means the application breaks when a provider goes down or changes pricing
4. Q: Why is provider failure so impactful? → A: because without circuit breakers and fallbacks, one failing provider takes down the entire application
5. Q: Why is that the root concern? → A: because the application's reliability should not depend on any single provider's availability — the router must degrade gracefully

**Root Motivation:** *the router must make multi-provider LLM usage as reliable as single-provider usage, with automatic fallback and graceful degradation*

### US-002: As a platform operator, I want to control LLM spending across providers while maintaining service quality

**Outcome:** budget tracking and cost-aware scoring prevent overspending without manual intervention

**Narrative:** the operator has daily and monthly budgets per provider and globally Then they manually check spend dashboards and disable providers when budgets approach limits Then the router's BudgetTracker automatically scores providers lower as budgets deplete, naturally shifting traffic to cheaper alternatives Finally spending stays within budget automatically; the operator only intervenes when ALL budgets are exhausted (deterministic-only mode)

**5 Whys:**
1. Q: Why do you need budget tracking in the router? → A: because LLM API costs are unpredictable and can spike suddenly
2. Q: Why are costs unpredictable? → A: because usage patterns vary by task complexity and provider pricing changes
3. Q: Why does that matter? → A: because a single expensive request to Opus when Groq would suffice wastes budget unnecessarily
4. Q: Why is that waste significant? → A: because the difference between Opus ($15/mtok) and Groq (free) is 3+ orders of magnitude per token
5. Q: Why is that the root concern? → A: because cost-optimal routing is the difference between a sustainable LLM operation and an unsustainable one

**Root Motivation:** *the router must make cost-optimal decisions automatically — no human can track 8 providers' budgets in real-time*

### US-003: As a engine integrator, I want to route generative requests through the canonical MBR→CBR→LBR cascade

**Outcome:** every generative request receives optimal provider selection based on capability, cost, and load

**Narrative:** the engine receives a DispatchOrder from Core with classified intent and pre-redacted IR Then MBR filters by capability tier, CBR scores by cost, LBR enforces rate limits and selects the best candidate Then the cascade produces a DispatchDecision and executes dispatch via the PAL adapter Finally an EngineResponse returns with cost telemetry, fallback tracking, and health feedback recorded

**5 Whys:**
1. Q: Why is the cascade always engaged? → A: because even simple requests should use the cheapest adequate provider, not the most expensive default
2. Q: Why does that matter? → A: because 48% of interactions are classified as deterministic but currently run through the full generative pipeline
3. Q: Why is that waste significant? → A: because that represents an estimated $55,739 in avoidable Opus-equivalent cost annually
4. Q: Why hasn't this been fixed before? → A: because without the cascade router, there's no mechanism to route based on complexity
5. Q: Why is that the root concern? → A: because intelligent routing IS the product — the router exists to make every request cost-optimal by default

**Root Motivation:** *every request must be routed through the cascade — the router is not optional, it is the core of Engine's intelligence*

---

## 4. Task Map (12 tasks)

| ID | Title | Depends | Phase | Complexity |
|----|-------|---------|-------|------------|
| TM-001 | implement MBR (Model Based Router) — capability filtering stage | — | test | standard |
| TM-002 | implement CBR (Cost Balancing Router) — cost scoring stage | TM-001 | test | standard |
| TM-003 | implement LBR (Load Balancing Router) — rate enforcement + final selection | TM-002 | test | standard |
| TM-004 | implement cascade dispatch (MBR→CBR→LBR composition + execution) | TM-003 | test | complex |
| TM-005 | implement GenerativeBackend adapters for all 8 providers | TM-004 | test | complex |
| TM-006 | implement context trust tier filtering (DIAN CECHT) | TM-004 | test | standard |
| TM-007 | implement cost governor + ScoringWeights per canonical spec | — | test | standard |
| TM-008 | implement health check background loop | TM-005 | test | standard |
| TM-009 | enhance HTTP API with dispatch endpoint and trust-tier headers | TM-004 | test | standard |
| TM-010 | wire RouterEngine.dispatch() method for engine consumers | TM-004, TM-009 | graft | standard |
| TM-011 | implement integration test suite for end-to-end cascade routing | TM-004, TM-005, TM-010 | code | standard |
| TM-012 | improve BudgetTracker with TPM and daily token cap enforcement | — | test | trivial |

### TM-001: implement MBR (Model Based Router) — capability filtering stage

The first stage of the three-function cascade. Takes DispatchOrder + BackendRegistry, filters candidates by complexity tier match + health. Implements adjacent-tier graceful upgrade (one tier higher if zero candidates). Outputs candidate set for CBR.

**Acceptance Criteria:**
- [ ] MBR filters candidates to those matching DispatchOrder.complexity_tier OR one tier above _(unit_test)_
- [ ] MBR excludes backends with circuit_open status _(unit_test)_
- [ ] MBR performs graceful upgrade to next tier when zero candidates in requested tier _(unit_test)_
- [ ] MBR NEVER downgrades — complex request never routed to medium/haiku tier _(unit_test)_
- [ ] MBR includes local providers as unlimited-rate in LBR passthrough _(unit_test)_

### TM-002: implement CBR (Cost Balancing Router) — cost scoring stage

The second stage. Takes MBR candidate set + budget tracker + cost governor, filters budget-exhausted providers, scores by projected cost with budget-pressure weight shifting. Implements CostGovernor activation logic.

**Acceptance Criteria:**
- [ ] CBR excludes providers with spent_usd >= budget_usd (hard filter) _(unit_test)_
- [ ] CBR scores candidates using ScoringWeights (cost=0.35, latency=0.25, priority=0.20, queue=0.10, health=0.10) _(unit_test)_
- [ ] Cost governor activates when daily/monthly spend exceeds cost_down_threshold _(unit_test)_
- [ ] When cost governor active, weights shift to cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05 _(unit_test)_
- [ ] When ALL providers exceed budget, returns BudgetExceededError _(unit_test)_

### TM-003: implement LBR (Load Balancing Router) — rate enforcement + final selection

The third stage. Takes CBR-scored candidates + rate limit tracker, filters rate-saturated providers, enforces rolling windows, breaks ties, implements privacy rotation for untrusted tier. Outputs final DispatchDecision.

**Acceptance Criteria:**
- [ ] LBR excludes providers where projected request count would exceed RPM/RPD limits _(unit_test)_
- [ ] LBR deprioritizes providers within 80% of rate limits (queue policy mode) _(unit_test)_
- [ ] LBR selects top candidate and returns DispatchDecision _(unit_test)_
- [ ] LBR treats local providers as unlimited-rate (hardware constraints only) _(unit_test)_
- [ ] Zero candidates remaining after LBR → RoutingError with diagnostic info _(unit_test)_

### TM-004: implement cascade dispatch (MBR→CBR→LBR composition + execution)

Wire the three stages together as the canonical route() function. Takes DispatchOrder, runs MBR→CBR→LBR, dispatches via selected backend adapter, returns EngineResponse or DispatchFailure. Implements fallback cascade with transactional dispatch log + budget update.

**Acceptance Criteria:**
- [ ] route() applies MBR→CBR→LBR in fixed order (not configurable) _(unit_test)_
- [ ] On primary backend failure, cascade advances to next candidate (fallback) _(unit_test)_
- [ ] EngineResponse.was_fallback=True when cascade advanced _(unit_test)_
- [ ] EngineResponse.fallback_chain lists all attempted backends _(unit_test)_
- [ ] Dispatch log write + budget update are transactional (both or neither) _(unit_test)_
- [ ] All backends exhausted → DispatchFailure (not exception) _(unit_test)_

### TM-005: implement GenerativeBackend adapters for all 8 providers

Create concrete implementations of the GenerativeBackend protocol for: NVIDIA NIM, Groq, OpenRouter, Cerebras, Gemini, Mistral, Anthropic, Ollama. Each adapter handles provider-specific auth headers, endpoint paths, streaming format, and health checks.

**Acceptance Criteria:**
- [ ] Each adapter implements generate(), health_check(), record_usage() per GenerativeBackend protocol _(unit_test)_
- [ ] All adapters use OpenAI-compatible wire format (chat completions) _(unit_test)_
- [ ] Anthropic adapter handles non-standard /v1/messages endpoint _(unit_test)_
- [ ] Ollama adapter requires no API key _(unit_test)_
- [ ] Each adapter reads its API key from the configured env_key _(unit_test)_

### TM-006: implement context trust tier filtering (DIAN CECHT)

Implement filter_context_for_provider() that strips system-level context based on provider trust tier. TIER 1 (TRUSTED): pass through. TIER 2 (SEMI_TRUSTED): omit behavioral rules, replace persona names, limit history. TIER 3 (UNTRUSTED): task instruction only. TIER 3-LOCAL: full context (no network egress).

**Acceptance Criteria:**
- [ ] TRUSTED providers receive full system-level context (minus PII, already absent) _(unit_test)_
- [ ] SEMI_TRUSTED providers receive context without behavioral rules or persona names _(unit_test)_
- [ ] UNTRUSTED providers receive task-specific instruction only _(unit_test)_
- [ ] LOCAL providers receive full context (no network egress risk) _(unit_test)_
- [ ] PII is never present regardless of tier (pre-redacted by CAL) _(unit_test)_

### TM-007: implement cost governor + ScoringWeights per canonical spec

Replace the current simple compute_composite_score (rank=0.6, budget=0.25, health=0.15) with the canonical ScoringWeights (cost=0.35, latency=0.25, priority=0.20, queue=0.10, health=0.10). Implement CostGovernorConfig, cost_governor_active(), cost_adjusted_weights().

**Acceptance Criteria:**
- [ ] ScoringWeights dataclass with canonical default values _(unit_test)_
- [ ] score_candidate() normalizes each dimension to [0.0, 1.0] _(unit_test)_
- [ ] cost_governor_active() returns True when daily or monthly spend exceeds threshold _(unit_test)_
- [ ] cost_adjusted_weights() shifts cost weight to 0.70 when governor active _(unit_test)_
- [ ] score is deterministic given same inputs _(unit_test)_
- [ ] score >= 0.0 always _(unit_test)_

### TM-008: implement health check background loop

Background asyncio task that runs every 30s, checking each enabled provider's health via health_check(). Updates BackendState status. Implements SLO enforcement: providers exceeding latency SLO for 3 consecutive checks transition to degraded.

**Acceptance Criteria:**
- [ ] health check loop runs every 30 seconds for all enabled providers _(unit_test)_
- [ ] providers exceeding latency SLO for 3 consecutive checks transition to degraded _(unit_test)_
- [ ] degraded providers receive ranking penalty but are not excluded _(unit_test)_
- [ ] health check failures do not crash the loop _(unit_test)_

### TM-009: enhance HTTP API with dispatch endpoint and trust-tier headers

Add POST /v1/dispatch endpoint for engine-style consumers. Add trust-tier and complexity-tier information to /v1/select and /v1/health responses. Add POST /v1/retire and POST /v1/reinstate endpoints for manual model lifecycle management.

**Acceptance Criteria:**
- [ ] POST /v1/dispatch accepts DispatchOrder JSON and returns EngineResponse or DispatchFailure _(integration_test)_
- [ ] /v1/select response includes trust_tier and complexity_tier per model _(integration_test)_
- [ ] POST /v1/retire and POST /v1/reinstate manage model lifecycle _(integration_test)_
- [ ] All endpoints return structured error responses (never raw exceptions) _(unit_test)_

### TM-010: wire RouterEngine.dispatch() method for engine consumers

Add dispatch() method to RouterEngine that takes a DispatchOrder and runs the full cascade. Wire it into the HTTP API. Preserve select_models() backward compatibility.

**Acceptance Criteria:**
- [ ] RouterEngine.dispatch(DispatchOrder) returns EngineResponse | DispatchFailure _(integration_test)_
- [ ] select_models() behavior unchanged (backward compatible) _(regression_test)_
- [ ] dispatch() integrates MBR→CBR→LBR + PAL adapter + fallback cascade _(integration_test)_
- [ ] no ImportError when loading modified RouterEngine _(unit_test)_

### TM-011: implement integration test suite for end-to-end cascade routing

Integration tests that exercise the full MBR→CBR→LBR→dispatch→response path with mock backends. Verify fallback cascade, circuit breaker integration, budget exhaustion, and graceful degradation scenarios.

**Acceptance Criteria:**
- [ ] integration test: full cascade with all healthy backends selects optimal provider _(integration_test)_
- [ ] integration test: primary failure triggers fallback to next candidate _(integration_test)_
- [ ] integration test: all backends failing returns DispatchFailure _(integration_test)_
- [ ] integration test: circuit breaker opens after 3 consecutive errors _(integration_test)_
- [ ] integration test: budget exhaustion deprioritizes expensive providers _(integration_test)_
- [ ] integration test: catalog refresh failure degrades gracefully _(integration_test)_

### TM-012: improve BudgetTracker with TPM and daily token cap enforcement

Extend BudgetTracker to track TPM (tokens per minute) and daily token cap. Add token-based sliding window for TPM. Wire into LBR rate-limit filtering.

**Acceptance Criteria:**
- [ ] BudgetTracker tracks TPM via sliding window (same as RPM) _(unit_test)_
- [ ] BudgetTracker enforces daily token cap (0 = unlimited) _(unit_test)_
- [ ] score() incorporates TPM headroom into budget score _(unit_test)_
- [ ] has_capacity() returns False when TPM or token cap exhausted _(unit_test)_

---

## 5. Module Boundary (24 modules)

| Module | Type | Status | Coverage |
|--------|------|--------|----------|
| `dragonlight_router.router` | graft | 🔧 existing — needs dispatch() graft (TM-010) | 80% |
| `dragonlight_router.selection.scoring` | standalone_module | 🔧 existing — needs canonical ScoringWeights and CostGovernor (TM-007) | 96% |
| `dragonlight_router.selection.interleave` | standalone_module | ✅ existing — complete, no changes needed | 100% |
| `dragonlight_router.selection.complexity` | standalone_module | ✅ existing — complete, used by MBR (TM-001) | 81% |
| `dragonlight_router.selection.mbr` | standalone_module | 🆕 NEW — to be implemented (TM-001) | — |
| `dragonlight_router.selection.cbr` | standalone_module | 🆕 NEW — to be implemented (TM-002) | — |
| `dragonlight_router.selection.lbr` | standalone_module | 🆕 NEW — to be implemented (TM-003) | — |
| `dragonlight_router.selection.context_filter` | standalone_module | 🆕 NEW — to be implemented (TM-006) | — |
| `dragonlight_router.dispatch.cascade` | standalone_module | 🆕 NEW — to be implemented (TM-004) | — |
| `dragonlight_router.health.tracker` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.health.circuit_breaker` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.budget.tracker` | standalone_module | 🔧 existing — needs TPM and daily token cap (TM-012) | 96% |
| `dragonlight_router.catalog.cache` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.catalog.refresher` | standalone_module | ✅ existing — complete | 76% |
| `dragonlight_router.core.types` | standalone_module | ✅ existing — complete, stable | 89% |
| `dragonlight_router.core.registry` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.core.state` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.config.schema` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.config.loader` | standalone_module | ✅ existing — complete | 84% |
| `dragonlight_router.cache.simple` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.cache.semantic` | standalone_module | ✅ existing — complete | 100% |
| `dragonlight_router.server.app` | graft | 🔧 existing — needs dispatch endpoint (TM-009) | 100% |
| `dragonlight_router.server.routes` | graft | 🔧 existing — needs new endpoints (TM-009) | 100% |
| `dragonlight_router.role_matrix` | standalone_module | ✅ existing — complete | 100% |

---

## 6. Graft Points

### GP-001: `dispatch` → `src/dragonlight_router/router.py`

- **Type:** method_addition
- **Integration:** Add dispatch() method to RouterEngine. Import dispatch.cascade.route. Wire into HTTP API via server/routes.py POST /v1/dispatch. No changes to select_models() — fully backward compatible.
- **Failure Mode:** ImportError at module load time if dispatch.cascade not available; dispatch() must handle case where no backends registered
- **Task:** TM-010

### GP-002: `ScoringWeights, score_candidate, CostGovernorConfig, cost_governor_active, cost_adjusted_weights` → `src/dragonlight_router/selection/scoring.py`

- **Type:** addition
- **Integration:** Add canonical ScoringWeights dataclass and score_candidate() alongside existing compute_composite_score(). Both coexist — simple weights for select_models, canonical for dispatch path.
- **Failure Mode:** score_candidate() normalization bug would produce wrong routing decisions silently — must have exhaustive normalization tests
- **Task:** TM-007

### GP-003: `POST /v1/dispatch, POST /v1/retire, POST /v1/reinstate` → `src/dragonlight_router/server/routes.py`

- **Type:** endpoint_addition
- **Integration:** Add new FastAPI routes to existing register_routes() function. Dispatch endpoint is async. Retire/reinstate are sync. All return structured JSON responses.
- **Failure Mode:** Malformed request body returns 422 with structured error; dispatch failure returns 503 with DispatchFailure JSON
- **Task:** TM-009

### GP-004: `has_tpm_capacity, record_tokens` → `src/dragonlight_router/budget/tracker.py`

- **Type:** method_addition
- **Integration:** Add TPM sliding window tracking (same deque pattern as RPM). Add daily token cap tracking. Wire into score() and has_capacity().
- **Failure Mode:** TPM tracking misconfiguration could over-restrict or under-restrict provider usage; token cap 0 must mean unlimited
- **Task:** TM-012

---

## 7. Call Graph

### Select Models Path

**Entry:** `RouterEngine.select_models(role, top_n, exclude_providers)`

| From | To | Reason | Condition |
|------|----|--------|-----------|
| `RouterEngine.select_models` | `RoleMatrix.reload_if_changed()` | hot-reload role matrix if file changed | — |
| `RouterEngine.select_models` | `RoleMatrix.get_ranked_models(role)` | get candidate models for role with ranks | — |
| `RouterEngine.select_models` | `CatalogCache.get()` | check if cached catalog is fresh (within TTL) | — |
| `RouterEngine.select_models` | `CatalogRefresher.refresh(providers)` | refresh stale catalogs concurrently | catalog is stale (None from cache) |
| `RouterEngine.select_models` | `BudgetTracker.score(provider)` | get budget availability score per provider | — |
| `RouterEngine.select_models` | `HealthTracker.score(model_id)` | get health score per model | — |
| `RouterEngine.select_models` | `compute_composite_score(rank, budget_score, health_score)` | calculate weighted composite score | — |
| `RouterEngine.select_models` | `interleave_providers(scored_models, max_consecutive)` | prevent consecutive same-provider runs | — |
| `RouterEngine.select_models` | `list[str] — ranked model IDs (top_n)` | return value | — |

### Dispatch Path

**Entry:** `RouterEngine.dispatch(order: DispatchOrder)`

| From | To | Reason | Condition |
|------|----|--------|-----------|
| `RouterEngine.dispatch` | `route(order, registry, budget_tracker, rate_tracker, health_cache, cost_governor, queue_depths, config)` | run MBR→CBR→LBR cascade | — |
| `route()` | `estimate_complexity(order)` | MBR: classify dispatch order into BackendTier | — |
| `route()` | `filter_by_capability(candidates, tier, health_tracker)` | MBR: filter by tier match + health | — |
| `route()` | `filter_by_budget(candidates, budget_tracker)` | CBR: hard-filter budget-exhausted | — |
| `route()` | `score_candidates(candidates, weights, budget_tracker, health_tracker)` | CBR: score by cost with governor | — |
| `route()` | `cost_governor_active(daily_spend, monthly_spend, config)` | CBR: check if cost governor should override weights | — |
| `route()` | `cost_adjusted_weights(base_weights)` | CBR: shift weights when governor active | cost governor is active |
| `route()` | `filter_by_rate_limit(candidates, budget_tracker)` | LBR: hard-filter rate-saturated providers | — |
| `route()` | `select_final_candidate(scored_candidates)` | LBR: pick top candidate, break ties | — |
| `route()` | `DispatchDecision or RoutingError` | return value | — |
| `RouterEngine.dispatch` | `filter_context_for_provider(order, provider)` | strip system-level context by trust tier before dispatch | — |
| `RouterEngine.dispatch` | `backend.generate(messages, max_tokens, temperature, stream)` | execute dispatch via selected backend adapter | — |
| `RouterEngine.dispatch` | `backend.record_usage(tokens_in, tokens_out)` | record token consumption for budget tracking | success |
| `RouterEngine.dispatch` | `BudgetTracker.record_request(provider)` | record request for rate limit tracking | — |
| `RouterEngine.dispatch` | `HealthTracker.record_success(model_id, latency_ms)` | record success and latency for health tracking | success |
| `RouterEngine.dispatch` | `HealthTracker.record_error(model_id, http_status=...)` | record error for health/circuit tracking | failure |
| `RouterEngine.dispatch` | `next candidate in cascade` | fallback to next candidate on failure | failure and more candidates exist |
| `RouterEngine.dispatch` | `EngineResponse or DispatchFailure` | return value | — |

### Feedback Path

**Entry:** `RouterEngine.record_request(provider, model_id, success, tokens_used, latency_ms)`

| From | To | Reason | Condition |
|------|----|--------|-----------|
| `RouterEngine.record_request` | `BudgetTracker.record_request(provider)` | track RPM window + increment RPD counter | — |
| `RouterEngine.record_request` | `BudgetTracker.record_tokens(provider, tokens_used)` | track token consumption (if TPM/daily cap enabled) | — |
| `RouterEngine.record_request` | `HealthTracker.record_success(model_id, latency_ms)` | reset error count, update EMA latency | success=True |
| `RouterEngine.record_request` | `HealthTracker.record_error(model_id, http_status=...)` | increment error count, may trip circuit or retire model | success=False |
| `HealthTracker.record_error` | `CircuitBreaker.record_error()` | may trip circuit breaker if threshold reached | — |
| `HealthTracker.record_error` | `HealthTracker._retire_model(model_id)` | immediate retirement on 404 | http_status==404 |
| `HealthTracker.record_success` | `CircuitBreaker.record_success()` | reset circuit to CLOSED, clear error timestamps | — |

### Catalog Refresh Path

**Entry:** `CatalogRefresher.refresh(providers)`

| From | To | Reason | Condition |
|------|----|--------|-----------|
| `CatalogRefresher.refresh` | `CatalogRefresher._fetch_provider(p)` | fetch catalog from single provider | — |
| `CatalogRefresher._fetch_provider` | `httpx.AsyncClient.get(url)` | GET /v1/models from provider endpoint | — |
| `CatalogRefresher.refresh` | `CatalogCache.save(catalog, path)` | persist fetched catalog to disk with atomic write | at least one provider succeeded |

---

## 8. References

### Canonical Specs

| Reference | Title | Relevance |
|-----------|-------|-----------|
| `canonical/daos-engine-canonical-spec.md#8` | Engine Canonical Spec §8 — Cascade Router / Multi-Provider Router | primary architecture authority — three-function composition, six routing dimensions, scoring functio... |
| `canonical/daos-engine-core-contract-spec.md#8.2` | Engine-Core Contract Spec §8.2 — Cascade Router Contract with Core | DispatchOrder format, IR boundary, PII pre-redaction guarantee |
| `canonical/dragonlight-security-implementation-spec.md#8.7` | Security Implementation Spec §8.7 — Context Filtering by Trust Tier | DIAN CECHT security pattern, trust tier definitions, PII boundary |

### Archived Specs

| Reference | Title | Relevance |
|-----------|-------|-----------|
| `archive/engine/cascade-router-spec.md` | Cascade Router Spec (archived) | original cascade router design — GenerativeBackend protocol, backend tier taxonomy, dispatch order f... |
| `archive/engine/multi-provider-router-spec.md` | Multi-Provider Router Spec (archived) | session 49 locked decision — always-engaged routing, complexity-driven selection, dual router clarif... |

### Project Files

| Reference | Title | Relevance |
|-----------|-------|-----------|
| `pyproject.toml` | Python project config | dependencies, mypy/ruff/pytest config, version |
| `config/router.yaml` | Router configuration | provider definitions, rate limits, budget settings, TTLs |
| `config/role-matrix.yaml` | Role-to-model mapping | which models serve which roles, ranking |
| `README.md` | Project README | v0.1 scope, installation, usage, API reference |
| `cascade-router-references.jsonl` | Reference corpus | source material for this live-spec |

### Design Decisions

| Reference | Title | Relevance |
|-----------|-------|-----------|
| `session-49-decision` | Session 49 — Always-Engaged Routing (locked) | cascade router is ALWAYS engaged, not bypassed when all backends online |
| `session-71-audit` | Phase 1 Audit — Implementation Reality | current state: bridge/cascade.py ~1K lines, 75+ tests, not yet cost-optimized |
| `session-77b-architecture` | CAL-PAL Architecture Exploration | OS analogues (MBR=capsched, CBR=cgroups, LBR=WFQ), Engine/Core boundary |
| `session-intelligence-analysis` | Session Intelligence Analysis | 48% deterministic traffic, $55,739 avoidable cost, cost-optimization driver |

---

## 9. Discovery Provenance

**Method:** reference_corpus_analysis

**Ideal State:** every LLM request is routed to the cheapest adequate provider automatically, with circuit breakers preventing cascading failures, budget tracking preventing overspend, and graceful degradation when providers fail

**Root 5-Why:** Why does the dragonlight-router exist?
1. because managing multiple LLM providers manually is unsustainable
2. because each provider has different rate limits, costs, and health states
3. because without intelligent routing, applications overpay or break
4. because the cost difference between providers is 3+ orders of magnitude
5. because cost-optimal routing is the difference between sustainable and unsustainable LLM operations
→ *the router must make every request cost-optimal by default while maintaining reliability across provider failures*

---

## 10. Requirements Provenance

**Completeness Score:** 0.92

**Gaps:**
- **[medium]** streaming response handling in dispatch: AsyncIterator[str] streaming dispatch path needs spec detail
- **[low]** PAL adapter protocol (prompt/payload construction): referenced in canonical spec but is a separate spec concern
- **[low]** privacy rotation algorithm for LBR untrusted tier: mentioned in canonical spec but details deferred to implementation

**Checklist:**
- ✅ operations_complete
- ✅ edge_cases_explored
- ✅ error_states_mapped
- ✅ empty_states_defined
- ✅ invariants_stated
- ✅ five_whys_traced
- ✅ signatures_defined
- ✅ module_boundary_complete
- ✅ graft_points_identified
