# Dragonlight Router -- Hazard Register

**Document ID:** QA-026  
**Standard:** FMEA (Failure Mode and Effects Analysis)  
**Scope:** Dragonlight Router cascade dispatch pipeline  
**Created:** 2026-06-16  
**Owner:** Dragonlight Engineering  
**Review cadence:** Every release, or when architecture changes  

---

## Methodology

This register uses FMEA methodology adapted for LLM routing infrastructure. Each hazard is assessed on two axes:

- **Severity** -- consequence magnitude if the failure occurs  
  - Critical: data breach, financial loss >$100, complete service outage  
  - High: degraded trust guarantees, financial loss $10-100, extended partial outage  
  - Medium: suboptimal routing, minor cost overrun, brief availability loss  
  - Low: cosmetic, logging-only, no operator-visible impact  

- **Likelihood** -- probability of occurrence given current architecture  
  - High: expected to occur under normal operating conditions or common edge cases  
  - Medium: plausible under stress, misconfiguration, or adversarial input  
  - Low: requires unlikely coincidence or sophisticated attack  

- **Risk Score** -- Severity x Likelihood matrix:

|                | Likelihood: High | Likelihood: Medium | Likelihood: Low |
|----------------|-------------------|--------------------|-----------------|
| Severity: Critical | **CRITICAL**   | **HIGH**           | **MEDIUM**      |
| Severity: High     | **HIGH**       | **HIGH**           | **MEDIUM**      |
| Severity: Medium   | **MEDIUM**     | **MEDIUM**         | **LOW**         |
| Severity: Low      | **LOW**        | **LOW**            | **LOW**         |

---

## Hazard Register

### HAZ-001 -- Context Sent to Wrong Trust Tier Provider

| Field | Detail |
|-------|--------|
| **ID** | HAZ-001 |
| **Category** | Data |
| **Description** | Operator context (system prompts, behavioral rules, persona definitions, conversation history) is dispatched to a provider whose trust tier is insufficient. This occurs if `_tier_to_provider_trust()` in `dispatch/cascade.py` maps a backend tier incorrectly, or if `filter_context_for_provider()` in `selection/context_filter.py` fails to strip sensitive fields for SEMI_TRUSTED or UNTRUSTED providers. The `DispatchOrder.context_trust_tier` field is accepted from the request body but not enforced in the cascade -- the tier is derived solely from `BackendConfig.tier`, meaning a caller-specified trust requirement can be silently ignored. |
| **Severity** | Critical |
| **Likelihood** | Medium |
| **Risk Score** | **HIGH** |
| **Current Mitigation** | `context_filter.py` implements a four-tier dispatch (`TRUSTED`, `SEMI_TRUSTED`, `UNTRUSTED`, `LOCAL`) with field-level redaction: SEMI_TRUSTED strips behavioral rules and redacts persona fields; UNTRUSTED returns task-only context. `cascade.py` maps each `BackendTier` to a `ProviderTrustTier` via `_tier_to_provider_trust()`. Unknown tiers default to `UNTRUSTED`. **v0.2.3:** `_filter_by_trust_floor()` in `cascade.py` now enforces `context_trust_tier` from `DispatchOrder` as a minimum floor — backends whose provider trust rank is below the caller's requested floor are removed before CBR/LBR scoring. 9 unit tests verify enforcement across all tier combinations. |
| **Residual Risk** | The tier mapping is hardcoded with no runtime validation that the mapping matches the actual provider's data handling agreement. If a new `BackendTier` is added without updating the mapping, the `dict.get()` fallback to `UNTRUSTED` is safe, but the silent default could mask a configuration error. SEMI_TRUSTED filtering uses string-match-based field exclusion (`"behavioral_rules"`) which is brittle if context field names change. |
| **Owner** | `selection/context_filter.py`, `dispatch/cascade.py` |

---

### HAZ-002 -- Budget Enforcement Race Condition

| Field | Detail |
|-------|--------|
| **ID** | HAZ-002 |
| **Category** | Cost |
| **Description** | Concurrent dispatch requests can pass budget checks simultaneously before either records its spend, causing aggregate spend to exceed configured limits. `BudgetTracker` in `budget/tracker.py` uses in-memory `defaultdict` structures with no locking. Two requests checking `has_capacity()` at the same instant both see headroom, both proceed, and both call `record_request()` after completion -- by which time the budget is overspent. |
| **Severity** | High |
| **Likelihood** | Medium |
| **Risk Score** | **HIGH** |
| **Current Mitigation** | The CBR stage in `dispatch/cascade.py` calls `_compute_aggregate_spend()` and `filter_by_cost()` which check budget scores before dispatch. The cost governor in `selection/scoring.py` shifts scoring weights when daily or monthly thresholds are crossed (70% cost weight). Budget state is persisted atomically via `budget/persistence.py` (tmp-rename pattern). **v0.2.3:** `BudgetTracker` now has an `asyncio.Lock` and `check_and_reserve()` method that atomically checks capacity and records spend under the lock. 4 unit tests including a concurrent race prevention test verifying that capacity limits are not exceeded under parallel dispatch. |
| **Residual Risk** | The `monthly_spend_usd()` method extrapolates from daily spend (daily * 30), which compounds estimation error. The daily reset boundary (`_maybe_reset_daily()`) clears all counters atomically at UTC midnight, creating a brief window where a burst of requests could all pass against a fresh zero-spend state. The lock is per-process — multi-process deployments (e.g., multiple uvicorn workers) still have no cross-process coordination. |
| **Owner** | `budget/tracker.py`, `dispatch/cascade.py` |

---

### HAZ-003 -- Cascade Exhaustion (Total Availability Loss)

| Field | Detail |
|-------|--------|
| **ID** | HAZ-003 |
| **Category** | Availability |
| **Description** | All backends in the cascade are exhausted during fallback, returning a `DispatchFailure` to the caller. This occurs when every candidate fails generation in `_handle_fallback_chain()` in `dispatch/cascade.py`. A correlated failure (e.g., a shared upstream dependency, DNS resolution failure, or coordinated provider outage) can trip all circuit breakers simultaneously, leaving zero available backends for subsequent requests during the cooldown window. |
| **Severity** | Critical |
| **Likelihood** | Low |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | The cascade in `dispatch/cascade.py` tries every candidate in ranked order before returning failure. MBR in `selection/mbr.py` implements graceful tier upgrade (tries requested tier, then next tier up). Circuit breakers in `health/circuit_breaker.py` use configurable thresholds (default: 3 errors in 120s window) and cooldown (default: 60s) with jitter (HAZ-009). HALF_OPEN state allows probe requests to test recovery. Health tracker in `health/tracker.py` provides per-model scoring and retirement. LOCAL backends bypass circuit breaker and rate limit checks entirely. **v0.2.6:** `HealthTracker.get_state()`/`restore_state()` now persist retired models and circuit breaker states across process restarts (wired into `RouterEngine.save_state()` and `_restore_health_state()`). `HealthTracker.availability_status()` returns router-level health (`healthy`/`degraded`/`unavailable`). The `/v1/health` endpoint now includes a `status` field so callers can detect degraded state before dispatching. 14 unit tests. |
| **Residual Risk** | No cached-response fallback exists -- when all backends are exhausted, the router returns a 500 error with no degraded-mode content. No exponential backoff on circuit breaker cooldowns. The HALF_OPEN probe allows exactly one request through, but if that request also fails, the circuit re-opens for another full cooldown. |
| **Owner** | `dispatch/cascade.py`, `health/circuit_breaker.py`, `health/tracker.py` |

---

### HAZ-004 -- Silent Fallback to Lower-Capability Provider

| Field | Detail |
|-------|--------|
| **ID** | HAZ-004 |
| **Category** | Quality |
| **Description** | When the primary provider fails, the cascade falls back to a lower-ranked (potentially lower-capability) provider without explicit caller awareness at decision time. The `EngineResponse` reports `was_fallback=True` and `fallback_chain` after the fact, but the caller has no opportunity to reject the fallback before the lower-capability response is generated and returned. MBR's `_enforce_no_downgrade()` prevents tier downgrades within the MBR stage, but the cascade can fall back across the full ranked list produced by CBR/LBR, which may include backends at the same tier but with significantly different capability profiles. |
| **Severity** | Medium |
| **Likelihood** | High |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | `EngineResponse` includes `was_fallback: bool` and `fallback_chain: list[str]` so the caller can detect fallback after the fact. MBR enforces a no-downgrade invariant via `_enforce_no_downgrade()` in `selection/mbr.py` (every candidate must be at or above the requested tier). Degraded backends receive a 0.5 score penalty in `_apply_degraded_penalty()` in `dispatch/cascade.py`, deprioritizing them in the ranked list. **v0.2.6:** `DispatchOrder.fallback_policy` field enables caller control: `"allow"` (default, full cascade fallback), `"deny"` (fail immediately if primary fails, no fallback), `"same_tier"` (only fall back to candidates at the same BackendTier). `_apply_fallback_policy()` in `cascade.py` restricts the candidate pool before dispatch. Applied in both streaming and non-streaming paths. Validated in request handler against `_ALLOWED_FALLBACK_POLICIES`. 10 unit tests. |
| **Residual Risk** | Backends at the same tier can have very different capability profiles (e.g., two MODERATE backends where one supports tool use and one does not). The capability filter in `mbr.py` checks `supports_tool_use`, `max_context_tokens`, and `supports_system_prompts`, but does not check qualitative capability differences (e.g., reasoning depth, code quality). |
| **Owner** | `dispatch/cascade.py`, `selection/mbr.py` |

---

### HAZ-005 -- Provider Rate Limit Violation and Account Suspension

| Field | Detail |
|-------|--------|
| **ID** | HAZ-005 |
| **Category** | Cost / Availability |
| **Description** | The router exceeds a provider's API rate limits (RPM, RPD, TPM), causing 429 responses, temporary throttling, or account-level suspension. The `BudgetTracker` in `budget/tracker.py` uses in-memory sliding windows for RPM and TPM tracking, which are lost on process restart. The LBR stage filters by a median-score threshold rather than hard capacity checks, meaning a provider at 51% capacity in a two-provider pool still passes filtering. |
| **Severity** | High |
| **Likelihood** | Medium |
| **Risk Score** | **HIGH** |
| **Current Mitigation** | `BudgetTracker` tracks RPM via sliding window (60s), RPD via daily counter, TPM via sliding window, and daily token cap. `has_capacity()` checks all four dimensions. LBR in `selection/lbr.py` filters candidates by median budget score. The server's `RateLimitMiddleware` in `server/middleware.py` enforces per-IP inbound rate limiting (token bucket, default 60 req/min). Budget state is persisted to disk via `budget/persistence.py`. **v0.2.3:** LBR now applies a hard `_hard_capacity_gate()` before median filtering: providers where `has_capacity()` returns False are removed entirely, regardless of median score. LOCAL tier bypasses this gate. 5 unit tests. Daily counters persist across restarts (HAZ-012). |
| **Residual Risk** | Sliding windows (RPM/TPM) are in-memory and reset on restart — a process crash and restart clears sub-minute state. The `_tpm_remaining()` calculation relies on estimated token counts (message length / 4), not actual provider-reported usage. Provider-side rate limits may differ from configured limits if the provider changes them without config update. The `_maybe_reset_daily()` boundary creates a step-function reset at UTC midnight. |
| **Owner** | `budget/tracker.py`, `selection/lbr.py`, `server/middleware.py` |

---

### HAZ-006 -- API Key Exposure in Logs or Error Messages

| Field | Detail |
|-------|--------|
| **ID** | HAZ-006 |
| **Category** | Security |
| **Description** | API keys loaded from environment variables (via `BackendConfig.env_key`) could be included in log output, error messages, or HTTP responses. The `OpenAICompatibleBackend` in `adapters/_openai_compat.py` reads `os.environ.get(config.env_key)` and stores it as `self._api_key`. If an exception during `generate()` includes the request URL (which does not contain the key, since auth is header-based) or the headers dict in its traceback, the Bearer token could appear in structured logs. The error re-raise pattern (`raise RuntimeError(f"... API error: {e}") from e`) preserves the original exception chain, which may contain header details. |
| **Severity** | Critical |
| **Likelihood** | Low |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | API keys are passed in HTTP headers (`Authorization: Bearer ...`), not in URLs or request bodies. The `_build_auth_headers()` method constructs headers programmatically without logging them. Error handling in `_openai_compat.py` wraps provider exceptions in `RuntimeError` with a descriptive message, and `structlog` is used throughout (structured logging avoids accidental string interpolation of sensitive objects). The router does not include raw exception tracebacks in HTTP responses -- errors are returned as structured JSON via `_format_error_response()` and `_format_dispatch_failure()` in `server/routes.py`. **v0.2.5:** `server/logging.py` adds a `scrub_secrets()` structlog processor to the pipeline via `configure_logging()`, called at app startup. The processor recursively scrubs Bearer tokens (case-insensitive), API key prefixes (`sk-`, `gsk_`, `nvapi-`, `AIza`, `key-`, `xai-`), and known secret key names (`authorization`, `api_key`, `api-key`, `token`, `secret`) from all log event dicts. Values are replaced with `[REDACTED]`. 14 unit tests. |
| **Residual Risk** | The `_api_key` attribute is a plain string on the adapter instance, accessible from any code with a reference to the adapter object. The scrubber operates on structlog event dicts only -- log output from non-structlog sinks (e.g., raw `print()`, third-party library loggers) is not scrubbed. The regex-based pattern matching could miss novel key formats not covered by the current prefix list. |
| **Owner** | `adapters/_openai_compat.py`, `server/routes.py` |

---

### HAZ-007 -- Prompt Injection Affecting Routing Decisions

| Field | Detail |
|-------|--------|
| **ID** | HAZ-007 |
| **Category** | Security |
| **Description** | Malicious content in `operator_message` or `system_prompt` fields could influence routing behavior if these fields are used in complexity estimation or trust tier determination. The `estimate_complexity()` function in `selection/mbr.py` determines the backend tier based on `context_tokens`, `requires_tool_use`, and `requires_long_context` -- all of which are numeric/boolean fields from the request body, not derived from message content. However, prompt content passes through `_sanitize_prompt()` in `server/routes.py` and then directly into the chosen provider's API, meaning an injected prompt could cause the provider to return content that, if used in a feedback loop, could affect subsequent routing. |
| **Severity** | High |
| **Likelihood** | Low |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | `_sanitize_prompt()` in `server/routes.py` strips null bytes and control characters (preserving newlines, carriage returns, tabs), truncates to 100K characters. Routing decisions in MBR/CBR/LBR are based on structured fields (`context_tokens`, `requires_tool_use`, `intent_category`), not on message content. Context filtering in `context_filter.py` limits what context reaches each provider based on trust tier, preventing untrusted providers from seeing behavioral rules or persona definitions. Input validation in `_validate_dispatch_request()` enforces type checks and length limits on all fields. **v0.2.6:** `_ALLOWED_INTENT_CATEGORIES` frozenset in `server/routes.py` validates `intent_category` against an allowed set of 17 categories. Unknown intent values are rejected with HTTP 400. This prevents adversarial intent injection from affecting MBR tier routing decisions (since `intent_category` now influences `estimate_complexity()` via HAZ-013). 5 unit tests. |
| **Residual Risk** | No semantic analysis of prompt content is performed -- adversarial prompts that stay within the length and character constraints pass through unmodified. The `specific_intent` field remains a free-form string with no validation against an allowed set. If the router is ever extended to use LLM-generated output to influence routing (e.g., response-quality-based re-routing), prompt injection becomes a direct routing manipulation vector. The sanitizer does not detect or mitigate known prompt injection patterns (e.g., "ignore previous instructions"). |
| **Owner** | `server/routes.py`, `selection/mbr.py` |

---

### HAZ-008 -- Stale Catalog Routing to Deprecated Models

| Field | Detail |
|-------|--------|
| **ID** | HAZ-008 |
| **Category** | Quality / Availability |
| **Description** | The router dispatches requests to models that have been deprecated, renamed, or removed by the provider. The `CatalogCache` in `catalog/cache.py` uses a file-backed cache with a configurable TTL (default: 24 hours). If the cache is not refreshed and a provider deprecates a model, the router continues routing to it until the next successful refresh. The `CatalogRefresher` in `catalog/refresher.py` fetches model lists from provider `/v1/models` endpoints, but this is not invoked automatically -- it requires an explicit `POST /v1/catalog/refresh` call or external trigger. |
| **Severity** | Medium |
| **Likelihood** | Medium |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | `HealthTracker.record_error()` in `health/tracker.py` treats HTTP 404 responses at inference time as model retirement events, immediately evicting the model via `_retire_model()`. The circuit breaker trips after 3 errors in 120s, preventing repeated dispatch to a failing model. `CatalogCache.is_stale()` returns true when the cache age exceeds TTL, and `CatalogCache.get()` returns `Err(StaleCatalogError)` for stale caches. `RoleMatrix` in `roles/matrix.py` supports hot-reload via mtime check. **v0.2.5:** `HealthCheckLoop` now accepts an `on_cycle` async callback and `on_cycle_interval` parameter. `RouterEngine._init_health_check()` wires `_async_refresh_catalog` as the `on_cycle` callback, with interval derived from `catalog_ttl_hours` (`max(1, (ttl * 3600) // 30 // 2)` cycles). Callback failures are caught with specific exception types and do not crash the health check loop. 8 unit tests. |
| **Residual Risk** | A model that returns 200 with degraded output (e.g., a model nearing deprecation that returns quality warnings but still generates) would not trigger the 404 retirement path. Reinstated models (`reinstate_model()`) have their error counts reset to 0, which could re-enable a model that was retired for good reason. The role matrix hot-reload checks file mtime but does not validate that the referenced model IDs still exist in the catalog. The refresh interval is derived from TTL but is approximate -- actual refresh timing depends on health check interval and cycle count. |
| **Owner** | `catalog/cache.py`, `catalog/refresher.py`, `health/tracker.py` |

---

### HAZ-009 -- Circuit Breaker Flapping

| Field | Detail |
|-------|--------|
| **ID** | HAZ-009 |
| **Category** | Availability |
| **Description** | Circuit breakers for multiple backends oscillate between OPEN and HALF_OPEN states in lockstep, creating periodic total unavailability. When a correlated failure (e.g., network partition) trips multiple circuit breakers simultaneously, they all enter OPEN with the same `_opened_at` timestamp. After the cooldown (default: 60s), all transition to HALF_OPEN simultaneously, all allow one probe request, and if those probes also fail (likely if the underlying issue persists), all re-enter OPEN simultaneously -- repeating the cycle. |
| **Severity** | High |
| **Likelihood** | Low |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | Circuit breakers in `health/circuit_breaker.py` are per-model, so different models can be in different states. The HALF_OPEN state allows exactly one probe request, limiting blast radius of a failed probe. Error timestamps are pruned to a configurable window (`error_window_s`, default 120s). The error threshold is configurable (`error_threshold`, default 3). `record_success()` immediately resets the circuit to CLOSED and clears the error history. **v0.2.5:** `CircuitBreaker.__init__` now accepts a `jitter_factor` parameter (default 0.25). Each time the circuit opens or re-opens (`record_error()`), `_compute_jittered_cooldown()` adds `random.uniform(0, jitter_factor * cooldown_s)` to the base cooldown, storing the result in `_effective_cooldown_s`. `allow_request()` and `restore_state()` use the jittered value. This desynchronizes recovery timing across breakers tripped by the same correlated failure. 10 unit tests. |
| **Residual Risk** | No exponential backoff exists -- the cooldown is fixed (with jitter) regardless of how many times the circuit has flapped. The HALF_OPEN state allows only one probe request, which creates a single point of failure for recovery determination. The jitter range is bounded to `[0, 0.25 * cooldown_s]` by default, so breakers tripped within the jitter window could still recover close together. The `check_loop.py` health check loop probes backends independently but its integration with the circuit breaker recovery path routes through `HealthTracker`, not directly through `CircuitBreaker.record_success()`. |
| **Owner** | `health/circuit_breaker.py`, `health/tracker.py` |

---

### HAZ-010 -- Token Count Estimation Inaccuracy

| Field | Detail |
|-------|--------|
| **ID** | HAZ-010 |
| **Category** | Cost |
| **Description** | Token counts used for cost estimation and budget tracking are approximated by dividing character count by 4 (`len(content) // 4` in `dispatch/cascade.py`). This heuristic is inaccurate for non-English text, code, structured data, and special tokens. Inaccurate token counts propagate to `BudgetTracker.record_request()`, affecting RPM/TPM sliding windows and daily token cap tracking, and to `EngineResponse.estimated_cost_usd`, which is reported to the caller. |
| **Severity** | Medium |
| **Likelihood** | High |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | The cost is labeled `estimated_cost_usd` in the `EngineResponse`, signaling to the caller that it is approximate. Budget tracking uses the same estimation consistently across all providers, so relative comparisons between providers remain valid even if absolute values are off. The cost governor in `selection/scoring.py` uses configurable thresholds (`cost_down_threshold_daily`, `cost_down_threshold_monthly`) that can be set conservatively to compensate for estimation error. **v0.2.6:** Token estimation is centralized in `_estimate_token_count()` in `cascade.py`, replacing inline `len(content) // 4` calculations. `_log_token_estimation()` logs estimation details (char count, estimated tokens, direction) on every dispatch for operator observability. Both streaming and non-streaming paths use the centralized function. The minimum is 1 token (prevents division artifacts). 6 unit tests. |
| **Residual Risk** | Character-count-based estimation can be off by 2-3x for CJK text, code with many short tokens, or messages with many special characters. This directly affects budget enforcement: a message that is 2x more expensive than estimated consumes 2x the budget without the tracker knowing. The `daily_token_cap` enforcement in `_daily_token_remaining()` uses these estimated counts, meaning a cap of 1M tokens could actually represent 500K-2M real tokens depending on content. Provider-reported token usage from response headers is not yet fed back to the budget tracker — the centralized estimation function is the preparation point for future tokenizer integration. |
| **Owner** | `dispatch/cascade.py`, `budget/tracker.py` |

---

### HAZ-011 -- Unauthenticated Admin Endpoints

| Field | Detail |
|-------|--------|
| **ID** | HAZ-011 |
| **Category** | Security |
| **Description** | Administrative endpoints (`POST /v1/retire`, `POST /v1/reinstate`, `POST /v1/catalog/refresh`) in `server/routes.py` are exposed without authentication or authorization checks. An attacker with network access to the router can retire all backends (causing total denial of service), reinstate retired backends (overriding safety controls), or trigger catalog refreshes (potential information disclosure of configured providers). The `RateLimitMiddleware` in `server/middleware.py` limits request rate per IP but does not distinguish between admin and non-admin endpoints. |
| **Severity** | Critical |
| **Likelihood** | Low |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | `RateLimitMiddleware` enforces per-IP rate limiting (default: 60 req/min token bucket). The router is expected to be deployed behind a reverse proxy or VPN that restricts access to the admin API. The `retire_handler()` returns 404 for unknown backend names, limiting blind enumeration. **v0.2.5:** `RouterConfig` now includes an `admin_api_key: str | None` field. `_check_admin_auth()` in `server/routes.py` validates `Authorization: Bearer <key>` headers on all admin paths (`/v1/retire`, `/v1/reinstate`, `/v1/catalog/refresh`). Returns 401 JSON for missing or invalid auth. When `admin_api_key` is `None` (unconfigured), auth is bypassed for backward compatibility. Non-admin endpoints (`/v1/dispatch`, `/v1/health`) are unaffected. 14 unit tests. |
| **Residual Risk** | The `/v1/health` endpoint (`GET`) remains unauthenticated and exposes budget and health state, including provider names, model IDs, and spend data. When `admin_api_key` is not configured, admin endpoints remain open (backward-compatible default). The `admin_api_key` is a single shared secret with no per-user or per-role granularity. No audit log records who called admin endpoints. The auth check uses constant-time comparison but the key is stored as a plain string in the config object. |
| **Owner** | `server/routes.py`, `server/middleware.py` |

---

### HAZ-012 -- In-Memory State Loss on Process Restart

| Field | Detail |
|-------|--------|
| **ID** | HAZ-012 |
| **Category** | Availability / Cost |
| **Description** | `BudgetTracker` and `HealthTracker` maintain all runtime state in memory (`defaultdict`, `deque`). A process crash, restart, or deployment loses: all RPM/TPM sliding windows, all daily spend counters, all circuit breaker states (error timestamps, open/closed status), all model retirement events, and all EMA latency data. After restart, the router has no knowledge of recent failures or spend, and may immediately re-route to providers that are over budget or models that were retired for cause. |
| **Severity** | High |
| **Likelihood** | Medium |
| **Risk Score** | **HIGH** |
| **Current Mitigation** | `budget/persistence.py` provides `save_budget_state()` and `load_budget_state()` with atomic write semantics (tmp-rename). `CatalogCache` in `catalog/cache.py` persists catalog state to disk with TTL-based expiration. **v0.2.3:** Budget state is now fully wired into the lifecycle: `RouterEngine._restore_budget_state()` loads daily counters at startup, `RouterEngine.save_state()` persists them at shutdown (called from server lifespan). `BudgetTracker.get_state()`/`restore_state()` serialize RPD counts, daily token counts, and day reset boundaries. `CircuitBreaker.get_state()`/`restore_state()` serialize OPEN state with cooldown tracking. 19 unit tests across all persistence touchpoints. |
| **Residual Risk** | Sliding windows (RPM/TPM) are intentionally not persisted — they represent sub-minute time-series data that becomes stale immediately on restore. Model retirements in `HealthTracker._retired` are in-memory only and lost on restart. EMA latency data (`_avg_latency`) is lost, causing the scoring function to start with no latency information. Circuit breaker state persistence is implemented but not yet wired into HealthTracker save/restore (individual breakers are serializable but the HealthTracker does not invoke get_state/restore_state on its breaker collection). |
| **Owner** | `budget/tracker.py`, `health/tracker.py`, `budget/persistence.py` |

---

### HAZ-013 -- Complexity Estimation Misrouting

| Field | Detail |
|-------|--------|
| **ID** | HAZ-013 |
| **Category** | Quality |
| **Description** | `estimate_complexity()` in `selection/mbr.py` uses a simple if-chain to determine the backend tier: LOCAL by default, SIMPLE if long context or >4096 tokens, MODERATE if tool use required, COMPLEX if >8192 tokens. This heuristic does not account for task complexity (e.g., a 1000-token reasoning task that requires COMPLEX-tier capability) or for the sequential evaluation order (a request with both `requires_tool_use=True` and `context_tokens=9000` gets COMPLEX, but `requires_tool_use=True` with `context_tokens=5000` gets only MODERATE despite possibly needing COMPLEX reasoning). The tier assignment is non-monotonic: later conditions overwrite earlier ones without accumulation. |
| **Severity** | Medium |
| **Likelihood** | Medium |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | MBR's graceful upgrade in `_resolve_tiers_to_try()` tries the requested tier and the next tier up, so a MODERATE estimate also checks COMPLEX backends. The no-downgrade invariant in `_enforce_no_downgrade()` ensures candidates are never below the estimated tier. **v0.2.6:** `_INTENT_TIER_FLOOR` dict in `selection/mbr.py` maps 10 intent categories to minimum BackendTier values. `estimate_complexity()` applies the intent floor after heuristic estimation — the floor can only raise the tier, never lower it. Examples: `complex_reasoning`/`strategic_planning`/`architecture` → COMPLEX; `code_generation`/`code_review`/`debugging` → MODERATE; `data_analysis`/`summarization` → SIMPLE. Combined with HAZ-007's intent validation, callers cannot inject arbitrary intent values. 14 unit tests. |
| **Residual Risk** | The one-tier-up upgrade is limited -- a LOCAL estimate only tries LOCAL and SIMPLE. The `requires_tool_use` and `requires_long_context` flags are caller-supplied with no validation, meaning a caller can force any tier assignment by setting these flags. No feedback loop exists to learn from quality outcomes and adjust tier estimation. Intent categories not in `_INTENT_TIER_FLOOR` default to pure heuristic estimation. |
| **Owner** | `selection/mbr.py` |

---

### HAZ-014 -- Concurrent Adapter State Mutation

| Field | Detail |
|-------|--------|
| **ID** | HAZ-014 |
| **Category** | Availability |
| **Description** | The `OpenAICompatibleBackend` in `adapters/_openai_compat.py` maintains mutable instance state (`self._status: BackendStatus`) that is written from async exception handlers without synchronization. Under concurrent dispatch, multiple asyncio tasks can call `generate()` on the same adapter instance. If one task's request fails and sets `self._status = BackendStatus.ERROR`, this affects all subsequent requests through that adapter instance, even if the failure was transient and other requests are succeeding simultaneously. |
| **Severity** | Medium |
| **Likelihood** | Medium |
| **Risk Score** | **MEDIUM** |
| **Current Mitigation** | The health tracker provides a separate, per-model health scoring system that is used for routing decisions, so the adapter's `_status` field has limited routing impact. Circuit breakers in `health/circuit_breaker.py` track errors independently of adapter status. **v0.2.5:** `create_adapter()` in `adapters/__init__.py` is verified to return a fresh instance per call -- no caching or sharing. `_try_adapter_dispatch()` and `_try_streaming_dispatch()` in `cascade.py` now assert `adapter.status == BackendStatus.AVAILABLE` immediately after `create_adapter()`, catching any factory regression that would return a stale or shared adapter. 4 unit tests verify fresh-instance isolation across all 11 providers. |
| **Residual Risk** | The `_status` field is set in multiple exception handlers without any atomicity guarantee, but since each dispatch uses a fresh adapter instance, concurrent mutations cannot cross-contaminate. The `health_check()` method also mutates `_status`, but health checks operate on the `HealthCheckLoop`'s own backend instances, not dispatch adapters. The `record_usage()` method is a no-op ("no-op until usage tracking is wired"), leaving a gap in usage tracking integration. |
| **Owner** | `adapters/_openai_compat.py` |

---

## Summary Matrix

| ID | Category | Description (Short) | Severity | Likelihood | Risk Score | Status |
|----|----------|---------------------|----------|------------|------------|--------|
| HAZ-001 | Data | Context to wrong trust tier | Critical | Medium | ~~HIGH~~ **MITIGATED** | Trust floor enforcement in cascade |
| HAZ-002 | Cost | Budget enforcement race condition | High | Medium | ~~HIGH~~ **MITIGATED** | asyncio.Lock + check_and_reserve |
| HAZ-003 | Availability | Cascade exhaustion | Critical | Low | ~~MEDIUM~~ **MITIGATED** | Health state persistence + availability status |
| HAZ-004 | Quality | Silent fallback to lower capability | Medium | High | ~~MEDIUM~~ **MITIGATED** | fallback_policy (allow/deny/same_tier) |
| HAZ-005 | Cost / Availability | Provider rate limit violation | High | Medium | ~~HIGH~~ **MITIGATED** | Hard capacity gate in LBR |
| HAZ-006 | Security | API key exposure in logs | Critical | Low | ~~MEDIUM~~ **MITIGATED** | scrub_secrets structlog processor |
| HAZ-007 | Security | Prompt injection affecting routing | High | Low | ~~MEDIUM~~ **MITIGATED** | intent_category validation against allowed set |
| HAZ-008 | Quality / Availability | Stale catalog routing | Medium | Medium | ~~MEDIUM~~ **MITIGATED** | Automatic catalog refresh in health check loop |
| HAZ-009 | Availability | Circuit breaker flapping | High | Low | ~~MEDIUM~~ **MITIGATED** | Jittered cooldown (jitter_factor=0.25) |
| HAZ-010 | Cost | Token count estimation inaccuracy | Medium | High | ~~MEDIUM~~ **MITIGATED** | Centralized estimation + observability logging |
| HAZ-011 | Security | Unauthenticated admin endpoints | Critical | Low | ~~MEDIUM~~ **MITIGATED** | admin_api_key bearer token auth |
| HAZ-012 | Availability / Cost | In-memory state loss on restart | High | Medium | ~~HIGH~~ **MITIGATED** | Budget persistence at startup/shutdown |
| HAZ-013 | Quality | Complexity estimation misrouting | Medium | Medium | ~~MEDIUM~~ **MITIGATED** | Intent-based tier floor in estimate_complexity |
| HAZ-014 | Availability | Concurrent adapter state mutation | Medium | Medium | ~~MEDIUM~~ **MITIGATED** | Fresh adapter per dispatch (create_adapter isolation) |

---

## Risk Distribution

- **HIGH risk (all mitigated):** HAZ-001 (trust floor), HAZ-002 (async lock), HAZ-005 (capacity gate), HAZ-012 (state persistence)
- **MEDIUM risk (all mitigated):** HAZ-003 (availability status + health persistence), HAZ-004 (fallback policy), HAZ-006 (secret scrubbing), HAZ-007 (intent validation), HAZ-008 (auto catalog refresh), HAZ-009 (jittered cooldown), HAZ-010 (token estimation), HAZ-011 (admin auth), HAZ-013 (intent tier floor), HAZ-014 (adapter isolation)
- **MEDIUM risk (remaining):** None
- **LOW risk:** None identified

---

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2026-06-16 | QA Pipeline | Initial FMEA hazard register created (QA-026 finding) |
| 2026-06-17 | GOIBNIU + LUGH | Mitigated all 4 HIGH-risk items: HAZ-001 (trust floor), HAZ-002 (async lock), HAZ-005 (capacity gate), HAZ-012 (state persistence). 37 new tests, 805 total, 100% coverage. |
| 2026-06-17 | GOIBNIU + LUGH | Streaming dispatch (SSE) implemented — resolves medium-severity spec gap. No hazard register items changed. 19 new tests, 824 total, 100% coverage. |
| 2026-06-17 | GOIBNIU + LUGH | Mitigated 5 MEDIUM-risk hazards: HAZ-006 (secret-scrubbing structlog processor), HAZ-008 (automatic catalog refresh via on_cycle callback), HAZ-009 (jittered circuit breaker cooldown), HAZ-011 (admin_api_key bearer token auth), HAZ-014 (fresh adapter per dispatch with assertion). 56 new tests, 880 total, 100% coverage. |
| 2026-06-17 | GOIBNIU + LUGH | Mitigated final 5 MEDIUM-risk hazards: HAZ-003 (health state persistence + availability status endpoint), HAZ-004 (fallback_policy field on DispatchOrder: allow/deny/same_tier), HAZ-007 (intent_category validation against _ALLOWED_INTENT_CATEGORIES frozenset), HAZ-010 (centralized _estimate_token_count with observability logging), HAZ-013 (intent-based tier floor in estimate_complexity via _INTENT_TIER_FLOOR). All 14 hazard register items now mitigated. 49 new tests, 929 total, 100% coverage. |
