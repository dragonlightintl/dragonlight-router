# Dragonlight Router -- Ground Truth Assessment
**Date:** 2026-06-16
**Assessor:** FIRINNE + GOIBNIU + GNOSIS + LUGH (co-embodied)
**Method:** Full codebase read (every source file, every test file), test execution, spec-to-code traceability

---

## Executive Summary

The Dragonlight Router is approximately **40% functional against its live-spec**. The core `select_models()` path works. The cascade dispatch path (MBR->CBR->LBR) has code but is **structurally broken** due to a critical dual-definition bug: `Ok`/`Err`/`Result` types are defined in both `result.py` AND `core/types.py` as separate classes, causing `isinstance` checks to silently fail across module boundaries. The 9 provider adapters exist but are mocks (7 of 9 are identical copy-paste stubs). The test suite has **55 failures out of 234 tests**, and 2 test files fail to even import. Coverage is 60% against an 80% gate. The delta-spec claims 10 of 12 task modules are complete; the actual number with working, tested implementations is closer to **4 of 12**.

---

## Critical Structural Bug: Dual Result Type Definition

**Severity: BLOCKER -- this undermines the entire Result-type migration.**

Two independent `Ok`/`Err`/`Result` class definitions exist:
- `src/dragonlight_router/result.py` (lines 17-63)
- `src/dragonlight_router/core/types.py` (lines 16-58)

They are **NOT the same class**. `isinstance(core.types.Ok(42), result.Ok)` returns `False`.

Modules importing from **different locations** will silently fail isinstance checks:
- `budget/tracker.py` imports `Ok, Err` from `core.types` (line 17)
- `catalog/cache.py` imports `Ok, Err` from `core.types` (line 18)
- `health/tracker.py` imports `Ok` from `core.types` (line 14)
- `router.py` imports `Ok, Err` from `result` (line 29)
- `dispatch/cascade.py` imports `Ok, Err` from `result` (line 10)
- `server/routes.py` imports `Ok, Err` from `result` (line 14)

**Concrete bug path:** `BudgetTracker.score()` returns `core.types.Ok(100.0)`. `RouterEngine._score_candidates()` checks `isinstance(budget_result, Ok)` where `Ok` is from `result.py`. This check **silently fails**, causing the code to fall through to the `else` branch and use the default value of `100.0` -- masking all budget scoring.

---

## Spec Traceability Matrix

### TM-001: MBR (Model Based Router) -- capability filtering stage

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Filters by complexity_tier OR one tier above | PARTIALLY IMPLEMENTED | `selection/mbr.py` lines 56-67 implement adjacent-tier upgrade. However, the `estimate_complexity()` in mbr.py (line 130) uses a different/simpler heuristic than `complexity.py` (line 34). Two competing complexity estimators exist. |
| AC-2 | Excludes backends with circuit_open | IMPLEMENTED, UNTESTED | `mbr.py` lines 85-104 check `state.status != BackendStatus.CIRCUIT_OPEN`. No unit tests exercise this path (the MBR unit tests all fail -- see below). |
| AC-3 | Graceful upgrade to next tier | IMPLEMENTED, UNTESTED | Lines 64-67. Tests all fail due to import/assertion errors. |
| AC-4 | NEVER downgrades | PARTIALLY IMPLEMENTED | The tier_order array goes LOCAL->SIMPLE->MODERATE->COMPLEX and only tries requested + next higher. But no explicit invariant assertion enforces never-downgrade. |
| AC-5 | Local providers as unlimited-rate passthrough | NOT STARTED | MBR does not handle local-provider rate exemption. That logic exists only in LBR. |

**Overall: PARTIALLY IMPLEMENTED.** Code exists and has real logic, but all 9 MBR unit tests fail (`tests/unit/selection/test_mbr.py`). The test failures are caused by assertion errors on mock data that doesn't satisfy new assertion checks.

### TM-002: CBR (Cost Balancing Router) -- cost scoring stage

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Hard filter: spent >= budget | IMPLEMENTED, UNTESTED | `cbr.py` lines 63-76 filter by budget_score > 0.0. But uses `hasattr(budget_result, 'value')` instead of isinstance due to dual-Ok bug. |
| AC-2 | Scores using ScoringWeights | IMPLEMENTED, UNTESTED | Lines 98-113 use `ScoringWeightsConfig` and `score_candidate()`. But `score_candidate` in `selection/scoring.py` (line 168) takes BackendConfig+order+weights, not the canonical dimensions. |
| AC-3 | Cost governor activates at threshold | IMPLEMENTED, UNTESTED | `scoring.py` line 241 `cost_governor_active()` checks daily/monthly spend. But uses hardcoded 0.0 daily_spend in cbr.py (line 95) -- so the governor never activates. |
| AC-4 | Weight shift when governor active | IMPLEMENTED, UNTESTED | `scoring.py` line 262 returns fixed weights (cost=0.70 etc). |
| AC-5 | All providers exceed budget -> BudgetExceededError | IMPLEMENTED, UNTESTED | `cbr.py` lines 85-87. |

**Overall: PARTIALLY IMPLEMENTED.** Code structure exists but daily_spend is always 0.0, meaning the cost governor can never activate in practice. 4 CBR unit tests fail. Two separate test files exist (`tests/unit/test_cbr.py` and `tests/unit/selection/test_cbr.py`); the former fails to import due to referencing `filter_by_cost_efficiency` which does not exist.

### TM-003: LBR (Load Balancing Router) -- rate enforcement

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Excludes providers exceeding RPM/RPD | PARTIALLY IMPLEMENTED | `lbr.py` lines 93-129 check rate limits but use budget_tracker.score() as a proxy for usage ratio rather than actual RPM/RPD remaining counts. The conversion logic (`1.0 - score/100.0`) is a rough approximation. |
| AC-2 | Deprioritizes within 80% of limits | IMPLEMENTED, UNTESTED | Lines 132-139. But deprioritized providers are still appended, not reordered. |
| AC-3 | Selects top candidate, returns DispatchDecision | STUB | `select_final_candidate()` (line 189) simply returns `candidates[0]` -- no tie-breaking, no WFQ scheduling, no DispatchDecision type. |
| AC-4 | Local providers unlimited-rate | IMPLEMENTED, UNTESTED | Lines 104-111 check for local providers. Heuristic is fragile (RPM=0 AND RPD=0 OR RPM>10000 AND RPD>100000). |
| AC-5 | Zero candidates -> RoutingError with diagnostics | IMPLEMENTED, UNTESTED | Lines 153-179 return LBRNoCapacityError with diagnostic dict. |

**Overall: PARTIALLY IMPLEMENTED.** The filtering logic has real code but uses budget score as a proxy rather than actual rate limit consumption tracking. 4 LBR unit tests fail.

### TM-004: Cascade Dispatch (MBR->CBR->LBR composition)

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | route() applies MBR->CBR->LBR in fixed order | IMPLEMENTED, UNTESTED | `dispatch/cascade.py` lines 27-137. The pipeline is MBR->CBR->LBR in that order. |
| AC-2 | Primary failure -> fallback to next candidate | NOT STARTED | No fallback logic exists. If MBR/CBR/LBR return Err, the error propagates immediately. No retry with next candidate. |
| AC-3 | EngineResponse.was_fallback=True when cascade advanced | NOT STARTED | The dispatch() function (line 140) creates a placeholder EngineResponse with `was_fallback=False` always. |
| AC-4 | fallback_chain lists all attempted backends | NOT STARTED | Always returns `fallback_chain=[]`. |
| AC-5 | Dispatch log + budget update transactional | NOT STARTED | No dispatch logging or transactional budget updates. |
| AC-6 | All backends exhausted -> DispatchFailure | PARTIALLY IMPLEMENTED | Returns Err(Exception) but not DispatchFailure type. |

**Overall: PARTIALLY IMPLEMENTED.** The cascade pipeline structure exists but the dispatch() function (line 140-210) is a **placeholder** that returns a mock EngineResponse with `content="[Generation would happen here via PAL adapter]"`. No actual generation occurs. No fallback chain logic.

### TM-005: Provider Adapters (8 providers)

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Each implements generate(), health_check(), record_usage() | PARTIALLY IMPLEMENTED | 9 adapter files exist (Anthropic, Cohere, Google, Groq, Local, Mistral, OpenAI, OpenRouter, Together). All implement the protocol. However, 7 of 9 are identical **mock stubs** -- they yield `"[Mock Provider] Response to: ..."` and never make real API calls. Only OpenRouter has real HTTP implementation. |
| AC-2 | OpenAI-compatible wire format | IMPLEMENTED (OpenRouter only) | OpenRouter adapter (lines 34-98) uses the chat completions wire format. Others are mocks. |
| AC-3 | Anthropic handles /v1/messages | NOT STARTED | `anthropic.py` is a mock stub identical to all others. No /v1/messages handling. |
| AC-4 | Ollama adapter no API key | NOT STARTED | There is no Ollama adapter. `local.py` exists but doesn't reference Ollama. |
| AC-5 | Each reads API key from env_key | PARTIALLY IMPLEMENTED | Mock stubs read env_key but only to generate mock error messages. Only OpenRouter actually uses the key. |

**Overall: STUB/SKELETON.** 8 of 9 adapters are mock stubs. Only OpenRouter has a real (though untested) implementation. The spec calls for 8 providers; the codebase has 9 (adding Together which is not in the original spec). No adapter unit tests exist.

### TM-006: Context Trust Tier Filtering (DIAN CECHT)

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | TRUSTED -> full context | IMPLEMENTED, UNTESTED | `context_filter.py` line 94-97. No tests cover this (0% coverage). |
| AC-2 | SEMI_TRUSTED -> no behavioral rules/persona | IMPLEMENTED, UNTESTED | Lines 99-124. Has real filtering logic. |
| AC-3 | UNTRUSTED -> task instruction only | IMPLEMENTED, UNTESTED | Lines 126-130. |
| AC-4 | LOCAL -> full context | IMPLEMENTED, UNTESTED | Lines 132-135. |
| AC-5 | PII never present | UNABLE TO VERIFY | No PII detection or enforcement. The assumption is PII is "pre-redacted by CAL" -- not enforced by this module. |

**Overall: IMPLEMENTED, UNTESTED.** The `filter_context_for_provider()` function has real logic, but the `filter_by_trust_tier()` function (line 30) operates on `TrustTier` enums, not `BackendConfig` objects. The context filter is NOT integrated into the cascade dispatch path. 0% test coverage. Note: `test_context_filter.py` exists but was created after the delta-spec audit and is not exercised.

### TM-007: ScoringWeights + CostGovernor

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | ScoringWeights dataclass with canonical defaults | IMPLEMENTED & VERIFIED | Two competing implementations: `scoring.py` (root, line 98) has `ScoringWeights` with 6 dimensions (rank, budget, health, cost, latency, queue_depth). `selection/scoring.py` has `ScoringWeightsConfig` (line 56) with 5 dimensions (cost, latency, priority, queue, health). Neither matches the canonical spec values exactly. |
| AC-2 | score_candidate() normalizes to [0,1] | IMPLEMENTED, UNTESTED | `selection/scoring.py` lines 168-238 normalizes and applies weights. However, `tests/unit/test_scoring.py` fails to import because it references `compute_budget_score` and `compute_health_score` which exist only in root `scoring.py`, not `selection/scoring.py`. |
| AC-3 | cost_governor_active() | IMPLEMENTED, UNTESTED | Two competing implementations: `scoring.py` line 203 takes `(config, daily_spend)`, `selection/scoring.py` line 241 takes `(daily_spend, monthly_spend, config)`. Different signatures, different semantics. |
| AC-4 | cost_adjusted_weights() | IMPLEMENTED, UNTESTED | Again two versions: `scoring.py` line 121 does proportional weight shifting, `selection/scoring.py` line 262 returns hardcoded values. |
| AC-5 | Score deterministic | UNABLE TO VERIFY | No tests verify determinism. |
| AC-6 | Score >= 0.0 | IMPLEMENTED | Postcondition assertions exist in both scoring modules. |

**Overall: PARTIALLY IMPLEMENTED.** Two competing scoring modules with incompatible APIs. The cascade path uses `selection/scoring.py`. The old path uses root `scoring.py`. Test file references non-existent functions, causing import failure.

### TM-008: Health Check Background Loop

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Loop runs every 30s | IMPLEMENTED, UNTESTED | `check_loop.py` lines 56-82. The asyncio loop exists. Constructor takes `interval_s=30.0`. |
| AC-2 | 3 consecutive SLO violations -> degraded | IMPLEMENTED, UNTESTED | Lines 189-219. Logic exists but has an **indentation bug** at line 189 -- the `if` block inside the `else` clause is indented one level too deep. Python parses it as valid syntax (since it's inside the else), but the logical flow may not match intent. |
| AC-3 | Degraded providers get ranking penalty | NOT STARTED | The health check sets `BackendStatus.DEGRADED` but nothing in MBR/CBR/LBR applies a penalty for degraded status. |
| AC-4 | Failures don't crash the loop | PARTIALLY IMPLEMENTED | Lines 77-82 catch `CancelledError`. But line 162 has `except Exception as exc` which catches broadly. |

**Overall: PARTIALLY IMPLEMENTED.** The loop infrastructure exists, SLO violation tracking exists, but the `except Exception` on line 162 violates coding standards. The degraded status is never consumed downstream. 4 check_loop tests fail.

### TM-009: HTTP API Dispatch Endpoints

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | POST /v1/dispatch | PARTIALLY IMPLEMENTED | `dispatch_handler()` exists in `routes.py` (line 152) but is **NOT registered** as a route in `app.py`. The handler exists but is unreachable. |
| AC-2 | /v1/select includes trust_tier and complexity_tier | NOT STARTED | `select_handler` returns model IDs and scores but no trust_tier or complexity_tier. |
| AC-3 | POST /v1/retire and /v1/reinstate | NOT STARTED | No retire or reinstate handlers exist in `routes.py`. No route registrations in `app.py`. |
| AC-4 | Structured error responses | PARTIALLY IMPLEMENTED | Existing handlers return `{"error": "..."}` JSON. But `dispatch_handler` line 217 has `except Exception as exc` -- the last violation remaining. |

**Overall: PARTIALLY IMPLEMENTED.** The dispatch handler code exists but is dead code (not routed). Retire/reinstate endpoints are completely absent.

### TM-010: RouterEngine.dispatch() graft

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | dispatch() returns EngineResponse or DispatchFailure | PARTIALLY IMPLEMENTED | `router.py` line 332 defines `dispatch()`. Returns `Result[EngineResponse, Exception]` but the cascade `dispatch()` returns a **placeholder** EngineResponse with mock content. |
| AC-2 | select_models() unchanged | IMPLEMENTED & VERIFIED | `select_models()` (line 120) is untouched from pre-dispatch era. 179 passing tests verify basic behavior. |
| AC-3 | dispatch() integrates MBR->CBR->LBR + PAL adapter + fallback | PARTIALLY IMPLEMENTED | Delegates to `cascade.dispatch()` which runs MBR->CBR->LBR but returns placeholder response (line 193: `content="[Generation would happen here via PAL adapter]"`). No PAL adapter integration. No fallback. |
| AC-4 | No ImportError | IMPLEMENTED & VERIFIED | The module imports without error. |

**Overall: PARTIALLY IMPLEMENTED.** The method exists, the cascade pipeline runs, but the actual generation step is a placeholder.

### TM-011: Integration Test Suite

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Full cascade selects optimal provider | STUB | `test_cascade_dispatch.py` line 37. Test exists but uses `patch.object(router_engine._registry, 'get_backend')` -- `BackendRegistry` has no `get_backend` method (it has `get()`). Test fails with AttributeError. |
| AC-2 | Primary failure triggers fallback | STUB | Line 68. Same `get_backend` issue. Fails. |
| AC-3 | All backends fail -> DispatchFailure | STUB | Line 86. Same issue. Fails. |
| AC-4 | Circuit breaker opens after 3 errors | STUB | Line 101. Passes trivially (just asserts `result is not None`). |
| AC-5 | Budget exhaustion deprioritizes | STUB | Line 113. Same trivial assertion. |
| AC-6 | Catalog refresh failure degrades | STUB | Line 124. Mocks catalog but dispatch runs through. |

**Overall: STUB/SKELETON.** 3 of 6 tests fail. The passing tests assert only `result is not None` which proves nothing. No real integration wiring.

### TM-012: BudgetTracker TPM + Daily Token Cap

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | TPM via sliding window | IMPLEMENTED & VERIFIED | `tracker.py` lines 39, 103, 165-186. Sliding window pattern identical to RPM. |
| AC-2 | Daily token cap (0 = unlimited) | IMPLEMENTED & VERIFIED | Lines 40, 104, 188-196. 0 handling verified by default return of 0 in `_daily_token_remaining`. |
| AC-3 | score() incorporates TPM | IMPLEMENTED & VERIFIED | Lines 60-61, 81-86, 94. Score uses `min(rpm_ratio, rpd_ratio, tpm_ratio, daily_token_ratio) * 100`. |
| AC-4 | has_capacity() checks TPM/token cap | NOT STARTED | `has_capacity()` (line 106) only checks RPM and RPD. Does NOT check TPM or daily token cap. |

**Overall: PARTIALLY IMPLEMENTED.** TPM tracking and daily token counting work. `score()` incorporates them. But `has_capacity()` does not enforce TPM or daily token cap limits, which means the hard-filter path is incomplete.

---

## Architecture Assessment (GOIBNIU)

### Load-Bearing vs. Ceremonial

**Load-bearing components (actually execute in production):**
- `RouterEngine.select_models()` -- the working entry point
- `BudgetTracker` -- RPM/RPD/TPM/daily token scoring
- `HealthTracker` + `CircuitBreaker` -- health scoring with circuit breaker
- `CatalogCache` + `CatalogRefresher` -- provider catalog management
- `RoleMatrix` -- role-to-model mapping with hot reload
- `interleave_providers()` -- provider diversity enforcement
- `compute_composite_score()` -- weighted scoring (rank/budget/health)
- Config loading pipeline (loader -> schema -> RouterConfig)

**Ceremonial/placeholder components (exist but don't function):**
- `dispatch/cascade.py dispatch()` -- returns placeholder EngineResponse
- All provider adapters except OpenRouter -- mock stubs
- `dispatch_handler` in routes.py -- defined but never registered
- `scoring.py` (root) -- orphaned duplicate of `selection/scoring.py`
- `result.py` -- orphaned duplicate of `core/types.py` Ok/Err/Result
- `caching/simple.py` + `caching/semantic.py` -- exist but not used by any routing path
- `selection/context_filter.py` -- exists but not integrated into cascade

### Structural Issues

1. **Dual Result type (CRITICAL):** `result.py` and `core/types.py` define incompatible Ok/Err classes. Cross-module isinstance checks silently fail. This is the most dangerous bug in the codebase because failures are silent.

2. **Dual scoring module:** `scoring.py` (root) and `selection/scoring.py` have overlapping but incompatible APIs. The cascade uses `selection/scoring.py`; the test file references functions from root `scoring.py` that don't exist in `selection/scoring.py`.

3. **Unreachable code:** `router.py` lines 292, 303-305 have postcondition assertions placed AFTER `return` statements, making them dead code.

4. **Dead dispatch handler:** `routes.py` defines `dispatch_handler` but `app.py` doesn't register it as a route. The endpoint is unreachable.

5. **check_loop.py indentation anomaly:** Line 189 has an `if` indented one level deeper than its containing `else` block. Python accepts it but the logical flow is suspect.

6. **Missing __init__.py:** `tests/unit/selection/` has test files but may lack proper test discovery configuration.

### Dependency Integrity

**Declared in pyproject.toml:**
- `pydantic>=2.0` -- used (config/schema.py)
- `pyyaml>=6.0` -- used (config/loader.py)
- `httpx>=0.25` -- used (catalog/refresher.py, adapters/openrouter.py)
- `structlog>=23.0` -- used (many modules)
- `returns>=0.22` -- declared but **NOT used**. The codebase has its own Ok/Err types.

**Used but not declared:**
- `aiohttp` -- used in `router.py` (import on line 16) and `check_loop.py` (line 18). Not in pyproject.toml dependencies.

---

## Implementation Quality (LUGH)

### Error Handling
- 5 `except Exception` violations remain: `check_loop.py:162`, `routes.py:217`, and the original violations have been partially fixed.
- The dual-Ok bug means Result-type error handling is silently bypassed in some paths.
- `hasattr(budget_result, 'value')` pattern in `cbr.py:70` and `lbr.py:70` is a workaround for the dual-Ok bug rather than proper isinstance dispatch.

### Edge Cases
- `has_capacity()` doesn't check TPM or daily token cap despite BudgetTracker supporting them.
- `select_final_candidate()` returns `candidates[0]` with no tie-breaking.
- Cost governor never activates because `daily_spend` is hardcoded to 0.0 in both `cascade.py:88` and `cbr.py:95`.
- `health_snapshot()` has unreachable postcondition assertions (lines 292-293 after return).
- `budget_snapshot()` has unreachable postcondition assertions (lines 303-305 after return).

### Interface Precision
- `dispatch()` returns `Result[EngineResponse, Exception]` but the spec calls for `EngineResponse | DispatchFailure`.
- `route()` returns `Result[BackendConfig, Exception]` rather than a more specific error type.
- The `ScoringWeightsConfig` (selection/scoring.py) has 5 dimensions while the canonical spec calls for 6.

---

## Delta Spec Accuracy

The delta-spec JSONL claims these tasks are complete:
**TM-001, TM-002, TM-003, TM-004, TM-005, TM-007, TM-009, TM-010, TM-011, TM-012**

Only **TM-006** and **TM-008** are implicitly NOT listed as complete.

### Where the delta spec is WRONG:

| Task | Delta Says | Ground Truth | Discrepancy |
|------|-----------|--------------|-------------|
| TM-001 | Complete | PARTIALLY IMPLEMENTED | All 9 MBR tests fail. Local-provider passthrough not implemented. |
| TM-002 | Complete | PARTIALLY IMPLEMENTED | cost_governor never activates (daily_spend=0.0). Test file imports non-existent function. 4 CBR tests fail. |
| TM-003 | Complete | PARTIALLY IMPLEMENTED | select_final_candidate is trivial stub. No WFQ scheduling. 4 LBR tests fail. |
| TM-004 | Complete | PARTIALLY IMPLEMENTED | dispatch() returns placeholder EngineResponse. No fallback logic. No actual generation. |
| TM-005 | Complete | STUB/SKELETON | 8 of 9 adapters are copy-paste mock stubs. Only OpenRouter has real code. |
| TM-007 | Complete | PARTIALLY IMPLEMENTED | Two competing scoring modules with incompatible APIs. Test file fails to import. |
| TM-009 | Complete | PARTIALLY IMPLEMENTED | dispatch_handler not registered. Retire/reinstate endpoints missing entirely. |
| TM-010 | Complete | PARTIALLY IMPLEMENTED | dispatch() exists but returns placeholder. No PAL adapter integration. |
| TM-011 | Complete | STUB/SKELETON | 3 of 6 integration tests fail. Passing tests assert only `is not None`. |
| TM-012 | Complete | PARTIALLY IMPLEMENTED | has_capacity() doesn't check TPM or daily token cap. |

### Where the delta spec is ACCURATE:

The delta-spec.md (the human-readable version) is actually more honest than the JSONL. The .md correctly identifies many gaps. The JSONL `completed_tasks` array overstates completion.

---

## Risk Assessment

### Would Break First Under Real Load

1. **Dual Result type bug** -- any code path crossing the result.py / core.types boundary will silently produce wrong results. Budget scoring in the select_models path is actively affected.

2. **Mock adapters** -- the system cannot actually dispatch requests. The placeholder EngineResponse means no real LLM calls can be made.

3. **Missing route registration** -- POST /v1/dispatch is dead code. Any client trying to use the dispatch endpoint gets a 404.

4. **aiohttp not declared** -- `pip install dragonlight-router` would fail to install aiohttp, breaking `router.py` and `check_loop.py` imports.

5. **Test suite instability** -- 55 of 234 tests fail. 2 test files don't import. Any CI pipeline (which the project has configured) would be red.

### Fragile Points

- `check_loop.py` has an indentation anomaly and `except Exception`
- The cost governor path is untestable without a spend tracking mechanism
- `_refresh_catalog()` uses `asyncio.run()` inside a sync method, which will fail if an event loop is already running

---

## Recommended Priority Actions

1. **Fix the dual Result type (BLOCKER).** Consolidate Ok/Err/Result into ONE location (either `result.py` or `core/types.py`, not both). Update all imports. This is a 30-minute fix that unblocks everything else.

2. **Fix the test suite.** The 55 failing tests indicate real bugs (import errors, assertion mismatches on mock data, wrong function references). Getting tests green reveals the actual state of the codebase.

3. **Add `aiohttp` to pyproject.toml dependencies.** It's imported but not declared.

4. **Remove `returns>=0.22` from dependencies.** It's declared but unused.

5. **Register the dispatch_handler route in app.py.** One line fix to make the endpoint reachable.

6. **Fix unreachable code in router.py.** Move postcondition assertions before `return` statements (lines 292, 303-305).

7. **Delete or consolidate duplicate scoring modules.** Either `scoring.py` or `selection/scoring.py` should be the canonical source. Having both causes confusion and test failures.

8. **Make has_capacity() check TPM and daily_token_cap.** Extend the 2-line method to check all 4 dimensions.

9. **Replace mock adapters with real implementations.** At minimum, Anthropic and OpenAI adapters need real HTTP client code. The OpenRouter adapter can serve as a template.

10. **Wire context_filter into the cascade dispatch path.** The module exists but is disconnected from the pipeline.

11. **Implement actual generation in dispatch().** The current placeholder response is meaningless. This requires working adapters (action 9).

12. **Implement fallback chain logic in cascade.py.** When the primary backend fails, try the next candidate in the LBR-ranked list.

---

## Test Execution Evidence

```
55 failed, 179 passed in 2.88s
2 collection errors (test_cbr.py, test_scoring.py fail to import)
Coverage: 60.35% (gate: 80% -- FAIL)
```

Key failure categories:
- 9 MBR tests: assertion errors on mock BackendConfig data
- 4 CBR tests (selection/): various type/assertion errors
- 4 LBR tests: assertion and type errors
- 3 integration tests: AttributeError on non-existent `get_backend` method
- 4 check_loop tests: various assertion/logic errors
- 4 persistence tests: Result type mismatch
- 5 router_engine tests: Result type cross-module isinstance failures
- 11 server tests: RuntimeError on event loop / Result type mismatches
- 4 types tests: ProviderConfig constructor signature mismatch
- 2 config tests: assertion failures on defaults
- 4 catalog tests: NameError / Result type mismatches
- 1 registry test: health_snapshot format mismatch

---

## Operational Readiness Assessment (FIRINNE)

*Adversarial ground-truth verification: does this thing actually work, or does it merely exist?*

### Overall Readiness Level

**DEV READY (lower end)**

The `select_models()` path functions as a Python library with real logic and 179 passing unit tests covering its subsystems individually. However, the test suite as a whole is broken (55 failures, 2 collection errors, 60% coverage vs. 80% gate), no test starts the actual server or sends a real HTTP request, the dispatch path returns placeholder content, and all provider adapters except one are mock stubs. An operator could import `RouterEngine` and call `select_models()` in a Python script today. They could not start the HTTP server, send a dispatch request, or receive an actual LLM response.

### Test Veracity

#### What the tests actually prove

The 179 passing tests prove that **individual subsystem internals work in isolation**: BudgetTracker correctly tracks sliding-window RPM/RPD/TPM/daily-token budgets and computes scores. HealthTracker correctly tracks errors, latencies, and circuit breaker state. CircuitBreaker correctly transitions between CLOSED/OPEN/HALF_OPEN states. CatalogCache correctly serializes/deserializes and enforces TTL. RoleMatrix correctly loads, ranks, and hot-reloads JSON files. Interleave correctly prevents provider concentration. SimpleCache and SemanticCache correctly store/retrieve/evict responses. Complexity estimation correctly maps intent+context to tiers. BackendState correctly tracks per-backend counters and circuit breakers.

These are **real behavioral tests**, not mock-testing-mocks. The BudgetTracker tests construct real BudgetTracker instances with real ProviderConfig objects, call real methods, and assert real outcomes. The CircuitBreaker tests use `time.sleep()` to exercise real timing behavior. The CatalogCache tests write real JSON files. This is genuine unit testing.

#### What the tests do NOT prove

1. **No test starts the router and sends a real HTTP request.** The server tests in `test_server.py` use Starlette's `TestClient` (which is close to an integration test), but all 11 of those tests fail. If they passed, they would be the most valuable tests in the suite because they would exercise the actual HTTP interface. They fail due to event loop issues and Result-type mismatches, meaning the HTTP path has never been successfully tested.

2. **No test exercises the cascade dispatch end-to-end.** The "integration" tests in `test_cascade_dispatch.py` are labeled as integration tests but are actually mock-heavy unit tests. They mock `_registry.get_backend` (which does not exist -- the method is `get()`), `_health.score`, `_budget.score`, and `_catalog.get`. Three of 6 tests fail with AttributeError. The 3 that pass assert only `result is not None`, which proves nothing about correctness.

3. **No test verifies that a request enters the system and a response exits.** The single most important test for a router -- "given this input, which backend does it select?" -- does not exist in a form that passes.

4. **No test verifies the dispatch function generates actual content.** `dispatch()` returns `content="[Generation would happen here via PAL adapter]"`. No test asserts anything about the content field because there is nothing real to assert.

#### Tests testing mocks vs. behavior

- `test_cascade_dispatch.py`: 6 tests, ALL mock-testing-mocks. They mock registry, health, budget, catalog, then assert the mock was called or that the result is not None. Even if they passed, they would prove wiring, not behavior. **3 fail, 3 pass trivially.**
- `test_check_loop.py`: 10 tests. 6 pass by mocking `_probe_backend`, `asyncio.sleep`, and `session.get`, then asserting mock state changes. These prove the HealthCheckLoop correctly updates internal state when mock probes return mock results. **4 fail.**
- `test_router_engine.py`: 17 tests. 5 use `patch.object` on `_refresher.refresh` to mock catalog refresh. The other 12 construct real RouterEngine instances with real config and test real behavior. The non-mocked tests are high-value. **7 fail.**

#### Coverage topology

| Category | Test Count | Passing | Failing | Import Error |
|----------|-----------|---------|---------|-------------|
| Unit tests (real behavior, no mocks) | ~170 | 155 | 15 | 0 |
| Unit tests (mock-heavy) | ~30 | 20 | 10 | 0 |
| Integration tests (labeled) | 6 | 3 | 3 | 0 |
| E2E / smoke tests (actual running system) | 0 | 0 | 0 | 0 |
| Tests that fail to import | ~30 (estimated across 2 files) | 0 | 0 | 2 files |

**Ratio: ~170 real unit / ~30 mock-heavy unit / 6 nominal integration / 0 e2e.**

The zero e2e tests is a significant finding. There is no test that starts the server and sends a curl-equivalent request.

#### Assertion-free tests

Three tests in `test_cascade_dispatch.py` assert only `result is not None`:
- `test_circuit_breaker_opens_after_3_consecutive_errors` (line 109)
- `test_budget_exhaustion_deprioritizes_expensive_providers` (line 121)
- `test_catalog_refresh_failure_degrades_gracefully` (line 135)

These tests pass, but they prove only that `dispatch()` does not crash -- they make no assertion about which backend was selected, what the response contains, or whether circuit breaker/budget/catalog logic was exercised.

#### Error path coverage

Error paths are **reasonably well tested** in the passing subsystems:
- BudgetTracker: tests cover RPM exhaustion, RPD exhaustion, TPM exhaustion, daily token cap exhaustion, unknown provider, zero RPM
- CircuitBreaker: tests cover error threshold, error window expiry, half-open recovery, half-open failure, success reset
- HealthTracker: tests cover 1/2/3 errors, circuit open, success after error
- CatalogCache: tests cover missing file, corrupt file, stale cache, empty cache
- Persistence: tests cover missing file, corrupt file, empty file (but all 4 load tests fail)
- Config: tests cover missing file (fails), default loading (fails)
- RoleMatrix: tests cover missing file, unknown role, hot reload

The error paths that are NOT tested are the ones that matter most at the system level: what happens when a real provider returns a 429? What happens when a real health check times out? What happens when the cascade exhausts all backends? These all exist only as mock-level tests, and most of those mock tests fail.

### Operator Experience

#### Can you install it?

`pyproject.toml` is well-structured with proper extras (`[server]`, `[adapters]`, `[cache]`, `[all]`, `[dev]`). The `pip install -e ".[all]"` command would work except for one problem: **`aiohttp` is imported by `router.py` and `check_loop.py` but is not declared as a dependency.** The install would succeed, but importing `RouterEngine` would fail with `ModuleNotFoundError: No module named 'aiohttp'` unless aiohttp is coincidentally installed in the environment. Additionally, `returns>=0.22` is declared as a dependency but is never used -- wasted install.

#### Can you start it?

There is a CLI entry point: `dragonlight-router = "dragonlight_router.server.app:main"`. The `main()` function in `app.py` calls `uvicorn.run(app, host=host, port=port)`. If aiohttp were installed (or the import fixed), this would likely start a server. However, `create_app()` on line 33 calls `loop = asyncio.get_event_loop()` followed by `loop.create_task(...)` -- this pattern is deprecated in Python 3.10+ and will emit a DeprecationWarning. In Python 3.12+, if no event loop is running, `get_event_loop()` raises a DeprecationWarning and may fail. The server tests all fail, which is indirect evidence that this path has problems.

#### Can you send it a request?

The README shows clear curl examples for `/v1/select`, `/v1/record`, `/v1/health`, `/v1/catalog`, and `/v1/catalog/refresh`. The request/response formats are documented. However, the `/v1/dispatch` endpoint (the most important one for engine-style consumers) is **not documented in the README** and is also **not registered as a route** in `app.py`. A client reading the README would know how to use `/v1/select` but would not know about dispatch, and even if they did, it would 404.

#### Is the README accurate?

The README is **well-written and largely accurate for the v0.1 scope**. It explicitly states in the "v0.1 scope" section: "dragonlight-router **selects** models. It does **not** dispatch requests." This is honest and matches reality. The README correctly describes `select_models()` as the primary interface and shows realistic usage patterns. The architecture table is accurate. The provider list is accurate. The "v0.1 scope" section even includes an ASCII diagram showing the expected flow.

The README is aspirational in one way: it says `pip install dragonlight-router[all]` as if it's a published package. It's not on PyPI. This is a local dev install.

#### Example configs and run scripts

- `config/router.yaml` exists with 8 real provider configurations. It references real API endpoints with real rate limits.
- `.env.example` exists with all required environment variables listed.
- `.github/workflows/ci.yml` exists with a proper CI pipeline (Python 3.11/3.12, ruff, mypy, pytest).
- No Docker, docker-compose, Makefile, or run scripts exist.
- No example scripts or Jupyter notebooks demonstrating usage.

#### Has this ever been run?

**No evidence of actual runtime execution exists.** There are no logs, no screenshots, no recorded demos. The git history (`git log --all --oneline --grep="run|deploy|test|demo|local"`) returns zero results. All commits are implementation commits ("feat:", "fix:", "test:"). A `.coverage` file exists (evidence that pytest was run with coverage), and `.pytest_cache` has cached data (evidence that pytest has been executed). `.ruff_cache` and `.mypy_cache` exist (evidence that linting/type-checking has been done). But there is no evidence that the server has ever been started, that a request has ever been sent, or that any provider API has ever been called.

### Production Concerns

#### Error handling

**Mixed quality.** The subsystems have good assertion-based precondition checking (every function validates its inputs). The `filter_by_cost`, `filter_by_rate_limit`, `filter_by_capabilities` functions all validate that candidates are lists of BackendConfig, orders are DispatchOrder, etc. However:
- 5 `except Exception` violations remain (broad catches that swallow errors)
- The dual Result type bug causes isinstance checks to silently fail, bypassing error handling entirely
- `dispatch()` in cascade.py has a comments-only placeholder where real error handling and fallback should be
- `cbr.py` line 70 uses `hasattr(budget_result, 'value')` as a workaround for the dual-Ok bug, which would also match any object with a `.value` attribute

There is no handling for: malformed HTTP request bodies beyond basic JSON parsing, request timeouts, connection draining, or partial failure in catalog refresh.

#### Logging/observability

**Structured logging is present and well-done.** The codebase uses `structlog` throughout with contextual key-value logging. Log lines include relevant context (`candidate_count`, `intent_category`, `budget_scores`, `error_type`). Debug-level logging traces the cascade path. This is one of the strongest aspects of the codebase. If the system were running, its logs would be useful. No metrics, no tracing, no Prometheus/StatsD/OpenTelemetry integration.

#### Configuration

**Well-designed.** External configuration via `router.yaml` with env var overrides. Hot-reloadable role matrix (no restart needed). Pydantic-validated config schema with defaults. Provider-specific rate limits. `DRAGONLIGHT_HOST` and `DRAGONLIGHT_PORT` env vars for server bind. No hardcoded secrets or URLs in source code.

#### Graceful shutdown

**Not implemented.** No signal handling (SIGTERM, SIGINT). No connection draining. The `HealthCheckLoop` has a `stop()` method that cancels the asyncio task, but nothing calls it on shutdown. Uvicorn provides some shutdown handling by default, but the application doesn't hook into it. The `aiohttp.ClientSession` created in `check_loop.py` is never explicitly closed, which would leak connections.

#### Health checks

**Partially implemented.** `GET /v1/health` returns a budget and health snapshot. This could serve as a readiness probe. However, there is no startup readiness check (e.g., "have all providers been probed at least once?"), no liveness probe separate from the health endpoint, and no self-test capability. The health check loop probes backends every 30 seconds but its results are not aggregated into the `/v1/health` response (the `health_snapshot()` method returns registry data, not health tracker data).

#### Security

**Absent.** No authentication on any endpoint. No TLS configuration. No input validation beyond basic JSON parsing and required-field checks. No rate limiting on the API endpoints themselves (ironic for a rate-limit-aware router). No CORS configuration. Any client on the network can call any endpoint. This is acceptable for a development sidecar but not for any shared or production deployment.

#### Dependencies

Declared in `pyproject.toml`:
- `pydantic>=2.0` -- used, appropriate, unpinned minor (acceptable for alpha)
- `pyyaml>=6.0` -- used, appropriate
- `httpx>=0.25` -- used by catalog refresher and OpenRouter adapter
- `structlog>=23.0` -- used throughout
- `returns>=0.22` -- **declared but unused**. The codebase has its own Ok/Err implementation.
- `aiohttp` -- **used but not declared**. This is a runtime import error waiting to happen.
- `starlette`, `uvicorn`, `openai`, `aiosqlite` -- correctly in optional extras
- Dev deps (`pytest`, `mypy`, `ruff`) correctly in `[dev]` extra

No lock file exists. Dependencies are floor-pinned but not ceiling-pinned, which is appropriate for a library but risky for a deployed service.

### The Bottom Line

If Korrigon needed to demo this to a client tomorrow, the demo would be: "here is a Python REPL where I import RouterEngine, call `select_models('coding')`, and get back a ranked list of model IDs." That works. The budget scoring, health tracking, circuit breaker, provider interleaving, and catalog management behind that call are real, tested, and functional.

The demo could NOT be: "here is a running HTTP service that accepts requests, routes them to the optimal LLM, and returns generated content." The server has never been started (no evidence in git history, test suite, or filesystem). The dispatch endpoint is dead code (defined but not routed). All provider adapters except OpenRouter are mock stubs that return `"[Mock Provider] Response to: ..."`. The cascade dispatch function returns `"[Generation would happen here via PAL adapter]"`. The test suite is red (55 failures, 60% coverage, 2 import errors).

What genuinely works: model selection. What merely exists: everything else. The gap between "select which model to use" and "actually use that model" is the entire value proposition of a router, and that gap is currently filled with placeholders. The `select_models()` path is a solid foundation. Building from here to a functional dispatch system requires fixing the dual Result type bug, replacing mock adapters with real ones, wiring the dispatch handler into routes, and writing the first test that starts the server and sends a real request.
