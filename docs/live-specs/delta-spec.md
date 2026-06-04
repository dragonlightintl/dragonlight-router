# Dragonlight Router — Implementation Delta & Remediation Plan

**Delta ID:** dragonlight-router-delta-v0.2.0-2026-05-30  
**Spec Baseline:** live-spec-v0.2.0-2026-05-30  
**Auditor:** Hermes Agent (Korrigon @ Dragonlight International)

---

## Executive Summary

The dragonlight-router currently delivers **~35% of live-spec acceptance criteria** and complies with **~48% of coding standards**. Test coverage is **37.77%** against an 80% gate. This document identifies 47 specific remediation items across 24 tasks, organized into 8 parallel execution waves.

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Spec Parity | 35% | 100% | 65% |
| Standards Compliance | 48% | 100% | 52% |
| Test Coverage | 37.77% | 80%+ | 42.23% |
| Critical Blockers | 7 | 0 | 7 |
| Total Remediation Items | 47 | 0 | 47 |
| Estimated Agent Hours | — | 28 | — |

---

## 1. Spec Parity Matrix

The live-spec defines 12 task modules. Only 3 have partial implementations; 9 have zero implementation.

| Task | Title | Status | AC Met | Parity |
|------|-------|--------|:------:|:------:|
| TM-001 | MBR (Model Based Router) — capability filtering stage | ❌ NOT_STARTED | 0/5 | 0% |
| TM-002 | CBR (Cost Balancing Router) — cost scoring stage | ❌ NOT_STARTED | 0/4 | 0% |
| TM-003 | LBR (Load Balancing Router) — rate-limit-aware dispatch | ❌ NOT_STARTED | 0/4 | 0% |
| TM-004 | Cascade Dispatch — MBR→CBR→LBR composition | ❌ NOT_STARTED | 0/5 | 0% |
| TM-005 | Provider Adapters — 8 provider implementations | ❌ NOT_STARTED | 0/5 | 0% |
| TM-006 | Context Trust Tier Filtering — DIAN CECHT | ❌ NOT_STARTED | 0/4 | 0% |
| TM-007 | Canonical ScoringWeights + CostGovernor | ⚠️ PARTIAL | 1/5 | 20% |
| TM-008 | Health Check Loop — periodic backend probing | ❌ NOT_STARTED | 0/4 | 0% |
| TM-009 | HTTP API Dispatch Endpoints | ⚠️ PARTIAL | 3/8 | 37% |
| TM-010 | RouterEngine.dispatch() graft | ❌ NOT_STARTED | 1/4 | 25% |
| TM-011 | Integration Tests — cascade dispatch pipeline | ❌ NOT_STARTED | 0/5 | 0% |
| TM-012 | BudgetTracker TPM + Daily Token Cap | ⚠️ PARTIAL | 2/6 | 33% |

### Key Findings

1. **The entire dispatch path is missing.** `dispatch/cascade.py` does not exist. The MBR→CBR→LBR cascade has zero implementation.
2. **All 8 provider adapters are missing.** `adapters/` is an empty `__init__.py`.
3. **MBR, CBR, LBR stages don't exist.** `selection/mbr.py`, `selection/cbr.py`, `selection/lbr.py` are absent.
4. **Context trust tier filtering (DIAN CECHT) is absent.** PII boundary enforcement is unimplemented.
5. **Health check loop is missing.** No periodic backend probing.
6. **BudgetTracker lacks TPM and daily token cap.** Only RPM/RPD tracked.
7. **RouterEngine.dispatch() doesn't exist.** Only `select_models()` is implemented.

### What Works Today

- ✅ `select_models()` — returns ranked model IDs
- ✅ Budget scoring (RPM/RPD), Health tracking + circuit breaker, Catalog caching + TTL
- ✅ Role matrix hot-reload, Provider interleaving, HTTP endpoints (select/record/health)
- ✅ Structured logging (7/24 modules), All frozen dataclasses in core/types.py
- ✅ Typed error dataclasses, BackendState invariant() function

---

## 2. Coding Standards Audit

Reference: `dragonlight-coding-standards-v2.md`

### Hard Rule Violations (Build-Fail Gates)

| Rule | Violations | RT Tasks |
|------|:----------:|----------|
| function_length_40_lines | 2 | see below |
| parameter_count_4 | 2 | see below |
| no_except_exception | 7 | see below |
| no_bare_except | 0 | see below |
| no_pass_in_error_handler | 1 | see below |
| frozen_dataclasses_for_data | 2 | see below |
| type_annotations_all_signatures | 1 | see below |
| mypy_strict_zero_warnings | 13 | see below |
| ruff_zero_warnings | 17 | see below |

### Advisory Rule Compliance

| Rule | Compliance | RT Tasks |
|------|:----------:|----------|
| assertion_density_2_per_function | 1% | — |
| hypothesis_property_based_testing | 0% | — |
| structured_logging_structlog | 30% | — |
| result_type_pattern | 0% | — |

### Detailed Violations

#### Function Length >40 lines (2 violations)

| File | Function | Lines | Fix |
|------|----------|:-----:|-----|
| `src/dragonlight_router/router.py` | `select_models()` | 71 | Decompose into _score_candidates(), _filter_by_catalog(), _build_ranked_list() |
| `src/dragonlight_router/selection/complexity.py` | `estimate_complexity()` | 50 | Extract tier classification logic into _classify_tier() helper |

#### except Exception (7 violations)

| File | Line | Context | Fix |
|------|------|---------|-----|
| `src/dragonlight_router/config/loader.py` | 47 | `except Exception as exc` | Catch yaml.YAMLError and OSError specifically |
| `src/dragonlight_router/catalog/cache.py` | 61 | `except Exception` | Catch (json.JSONDecodeError, OSError, ValueError) specifically |
| `src/dragonlight_router/server/routes.py` | 21 | `except Exception` | Catch json.JSONDecodeError specifically for JSON parse |
| `src/dragonlight_router/server/routes.py` | 55 | `except Exception` | Catch specific validation errors |
| `src/dragonlight_router/server/routes.py` | 129 | `except Exception as exc` | Catch specific HTTP/provider errors |
| `src/dragonlight_router/router.py` | 198 | `except Exception as exc` | Catch (aiohttp.ClientError, asyncio.TimeoutError) specifically |
| `src/dragonlight_router/budget/persistence.py` | 38 | `except Exception` | Catch (OSError, json.JSONDecodeError, TypeError) specifically |

#### Non-Frozen Dataclasses (2 violations)

| File | Class | Current | Fix |
|------|-------|---------|-----|
| `src/dragonlight_router/core/registry.py` | `BackendRegistry` | @dataclass (no frozen=True) | Add deviation record OR refactor to frozen + separate mutable BackendRegistryState |
| `src/dragonlight_router/core/state.py` | `BackendState` | @dataclass (no frozen=True) | Add deviation record with explicit justification + mitigations |

#### mypy Strict (13 errors)

| File | Line(s) | Issue | Fix |
|------|---------|-------|-----|
| `src/dragonlight_router/core/registry.py` | 41,43 | Missing type arguments for generic type dict | Add dict[str, ...] type parameters |
| `src/dragonlight_router/caching/simple.py` | 48 | Returning Any from function declared to return str | None | Add explicit type cast or narrow return type |
| `src/dragonlight_router/caching/simple.py` | 68 | Missing type arguments for generic type dict | Add dict[str, ...] type parameters |
| `src/dragonlight_router/catalog/cache.py` | 78,83,94 | Returning Any / Missing type arguments for dict | Add proper type annotations for dict and return values |
| `src/dragonlight_router/budget/persistence.py` | 18,47,60 | Missing type arguments for dict / Returning Any | Add dict[str, Any] annotations and explicit casts |
| `src/dragonlight_router/catalog/refresher.py` | 44 | Incompatible types in assignment (list | BaseException -> list) | Add proper type narrowing with isinstance check before assignment |
| `src/dragonlight_router/server/app.py` | 23 | Function missing type annotation for parameter | Add parameter type annotation |

#### ruff (17 errors)

| Code | Count | Description | Fix |
|------|:-----:|-------------|-----|
| F401 | 5 | unused-import | Remove unused imports |
| SIM105 | 2 | suppressible-exception (try-except-pass) | Replace with contextlib.suppress() |
| SIM114 | 2 | if-with-same-arms | Combine if branches with logical or |
| UP017 | 2 | datetime-timezone-utc | Use datetime.UTC alias |
| B905 | 1 | zip-without-explicit-strict | Add strict= to zip() calls |
| C401 | 1 | unnecessary-generator-set | Use set literal instead of set(generator) |
| E501 | 1 | line-too-long | Break long lines |
| I001 | 1 | unsorted-imports | Sort imports with isort |
| UP035 | 1 | deprecated-import | Update deprecated import paths |
| S101 | 1 | assert used outside test | Use invariant() function or add noqa with deviation record |

### Result Type Gap (Critical)

Zero Ok/Err Result types exist. 7 fallible functions need migration (RT-003).

---

## 3. Test Gap Analysis

| Metric | Value |
|--------|-------|
| Total Tests | 188 |
| Coverage | **37.77%** (FAIL — gate 80%) |
| Hypothesis Tests | **0** |
| Integration Tests | **0** |
| Missing-Assertion Tests | 3 |

### Low-Coverage Modules

| Module | Coverage | Missing Lines | Priority |
|--------|:--------:|---------------|:--------:|
| `src/dragonlight_router/catalog/refresher.py` | 36% | 32-46, 50-71 | high |
| `src/dragonlight_router/server/app.py` | 68% | 45-51 | medium |
| `src/dragonlight_router/health/tracker.py` | 74% | 53,60,87-88,94-95,99,103-108 | medium |
| `src/dragonlight_router/roles/matrix.py` | 73% | 46,52-53,70-73,76-84 | medium |
| `src/dragonlight_router/server/routes.py` | 16% | 17-46,51-79,84-86,94-105,114-130 | critical |

### Required Property-Based Tests (0 of 6 exist)

| Module | Properties | Priority |
|--------|-----------|:--------:|
| `selection/scoring.py` | composite_score_in_0_100, budget_score_monotonic_with_remaining, health_score_known_values | high |
| `selection/interleave.py` | output_subset_of_input, max_consecutive_bounded, length_preserved | high |
| `selection/complexity.py` | tier_never_downgrades, token_thresholds_monotonic | high |
| `budget/tracker.py` | score_in_0_100, score_decreases_with_usage, rpm_window_sliding | high |
| `caching/simple.py` | get_after_set_roundtrip, ttl_expiration, lru_eviction_under_max | medium |
| `core/types.py` | frozen_dataclass_immutability, serialization_roundtrip | medium |


---

## 4. Remediation Task Map

### Critical Path (Sequential)


1. **RT-003**: Introduce canonical Result type (Ok/Err) for all fallible operations
1. **RT-012**: Add TPM + daily token cap tracking to BudgetTracker
1. **RT-013**: Create selection/mbr.py — MBR capability filtering stage
1. **RT-014**: Create selection/cbr.py — CBR cost balancing stage
1. **RT-015**: Create selection/lbr.py — LBR rate-limit-aware dispatch
1. **RT-016**: Create dispatch/cascade.py — MBR→CBR→LBR composition
1. **RT-021**: Add RouterEngine.dispatch() method graft
1. **RT-022**: Create integration test suite for cascade dispatch

**Critical path:** 8 tasks, ~28 hours

### Full Task Details

**RT-001**: Decompose RouterEngine.select_models() from 71 to ≤40 lines — hard, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/router.py`

Changes:
- Extract _score_candidates(candidates, budget, health, exclude) -> list[ModelScore]
- Extract _filter_by_catalog(candidates, catalog, fetched_providers) -> list[tuple[str,int]]
- Extract _build_ranked_list(scored, max_consecutive, top_n) -> list[str]

Acceptance Criteria:
- [ ] select_models() body ≤ 40 lines
- [ ] All extracted helpers have type annotations and return Result types
- [ ] All existing tests pass unchanged
- [ ] New helper functions have precondition assertions

Agent: Refactor RouterEngine.select_models() into 3-4 smaller private methods. Each method must be independently testable. Preserve the exact public behavior. Add assertions per coding standard.

---


**RT-002**: Fix 7 except Exception violations — replace with specific exception types — hard, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/config/loader.py`, `src/dragonlight_router/catalog/cache.py`, `src/dragonlight_router/server/routes.py`, `src/dragonlight_router/router.py`, `src/dragonlight_router/budget/persistence.py`

Changes:
- loader.py:47 — except Exception → except (yaml.YAMLError, OSError)
- cache.py:61 — except Exception → except (json.JSONDecodeError, OSError, ValueError)
- routes.py:21 — except Exception → except json.JSONDecodeError
- routes.py:55 — except Exception → except (KeyError, TypeError, ValueError)
- routes.py:129 — except Exception → except (aiohttp.ClientError, asyncio.TimeoutError)
- router.py:198 — except Exception → except (aiohttp.ClientError, asyncio.TimeoutError)
- persistence.py:38 — except Exception → except (OSError, json.JSONDecodeError, TypeError)

Acceptance Criteria:
- [ ] Zero `except Exception` in codebase (confirmed by ruff/grep)
- [ ] Each catch block handles specific exception types appropriate to the operation
- [ ] No silent swallowing — every handler either returns Err, logs, or re-raises
- [ ] All existing tests pass

Agent: Replace each `except Exception` with the specific exception types listed. Ensure each handler either returns an Err Result, logs with structlog, or re-raises. Do NOT catch Exception anywhere. Run ruff to verify.

---


**RT-003**: Introduce canonical Result type (Ok/Err) for all fallible operations — advisory, complex effort, parallelizable=False

- Depends on: RT-002
- Targets: `src/dragonlight_router/core/types.py`, `src/dragonlight_router/budget/tracker.py`, `src/dragonlight_router/health/tracker.py`, `src/dragonlight_router/catalog/cache.py`, `src/dragonlight_router/catalog/refresher.py`, `src/dragonlight_router/config/loader.py`, `src/dragonlight_router/budget/persistence.py`

Changes:
- Add Ok[T], Err[E], Result[T,E] to core/types.py per canonical pattern
- Migrate BudgetTracker.score() → Result[float, ProviderNotFoundError]
- Migrate HealthTracker.score() → Result[float, ModelNotFoundError]
- Migrate CatalogCache.get() → Result[dict, StaleCatalogError]
- Migrate ConfigLoader.load_config() → Result[RouterConfig, RouterConfigError]
- Migrate BudgetPersistence.save/load → Result[None, StatePersistenceError]

Acceptance Criteria:
- [ ] Ok/Err frozen dataclasses exist in core/types.py
- [ ] All 7 target functions return Result type
- [ ] Callers use isinstance(x, Ok) / isinstance(x, Err) for dispatch
- [ ] No function returns both None and a value — None cases become Err
- [ ] Hypothesis test: Ok.value and Err.error are always present

Agent: Add the canonical Result type to core/types.py first. Then migrate each function one at a time, updating callers to use isinstance dispatch. Every Result-returning function needs both Ok and Err test cases.

---


**RT-004**: Fix mypy strict violations (13 errors) — hard, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/core/registry.py`, `src/dragonlight_router/caching/simple.py`, `src/dragonlight_router/catalog/cache.py`, `src/dragonlight_router/budget/persistence.py`, `src/dragonlight_router/catalog/refresher.py`, `src/dragonlight_router/server/app.py`

Changes:
- registry.py:41,43 — add dict[str, GenerativeBackend] and dict[str, BackendState] type params
- caching/simple.py:48 — add explicit cast for Any return from sqlite3
- caching/simple.py:68 — add dict[str, Any] type param
- catalog/cache.py:78,83,94 — add proper dict type params, fix Any return
- budget/persistence.py:18,47,60 — add dict type params, fix Any return
- catalog/refresher.py:44 — add isinstance narrow before assignment
- server/app.py:23 — add parameter type annotation
- Install types-PyYAML stubs

Acceptance Criteria:
- [ ] mypy --strict src/ produces zero errors
- [ ] pip install types-PyYAML added to project deps or dev deps
- [ ] All existing tests pass

Agent: Fix each mypy error by adding proper type annotations. Run mypy after each fix to confirm. Install types-PyYAML. Target: mypy src/ — zero errors.

---


**RT-005**: Fix ruff violations (17 errors) and add missing docstrings — hard, simple effort, parallelizable=True

- Depends on: —
- Targets: `ALL source files`

Changes:
- Remove 5 unused imports (F401)
- Replace 2 try-except-pass with contextlib.suppress (SIM105)
- Combine 2 if-with-same-arms with logical or (SIM114)
- Replace 2 datetime.timezone.utc with datetime.UTC (UP017)
- Add strict= to zip() call (B905)
- Replace set(generator) with set literal (C401)
- Fix line-too-long (E501)
- Sort imports (I001)
- Update deprecated import (UP035)
- Add S101 noqa or use invariant() for non-test assert

Acceptance Criteria:
- [ ] ruff check src/ produces zero errors
- [ ] All auto-fixable issues applied via ruff --fix
- [ ] Remaining issues manually resolved
- [ ] All existing tests pass

Agent: Run ruff check src/ --fix first for auto-fixable items. Then manually fix remaining items. Run ruff check src/ to confirm zero errors.

---


**RT-006**: Add precondition/postcondition assertions to all 93 unasserted functions — advisory, standard effort, parallelizable=True

- Depends on: RT-005
- Targets: `ALL source files`

Changes:


Acceptance Criteria:
- [ ] Every non-trivial function has ≥1 precondition assertion
- [ ] Functions that compute values have ≥1 postcondition assertion
- [ ] Assertion messages are descriptive: assert x > 0, 'rpm_limit must be positive'
- [ ] state.py invariant() function is used for critical invariants

Agent: Add precondition assertions at function entry for all parameter constraints. Add postcondition assertions before return for computed values. Use the pattern: assert condition, 'descriptive message'. Follow state.py invariant() for critical checks.

---


**RT-007**: Add Hypothesis property-based tests for 6 target modules — advisory, complex effort, parallelizable=True

- Depends on: RT-006
- Targets: `tests/unit/test_scoring_properties.py`, `tests/unit/test_interleave_properties.py`, `tests/unit/test_complexity_properties.py`, `tests/unit/test_budget_properties.py`, `tests/unit/test_caching_properties.py`, `tests/unit/test_types_properties.py`

Changes:
- Create 6 new Hypothesis test files with @given decorators
- scoring: composite_score_in_0_100, budget_score_monotonic, health_score_known_values
- interleave: output_subset_of_input, max_consecutive_bounded, length_preserved
- complexity: tier_never_downgrades, token_thresholds_monotonic
- budget: score_in_0_100, score_decreases_with_usage, rpm_window_sliding
- caching: get_after_set_roundtrip, ttl_expiration, lru_eviction
- types: frozen_immutability, serialization_roundtrip

Acceptance Criteria:
- [ ] 6 new test files with Hypothesis @given strategies
- [ ] Each file has ≥3 properties testing different categories (roundtrip, invariant, idempotency)
- [ ] All property tests pass with 100+ examples per property
- [ ] Property docstrings document which of the 7 canonical categories they test
- [ ] Tests run in <1s each

Agent: Create Hypothesis test files following the dragonlight-property-based-testing-strategy.md. Each property test must document its category (round-trip, invariant preservation, idempotency, commutativity, subset, monotonicity, no-counterexample). Use 100+ examples. Use assume() to filter invalid inputs.

---


**RT-008**: Add structlog logging to 17 modules currently without logging — advisory, standard effort, parallelizable=True

- Depends on: RT-005
- Targets: `src/dragonlight_router/core/registry.py`, `src/dragonlight_router/core/state.py`, `src/dragonlight_router/health/circuit_breaker.py`, `src/dragonlight_router/server/app.py`, `src/dragonlight_router/server/routes.py`, `src/dragonlight_router/selection/complexity.py`, `src/dragonlight_router/selection/scoring.py`, `src/dragonlight_router/selection/interleave.py`, `src/dragonlight_router/caching/store.py`, `src/dragonlight_router/caching/semantic.py`, `src/dragonlight_router/caching/simple.py`, `src/dragonlight_router/budget/tracker.py`

Changes:
- Add `import structlog; logger = structlog.get_logger()` to each module
- Add logger.info() at function entry for state-changing operations
- Add logger.warning() for degraded states
- Add logger.error() for failures before returning Err
- Event names in snake_case, include correlation fields

Acceptance Criteria:
- [ ] All operational modules use structlog
- [ ] Log entries include event name, relevant IDs, typed fields
- [ ] No sensitive data (API keys, PII) in log output
- [ ] Logging does not change function behavior

Agent: Add structlog to each listed module. Log at entry of state-changing functions. Use snake_case event names. Include model_id, provider_name, or other context fields. Never log env_key values or request content.

---


**RT-009**: Add deviation records for 2 non-frozen dataclasses — hard, simple effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/core/registry.py`, `src/dragonlight_router/core/state.py`

Changes:
- Add DEVIATION_RECORD docstring to BackendRegistry class
- Add DEVIATION_RECORD docstring to BackendState class
- Record: rule violated, justification, approved by, mitigations, scope, expiration

Acceptance Criteria:
- [ ] Each non-frozen dataclass has a deviation record in its docstring
- [ ] Deviation records contain all 6 required fields
- [ ] Deviation records reference the specific coding standard rule

Agent: Add a DEVIATION_RECORD section to the class docstring of BackendRegistry and BackendState. Include: rule_violated, justification, approved_by, mitigations, scope, expiration. BackendState is justified as runtime mutable state. BackendRegistry needs justification for mutable dict defaults.

---


**RT-010**: Fix record_request 5-parameter violation — hard, simple effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/router.py`

Changes:
- Create RequestOutcome frozen dataclass with provider, model_id, success, tokens_used, latency_ms
- Change record_request(self, outcome: RequestOutcome) -> None
- Update all callers in server/routes.py and tests

Acceptance Criteria:
- [ ] record_request() has ≤4 parameters
- [ ] RequestOutcome is frozen dataclass with all required fields
- [ ] All callers updated
- [ ] All tests pass

Agent: Create a RequestOutcome frozen dataclass in core/types.py. Change record_request signature to accept it. Update routes.py record_handler and test_router_engine.py.

---


**RT-011**: Fix 3 missing-assertion test functions — hard, simple effort, parallelizable=True

- Depends on: —
- Targets: `tests/unit/test_budget_tracker.py`, `tests/unit/test_router_engine.py`

Changes:
- test_unknown_provider_no_error — add assert on return value or state
- test_record_success — add assert verifying health/budget state changed
- test_record_failure — add assert verifying error count incremented

Acceptance Criteria:
- [ ] All 3 test functions contain at least one assert or pytest.raises
- [ ] Tests verify actual behavior, not just 'does not raise'

Agent: Add meaningful assertions to each of the 3 test functions. For test_record_success, verify health_tracker.score() returns expected value. For test_record_failure, verify error_count. For test_unknown_provider, verify score() returns 100.0.

---


**RT-012**: Add TPM + daily token cap tracking to BudgetTracker — critical, standard effort, parallelizable=False

- Depends on: RT-003
- Targets: `src/dragonlight_router/budget/tracker.py`

Changes:
- Add _tpm_windows: dict[str, deque[float]] sliding window (same pattern as RPM)
- Add _tokens_today: dict[str, int] counter
- Add _tpm_remaining(provider) method
- Extend has_capacity() to check RPM + RPD + TPM + token_cap
- Extend score() to include TPM ratio in composite
- Extend record_request() to track tokens in TPM window and daily token counter

Acceptance Criteria:
- [ ] TPM sliding window tracking works identically to RPM tracking
- [ ] Daily token cap enforced — has_capacity() returns False when cap reached
- [ ] score() returns min(rpm_ratio, rpd_ratio, tpm_ratio, token_ratio) * 100
- [ ] 0 token_cap means unlimited (ratio=1.0)
- [ ] Unit tests for all new capacity checks
- [ ] Property test: score() in [0, 100] for all valid inputs

Agent: Add TPM sliding window to BudgetTracker following the exact pattern of RPM tracking. Add daily token cap counter following RPD pattern. Extend has_capacity() and score() to include these new dimensions. The ProviderConfig already has tpm_limit field. Add comprehensive tests.

---


**RT-013**: Create selection/mbr.py — MBR capability filtering stage — critical, complex effort, parallelizable=True

- Depends on: RT-003
- Targets: `src/dragonlight_router/selection/mbr.py`

Changes:
- Create filter_by_capability(candidates, tier, health_cache) -> Result[list[Candidate], MBRNoCandidatesError]
- Create estimate_complexity(order: DispatchOrder) -> BackendTier
- Implement adjacent-tier graceful upgrade logic
- Implement circuit_open exclusion
- Implement never-downgrade invariant
- Implement local-provider unlimited-rate passthrough

Acceptance Criteria:
- [ ] All 5 TM-001 acceptance criteria met
- [ ] Function bodies ≤40 lines
- [ ] All functions have ≥2 assertions
- [ ] structlog logging at entry/exit
- [ ] Hypothesis property test: tier_never_downgrades
- [ ] Unit tests for each AC criterion

Agent: Create mbr.py from scratch. Implement filter_by_capability and estimate_complexity. Use Result type for return. Add guard clauses, assertions, structlog. Follow the complexity.py pattern for tier classification. Create tests/unit/test_mbr.py alongside.

---


**RT-014**: Create selection/cbr.py — CBR cost balancing stage — critical, complex effort, parallelizable=False

- Depends on: RT-003, RT-013
- Targets: `src/dragonlight_router/selection/cbr.py`

Changes:
- Create CostGovernorConfig frozen dataclass
- Create cost_governor_active(daily_spend, daily_budget, config) -> bool
- Create cost_adjusted_weights(base_weights, daily_spend, daily_budget) -> ScoringWeights
- Create filter_by_budget(candidates, budget_tracker) -> Result[list[Candidate], CBRBudgetExhaustedError]
- Create score_by_cost(candidates, budget_tracker, cost_governor) -> list[ScoredCandidate]

Acceptance Criteria:
- [ ] All 4 TM-002 acceptance criteria met
- [ ] CostGovernor activates at 80% daily_budget threshold
- [ ] cost_adjusted_weights smoothly shifts rank→budget weight
- [ ] All functions ≤40 lines, ≥2 assertions, Result returns, structlog
- [ ] Hypothesis test: cost_adjusted_weights_sum_to_1
- [ ] Unit tests for each AC criterion

Agent: Create cbr.py. Depends on ScoringWeights from scoring.py (RT-003 graft) and MBR output format (RT-013). Create tests/unit/test_cbr.py alongside. Follow coding standards strictly.

---


**RT-015**: Create selection/lbr.py — LBR rate-limit-aware dispatch — critical, complex effort, parallelizable=False

- Depends on: RT-012, RT-014
- Targets: `src/dragonlight_router/selection/lbr.py`

Changes:
- Create filter_by_rate_limits(candidates, budget_tracker) -> Result[list[Candidate], LBRNoCapacityError]
- Create schedule_by_load(candidates, rate_tracker) -> list[ScoredCandidate]
- Implement WFQ-inspired scheduling
- Enforce RPM, RPD, TPM, and daily token cap per provider

Acceptance Criteria:
- [ ] All 4 TM-003 acceptance criteria met
- [ ] LBR checks all 4 rate limit dimensions
- [ ] WFQ scheduling produces deterministic ordering for same inputs
- [ ] All functions ≤40 lines, ≥2 assertions, Result returns, structlog
- [ ] Hypothesis test: schedule_by_load_produces_permutation
- [ ] Unit tests for each AC criterion

Agent: Create lbr.py. Depends on BudgetTracker with TPM support (RT-012) and CBR output format (RT-014). Create tests/unit/test_lbr.py alongside.

---


**RT-016**: Create dispatch/cascade.py — MBR→CBR→LBR composition — critical, complex effort, parallelizable=False

- Depends on: RT-013, RT-014, RT-015
- Targets: `src/dragonlight_router/dispatch/cascade.py`

Changes:
- Create route(order, registry, budget, rate, health, cost_governor, queue_depths, config) -> Result[EngineResponse, DispatchFailure]
- Implement MBR→CBR→LBR pipeline with Result composition
- Each stage returns Result[CandidateSet, StageError]
- Empty candidate set → Err with diagnostics
- Construct fallback chain from LBR survivors

Acceptance Criteria:
- [ ] All 5 TM-004 acceptance criteria met
- [ ] Pipeline composition order: MBR then CBR then LBR (load-bearing, not configurable)
- [ ] Each stage failure produces typed Err with context
- [ ] Fallback chain preserves LBR ordering
- [ ] Unit tests for each stage failure path
- [ ] Integration test for full pipeline

Agent: Create dispatch/cascade.py with route() function. Compose MBR→CBR→LBR using guard-clause Result composition. Each stage failure short-circuits. Create tests/unit/test_cascade.py alongside.

---


**RT-017**: Create selection/context_filter.py — DIAN CECHT trust tier filtering — critical, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/selection/context_filter.py`

Changes:
- Create TrustTier enum (LOCAL, HAIKU, SONNET, OPUS)
- Create filter_by_trust_tier(candidates, required_tier) -> list[Candidate]
- Implement trust hierarchy: LOCAL trusts all, OPUS trusts only OPUS
- Implement untrusted-provider PII exclusion

Acceptance Criteria:
- [ ] All 4 TM-006 acceptance criteria met
- [ ] Trust hierarchy is strict: no tier trusts below its level
- [ ] PII context never sent to untrusted tier providers
- [ ] Unit tests for each tier combination
- [ ] Hypothesis test: filter_by_trust_tier_output_subset_of_input

Agent: Create context_filter.py with TrustTier enum and filter_by_trust_tier. This is independent of MBR/CBR/LBR and can be built in parallel. Add tests/unit/test_context_filter.py.

---


**RT-018**: Add canonical ScoringWeights + CostGovernor to scoring.py — critical, standard effort, parallelizable=True

- Depends on: RT-005
- Targets: `src/dragonlight_router/selection/scoring.py`

Changes:
- Add ScoringWeights frozen dataclass with 6 canonical dimensions
- Add score_candidate(candidate, weights, budget_score, health_score) -> float
- Add CostGovernorConfig frozen dataclass
- Add cost_governor_active(spend, budget, config) -> bool
- Add cost_adjusted_weights(base, spend, budget) -> ScoringWeights

Acceptance Criteria:
- [ ] All 4 unmet TM-007 acceptance criteria now met
- [ ] ScoringWeights defaults match canonical spec (rank, budget, health, cost, latency, queue_depth)
- [ ] score_candidate() normalizes all inputs to [0,1] before weighting
- [ ] CostGovernor activates at 80% threshold
- [ ] Coexists with existing compute_composite_score() — backward compatible
- [ ] Hypothesis test: weights_sum_to_1, score_in_0_100

Agent: Add canonical types to scoring.py alongside existing simple scoring. Do NOT modify compute_composite_score(). Add ScoringWeights, score_candidate, CostGovernorConfig, cost_governor_active, cost_adjusted_weights. Add tests to test_scoring.py.

---


**RT-019**: Create health/check_loop.py — periodic backend probing — high, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/health/check_loop.py`

Changes:
- Create HealthCheckLoop class with configurable interval
- Implement periodic probe that updates BackendState.status
- Integrate with CircuitBreaker — probe success closes circuit
- Implement 404-based permanent retirement

Acceptance Criteria:
- [ ] All 4 TM-008 acceptance criteria met
- [ ] Check loop runs as asyncio background task
- [ ] Probe results update BackendState within 1 interval
- [ ] 404 response triggers permanent retirement
- [ ] CircuitBreaker half-open → closed on probe success

Agent: Create check_loop.py. Use asyncio.create_subprocess_exec for HTTP probes. Integrate with existing CircuitBreaker and HealthTracker. Add tests/unit/test_check_loop.py.

---


**RT-020**: Add dispatch/retire/reinstate endpoints to routes.py — high, standard effort, parallelizable=False

- Depends on: RT-002, RT-016
- Targets: `src/dragonlight_router/server/routes.py`

Changes:
- Add POST /v1/dispatch handler
- Add POST /v1/retire handler
- Add POST /v1/reinstate handler
- Fix except Exception in existing handlers (routes.py:21,55,129)
- Add input validation at system boundaries

Acceptance Criteria:
- [ ] All 8 TM-009 acceptance criteria met
- [ ] 3 new endpoints functional
- [ ] All error responses are structured JSON
- [ ] No except Exception in any route handler
- [ ] Integration test for each endpoint

Agent: Add POST /v1/dispatch, /v1/retire, /v1/reinstate to routes.py. Dispatch delegates to RouterEngine.dispatch(). Retire/reinstate manage backend health state. Fix all except Exception blocks. Add tests.

---


**RT-021**: Add RouterEngine.dispatch() method graft — critical, standard effort, parallelizable=False

- Depends on: RT-001, RT-016
- Targets: `src/dragonlight_router/router.py`

Changes:
- Add async def dispatch(self, order: DispatchOrder) -> EngineResponse | DispatchFailure
- Import from dragonlight_router.dispatch.cascade
- Wire dispatch() to route() function
- Ensure backward compatibility with select_models()

Acceptance Criteria:
- [ ] All 3 unmet TM-010 acceptance criteria met
- [ ] dispatch() delegates to cascade.route()
- [ ] select_models() behavior unchanged
- [ ] dispatch() handles case where no backends registered (returns DispatchFailure)

Agent: Add dispatch() method to RouterEngine class. Import route from dispatch.cascade. Handle empty registry case. Do NOT modify select_models() behavior. Add test_dispatch to test_router_engine.py.

---


**RT-022**: Create integration test suite for cascade dispatch — high, standard effort, parallelizable=False

- Depends on: RT-016, RT-021
- Targets: `tests/integration/test_cascade_dispatch.py`

Changes:
- Create full pipeline integration test
- Test circuit breaker triggering under failure load
- Test fallback chain advancement
- Test budget enforcement
- Test catalog refresh

Acceptance Criteria:
- [ ] All 5 TM-011 acceptance criteria met
- [ ] Integration tests use real component wiring (not mocks for internal functions)
- [ ] Mock only external HTTP calls at the seam
- [ ] Tests run in <5s each
- [ ] Coverage for dispatch/cascade.py ≥ 80%

Agent: Create integration test suite. Wire real components together. Mock only aiohttp calls at the HTTP boundary. Test full cascade: order in → response out. Verify fallback, circuit breaker, budget enforcement.

---


**RT-023**: Add structlog to server routes for request/response observability — advisory, simple effort, parallelizable=True

- Depends on: RT-002
- Targets: `src/dragonlight_router/server/routes.py`

Changes:
- Add structlog import and logger
- Log request entry with role/top_n/dispatch_order
- Log response with model count and latency
- Log errors with structured fields

Acceptance Criteria:
- [ ] Every route handler logs entry and exit
- [ ] Log entries include request_id or equivalent correlation
- [ ] No sensitive data logged

Agent: Add structlog to routes.py. Log at handler entry with request details. Log at exit with response summary. Include request_id for correlation.

---


**RT-024**: Raise test coverage from 38% to ≥80% — hard, complex effort, parallelizable=True

- Depends on: RT-011, RT-007
- Targets: `tests/unit/test_catalog.py — cover refresher.py lines 32-46,50-71`, `tests/unit/test_server.py — cover routes.py lines 17-46,51-79`, `tests/unit/test_health_tracker.py — cover tracker.py lines 53,60,87-88,94-95`, `tests/unit/test_roles.py — cover matrix.py lines 46,52-53,70-73,76-84`, `tests/unit/test_types.py — cover errors.py (currently 0%)`

Changes:
- Add tests for catalog refresher success/failure paths
- Add tests for server route error handling paths
- Add tests for health tracker edge cases (retirement, recovery)
- Add tests for role matrix file reload and missing role
- Add tests for core/errors.py typed error construction and __str__

Acceptance Criteria:
- [ ] Overall coverage ≥ 80%
- [ ] No module below 50% coverage
- [ ] All new tests have meaningful assertions
- [ ] Coverage gaps for new modules (mbr, cbr, lbr, etc.) addressed by their RT tasks

Agent: Target the specific uncovered lines listed. Add tests for each untested path. Focus on error handling, edge cases, and boundary conditions. Do NOT add tests that just call functions without asserting results.

---


## 5. Execution Plan

| Wave | Description | Tasks | Agents | Hours |
|:----:|-------------|-------|:------:|:-----:|
| 0 | Independent hard-rule fixes — no dependencies, all parallelizable | RT-002, RT-004, RT-005, RT-009, RT-010, RT-011, RT-017, RT-019 | 4 | 4 |
| 1 | Result type introduction — unblocks all cascade stages | RT-003 | 1 | 3 |
| 2 | Function decomposition + canonical scoring + logging + TPM | RT-001, RT-006, RT-007, RT-008, RT-012, RT-018, RT-023 | 4 | 6 |
| 3 | MBR stage — unblocks CBR and cascade | RT-013 | 1 | 3 |
| 4 | CBR stage — unblocks LBR | RT-014 | 1 | 3 |
| 5 | LBR stage — unblocks cascade composition | RT-015 | 1 | 3 |
| 6 | Cascade composition + dispatch graft + routes + coverage | RT-016, RT-020, RT-021, RT-024 | 3 | 6 |
| 7 | Integration tests — final validation | RT-022 | 1 | 2 |

```
Wave 0:  RT-002  RT-004  RT-005  RT-009  RT-010  RT-011  RT-017  RT-019
              \                          |       |
Wave 1:                                RT-003 (Result type)
              /         |              \
Wave 2:  RT-001  RT-006  RT-008  RT-012  RT-018  RT-023
                                      |
Wave 3:                              RT-013 (MBR)
                                      |
Wave 4:                              RT-014 (CBR)
                                      |
Wave 5:                              RT-015 (LBR)
                               /      |
Wave 6:                 RT-016  RT-020  RT-021
                            \          |
Wave 7:                RT-022  RT-024  RT-007
```

**Total:** ~28 agent-hours, 4 concurrent agents, 8 waves

---

## 6. Risk Assessment

| Risk | Tasks | Description |
|:----:|-------|-------------|
| 🔴 | RT-003 | Result type migration — 7 files, cascading API changes |
| 🔴 | RT-016 | Cascade composition — 3 new modules, load-bearing |
| 🟡 | RT-001 | select_models decomposition — must preserve behavior |
| 🟡 | RT-012 | BudgetTracker TPM extension |
| 🟢 | RT-005, RT-009, RT-010, RT-011 | Mechanical fixes |

---

## 7. Acceptance Gate

- [ ] `ruff check src/` → zero errors
- [ ] `mypy src/` → zero errors
- [ ] `pytest --cov-fail-under=80` → all pass
- [ ] All 12 TM-* acceptance criteria met
- [ ] Zero `except Exception`
- [ ] Zero functions >40 lines or >4 params
- [ ] Result type for all fallible operations
- [ ] Hypothesis property tests for 6 modules
- [ ] structlog in all operational modules
- [ ] Deviation records for non-frozen dataclasses

---

*Generated by Hermes Agent — Delta Audit of dragonlight-router against live-spec v0.2.0 and Dragonlight Coding Standards v2*
