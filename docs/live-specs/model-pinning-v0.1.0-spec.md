# Model Pinning v0.1.0 Live Spec

**Version:** 0.1.0
**Effective:** 2026-06-18
**Status:** Design (pre-implementation)
**Depends on:** Dragonlight Router v0.3.0 spec

## 1. Problem Statement

The cascade pipeline (MBR -> IBR -> CBR -> LBR) is the correct default for most dispatch requests -- it balances capability, cost, health, and load across the backend pool. However, certain callers need deterministic model selection:

- **Integration tests** that must hit a specific backend for reproducibility.
- **Benchmarking** that isolates a single model's performance characteristics.
- **Operator overrides** where a human knows the right model for a task and wants to skip the selection heuristics.
- **Factory pipelines** that pin specific models per pipeline stage for consistency across runs.

Today, callers cannot express "use exactly this model." The only path is the full cascade, which may select a different backend on every request depending on health, budget, and load state. Model pinning provides a direct-dispatch escape hatch while preserving the router's operational guardrails.

## 2. Design

### 2.1 DispatchOrder Extension

`DispatchOrder` gains one optional field:

```python
@dataclass(frozen=True)
class DispatchOrder:
    # ... existing fields ...
    model: str | None = None  # e.g. "anthropic/claude-sonnet-4-20250514"
```

When `model` is not None, the request is a **pinned dispatch**. The value is the backend name as registered in `BackendRegistry` -- the same string accepted by `registry.get()` (provider-prefixed model ID, e.g. `"groq/llama-3.3-70b-versatile"`).

### 2.2 Dispatch Flow (Pinned vs Cascade)

```
DispatchOrder
    |
    +-- model is None? --> [MBR -> IBR -> CBR -> LBR] (cascade, unchanged)
    |
    +-- model is set? ---> [Registry Lookup]
                               |
                               +-- Not found --> 400 error
                               +-- Found --> [Health Check]
                                                |
                                                +-- Circuit open + honor_pinned_health=true --> 503 error
                                                +-- Circuit open + honor_pinned_health=false --> attempt anyway
                                                +-- Healthy --> [Budget Check]
                                                                   |
                                                                   +-- Over budget --> 429 error
                                                                   +-- Within budget --> [Dispatch]
                                                                                            |
                                                                                            v
                                                                                    EngineResponse
```

**Bypassed stages:** MBR (role matrix lookup), IBR (intent classification), CBR (cost-aware scoring), LBR (load-balanced selection). The cascade is entirely skipped.

**Preserved operations:** Registry lookup, health/circuit-breaker check, budget enforcement (`check_and_reserve`), rate limit check (`has_capacity`), context filtering (trust tier mapping), adapter creation (fresh instance per HAZ-014), token estimation, cost tracking (`record_request`), health recording (`record_success`/`record_error`), response validation, cache check/store, structured logging.

### 2.3 Pinned Dispatch Implementation Path

The pinned path is implemented as an early branch in `cascade.dispatch()`, before `_run_cascade()` is called:

1. `registry.get(order.model)` -- retrieve the backend and state.
2. If `(None, None)` returned: return `Err(ModelNotFoundError(...))`.
3. If `state.status == BackendStatus.RETIRED`: return `Err(ModelRetiredError(...))`.
4. If `state.is_circuit_open()` and `honor_pinned_health` is true: return `Err(ModelUnhealthyError(...))`.
5. Budget check via `budget_tracker.has_capacity(backend.config.provider)`: if false, return `Err(BudgetExhaustedError(...))`.
6. Rate limit check via `budget_tracker.check_and_reserve(backend.config.provider)`: if false, return `Err(RateLimitExhaustedError(...))`.
7. Build context via `_build_dispatch_context(order)`, apply context filtering based on backend tier.
8. Create fresh adapter, dispatch, record outcome. Same as `_try_adapter_dispatch()`.
9. On success: build `EngineResponse` with `dispatch_mode="pinned"`, `was_fallback=False`, empty `fallback_chain`.
10. On adapter failure: return `Err(...)`. No fallback -- pinned dispatch has exactly one candidate.

The streaming path (`dispatch_stream`) follows the same early-branch pattern.

### 2.4 Error Types

Three new error dataclasses in `core/types.py`:

```python
@dataclass(frozen=True)
class ModelNotFoundError:
    """Pinned model not found in registry."""
    model: str
    message: str

@dataclass(frozen=True)
class ModelUnhealthyError:
    """Pinned model is circuit-open or retired."""
    model: str
    status: str  # "circuit_open" | "retired"
    message: str

@dataclass(frozen=True)
class BudgetExhaustedError:
    """Pinned model's provider has exhausted its budget or rate limits."""
    model: str
    provider: str
    message: str
```

### 2.5 EngineResponse Extension

`EngineResponse` gains one field:

```python
@dataclass(frozen=True)
class EngineResponse:
    # ... existing fields ...
    dispatch_mode: str = "cascade"  # "cascade" | "pinned"
```

Default `"cascade"` preserves backward compatibility. Pinned dispatches set `"pinned"`.

`StreamChunk` gains the same field on metadata events:

```python
@dataclass(frozen=True)
class StreamChunk:
    # ... existing fields ...
    dispatch_mode: str = "cascade"
```

### 2.6 API Change

**POST /v1/dispatch** request body gains:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | `string` | No | `null` | Backend name to pin. When set, bypasses cascade. |

When `model` is present, `intent_category` and `specific_intent` become optional (they are not used for routing but are still logged if provided). When `model` is absent, existing required-field validation is unchanged.

**Response** gains `dispatch_mode` field:

```json
{
  "content": "...",
  "backend_used": "anthropic/claude-sonnet-4-20250514",
  "dispatch_mode": "pinned",
  "...": "..."
}
```

**Error responses** for pinned dispatch:

| Condition | HTTP Status | Error Message |
|-----------|-------------|---------------|
| Model not in registry | 400 | `"pinned model not found in registry: {model}"` |
| Model retired | 400 | `"pinned model is retired: {model}"` |
| Model circuit-open (honored) | 503 | `"pinned model is unhealthy (circuit open): {model}"` |
| Provider over budget | 429 | `"pinned model's provider budget exhausted: {provider}"` |
| Provider rate-limited | 429 | `"pinned model's provider rate limit exhausted: {provider}"` |
| Adapter failure | 502 | `"pinned model dispatch failed: {model}"` |

### 2.7 Configuration

One new field in `router.yaml`:

```yaml
pinned_dispatch:
  honor_health: true  # When true, circuit-open models return 503. When false, attempt anyway.
```

Pydantic config model:

```python
class PinnedDispatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    honor_health: bool = True
```

Added to `RouterConfig` as:

```python
class RouterConfig(BaseModel):
    # ... existing fields ...
    pinned_dispatch: PinnedDispatchConfig = PinnedDispatchConfig()
```

### 2.8 Observability

**Structured log events:**

- `pinned_dispatch_start`: Emitted at entry. Fields: `model`, `request_id`.
- `pinned_dispatch_rejected`: Emitted on pre-flight failure. Fields: `model`, `reason` (`"not_found"` | `"retired"` | `"circuit_open"` | `"budget_exhausted"` | `"rate_limited"`).
- `pinned_dispatch_complete`: Emitted on success. Fields: `model`, `latency_ms`, `tokens_in`, `tokens_out`, `estimated_cost_usd`.
- `pinned_dispatch_failed`: Emitted on adapter failure. Fields: `model`, `error_type`, `error_message`.

All existing dispatch log events (`dispatch_pipeline_start`, etc.) continue to fire for cascade dispatches. The `dispatch_mode` field is included in all dispatch-related log events to distinguish the two paths.

**Metrics:**

`/metrics` gains:
- `pinned_dispatch_count` -- total pinned dispatch attempts.
- `pinned_dispatch_rejected_count` -- broken down by reason.
- `pinned_dispatch_success_count` -- successful completions.

## 3. Acceptance Criteria

- AC-PIN-001: When `model` is set in DispatchOrder, the cascade (MBR/IBR/CBR/LBR) MUST NOT execute.
- AC-PIN-002: When `model` is set, `registry.get(model)` MUST be used to resolve the backend.
- AC-PIN-003: When the pinned model is not found in the registry, the router MUST return HTTP 400 with a clear error message.
- AC-PIN-004: When the pinned model is retired, the router MUST return HTTP 400.
- AC-PIN-005: When the pinned model is circuit-open and `honor_health` is true, the router MUST return HTTP 503.
- AC-PIN-006: When the pinned model is circuit-open and `honor_health` is false, the router MUST attempt dispatch.
- AC-PIN-007: Budget enforcement (`has_capacity`) MUST run for pinned dispatches. Exhausted budget MUST return HTTP 429.
- AC-PIN-008: Rate limit enforcement (`check_and_reserve`) MUST run for pinned dispatches. Exhausted rate limits MUST return HTTP 429.
- AC-PIN-009: Cost tracking (`record_request`) MUST execute after pinned dispatch completes.
- AC-PIN-010: Health recording (`record_success`/`record_error`) MUST execute after pinned dispatch completes.
- AC-PIN-011: Context filtering MUST apply based on the pinned backend's trust tier.
- AC-PIN-012: A fresh adapter MUST be created per pinned dispatch attempt (HAZ-014).
- AC-PIN-013: Response validation (`_validate_llm_response`) MUST apply to pinned dispatch output.
- AC-PIN-014: `EngineResponse.dispatch_mode` MUST be `"pinned"` for pinned dispatches and `"cascade"` for cascade dispatches.
- AC-PIN-015: `EngineResponse.was_fallback` MUST be `false` for pinned dispatches (no fallback chain exists).
- AC-PIN-016: When `model` is set in the API request, `intent_category` and `specific_intent` MUST be optional.
- AC-PIN-017: When `model` is absent, existing required-field validation MUST be unchanged.
- AC-PIN-018: Cache check MUST run before pinned dispatch (return cached response if available).
- AC-PIN-019: Cache store MUST run after successful pinned dispatch.
- AC-PIN-020: The `dispatch_mode` field MUST be included in all dispatch-related structured log events.
- AC-PIN-021: Pinned dispatch MUST work with both streaming and non-streaming paths.
- AC-PIN-022: `pinned_dispatch.honor_health` MUST default to `true`.

## 4. Hazard Analysis

| ID | Hazard | Severity | Likelihood | Risk Score | Mitigation |
|----|--------|----------|------------|------------|------------|
| HAZ-PIN-001 | Pinned dispatch bypasses cost optimization, caller unknowingly routes all traffic to expensive model | Medium | Medium | MEDIUM | Budget enforcement still runs. Cost governor thresholds still apply at the provider level. `dispatch_mode: "pinned"` in response and logs makes the bypass visible. |
| HAZ-PIN-002 | Pinned dispatch to circuit-open model causes repeated failures and degrades health metrics | Medium | Low | LOW | `honor_health: true` default rejects circuit-open models. When overridden, failures still feed circuit breaker -- the model will trip open and subsequent non-override requests will be rejected. |
| HAZ-PIN-003 | Caller pins a model that exists in registry but whose API key is invalid or missing | Medium | Medium | MEDIUM | `BackendStatus.KEY_INVALID` is treated the same as `RETIRED` -- pinned dispatch returns 400. The adapter creation path checks for key presence. |
| HAZ-PIN-004 | Pinned dispatch bypasses intent-based tier floors, allowing a SIMPLE model for complex reasoning | Medium | Medium | MEDIUM | This is intentional -- the caller is explicitly choosing the model. The `intent_category` (if provided) is logged for observability so mismatches can be detected in post-hoc analysis. |

## 5. Migration Path

Model pinning is fully additive. Migration from v0.3.0 or v0.3.0+IBR:

1. **No config change required** -- `pinned_dispatch` section is optional with sensible defaults.
2. **No API breaking changes** -- `model` field is optional; existing requests without it behave identically.
3. **Response changes are additive** -- `dispatch_mode` defaults to `"cascade"`, existing consumers see no difference.
4. **Rollback** -- remove `model` from request bodies. No server-side change needed.

All type changes use optional fields with backward-compatible defaults:
- `DispatchOrder.model: str | None = None`
- `EngineResponse.dispatch_mode: str = "cascade"`
- `StreamChunk.dispatch_mode: str = "cascade"`
- `RouterConfig.pinned_dispatch: PinnedDispatchConfig = PinnedDispatchConfig()`

## 6. Testing Strategy

### 6.1 Unit Tests

| Area | Tests | Approach |
|------|-------|----------|
| Pinned dispatch happy path | ~10 | Mock registry, verify cascade not called, verify response has `dispatch_mode="pinned"` |
| Model not found | ~5 | Verify 400 error with clear message for missing, empty, malformed model names |
| Model retired | ~3 | Set BackendStatus.RETIRED, verify 400 |
| Model circuit-open (honor) | ~5 | Open circuit, honor_health=true, verify 503 |
| Model circuit-open (override) | ~5 | Open circuit, honor_health=false, verify dispatch attempted |
| Budget exhaustion | ~5 | Exhaust provider budget, verify 429 |
| Rate limit exhaustion | ~5 | Exhaust provider rate limits, verify 429 |
| Cost tracking | ~5 | Verify record_request called with correct provider after pinned dispatch |
| Health recording | ~5 | Verify record_success/record_error called after pinned dispatch |
| Context filtering | ~5 | Verify trust tier filtering applied based on pinned backend tier |
| Cache integration | ~5 | Verify cache check before dispatch, cache store after success |
| Validation relaxation | ~5 | Verify intent_category/specific_intent optional when model is set |
| Streaming path | ~5 | Verify streaming dispatch works with pinned model |
| dispatch_mode field | ~5 | Verify field in EngineResponse, StreamChunk, log events |

### 6.2 Property-Based Tests (Hypothesis)

- **Operational invariant:** For any valid pinned dispatch, budget tracking and health recording always execute regardless of dispatch outcome.
- **Cascade isolation:** When `model` is set, no MBR/IBR/CBR/LBR function is called.
- **Error determinism:** The same (model, registry state, budget state, health state) tuple always produces the same error category.

### 6.3 Integration Tests

- Full pinned dispatch through RouterEngine.dispatch() with a mock adapter.
- Pinned dispatch followed by cascade dispatch on the same engine instance -- verify no state leakage.
- Pinned dispatch through HTTP handler (`dispatch_handler`) with model field in request body.
- Streaming pinned dispatch through SSE path.

## 7. Data Model Summary

### 7.1 New Types

| Type | Location | Fields | Purpose |
|------|----------|--------|---------|
| `ModelNotFoundError` | `core/types.py` | model, message | Pinned model not in registry |
| `ModelUnhealthyError` | `core/types.py` | model, status, message | Pinned model circuit-open or retired |
| `BudgetExhaustedError` | `core/types.py` | model, provider, message | Provider budget or rate limit exhausted |
| `PinnedDispatchConfig` | `config/schema.py` | honor_health | Pinned dispatch configuration |

### 7.2 Modified Types

| Type | Change | Backward Compatible |
|------|--------|---------------------|
| `DispatchOrder` | Add `model: str \| None = None` | Yes (default None) |
| `EngineResponse` | Add `dispatch_mode: str = "cascade"` | Yes (default "cascade") |
| `StreamChunk` | Add `dispatch_mode: str = "cascade"` | Yes (default "cascade") |
| `RouterConfig` | Add `pinned_dispatch: PinnedDispatchConfig` | Yes (default config) |
