# Intent Based Router v0.1.0 Live Spec

**Version:** 0.1.0
**Effective:** 2026-06-18
**Status:** Implemented (v0.3.0)
**Depends on:** Dragonlight Router v0.3.0 spec

## 1. System Overview

The Intent Based Router (IBR) adds two capabilities to the existing cascade pipeline:

1. **Intent classification** -- a lightweight LLM call that analyzes what a request *actually needs* beyond its declared role. A `code_review` request might need "deep architectural reasoning" or "quick syntax lint." These are different model requirements.
2. **Model flavor profiling** -- a structured representation of each model's strengths and weaknesses across intent dimensions. Two models that both pass MBR filtering for the same tier have measurably different "flavors" -- one excels at refactoring, another at documentation.

IBR sits between MBR and CBR in the cascade. It does not replace the existing pipeline -- it augments CBR scoring with an intent-flavor match signal. When IBR is disabled or fails, the pipeline degrades to v0.3.0 behavior with zero functional difference.

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
[IBR] Intent classification + flavor match scoring    <-- NEW
    |
    v
[CBR] Budget filter + cost-aware scoring (now includes flavor_match weight)
    |
    v
[LBR] Rate-limit filter + weighted random selection
    |
    v
[Dispatch] Fallback chain + context filtering + cache
    |
    v
EngineResponse / StreamChunk[]
```

### AC-IBR-SYS

- IBR-SYS-01: IBR MUST be opt-in via `intent_classification.enabled` in router.yaml (default: false).
- IBR-SYS-02: When disabled, the pipeline MUST behave identically to v0.3.0.
- IBR-SYS-03: When classification fails or times out, the pipeline MUST fall back to v0.3.0 behavior for that request (no flavor_match scoring applied).
- IBR-SYS-04: IBR MUST NOT increase median dispatch latency by more than 100ms when enabled.

## 2. Intent Classification Engine

### 2.1 Taxonomy

Classification produces a `ClassifiedIntent` with three orthogonal dimensions:

| Dimension | Values | Description |
|-----------|--------|-------------|
| **task_type** (8) | `generation`, `analysis`, `refactoring`, `summarization`, `creative`, `reasoning`, `lookup`, `translation` | What the request needs done |
| **domain** (6) | `code`, `technical`, `legal`, `business`, `creative_writing`, `general` | Subject matter signal |
| **quality_speed** (3) | `quality`, `balanced`, `speed` | Latency-quality tradeoff preference |

The taxonomy is fixed at the system level. Operators configure model flavor profiles against it, not the taxonomy itself.

### 2.2 Classification Model

**Decision: Local-first with hosted fallback.** The classifier runs as a structured-output prompt against a Haiku-class model. The prompt template ships with the router (not operator-configurable). The classifier uses backends registered under the `classification` role in the model-role matrix. If no classification backend is available or healthy, classification is skipped entirely (v0.3.0 behavior). The classifier receives only `operator_message` (not system_prompt or context) to bound input size and avoid information leakage.

### 2.3 Latency Budget

- **Hard timeout:** 100ms. Cancelled regardless of progress.
- **Soft target:** 50ms P95. Exceeding emits structured log warning.
- **Timeout behavior:** Set `classified_intent = None`, proceed without flavor_match signal.
- **Caching:** SHA-256 of `operator_message`. TTL: 300s. Max entries: 5000.

### 2.4 Classification Output

```python
@dataclass(frozen=True)
class ClassifiedIntent:
    task_type: str         # One of the 8 task_type values
    domain: str            # One of the 6 domain values
    quality_speed: str     # One of the 3 quality_speed values
    confidence: float      # 0.0-1.0 classifier self-reported confidence
    latency_ms: float      # Wall-clock time for classification
    from_cache: bool       # Whether this was a cache hit
```

### AC-IBR-CLS

- IBR-CLS-01: Classification MUST complete within 100ms hard timeout.
- IBR-CLS-02: Classification MUST receive only `operator_message` -- never system_prompt or context.
- IBR-CLS-03: Classification results MUST be cached by SHA-256 of operator_message.
- IBR-CLS-04: On timeout or error, classification MUST return None (not a default/guess).
- IBR-CLS-05: task_type, domain, and quality_speed MUST be validated against allowed values.
- IBR-CLS-06: All classification results MUST include latency_ms for observability.
- IBR-CLS-07: The `classification` role MUST exist in model_role_matrix.json when IBR is enabled.

## 3. Model Flavor Profile System

### 3.1 Profile Schema

A model flavor profile maps a model_id to its relative strengths across intent dimensions.

```python
@dataclass(frozen=True)
class FlavorScore:
    score: float           # 0.0-1.0 relative strength
    confidence: float      # 0.0-1.0 how much data backs this score
    sample_count: int      # Number of observations behind this score

@dataclass(frozen=True)
class ModelFlavorProfile:
    model_id: str
    version: int           # Profile schema version (for migration)
    updated_at: str        # ISO-8601 timestamp of last update

    # Task type scores (8 entries)
    task_scores: dict[str, FlavorScore]

    # Domain scores (6 entries)
    domain_scores: dict[str, FlavorScore]

    # Quality-speed alignment (3 entries)
    qs_scores: dict[str, FlavorScore]
```

Scores are relative within a model's profile (not absolute cross-model rankings). Missing dimensions default to `FlavorScore(score=0.5, confidence=0.0, sample_count=0)`.

### 3.2 Discovery Methods

Three methods, ranked by priority and reliability:

**Method 1: Operator-declared profiles (v0.1.0, primary)**

Operators author profiles in `config/model_flavor_profiles.yaml`. Each model maps task_scores, domain_scores, and qs_scores as `dimension_name: float` pairs (0.0-1.0). Unlisted dimensions inherit neutral default. Loaded at boot, reloaded on file change (same mechanism as RoleMatrix). Ships with starter profiles for default matrix models.

**Method 2: Feedback-loop learning (v0.2.0)**

When a request has a `ClassifiedIntent` AND the outcome includes `quality_rating` (1-5), the system updates the model's profile via EMA: `new = 0.1 * observation + 0.9 * old`. Confidence grows: `min(1.0, sample_count / 50)`. Stored in SQLite (`state_dir/flavor_profiles.db`), overlays operator-declared profiles. Feedback can raise scores but never lower below 80% of operator-declared value (intent preservation).

**Method 3: Automated benchmarking (v0.3.0)**

CLI tool (`dragonlight-router benchmark-flavors`) runs ~50 standardized eval prompts per model, scored by LLM-as-judge. Produces high-confidence profiles. Stale benchmarks (>30 days) decay toward 0.5 at 0.01/day.

### 3.3 Profile Storage and Lifecycle

| Source | Storage | Loaded at | Update frequency | Priority |
|--------|---------|-----------|-----------------|----------|
| Operator-declared | `config/model_flavor_profiles.yaml` | Boot + file watch | Manual edit | Base layer |
| Feedback-learned | `state_dir/flavor_profiles.db` | Boot | Per-request | Overlay |
| Benchmark | `state_dir/benchmark_profiles.json` | Boot | On benchmark run | Overlay |

Resolution order: benchmark > feedback > operator-declared > neutral default.

### AC-IBR-FLV

- IBR-FLV-01: Model flavor profiles MUST load from `config/model_flavor_profiles.yaml`.
- IBR-FLV-02: Missing dimensions MUST default to FlavorScore(0.5, 0.0, 0).
- IBR-FLV-03: Feedback-learned scores MUST NOT lower a dimension below 80% of the operator-declared value.
- IBR-FLV-04: Flavor profiles MUST be reloaded on file change (same mechanism as RoleMatrix).
- IBR-FLV-05: All FlavorScore.score values MUST be in [0.0, 1.0].
- IBR-FLV-06: Benchmark profiles older than 30 days MUST decay toward 0.5.

## 4. Scoring Integration

### 4.1 Flavor Match Score

Given a `ClassifiedIntent` and a `ModelFlavorProfile`, the flavor match score is:

```
flavor_match = (
    task_weight * profile.task_scores[intent.task_type].score +
    domain_weight * profile.domain_scores[intent.domain].score +
    qs_weight * profile.qs_scores[intent.quality_speed].score
)
```

Where `task_weight = 0.50`, `domain_weight = 0.30`, `qs_weight = 0.20`. These weights are not operator-configurable in v0.1.0.

The flavor_match is a value in [0.0, 1.0].

**Confidence gating:** If the classifier confidence < 0.6 OR the average FlavorScore.confidence across matched dimensions < 0.3, the flavor_match is discarded (treated as if IBR were disabled for this request). This prevents low-confidence classifications from distorting routing.

### 4.2 CBR Weight Integration

When IBR is active and produces a valid flavor_match, CBR scoring gains a sixth dimension:

**IBR-enabled weights (default):**
- cost: 0.30 (was 0.35)
- latency: 0.20 (was 0.25)
- priority: 0.15 (was 0.20)
- queue: 0.10 (unchanged)
- health: 0.10 (unchanged)
- flavor_match: 0.15 (new)

Sum = 1.0.

When IBR is disabled, classification fails, or confidence gating triggers, the original v0.3.0 weights apply (sum = 1.0 without flavor_match).

**Cost governor interaction:** When cost governor is active, flavor_match weight drops to 0.05 and cost absorbs the difference:
- cost: 0.65, latency: 0.10, priority: 0.10, queue: 0.05, health: 0.05, flavor_match: 0.05.

### AC-IBR-SCORE

- IBR-SCORE-01: Flavor match MUST be computed as weighted sum of task, domain, and qs scores.
- IBR-SCORE-02: When IBR produces valid flavor_match, CBR weights MUST sum to 1.0 across 6 dimensions.
- IBR-SCORE-03: When IBR is disabled or fails, CBR weights MUST revert to v0.3.0 5-dimension values.
- IBR-SCORE-04: Confidence gating MUST discard flavor_match when classifier confidence < 0.6 OR profile confidence < 0.3.
- IBR-SCORE-05: Cost governor MUST reduce flavor_match weight to 0.05 (cost preservation priority).

## 5. Pipeline Integration

### 5.1 Execution Flow

IBR classification runs **after MBR and trust floor filtering** but **before CBR scoring**. The MBR-filtered candidate list is needed to look up flavor profiles. Classification and profile lookup are independent and can run concurrently.

```
MBR candidates (list[BackendConfig])
    |
    +--[async] classify_intent(order.operator_message)
    |     -> ClassifiedIntent | None
    |
    +--[sync] load_flavor_profiles(candidate.name for candidate in mbr_candidates)
    |     -> dict[str, ModelFlavorProfile]
    |
    v
compute_flavor_scores(intent, profiles)
    -> dict[str, float]  # model_id -> flavor_match score
    |
    v
CBR scoring (with flavor_match as 6th dimension)
```

### 5.2 Relationship to Existing Systems

**ComplexityEstimator:** Unchanged. Drives MBR tier estimation (vertical axis: "how powerful?"). IBR operates on the horizontal axis ("among peers at this tier, which flavor?"). `quality_speed` from ClassifiedIntent influences CBR weight balance (speed preference increases latency weight, quality preference increases flavor_match weight) but never overrides tier selection.

**Model Role Matrix:** Remains source of truth for which models serve which roles. IBR re-ranks MBR-filtered candidates; it never adds models absent from the matrix. The matrix gains one new role: `classification`.

### AC-IBR-PIPE

- IBR-PIPE-01: Classification MUST run after MBR filtering, before CBR scoring.
- IBR-PIPE-02: Classification and profile lookup MUST be runnable concurrently.
- IBR-PIPE-03: IBR MUST NOT modify MBR tier estimation or candidate filtering.
- IBR-PIPE-04: IBR MUST NOT add models that are absent from the role matrix.
- IBR-PIPE-05: The `classification` role in model_role_matrix.json MUST be required when IBR is enabled.

## 6. Configuration

### 6.1 router.yaml Extension

```yaml
# New section in router.yaml
intent_classification:
  enabled: false                    # Opt-in (IBR-SYS-01)
  timeout_ms: 100                   # Hard classification timeout
  cache_ttl_s: 300                  # Classification cache TTL
  cache_max_entries: 5000           # Classification cache size
  confidence_threshold: 0.6         # Minimum classifier confidence
  profile_confidence_threshold: 0.3 # Minimum flavor profile confidence
  flavor_match_weight: 0.15         # Weight in CBR scoring (6th dimension)
  flavor_match_weight_governor: 0.05 # Weight when cost governor active
```

### 6.2 model_flavor_profiles.yaml

New file at `config/model_flavor_profiles.yaml`. Schema defined in section 3.2. Loaded by a new `FlavorProfileLoader` that mirrors `RoleMatrix` behavior (load at boot, watch for changes).

### 6.3 model_role_matrix.json Addition

```json
{
  "classification": [
    {"model_id": "ollama/phi-3-mini", "rank": 95},
    {"model_id": "groq/llama-3.3-70b-versatile", "rank": 80},
    {"model_id": "gemini/gemini-2.5-flash", "rank": 75}
  ]
}
```

### AC-IBR-CFG

- IBR-CFG-01: All IBR configuration MUST have sensible defaults (disabled by default).
- IBR-CFG-02: `intent_classification` section MUST be optional in router.yaml.
- IBR-CFG-03: `model_flavor_profiles.yaml` MUST be optional (missing file = all neutral profiles).
- IBR-CFG-04: Configuration MUST be validated at boot via Pydantic schema.

## 7. Observability

### 7.1 Structured Logging

Two log events per IBR-active dispatch: `ibr_classification` (task_type, domain, quality_speed, confidence, latency_ms, from_cache, classifier_model) and `ibr_flavor_match` (model_id, flavor_match_score, per-dimension scores, confidence_gated).

### 7.2 EngineResponse Extension

EngineResponse and StreamChunk gain optional fields: `classified_intent: ClassifiedIntent | None`, `flavor_match_score: float | None`, `ibr_active: bool`. All default to None/False when IBR is disabled.

### 7.3 Metrics

New `/metrics` counters: `ibr_classification_count`, `ibr_classification_cache_hit_rate`, `ibr_classification_timeout_count`, `ibr_classification_p50_ms`, `ibr_classification_p95_ms`, `ibr_confidence_gate_count`.

### AC-IBR-OBS

- IBR-OBS-01: Every classification MUST emit a structured log event with all ClassifiedIntent fields.
- IBR-OBS-02: Every flavor match computation MUST emit a structured log event with per-dimension scores.
- IBR-OBS-03: EngineResponse MUST include `classified_intent`, `flavor_match_score`, and `ibr_active`.
- IBR-OBS-04: Classification timeout events MUST be logged at warning level.
- IBR-OBS-05: `/metrics` MUST include IBR-specific counters when IBR is enabled.

## 8. API Changes

### 8.1 Modified Endpoints

**POST /v1/dispatch** -- response gains optional `classified_intent` (ClassifiedIntent object), `flavor_match_score` (float), `ibr_active` (bool). Absent when IBR is disabled.

**POST /v1/record** -- request gains optional `quality_rating` (int 1-5). Feeds feedback learning (v0.2.0).

### 8.2 New Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/flavor-profiles` | None | All loaded flavor profiles |
| GET | `/v1/flavor-profiles/{model_id}` | None | Single model profile (404 if missing) |
| POST | `/v1/flavor-profiles/{model_id}` | Admin | Upsert profile at runtime |

### AC-IBR-API

- IBR-API-01: `/v1/dispatch` response MUST include IBR fields only when IBR is active.
- IBR-API-02: `/v1/flavor-profiles` MUST return profiles without authentication.
- IBR-API-03: POST `/v1/flavor-profiles/{model_id}` MUST require admin auth.
- IBR-API-04: `quality_rating` in `/v1/record` MUST be validated as integer 1-5.

## 9. Data Model

### 9.1 New Types

All new types are frozen dataclasses in `core/types.py`:

| Type | Fields | Purpose |
|------|--------|---------|
| `ClassifiedIntent` | task_type, domain, quality_speed, confidence, latency_ms, from_cache | Classification output |
| `FlavorScore` | score, confidence, sample_count | Single dimension score |
| `ModelFlavorProfile` | model_id, version, updated_at, task_scores, domain_scores, qs_scores | Full model profile |
| `IBRScoringContext` | classified_intent, flavor_profiles, flavor_match_weight | Passed into CBR |

### 9.2 Modified Types

| Type | Change | Backward Compatible |
|------|--------|---------------------|
| `EngineResponse` | Add optional `classified_intent`, `flavor_match_score`, `ibr_active` | Yes (defaults to None/False) |
| `StreamChunk` | Add optional `classified_intent_json`, `flavor_match_score`, `ibr_active` | Yes (defaults to empty/None/False) |
| `ScoringWeightsConfig` | Add optional `flavor_match: float = 0.0` | Yes (0.0 = no effect) |
| `RequestOutcome` | Add optional `quality_rating: int | None = None` | Yes (default None) |

### 9.3 New Modules

| Module | Purpose |
|--------|---------|
| `selection/ibr.py` | Intent classification + flavor matching orchestration |
| `selection/classifier.py` | Classification prompt, LLM call, caching, timeout |
| `selection/flavor.py` | FlavorProfileLoader, match scoring, feedback updates |

### AC-IBR-DATA

- IBR-DATA-01: All new types MUST be frozen dataclasses.
- IBR-DATA-02: All modified types MUST remain backward compatible (optional fields with defaults).
- IBR-DATA-03: `ScoringWeightsConfig.__post_init__` MUST enforce sum = 1.0 including flavor_match.

## 10. Testing Strategy

### 10.1 Unit Tests

| Area | Tests | Approach |
|------|-------|----------|
| Classification taxonomy | ~30 | Validate all allowed values, reject invalid |
| Classification timeout | ~10 | Mock classifier with sleep, verify None return |
| Classification caching | ~15 | Cache hit/miss, TTL expiry, key computation |
| Flavor profile loading | ~20 | YAML parse, defaults, file watch, missing file |
| Flavor match scoring | ~25 | All dimension combinations, weight validation |
| Confidence gating | ~15 | Below threshold, above threshold, edge cases |
| CBR weight integration | ~20 | 6-dimension scoring, fallback to 5-dimension |
| Cost governor interaction | ~10 | Governor + IBR weight adjustment |

### 10.2 Property-Based Tests (Hypothesis)

- **Scoring invariant:** flavor_match score is always in [0.0, 1.0] for any valid ClassifiedIntent and ModelFlavorProfile.
- **Weight sum invariant:** ScoringWeightsConfig with flavor_match always sums to 1.0.
- **Confidence gating invariant:** flavor_match is never applied when confidence < threshold.
- **Degradation invariant:** IBR failure always produces identical behavior to IBR-disabled.
- **Monotonicity:** Higher flavor_match scores produce higher CBR composite scores, all else equal.

### 10.3 Integration Tests

- Full cascade with IBR enabled: MBR -> IBR -> CBR -> LBR -> dispatch.
- Classification timeout mid-cascade: verify graceful degradation.
- Flavor profile hot-reload: update YAML, verify next dispatch uses new profiles.
- IBR disabled: verify zero overhead and identical behavior to v0.3.0.

### 10.4 Benchmark Tests

- Classification latency P50/P95 against local Ollama model (must be <50ms/<100ms).
- End-to-end dispatch latency delta with IBR enabled vs disabled (must be <100ms).
- Classification cache hit rate under synthetic load (target >60% after warmup).

### AC-IBR-TEST

- IBR-TEST-01: All existing 1079 tests MUST continue to pass with IBR disabled.
- IBR-TEST-02: Property-based tests MUST cover scoring, confidence gating, and degradation invariants.
- IBR-TEST-03: Integration tests MUST verify graceful degradation on classification failure.
- IBR-TEST-04: Benchmark tests MUST verify P95 classification latency < 100ms.

## 11. Migration Path

IBR is fully opt-in. Migration from v0.3.0: (1) no config change required -- IBR defaults to disabled, (2) add `intent_classification: { enabled: true }` to router.yaml, (3) add `classification` role to model_role_matrix.json, (4) optionally create `config/model_flavor_profiles.yaml`. Rollback: set `enabled: false`. All type changes are additive (optional fields with defaults). No fields removed or renamed.

### AC-IBR-MIG

- IBR-MIG-01: Enabling IBR MUST NOT require changes to existing router.yaml fields.
- IBR-MIG-02: Disabling IBR MUST immediately revert to v0.3.0 behavior.
- IBR-MIG-03: Existing API consumers MUST NOT break when IBR fields are added to responses.

## 12. Open Questions

**OQ-1: Classification model selection.** Benchmark Ollama phi-3-mini (3.8B), Groq llama-3.3-8b, and Gemini Flash against 200 sample prompts. Measure latency P50/P95 and accuracy vs human labels. Hypothesis: local 3-4B model achieves >80% accuracy at <30ms P95.

**OQ-2: Feedback convergence rate.** Simulate EMA feedback loop with synthetic ratings. Hypothesis: 30-50 observations per dimension for stability. High-traffic dimensions stabilize in ~1 week; long-tail dimensions may need months.

**OQ-3: Taxonomy granularity.** Ship with 8 task types. Instrument classification distribution. Review after 30 days: subdivide types receiving >40% of traffic, merge types receiving <5%.

**OQ-4: Cross-model calibration.** v0.1.0 does not calibrate across models. Scores are model-relative; cross-model comparison happens implicitly through rank signal in the model-role matrix. Revisit with automated benchmarking in v0.2.0.

## 13. Hazard Analysis

| ID | Hazard | Severity | Mitigation |
|----|--------|----------|------------|
| HAZ-015 | Classification increases dispatch latency beyond acceptable | High | 100ms hard timeout + cache + graceful degradation |
| HAZ-016 | Misclassification routes to wrong model flavor | Medium | Confidence gating (discard low-confidence), no-worse-than-v0.3.0 guarantee |
| HAZ-017 | Feedback loop converges on wrong profiles | Medium | Operator floor (80% of declared value), decay toward neutral |
| HAZ-018 | Classification model unavailable | Low | Skip classification entirely, v0.3.0 behavior |
| HAZ-019 | Flavor profile YAML syntax error blocks boot | Medium | Load with empty profiles on parse error, log warning |
