# IBR Cascade Activation: Spectrograph, Intent Categories, Per-Request CBR Weights

**Status:** Design  
**Authors:** GOIBNIU + LUGH  
**Date:** 2026-06-24  
**Scope:** Three enhancements to the dispatch pipeline — spectrograph activation audit, finer intent categories, per-request CBR weight adjustment.

---

## 1. Spectrograph Activation Audit

### 1.1 Current State

The spectrography system is **fully wired and active**. The end-to-end path is:

1. **Profiles exist** at `config/model_spectrograph_profiles.yaml` with empirical data from spectrography runs (40+ models profiled across task_type, domain, qs_scores).
2. **IBR is enabled** in `config/router.yaml` (`intent_classification.enabled: true`).
3. **Boot path:** `RouterEngine.__init__` calls `_init_ibr()`, which:
   - Creates `SpectrographProfileLoader` from `model_spectrograph_profiles.yaml`
   - Creates `FeedbackStore` at `{state_dir}/spectrograph_feedback.db`
   - Resolves a `classification_adapter` from the `classification` role in the matrix
4. **Dispatch path:** `cascade.dispatch()` -> `_run_cascade()` -> `_run_ibr_stage()`:
   - Calls `run_ibr_stage()` in `selection/ibr.py`
   - Classifies intent via LLM call (with SHA-256 cache)
   - Loads spectrograph profiles (with mtime hot-reload)
   - Computes `spectrograph_match` scores per candidate
   - Returns `IBRResult` with `ibr_active=True` and `spectrograph_scores` dict
5. **Scoring path:** `_run_cbr_stage()` calls `_resolve_cbr_weights()`, which checks `IBRResult`:
   - If IBR is active, uses 6-dimension `ScoringWeightsConfig` with `spectrograph_match=0.15`
   - `_score_and_rank_candidates()` passes `spectrograph_match` to `score_candidate()`
   - `_apply_weights()` multiplies `normalized.spectrograph_match * weights.spectrograph_match`
6. **Feedback loop:** `RouterEngine.record_ibr_feedback()` -> `FeedbackStore.record_feedback()` applies EMA updates to learned profiles with floor enforcement.

### 1.2 Gaps Identified

| Gap | Severity | Description |
|-----|----------|-------------|
| GAP-SPEC-01 | Medium | `FeedbackStore.get_learned_profiles()` is never called during dispatch. Learned profiles from feedback are persisted to SQLite but never merged with operator-declared profiles at scoring time. The `SpectrographProfileLoader.get_merged_profiles()` method exists but is not invoked in the pipeline. |
| GAP-SPEC-02 | Low | The spectrography runner (`spectrography/runner.py`) writes profiles to `config/model_spectrograph_profiles.yaml` via `--write-profiles`, but there is no automated schedule for re-running spectrography. Profiles go stale as new models appear. |
| GAP-SPEC-03 | Low | The `classification` role must exist in the role matrix for IBR to activate. If no model has this role, `_resolve_classification_adapter()` returns None and IBR silently degrades. No health check or alert for this condition. |

### 1.3 Required Changes for GAP-SPEC-01

**Goal:** Wire feedback-learned profiles into the scoring pipeline so that operational feedback improves routing over time.

**File:** `src/dragonlight_router/selection/ibr.py`

In `_build_ibr_result()`, after loading `profiles = spectrograph_loader.profiles`, merge with feedback-learned profiles:

```python
# Current (line 175):
profiles = spectrograph_loader.profiles

# Proposed:
profiles = spectrograph_loader.profiles
if feedback_store is not None:
    learned = feedback_store.get_learned_profiles()
    if learned:
        profiles = spectrograph_loader.get_merged_profiles(learned)
```

This requires threading `feedback_store: FeedbackStore | None` through:
1. `run_ibr_stage()` signature — add `feedback_store` param
2. `_execute_ibr()` signature — add `feedback_store` param
3. `_build_ibr_result()` signature — add `feedback_store` param
4. `cascade._run_ibr_stage()` — pass `ctx.feedback_store` (requires adding `feedback_store` to `DispatchContext`)

**File:** `src/dragonlight_router/dispatch/cascade.py`

Add `feedback_store` to `DispatchContext`:

```python
@dataclass(frozen=True)
class DispatchContext:
    # ... existing fields ...
    feedback_store: FeedbackStore | None = None
```

**File:** `src/dragonlight_router/router.py`

In `dispatch()` and `dispatch_stream()`, pass `feedback_store=self._feedback_store` to cascade functions.

### 1.4 Required Changes for GAP-SPEC-03

**File:** `src/dragonlight_router/router.py`

In `_init_ibr()`, after `_resolve_classification_adapter()` returns None, emit a structured warning that can be picked up by monitoring:

```python
if self._classification_adapter is None:
    logger.warning(
        "ibr_degraded_no_classifier",
        reason="no_classification_role_or_no_available_backend",
        ibr_enabled=True,
        impact="IBR will be inactive for all requests until a classification backend is available",
    )
```

---

## 2. Finer Intent Categories

### 2.1 Current State

There are **two separate intent taxonomies** in the codebase, serving different purposes:

**IBR Classifier Taxonomy** (`selection/classifier.py`, `core/types.py`):
- `TASK_TYPES`: generation, analysis, refactoring, summarization, creative, reasoning, lookup, translation
- `DOMAINS`: code, technical, legal, business, creative_writing, general
- `QUALITY_SPEED`: quality, balanced, speed

These are the dimensions the LLM classifier outputs. They map to spectrograph profile dimensions for model-capability matching.

**Factory Intent Categories** (`DispatchOrder.intent_category`):
- Used by MBR tier floors (`selection/mbr.py` `_INTENT_TIER_FLOOR`)
- Used by CBR weight profiles (`selection/scoring.py` `_LOW_STAKES_INTENTS`, `_HIGH_STAKES_INTENTS`)
- Current recognized categories:
  - **MBR tier floors:** complex_reasoning, strategic_planning, architecture, implementation_complex (COMPLEX); engineering_build, code_review, debugging, spec_writing, code_generation, implementation, coherence_merge (MODERATE); test_generation, test_property, audit, data_analysis, summarization (SIMPLE)
  - **CBR weight profiles:** test_generation, test_property, audit, data_analysis, summarization (low-stakes); implementation, implementation_complex, coherence_merge, complex_reasoning, strategic_planning, architecture (high-stakes)

### 2.2 Gap Analysis

The factory sends `intent_category` values like `test_generation`, `implementation`, `debugging`. The MBR stage uses these for tier floors. The CBR stage uses them for weight profiles. But many categories that MBR recognizes have no corresponding CBR weight profile (they fall through to default weights):

| intent_category | MBR tier floor | CBR weight profile |
|----------------|---------------|-------------------|
| engineering_build | MODERATE | **default** (gap) |
| code_review | MODERATE | **default** (gap) |
| debugging | MODERATE | **default** (gap) |
| spec_writing | MODERATE | **default** (gap) |
| code_generation | MODERATE | **default** (gap) |
| session_lifecycle | (uses complexity.py OPUS) | **default** (gap) |

### 2.3 Proposed Intent Category Additions

Add factory-facing intent categories that the factory can send, and wire them into both MBR tier floors and CBR weight profiles.

#### 2.3.1 New MBR Tier Floors

**File:** `src/dragonlight_router/selection/mbr.py`

Add to `_INTENT_TIER_FLOOR`:

```python
_INTENT_TIER_FLOOR: dict[str, BackendTier] = {
    # Existing entries unchanged...

    # New categories
    "refactoring": BackendTier.MODERATE,
    "documentation": BackendTier.SIMPLE,
    "test_fix": BackendTier.SIMPLE,
    "security_review": BackendTier.COMPLEX,
    "performance_optimization": BackendTier.COMPLEX,
    "migration": BackendTier.MODERATE,
    "api_design": BackendTier.MODERATE,
}
```

#### 2.3.2 New CBR Weight Profile Classifications

**File:** `src/dragonlight_router/selection/scoring.py`

Expand the stakes classifications to cover all recognized intent categories:

```python
_LOW_STAKES_INTENTS: frozenset[str] = frozenset({
    # Existing
    "test_generation",
    "test_property",
    "audit",
    "data_analysis",
    "summarization",
    # New
    "documentation",
    "test_fix",
    "code_generation",      # simple code gen is low-stakes (templates, boilerplate)
})

_HIGH_STAKES_INTENTS: frozenset[str] = frozenset({
    # Existing
    "implementation",
    "implementation_complex",
    "coherence_merge",
    "complex_reasoning",
    "strategic_planning",
    "architecture",
    # New
    "security_review",
    "performance_optimization",
    "debugging",             # debugging requires capability, not speed
    "migration",
})

# New: mid-stakes category for intents that need balanced weights
# but with a slight capability bias over default.
_MID_STAKES_INTENTS: frozenset[str] = frozenset({
    "engineering_build",
    "code_review",
    "spec_writing",
    "refactoring",
    "api_design",
})

_MID_STAKES_WEIGHTS = ScoringWeightsConfig(
    cost=0.15,
    latency=0.20,
    priority=0.25,
    spectrograph_match=0.20,
    queue=0.10,
    health=0.10,
)
```

Update `intent_weights_for_category()`:

```python
def intent_weights_for_category(intent_category: str) -> ScoringWeightsConfig:
    if intent_category in _LOW_STAKES_INTENTS:
        return _LOW_STAKES_WEIGHTS
    if intent_category in _HIGH_STAKES_INTENTS:
        return _HIGH_STAKES_WEIGHTS
    if intent_category in _MID_STAKES_INTENTS:
        return _MID_STAKES_WEIGHTS
    return ScoringWeightsConfig()
```

### 2.4 Intent-to-IBR Mapping

When the factory sends `intent_category`, the router also runs the IBR classifier to get `ClassifiedIntent` (task_type, domain, quality_speed). These serve complementary purposes:

- `intent_category` -> CBR weight profile selection (which scoring dimensions matter)
- `ClassifiedIntent` -> spectrograph match scoring (which models are best suited)

No additional mapping between these two systems is needed. They operate on orthogonal axes: intent_category controls *how much* spectrograph matching matters, while ClassifiedIntent controls *what the match score is*.

---

## 3. Per-Request CBR Weight Adjustment

### 3.1 Architecture

The weight adjustment system operates on a **three-tier stakes classification** derived from `intent_category` (Section 2.3.2) plus optional context signals from the `DispatchOrder`.

The existing `_resolve_cbr_weights()` function in `cascade.py` already has the hook point. The enhancement extends it with a richer signal set and a third weight tier.

### 3.2 Stakes Classification

#### 3.2.1 Primary Signal: intent_category

The `intent_category` field on `DispatchOrder` is the primary stakes classifier. Section 2.3.2 defines the three sets: `_LOW_STAKES_INTENTS`, `_MID_STAKES_INTENTS`, `_HIGH_STAKES_INTENTS`.

#### 3.2.2 Secondary Signals: Context Escalation

Context signals can **escalate** stakes (never reduce). This prevents a high-context-count test generation request from being routed to a cheap model that will truncate.

**File:** `src/dragonlight_router/selection/scoring.py`

New function:

```python
def classify_request_stakes(order: DispatchOrder) -> str:
    """Classify a request into cost-optimized / balanced / capability-optimized.

    Primary signal: intent_category mapping to LOW/MID/HIGH stakes.
    Secondary signals (escalation only):
    - context_tokens > 8000 -> escalate to at least balanced
    - requires_tool_use -> escalate to at least balanced
    - context_tokens > 32000 -> escalate to capability-optimized
    - context_trust_tier == "local" or "trusted" -> escalate to capability-optimized
      (security-sensitive context implies high-stakes work)

    Returns:
        One of "cost_optimized", "balanced", "capability_optimized"
    """
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder"

    # Primary classification from intent_category
    if order.intent_category in _HIGH_STAKES_INTENTS:
        base_tier = "capability_optimized"
    elif order.intent_category in _MID_STAKES_INTENTS:
        base_tier = "balanced"
    elif order.intent_category in _LOW_STAKES_INTENTS:
        base_tier = "cost_optimized"
    else:
        base_tier = "balanced"  # unknown intents get balanced

    # Escalation (never reduces)
    _TIER_RANK = {"cost_optimized": 0, "balanced": 1, "capability_optimized": 2}
    current_rank = _TIER_RANK[base_tier]

    if order.context_tokens > 32000:
        current_rank = max(current_rank, 2)
    elif order.context_tokens > 8000:
        current_rank = max(current_rank, 1)

    if order.requires_tool_use:
        current_rank = max(current_rank, 1)

    if order.context_trust_tier in ("local", "trusted"):
        current_rank = max(current_rank, 2)

    _RANK_TO_TIER = {0: "cost_optimized", 1: "balanced", 2: "capability_optimized"}
    return _RANK_TO_TIER[current_rank]
```

### 3.3 Weight Profiles

Three profiles that shift scoring emphasis across the six scoring dimensions. All sum to 1.0.

#### 3.3.1 Cost-Optimized Profile

**Use case:** test_generation, documentation, data_analysis, audit, summarization, test_fix, code_generation.

**Philosophy:** Fastest correct answer wins. Cost and latency dominate. Spectrograph match is low because we are willing to use any model that is cheap and fast.

```python
_COST_OPTIMIZED_WEIGHTS = ScoringWeightsConfig(
    cost=0.35,           # +0.15 vs default: maximize cheapness
    latency=0.30,        # +0.05 vs default: maximize speed
    priority=0.10,       # -0.10 vs default: rank matters less
    spectrograph_match=0.10,  # -0.05 vs default: capability matching less important
    queue=0.10,          # +0.00 vs default: unchanged
    health=0.05,         # -0.05 vs default: willing to try degraded backends
)
```

This is the existing `_LOW_STAKES_WEIGHTS` with the same values. No change needed.

#### 3.3.2 Balanced Profile (new)

**Use case:** engineering_build, code_review, spec_writing, refactoring, api_design, unknown intents.

**Philosophy:** Quality matters but we are not burning money. Spectrograph match has moderate influence. Priority (rank) is weighted higher to respect the role matrix ordering.

```python
_BALANCED_WEIGHTS = ScoringWeightsConfig(
    cost=0.15,           # -0.05 vs default: cost matters but is not primary
    latency=0.20,        # -0.05 vs default: speed matters but not critical
    priority=0.25,       # +0.05 vs default: trust role matrix rankings
    spectrograph_match=0.20,  # +0.05 vs default: use capability signal
    queue=0.10,          # +0.00 vs default: unchanged
    health=0.10,         # +0.00 vs default: unchanged
)
```

This is the proposed `_MID_STAKES_WEIGHTS` from Section 2.3.2.

#### 3.3.3 Capability-Optimized Profile

**Use case:** implementation, architecture, security_review, complex_reasoning, strategic_planning, debugging, migration, performance_optimization.

**Philosophy:** Getting the right answer matters more than speed or cost. Spectrograph match is heavily weighted because we need the model that is actually good at this task type. Priority (rank) is weighted high because operator-curated rankings encode domain knowledge.

```python
_CAPABILITY_OPTIMIZED_WEIGHTS = ScoringWeightsConfig(
    cost=0.10,           # -0.10 vs default: willing to pay for quality
    latency=0.10,        # -0.15 vs default: willing to wait for quality
    priority=0.30,       # +0.10 vs default: operator rankings are gospel
    spectrograph_match=0.25,  # +0.10 vs default: spectrograph is primary signal
    queue=0.10,          # +0.00 vs default: unchanged
    health=0.15,         # +0.05 vs default: avoid unstable backends for critical work
)
```

This is the existing `_HIGH_STAKES_WEIGHTS` with the same values. No change needed.

### 3.4 Interaction with Existing Scoring Dimensions

The six dimensions in `ScoringWeightsConfig` map to raw scores extracted in `_extract_raw_scores()`:

| Dimension | Raw score source | What weight shift does |
|-----------|-----------------|----------------------|
| `cost` | `min(100 / (avg_cost + 1), 100)` — inverse of $/Mtok | Higher weight = prefer cheaper models |
| `latency` | Health score proxy (reuses `health_score`) | Higher weight = prefer historically fast models |
| `priority` | `BackendConfig.priority` — from role matrix rank | Higher weight = trust operator curation |
| `queue` | Budget availability inverted (`100 - budget_score`) | Higher weight = avoid providers near limits |
| `health` | Health tracker score (error count, circuit state) | Higher weight = avoid error-prone models |
| `spectrograph_match` | `compute_spectrograph_match()` — weighted match across task/domain/qs | Higher weight = prefer models proven good at this task |

### 3.5 Interaction with Cost Governor

The cost governor (`cost_governor_active()`) overrides weight profiles when daily/monthly spend exceeds thresholds. This is a **hard override** that takes priority over per-request weight adjustment.

The existing `cost_adjusted_weights()` already handles both IBR-active and IBR-inactive cases. No changes needed. The cost governor activates when thresholds are hit and squashes all weight profiles down to cost-heavy, regardless of intent.

This is correct behavior: budget exhaustion is a system-level concern that overrides per-request optimization.

### 3.6 Implementation Plan

#### 3.6.1 Changes to `scoring.py`

1. Add `_MID_STAKES_INTENTS` frozenset (Section 2.3.2)
2. Add `_MID_STAKES_WEIGHTS` constant (Section 3.3.2)
3. Update `intent_weights_for_category()` to check `_MID_STAKES_INTENTS`
4. Add `classify_request_stakes()` function (Section 3.2.2)
5. Update `_LOW_STAKES_INTENTS` and `_HIGH_STAKES_INTENTS` with new categories (Section 2.3.2)

#### 3.6.2 Changes to `cascade.py`

Update `_resolve_cbr_weights()` to use the new three-tier system. The function already has the right shape. The change replaces the binary `intent_weights_for_category` check with the richer `classify_request_stakes` pathway:

```python
def _resolve_cbr_weights(
    ibr_result: IBRResult | None,
    ctx: DispatchContext,
    order: DispatchOrder | None = None,
) -> ScoringWeightsConfig:
    # Per-intent-category weight profiles take priority
    if order is not None:
        intent_weights = intent_weights_for_category(order.intent_category)
        default_weights = ScoringWeightsConfig()
        if intent_weights != default_weights:
            logger.debug(
                "cbr_weights_from_intent_category",
                intent_category=order.intent_category,
                stakes=classify_request_stakes(order),
                cost=intent_weights.cost,
                spectrograph_match=intent_weights.spectrograph_match,
            )
            return intent_weights

    # ... rest unchanged (IBR fallback, default)
```

The `classify_request_stakes()` call in the log is for observability. The actual weight selection still routes through `intent_weights_for_category()`, which now has three tiers. The context-based escalation from `classify_request_stakes()` can override the intent-based classification:

```python
def _resolve_cbr_weights(
    ibr_result: IBRResult | None,
    ctx: DispatchContext,
    order: DispatchOrder | None = None,
) -> ScoringWeightsConfig:
    if order is not None:
        # Try intent-based classification first
        intent_weights = intent_weights_for_category(order.intent_category)
        default_weights = ScoringWeightsConfig()

        # Context-based escalation
        stakes = classify_request_stakes(order)
        stakes_weights = _STAKES_TO_WEIGHTS.get(stakes)

        if intent_weights != default_weights and stakes_weights is not None:
            # Both signals active: use whichever is higher-stakes
            # (higher spectrograph_match weight = higher stakes)
            if stakes_weights.spectrograph_match > intent_weights.spectrograph_match:
                logger.debug(
                    "cbr_weights_escalated_by_context",
                    intent_category=order.intent_category,
                    stakes=stakes,
                )
                return stakes_weights
            return intent_weights
        elif intent_weights != default_weights:
            return intent_weights
        elif stakes_weights is not None:
            return stakes_weights

    if ibr_result is not None and ibr_result.ibr_active:
        # ... existing IBR fallback ...
    return ScoringWeightsConfig()
```

New constant in `scoring.py`:

```python
_STAKES_TO_WEIGHTS: dict[str, ScoringWeightsConfig] = {
    "cost_optimized": _LOW_STAKES_WEIGHTS,
    "balanced": _MID_STAKES_WEIGHTS,
    "capability_optimized": _HIGH_STAKES_WEIGHTS,
}
```

#### 3.6.3 Changes to `mbr.py`

Add new intent categories to `_INTENT_TIER_FLOOR` (Section 2.3.1).

#### 3.6.4 Changes to `ibr.py`

Wire `feedback_store` through the IBR pipeline (Section 1.3).

#### 3.6.5 Changes to `router.py`

1. Pass `feedback_store` to cascade dispatch calls
2. Add degraded-IBR warning (Section 1.4)

---

## 4. File Change Summary

| File | Change Type | Description |
|------|------------|-------------|
| `src/dragonlight_router/selection/scoring.py` | Modify | Add `_MID_STAKES_INTENTS`, `_MID_STAKES_WEIGHTS`, `_STAKES_TO_WEIGHTS`. Update `intent_weights_for_category()` for three tiers. Add `classify_request_stakes()`. Expand `_LOW_STAKES_INTENTS` and `_HIGH_STAKES_INTENTS`. |
| `src/dragonlight_router/selection/mbr.py` | Modify | Add new intent categories to `_INTENT_TIER_FLOOR`: refactoring, documentation, test_fix, security_review, performance_optimization, migration, api_design. |
| `src/dragonlight_router/dispatch/cascade.py` | Modify | Add `feedback_store` to `DispatchContext`. Update `_resolve_cbr_weights()` for three-tier escalation. Import `classify_request_stakes`. |
| `src/dragonlight_router/selection/ibr.py` | Modify | Thread `feedback_store` through `run_ibr_stage()` -> `_execute_ibr()` -> `_build_ibr_result()`. Merge learned profiles with operator profiles before scoring. |
| `src/dragonlight_router/router.py` | Modify | Pass `feedback_store=self._feedback_store` to cascade dispatch. Add degraded-IBR warning in `_init_ibr()`. |

---

## 5. Test Plan

### 5.1 Unit Tests

| Test | File | Validates |
|------|------|----------|
| `test_classify_request_stakes_low` | `tests/test_scoring.py` | test_generation -> cost_optimized |
| `test_classify_request_stakes_mid` | `tests/test_scoring.py` | code_review -> balanced |
| `test_classify_request_stakes_high` | `tests/test_scoring.py` | architecture -> capability_optimized |
| `test_classify_request_stakes_context_escalation` | `tests/test_scoring.py` | test_generation + 40000 tokens -> capability_optimized |
| `test_classify_request_stakes_tool_use_escalation` | `tests/test_scoring.py` | documentation + requires_tool_use -> balanced |
| `test_classify_request_stakes_trust_tier_escalation` | `tests/test_scoring.py` | summarization + context_trust_tier="trusted" -> capability_optimized |
| `test_intent_weights_mid_stakes` | `tests/test_scoring.py` | code_review -> _MID_STAKES_WEIGHTS |
| `test_intent_weights_new_low_stakes` | `tests/test_scoring.py` | documentation -> _LOW_STAKES_WEIGHTS |
| `test_intent_weights_new_high_stakes` | `tests/test_scoring.py` | security_review -> _HIGH_STAKES_WEIGHTS |
| `test_mid_stakes_weights_sum` | `tests/test_scoring.py` | _MID_STAKES_WEIGHTS sums to 1.0 |
| `test_stakes_to_weights_complete` | `tests/test_scoring.py` | All three stakes map to valid weights |
| `test_mbr_tier_floor_new_categories` | `tests/test_mbr.py` | New categories produce correct tier floors |
| `test_feedback_merge_in_ibr` | `tests/test_ibr.py` | Learned profiles merge with operator profiles in scoring |
| `test_ibr_feedback_store_none` | `tests/test_ibr.py` | None feedback_store degrades gracefully |

### 5.2 Integration Tests

| Test | Validates |
|------|----------|
| Dispatch with `intent_category="test_generation"` uses cost-optimized weights | End-to-end weight selection for low-stakes |
| Dispatch with `intent_category="architecture"` uses capability-optimized weights | End-to-end weight selection for high-stakes |
| Dispatch with `intent_category="code_review"` uses balanced weights | End-to-end weight selection for mid-stakes |
| Dispatch with `intent_category="test_generation"` and `context_tokens=50000` escalates to capability-optimized | Context escalation override |
| Cost governor override supersedes per-request weights | Governor takes precedence over all profiles |

---

## 6. Observability

All weight profile selections are logged via structlog with:

```python
logger.debug(
    "cbr_weights_resolved",
    intent_category=order.intent_category,
    stakes_classification=stakes,
    weights_source="intent_category" | "context_escalation" | "ibr_active" | "default",
    cost_weight=weights.cost,
    spectrograph_match_weight=weights.spectrograph_match,
    governor_active=False,
)
```

The `stakes_classification` field in logs enables monitoring of the distribution of request stakes across the fleet, which informs budget planning and model procurement decisions.

---

## 7. Migration / Backward Compatibility

All changes are **additive and backward-compatible**:

- Unknown `intent_category` values continue to receive default weights (unchanged)
- The `classify_request_stakes()` function only escalates, never reduces (monotonic)
- The `_MID_STAKES_WEIGHTS` profile is new; no existing behavior changes unless a category is explicitly added to `_MID_STAKES_INTENTS`
- The feedback merge in IBR adds information; if no feedback exists, behavior is identical to current
- The cost governor override is unchanged and still takes absolute priority
- IBR disable (`intent_classification.enabled: false`) still produces identical v0.3.0 behavior
