> **HISTORICAL -- SUPERSEDED BY v0.3.0 IMPLEMENTATION**
>
> This spec (v0.1.0) was written during initial design. The implemented system
> follows this spec's architecture and probe design, but the codebase has evolved
> beyond this document. For current behavior, see `docs/spectrography.md` (user
> guide) and the source in `src/dragonlight_router/spectrography/`. A full v0.3.0
> spec rewrite is tracked as SPEC-01 in the audit tracker.

# Model Spectrography v0.1.0 Live Spec

**Version:** 0.1.0
**Effective:** 2026-06-18
**Status:** Historical (superseded by v0.3.0 implementation)
**Depends on:** Model Pinning v0.1.0, Dragonlight Router v0.3.0, IBR v0.1.0

## 1. Problem Statement

The IBR routes requests using flavor profiles — per-model scores across task_type (8), domain (6), and quality_speed (3) dimensions. Today, profiles come from two sources:

1. **Operator-declared** (`model_flavor_profiles.yaml`) — hand-tuned estimates. Fast to create, but subjective and potentially inaccurate.
2. **Feedback-learned** (FeedbackStore EMA) — aggregated from production quality ratings. Accurate over time, but cold-start requires many observations before profiles stabilize.

Neither answers the fundamental question: **what are the actual, empirical differences between models of similar capability?**

Kimi K2.6 vs DeepSeek V4 Pro vs Gemini Flash — all strong coding models, but which one is better at *refactoring* vs *generation* vs *analysis*? Qwen 3.5 vs Llama 3.3 — how do their reasoning styles actually compare on legal vs technical domains?

The Model Spectrography system closes this gap. It runs controlled, head-to-head evaluations across all IBR dimensions, uses LLM-as-judge scoring, and produces empirical flavor fingerprints that can replace or validate operator-declared profiles. It also provides a lifecycle mechanism to keep profiles current as models are added, updated, or deprecated.

**Distinction from calibration audit:** The calibration audit (`benchmark/calibration_audit.py`) tests the *router's operational machinery* — budget enforcement, rate limiting, health monitoring, cost tracking. Model spectrography tests the *models themselves* — their relative strengths, weaknesses, and personality across intent dimensions. The calibration audit routes through the router's HTTP API. Model spectrography calls model adapters directly to isolate model performance from router performance.

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│           Model Spectrography System                  │
│                                                       │
│  spectrography/                                       │
│    probes.py      — discriminative probe prompts      │
│    runner.py      — orchestrator (eval + judge)       │
│    analyzer.py    — fingerprint computation + delta   │
│    lifecycle.py   — staleness, scheduling, triggers   │
│                                                       │
│  Output:                                              │
│    spectrography_results/<run_id>/report.json          │
│    spectrography_results/<run_id>/fingerprints.yaml    │
│    config/model_flavor_profiles.yaml  (optional write) │
└──────────────────────────────────────────────────────┘
         |                          |
         v                          v
  Model Adapters (direct)    LLM-as-Judge (direct)
  (isolate model behavior    (score via adapter,
   from router machinery)     not router dispatch)
```

### 2.1 Direct Adapter Calls (Not Router HTTP)

Unlike the calibration audit which routes through `POST /v1/dispatch`, model spectrography calls adapters directly via the `GenerativeBackend` protocol. This isolates model-specific behavior from router-specific behavior (rate limiting, budget, circuit breakers). The spectrography system manages its own pacing and retry logic.

### 2.2 Discriminative Probe Design

The existing eval prompts in `benchmark/prompts.py` test general capability within each dimension. Model spectrography needs something different: probes that **discriminate** between models of similar capability. Two models might both score 4/5 on a general coding prompt, but differ on:

- Style preference (verbose vs terse)
- Error handling approach (defensive vs optimistic)
- Reasoning chain transparency (show-work vs direct-answer)
- Domain-specific vocabulary precision
- Edge case awareness

Discriminative probes are designed to surface these differences, not just measure absolute quality.

## 3. Probe Design

### 3.1 Probe Structure

```python
@dataclass(frozen=True)
class SpectrographyProbe:
    id: str                    # e.g. "disc-refactoring-code-001"
    task_type: str             # IBR task_type dimension
    domain: str                # IBR domain dimension
    quality_speed: str         # IBR quality_speed dimension
    prompt: str                # The probe prompt
    judge_criteria: str        # What the judge scores
    discrimination_axis: str   # What model difference this probes
    difficulty: str            # "easy" | "medium" | "hard"
```

The `discrimination_axis` field documents what model personality trait the probe is designed to surface. This is metadata for humans reviewing results — the judge does not use it.

### 3.2 Probe Categories

Each category targets a specific dimension of model personality:

| Category | What it tests | Example |
|----------|--------------|---------|
| Style probes | Verbosity, structure, formatting | "Write a function. Do NOT include comments or docstrings." (tests compliance vs instinct to document) |
| Edge-case probes | Defensive thinking, corner-case awareness | "Find all bugs in this code." with 3 obvious + 2 subtle bugs. Scoring weights subtle-bug detection. |
| Reasoning-depth probes | Chain of thought, step decomposition | "Explain why X is O(n log n)" — scores reasoning chain quality, not just final answer |
| Domain-cross probes | Knowledge boundary identification | Technical question with legal implications — tests whether model surfaces the cross-domain concern |
| Instruction-following probes | Constraint adherence | Strict format requirements. Tests exact-compliance vs "spirit of the request" models |
| Speed-quality tradeoff probes | Response under time-pressure framing | "Quick answer: ..." vs "Think carefully: ..." — same question, different framing. Tests calibration |

### 3.3 Probe Count

Target: **80 probes** spanning all 8 task_types x 6 domains, with emphasis on high-value combinations (code/generation, code/analysis, code/refactoring, technical/reasoning). At least 3 probes per task_type, at least 2 per domain, at least 2 per discrimination_axis category.

### 3.4 Probe Bank Location

`src/dragonlight_router/spectrography/probes.py` — same pattern as `benchmark/prompts.py`.

## 4. Execution Flow

### 4.1 Model Target Selection

Same deduplication logic as the calibration audit: read the role matrix, collect all model IDs, deduplicate by base model identity (strip provider prefix), keep the highest-priority provider per model. The `--models` flag allows subsetting.

### 4.2 Evaluation Phase

For each model, for each probe:

1. Create a fresh adapter instance for the model.
2. Send the probe prompt via `adapter.generate()` with `max_tokens=512, temperature=0.7, stream=True`.
3. Collect the full response text.
4. Record: response text, latency_ms, tokens_in, tokens_out.
5. On adapter failure: record null response. Do not retry — adapter failures are themselves signal about model reliability.

### 4.3 Judge Phase

For each (model, probe, response) triple:

1. Build judge messages using the same `_JUDGE_USER_TEMPLATE` from `benchmark/judge.py`.
2. Send to judge adapter (direct call, not router dispatch).
3. Parse JSON scoring response (accuracy, completeness, clarity, relevance on 1-5 scale).
4. Normalize to 0.0-1.0 via `_normalize_scores`.
5. On judge failure: score 0.5 (neutral). Log the failure.

### 4.4 Fingerprint Computation

After all evaluations complete:

1. **Per-dimension aggregation.** For each model, group scores by task_type, domain, and quality_speed. Compute mean and standard deviation per dimension. Map to `FlavorScore` format: `score=mean, confidence=1.0-stddev, sample_count=probe_count`.

2. **Cross-model normalization.** Within each dimension, rank-normalize scores so that the best model gets 1.0 and worst gets 0.0 on each dimension. This ensures flavor profiles express *relative* strength, not absolute quality. Raw scores are preserved in the detailed report for absolute comparison.

3. **Fingerprint assembly.** Each model gets a `ModelFlavorProfile` with rank-normalized task_scores, domain_scores, and qs_scores. These are "empirical profiles" — they can replace or validate operator-declared profiles.

### 4.5 Calibration Delta

Compare each empirical fingerprint against the operator-declared profile from `model_flavor_profiles.yaml`:

- For each dimension, compute `delta = abs(empirical - declared)`.
- Flag any dimension where `delta > 0.15` (same threshold as calibration audit).
- Produce a calibration table: `{model, dimension, declared, empirical, delta, recommendation}`.
- Recommendation is one of: `"confirm"` (delta ≤ 0.05), `"review"` (0.05 < delta ≤ 0.15), `"update"` (delta > 0.15).

## 5. Pacing Strategy

### 5.1 Per-Provider Rate Limiting

Same pattern as calibration audit's `ProviderPacer`:

| Provider | Min delay between requests | Rationale |
|----------|---------------------------|-----------|
| gemini | 1.0s | 2 models, moderate RPM limits |
| groq | 1.5s | Aggressive rate limiting on free tier |
| nvidia_nim | 1.0s | 4 models sharing one key |
| openrouter | 2.0s | Free tier, strictest limits |

### 5.2 Provider-Interleaved Scheduling

Same round-robin approach: rotate across providers per probe round to avoid slamming any single provider.

### 5.3 Retry Policy

On adapter error: **no retry for eval calls** (failure is signal). For judge calls: retry once with 5s backoff, then score 0.5.

## 6. Output Format

### 6.1 JSON Report (`spectrography_results/<run_id>/report.json`)

```json
{
  "run_id": "disc-20260618-a1b2c3",
  "started_at": "2026-06-18T10:00:00Z",
  "completed_at": "2026-06-18T11:30:00Z",
  "judge_model": "gemini/gemini-2.5-pro",
  "models_evaluated": 10,
  "probes_per_model": 80,
  "total_eval_calls": 800,
  "total_judge_calls": 800,
  "fingerprints": {
    "gemini/gemini-2.5-flash": {
      "task_scores": {"generation": {"score": 0.82, "confidence": 0.91, "sample_count": 12}, ...},
      "domain_scores": {...},
      "qs_scores": {...}
    },
    ...
  },
  "calibration_deltas": {
    "gemini/gemini-2.5-flash": {
      "task/generation": {"declared": 0.80, "empirical": 0.82, "delta": 0.02, "recommendation": "confirm"},
      "task/refactoring": {"declared": 0.70, "empirical": 0.55, "delta": 0.15, "recommendation": "update"},
      ...
    },
    ...
  },
  "per_probe_results": [
    {"model": "...", "probe_id": "...", "latency_ms": 1200, "tokens_in": 50, "tokens_out": 320,
     "judge_scores": {"accuracy": 4, "completeness": 5, "clarity": 4, "relevance": 5},
     "normalized_score": 0.81, "error": null},
    ...
  ],
  "model_rankings": {
    "task/generation": ["gemini/gemini-2.5-pro", "nvidia_nim/moonshotai/kimi-k2.6", ...],
    "task/analysis": [...],
    ...
  }
}
```

### 6.2 Fingerprints YAML (`spectrography_results/<run_id>/fingerprints.yaml`)

Drop-in replacement for `config/model_flavor_profiles.yaml`. Same schema, empirical values. The operator can review and copy this file to replace the hand-tuned profiles.

```yaml
version: 1
source: "spectrography-run-disc-20260618-a1b2c3"
generated_at: "2026-06-18T11:30:00Z"

profiles:
  "gemini/gemini-2.5-flash":
    task_scores:
      generation: 0.82
      analysis: 0.75
      refactoring: 0.55
      # ...
    domain_scores:
      code: 0.80
      # ...
    qs_scores:
      speed: 0.90
      # ...
```

### 6.3 Markdown Summary (`spectrography_results/<run_id>/summary.md`)

Human-readable report:
- Run metadata (timestamp, duration, judge model, model count)
- **Per-model fingerprint table** (sorted by overall average score)
- **Strengths and weaknesses** per model (top 3 highest/lowest dimensions)
- **Head-to-head comparison table** for same-tier models (e.g., all coding models compared on code dimensions)
- **Calibration delta table** (declared vs empirical, flagging updates)
- **Discrimination findings** — which probes showed the most variance across models (most discriminative)

## 7. Lifecycle Management

### 7.1 Staleness Detection

Spectrography profiles include `generated_at` timestamps. The same decay logic from `benchmark/runner.py` applies (IBR-FLV-06):

- Profiles older than 30 days decay toward 0.5 at 0.01/day.
- The spectrography runner checks profile age at boot. If any model's profile is >30 days old, it logs a warning: `"stale_spectrography_profile"`.

### 7.2 Triggers for Re-Spectrography

The system should be re-run when:

1. **New model added** to the router's backend registry or role matrix.
2. **Model version updated** (e.g., model provider releases a new version behind the same model ID).
3. **Significant calibration drift** detected by the calibration audit (delta > 0.15 on multiple dimensions).
4. **Scheduled cadence** — monthly re-spectrography recommended. The `--models` flag allows targeting only new/changed models.

### 7.3 Incremental Spectrography

The `--models` flag allows running spectrography on a subset. New model fingerprints are merged into the existing `fingerprints.yaml`:

- New models are added.
- Existing models are updated only if the new run includes them.
- Models not in the current run are preserved unchanged.

### 7.4 Profile Integration Path

Spectrography fingerprints feed into the IBR through the existing profile merge hierarchy:

```
Operator-declared (YAML)  ←  Spectrography can update this
        ↓
Feedback-learned (EMA)    ←  Production quality signal
        ↓
Merged profile (runtime)  ←  FlavorProfileLoader.get_merged_profiles()
```

Spectrography results can either:
- **Replace** operator-declared profiles (copy `fingerprints.yaml` → `model_flavor_profiles.yaml`)
- **Validate** operator-declared profiles (review calibration deltas, manually adjust)
- **Seed** feedback learning (provide high-confidence starting points that EMA refines)

## 8. Judge Model Selection

Same as calibration audit:

- **Primary judge:** `gemini/gemini-2.5-pro` — broadest coverage, highest review rank.
- **Fallback judge:** `nvidia_nim/qwen/qwen3.5-397b-a17b` — top reasoning rank.
- **Self-evaluation flag:** When Gemini 2.5 Pro judges its own responses, mark those scores as `"self_evaluated": true` in per_probe_results.
- The `--judge-model` CLI flag overrides the default.

## 9. CLI Interface

```
python -m dragonlight_router.spectrography.runner \
  --judge-model gemini/gemini-2.5-pro \
  --output-dir spectrography_results/ \
  --models gemini/gemini-2.5-flash groq/llama-3.3-70b-versatile \
  --provider-delay groq=2.0 openrouter=3.0 \
  --write-profiles          # Write fingerprints.yaml to config/
  --resume                  # Resume from checkpoint
```

| Flag | Default | Description |
|------|---------|-------------|
| `--judge-model` | `gemini/gemini-2.5-pro` | Model to use for judging |
| `--output-dir` | `spectrography_results/` | Output directory |
| `--models` | all from role matrix | Subset of models to evaluate |
| `--provider-delay` | see section 5.1 | Per-provider delay overrides |
| `--write-profiles` | off | Write fingerprints.yaml into `config/` |
| `--resume` | off | Resume from checkpoint |
| `--dry-run` | off | List models and probes, no execution |

## 10. Error Handling and Resumption

### 10.1 Checkpoint File

Same pattern as calibration audit: `spectrography_results/<run_id>/checkpoint.jsonl`. One JSON line per completed (model, probe) evaluation. On `--resume`, skip completed pairs.

### 10.2 Failure Modes

| Failure | Handling |
|---------|----------|
| Adapter creation fails | Exclude model, log warning |
| Model returns empty response | Score 0.0, record as signal |
| Model adapter timeout | Record null, score 0.0 |
| Judge parse failure | Score 0.5 (neutral), log warning |
| Judge adapter failure | Retry once, then score 0.5 |
| Script crash | Resume from checkpoint |

### 10.3 Graceful Shutdown

On SIGINT/SIGTERM: flush checkpoint, write partial report from available data. Mark report as incomplete.

## 11. Acceptance Criteria

- AC-DISC-001: Spectrography probes MUST span all 8 IBR task_types.
- AC-DISC-002: Spectrography probes MUST span all 6 IBR domains.
- AC-DISC-003: At least 80 spectrography probes MUST exist.
- AC-DISC-004: Each probe MUST specify a `discrimination_axis`.
- AC-DISC-005: Model evaluation MUST use direct adapter calls, NOT router HTTP dispatch.
- AC-DISC-006: Judge scoring MUST use the same 4-criterion rubric (accuracy, completeness, clarity, relevance).
- AC-DISC-007: Fingerprint scores MUST be rank-normalized across models per dimension.
- AC-DISC-008: Output fingerprints.yaml MUST be schema-compatible with `model_flavor_profiles.yaml`.
- AC-DISC-009: Calibration deltas MUST be computed against operator-declared profiles.
- AC-DISC-010: Dimensions with delta > 0.15 MUST be flagged with recommendation `"update"`.
- AC-DISC-011: Self-evaluation (judge evaluating its own model) MUST be flagged.
- AC-DISC-012: Checkpoint/resume MUST work correctly for interrupted runs.
- AC-DISC-013: `--write-profiles` MUST produce a valid drop-in YAML file.
- AC-DISC-014: The system MUST NOT modify the existing calibration audit or benchmark runner.
- AC-DISC-015: Per-provider pacing MUST be enforced to avoid rate-limit exhaustion.
- AC-DISC-016: Graceful shutdown MUST flush checkpoint and write partial report.
- AC-DISC-017: The `--models` flag MUST allow incremental spectrography of new models only.
- AC-DISC-018: Spectrography profiles older than 30 days MUST decay toward 0.5 (reuse IBR-FLV-06).

## 12. File Layout

```
src/dragonlight_router/spectrography/
    __init__.py
    probes.py          — 80+ discriminative probe prompts
    runner.py          — main orchestrator (CLI entry point)
    analyzer.py        — fingerprint computation, normalization, delta
    lifecycle.py       — staleness check, merge, profile writing
```

All new files. No modifications to existing `benchmark/` module.
