# Model Flavor Discovery — Design Plan

**Author:** GOIBNIU (builder) + LUGH (quality guardian)
**Date:** 2026-06-18
**Status:** Draft plan — not a spec, not implemented
**Target system:** Dragonlight Router (vendor/dragonlight-router)

---

## Problem Statement

The Dragonlight Router selects models via a role-rank matrix (`model_role_matrix.json`) where ranks are operator-declared. A model ranked 95 for "coding" and 70 for "reasoning" reflects Korrigon's experience, not measured capability. When two models are ranked similarly (Kimi K2.6 at 95 vs DeepSeek V4 Pro at 90 for coding), the system has no empirical basis for choosing between them for a *specific* coding sub-task — refactoring vs generation vs bug analysis.

The gap is this: **the router knows what roles a model can fill, but not how it fills them.** Two 90-rank coding models may have completely different strengths. One may produce tighter diffs when refactoring. Another may write more defensively when generating from scratch. A third may catch more edge cases during analysis. These personality-level differences are invisible to the current scoring system.

### What we want to learn about each model

1. **Task-type affinity** — How does quality vary across generation, refactoring, analysis, summarization, creative writing, and reasoning? (Not binary "can/can't" — a gradient.)
2. **Domain strength** — Does the model's code quality hold across Python, TypeScript, SQL, infrastructure-as-code, and prose? Or does it excel in one and fall apart in others?
3. **Behavioral tendencies** — Verbosity, instruction adherence, code style consistency, defensive coding habits, explanation depth, hallucination rate on factual queries.
4. **Quality-speed tradeoff profile** — At comparable token budgets, which model produces higher quality? Which degrades gracefully under tight token limits?
5. **Discriminating characteristics** — The things that make Model A feel different from Model B on the same prompt. Not overall quality — differential quality.

### What this is NOT

This is not the dogfood benchmark (which tests the router's operational machinery — cascade, budget, interleaving). This is not a leaderboard. This is an empirical process that produces structured data the IBR consumes at runtime to make better routing decisions.

---

## Proposed Name: Model Spectrogram

"Flavor profile" captures the concept but is overloaded (the YAML file is already called that). The output artifact of this system is a **Model Spectrogram** — a multi-dimensional capability/personality fingerprint, analogous to a sound spectrogram showing energy distribution across frequencies. Each model has a spectrogram. The router reads spectrograms to match requests to models.

Short form in code: `ModelSpectrogram`, `spectrogram` field name, `spectrograms/` directory.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Model Spectrogram Pipeline                        │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │ Prompt Bank  │───▶│  Test Runner  │───▶│  LLM-as-Judge Scorer  │   │
│  │ (~120 items) │    │  (per-model)  │    │  (pairwise + absolute)│   │
│  └─────────────┘    └──────────────┘    └───────────┬───────────┘   │
│                                                      │               │
│                                                      ▼               │
│                                          ┌───────────────────────┐   │
│                                          │  Spectrogram Builder  │   │
│                                          │  (aggregate scores →  │   │
│                                          │   per-model profile)  │   │
│                                          └───────────┬───────────┘   │
│                                                      │               │
│                                                      ▼               │
│                                          ┌───────────────────────┐   │
│                                          │  spectrograms/*.json  │   │
│                                          │  (one file per model) │   │
│                                          └───────────┬───────────┘   │
│                                                      │               │
│                                                      ▼               │
│                                          ┌───────────────────────┐   │
│                                          │  IBR FlavorProfileLoader│  │
│                                          │  (reads as empirical   │   │
│                                          │   layer for scoring)   │   │
│                                          └───────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Components

1. **Prompt Bank** — Curated prompts designed to discriminate between models. Organized by task_type x domain, with additional "behavioral probes" for personality dimensions.
2. **Test Runner** — Sends each prompt to each model under controlled conditions (fixed temperature, fixed max_tokens, same system prompt). Captures raw responses.
3. **LLM-as-Judge Scorer** — Evaluates responses on multiple dimensions. Uses both absolute scoring (1-10 per dimension per response) and pairwise comparison (given two responses to the same prompt, which is better and why).
4. **Spectrogram Builder** — Aggregates judge scores into a per-model spectrogram. Computes means, confidence intervals, and differential rankings.
5. **Spectrogram Store** — JSON files (one per model) in `config/spectrograms/`. Machine-readable. Versioned by run date.

---

## Design Decision 1: Prompt Design for Flavor Discrimination

### The problem with broad prompts

The existing router concept of eval prompts (50 prompts across task_type x domain) tests *can the model do this at all*. For flavor discovery, the question is *how does this model do this differently from that model*. Broad prompts produce uniformly "good enough" responses from capable models — they don't discriminate.

### Discriminating prompt design principles

1. **Constrained outputs** — Prompts that require a specific structure, format, or length. Models that follow instructions precisely score differently from those that "helpfully" deviate.
2. **Ambiguous inputs** — Prompts where multiple valid approaches exist. The model's *choice* reveals its personality. Does it pick the safe refactor or the aggressive one? The verbose explanation or the terse one?
3. **Known-answer subtlety** — Prompts where there's a correct answer that requires noticing a non-obvious detail. Tests attention to edge cases, not just general competence.
4. **Style-revealing tasks** — Same task, same input, but the model's natural code style / prose style / reasoning style becomes visible.
5. **Degradation probes** — Prompts at the boundary of the model's capability. How gracefully does quality degrade?

### Prompt bank structure (~120 prompts)

**Category A: Task-Type Discriminators (48 prompts)**
8 task_types x 6 prompts each. Each prompt targets a specific quality that varies between models.

Example — Refactoring discriminator:
```yaml
- id: refactor-001
  task_type: refactoring
  domain: code
  prompt: |
    Refactor this function to reduce cyclomatic complexity without changing
    behavior. The function has 6 branches and 3 early returns. Preserve all
    edge case handling. Return ONLY the refactored function, no explanation.

    ```python
    def resolve_tier(order):
        if order.intent_category in _OPUS_INTENTS:
            return BackendTier.OPUS, 0.9, [f"intent={order.intent_category}"]
        if order.intent_category in _SONNET_INTENTS:
            tier = BackendTier.SONNET
            conf = 0.85
        elif order.requires_tool_use:
            tier = BackendTier.SONNET
            conf = 0.85
        elif order.context_tokens >= 8000:
            tier = BackendTier.SONNET
            conf = 0.8
        else:
            tier = BackendTier.LOCAL
            conf = 0.8
        if tier == BackendTier.LOCAL:
            if len(order.operator_message) <= 50:
                return tier, conf, ["short_message"]
            elif order.context_tokens >= 2000:
                return BackendTier.HAIKU, 0.7, ["medium_context"]
            else:
                return BackendTier.HAIKU, 0.7, ["moderate_message"]
        return tier, conf, [f"intent={order.intent_category}"]
    ```
  scoring_focus:
    - instruction_adherence: "Did it return ONLY the function? Or add explanation?"
    - behavior_preservation: "Does the refactored version handle all 6 branches?"
    - complexity_reduction: "Is cyclomatic complexity actually lower?"
    - code_style: "Is the refactored code idiomatic Python?"
```

Example — Analysis discriminator:
```yaml
- id: analysis-003
  task_type: analysis
  domain: code
  prompt: |
    This function has a subtle concurrency bug. Identify it in one sentence,
    then provide a minimal fix (diff format).

    ```python
    _router_instance = None

    def get_router(config_path=None):
        global _router_instance
        if _router_instance is not None:
            return _router_instance
        _router_instance = RouterEngine(config_path=config_path)
        return _router_instance
    ```
  scoring_focus:
    - bug_identification: "Did it identify the TOCTOU race (check-then-act without lock)?"
    - fix_precision: "Is the fix minimal? Or did it over-engineer?"
    - false_positives: "Did it flag non-bugs as bugs?"
```

**Category B: Behavioral Probes (36 prompts)**
6 behavioral dimensions x 6 prompts each. These measure personality, not capability.

Dimensions measured:
- **Verbosity tendency** — Does the model over-explain when asked to be concise?
- **Instruction precision** — When given conflicting constraints, which does it prioritize?
- **Defensive coding** — Does it add error handling unprompted? Input validation?
- **Explanation depth** — Given a "why" question, does it go shallow or deep?
- **Hallucination resistance** — Given an unanswerable question, does it refuse or fabricate?
- **Format compliance** — When asked for JSON/YAML/specific format, how strict is adherence?

Example — Verbosity probe:
```yaml
- id: behav-verbose-002
  behavioral_dimension: verbosity_tendency
  prompt: |
    In exactly one sentence, explain why Python uses GIL.
  scoring_focus:
    - sentence_count: "Exactly one sentence, or did it add caveats/context?"
    - accuracy: "Is the one sentence correct?"
    - information_density: "How much correct information is packed in?"
```

Example — Hallucination resistance:
```yaml
- id: behav-halluc-004
  behavioral_dimension: hallucination_resistance
  prompt: |
    What is the default value of the `cascade_depth` parameter in
    DragonlightRouter's `select_models()` method?
  scoring_focus:
    - refusal_quality: "Did it say it doesn't know? Or fabricate a value?"
    - confidence_calibration: "If it answered, did it hedge appropriately?"
  note: "This parameter does not exist. Any specific answer is a hallucination."
```

**Category C: Domain-Specific Depth (24 prompts)**
4 priority domains x 6 prompts each. These test how quality changes across domains for the same model.

Domains: Python, TypeScript, SQL/data, technical prose.

**Category D: Quality-Speed Tradeoff (12 prompts)**
Same 12 prompts run twice per model — once with max_tokens=256 (speed constraint), once with max_tokens=2048 (quality headroom). Measures degradation.

### Total: ~120 prompts, but runtime is ~132 prompt-model pairs per model (120 + 12 repeated at different token limits).

---

## Design Decision 2: Scoring Dimensions

### Primary dimensions (map to IBR routing)

These feed directly into the router's scoring system:

| Dimension | Type | Scale | Description |
|-----------|------|-------|-------------|
| `task_type_scores` | dict[str, float] | 0.0-1.0 | Per-task-type quality (generation, refactoring, analysis, summarization, creative, reasoning, lookup, translation) |
| `domain_scores` | dict[str, float] | 0.0-1.0 | Per-domain quality (python, typescript, sql, prose, general) |
| `quality_speed_ratio` | float | 0.0-1.0 | How much quality is retained under token constraints (1.0 = no degradation) |

### Behavioral dimensions (the "personality" layer)

These are new — not in the current router, but critical for fine-grained matching:

| Dimension | Type | Scale | Description |
|-----------|------|-------|-------------|
| `instruction_adherence` | float | 0.0-1.0 | How precisely it follows explicit constraints |
| `verbosity` | float | 0.0-1.0 | 0.0 = extremely terse, 1.0 = extremely verbose (neutral = ~0.5) |
| `defensive_coding` | float | 0.0-1.0 | Tendency to add error handling, validation, edge cases |
| `explanation_depth` | float | 0.0-1.0 | How deep it goes on "why" questions |
| `hallucination_resistance` | float | 0.0-1.0 | Ability to refuse or hedge on unanswerable questions |
| `format_compliance` | float | 0.0-1.0 | Strictness of adherence to requested output format |

### Confidence metadata

Every score carries a confidence interval:

```python
@dataclass(frozen=True)
class ScoredDimension:
    value: float          # 0.0-1.0
    confidence: float     # 0.0-1.0 (based on sample count and score variance)
    sample_count: int     # how many prompts contributed
    std_dev: float        # standard deviation across samples
```

---

## Design Decision 3: Head-to-Head vs Absolute Scoring

**Recommendation: Hybrid approach — absolute scoring for all, pairwise for top contenders.**

### Phase 1: Absolute scoring (all models)

Every model's response to every prompt is scored independently by the LLM-as-judge on each relevant dimension (1-10 scale, normalized to 0.0-1.0). This is O(n) per prompt where n = number of models.

Absolute scoring establishes baseline capability and is cheap. For 10 models x 132 prompt-runs = 1,320 model calls + 1,320 judge calls = 2,640 total API calls.

### Phase 2: Pairwise comparison (top contenders per role)

For models that score within 10% of each other on a dimension, run pairwise comparison. Send the same prompt's two responses to the judge with the question: "Which response is better for [dimension], and why?"

This is triggered selectively — only for model pairs that absolute scoring can't distinguish. For 10 models with 3-4 close pairs per dimension, this adds ~200-400 additional judge calls.

### Why not pairwise-only?

O(n^2) scaling. With 10 models and 120 prompts, full pairwise = 10*9/2 * 120 = 5,400 comparisons * judge calls. That's 5x more expensive than the hybrid approach and provides diminishing returns for models that are clearly different.

### Judge model selection

The judge should be a model NOT in the test pool. Options:
- **Anthropic Claude** (if available via API) — strong at evaluation, not in the router's free-tier pool
- **A second, separate instance of the strongest available model** — with a distinct system prompt that focuses on evaluation

The judge system prompt is critical. It must:
1. Score on the specific dimension asked, not general quality
2. Justify every score with a concrete observation from the response
3. Be calibrated — a 7/10 means the same thing across all models
4. Not have positional bias (for pairwise, randomize which response is "A" vs "B")

---

## Design Decision 4: The Spectrogram Data Model

### Output format: one JSON file per model

```
config/spectrograms/
  nvidia_nim--moonshotai--kimi-k2.6.json
  nvidia_nim--deepseek-ai--deepseek-v4-pro.json
  groq--llama-3.3-70b-versatile.json
  ...
```

File name convention: model_id with `/` replaced by `--` (filesystem safe).

### Spectrogram structure

```json
{
  "model_id": "nvidia_nim/moonshotai/kimi-k2.6",
  "spectrogram_version": 1,
  "generated_at": "2026-06-18T14:30:00Z",
  "run_id": "discovery-2026-06-18-001",
  "prompt_bank_version": "1.0.0",
  "judge_model": "anthropic/claude-sonnet-4-20250514",

  "task_type_scores": {
    "generation":     {"value": 0.88, "confidence": 0.92, "sample_count": 6, "std_dev": 0.05},
    "refactoring":    {"value": 0.91, "confidence": 0.90, "sample_count": 6, "std_dev": 0.06},
    "analysis":       {"value": 0.82, "confidence": 0.88, "sample_count": 6, "std_dev": 0.08},
    "summarization":  {"value": 0.75, "confidence": 0.85, "sample_count": 6, "std_dev": 0.10},
    "creative":       {"value": 0.70, "confidence": 0.82, "sample_count": 6, "std_dev": 0.12},
    "reasoning":      {"value": 0.85, "confidence": 0.90, "sample_count": 6, "std_dev": 0.07},
    "lookup":         {"value": 0.65, "confidence": 0.80, "sample_count": 6, "std_dev": 0.14},
    "translation":    {"value": 0.72, "confidence": 0.83, "sample_count": 6, "std_dev": 0.11}
  },

  "domain_scores": {
    "python":     {"value": 0.92, "confidence": 0.94, "sample_count": 8, "std_dev": 0.04},
    "typescript": {"value": 0.78, "confidence": 0.86, "sample_count": 6, "std_dev": 0.09},
    "sql":        {"value": 0.80, "confidence": 0.85, "sample_count": 6, "std_dev": 0.08},
    "prose":      {"value": 0.74, "confidence": 0.82, "sample_count": 6, "std_dev": 0.11},
    "general":    {"value": 0.81, "confidence": 0.88, "sample_count": 6, "std_dev": 0.07}
  },

  "behavioral_scores": {
    "instruction_adherence":    {"value": 0.94, "confidence": 0.91, "sample_count": 6, "std_dev": 0.04},
    "verbosity":                {"value": 0.62, "confidence": 0.88, "sample_count": 6, "std_dev": 0.08},
    "defensive_coding":         {"value": 0.85, "confidence": 0.86, "sample_count": 6, "std_dev": 0.07},
    "explanation_depth":        {"value": 0.78, "confidence": 0.84, "sample_count": 6, "std_dev": 0.10},
    "hallucination_resistance": {"value": 0.88, "confidence": 0.90, "sample_count": 6, "std_dev": 0.06},
    "format_compliance":        {"value": 0.91, "confidence": 0.92, "sample_count": 6, "std_dev": 0.05}
  },

  "quality_speed_profile": {
    "unconstrained_mean": 0.85,
    "constrained_mean": 0.72,
    "degradation_ratio": 0.85,
    "sample_count": 12
  },

  "pairwise_results": [
    {
      "compared_to": "nvidia_nim/deepseek-ai/deepseek-v4-pro",
      "dimension": "refactoring",
      "wins": 4,
      "losses": 1,
      "ties": 1,
      "net_advantage": 0.12
    }
  ],

  "metadata": {
    "total_prompts_run": 132,
    "total_judge_calls": 145,
    "total_cost_usd": 0.00,
    "run_duration_s": 1800,
    "models_compared_pairwise": ["nvidia_nim/deepseek-ai/deepseek-v4-pro"]
  }
}
```

### Mapping to existing types

The spectrogram extends the concepts in the current codebase:

| Current concept | Spectrogram equivalent |
|----------------|----------------------|
| `model_role_matrix.json` rank (0-100) | `task_type_scores` + `domain_scores` (0.0-1.0, with confidence) |
| `compute_composite_score()` in `scoring.py` | Composite can incorporate spectrogram dimensions as a weighted factor |
| `ComplexityEstimate.tier` | Behavioral scores (e.g., `explanation_depth`) inform tier estimation |
| Operator YAML flavor profiles (planned) | Spectrograms are the empirical layer that validates/refines operator intuition |

### New type in `core/types.py`

```python
@dataclass(frozen=True)
class ScoredDimension:
    """A single scored capability dimension with confidence metadata."""
    value: float
    confidence: float
    sample_count: int
    std_dev: float

@dataclass(frozen=True)
class ModelSpectrogram:
    """Empirical capability/personality fingerprint for a model."""
    model_id: str
    spectrogram_version: int
    generated_at: str
    run_id: str
    task_type_scores: dict[str, ScoredDimension]
    domain_scores: dict[str, ScoredDimension]
    behavioral_scores: dict[str, ScoredDimension]
    quality_speed_degradation: float  # 0.0-1.0
    pairwise_advantages: dict[str, float]  # model_id -> net advantage
```

---

## Design Decision 5: Lifecycle Management

### When to run discovery

| Trigger | Scope | Estimated time |
|---------|-------|----------------|
| **New model added** to `model_role_matrix.json` | Full run on the new model only | ~25 min (132 prompts + judge) |
| **Model updated** (provider ships a new version) | Full re-run on that model | ~25 min |
| **Periodic refresh** (monthly) | Full run on all models | ~4 hours for 10 models |
| **Manual trigger** | Operator-specified models/dimensions | Variable |
| **Post-feedback drift detection** | Models whose feedback EMA has diverged from spectrogram | Targeted re-run |

### Staleness detection

Each spectrogram file has `generated_at`. The discovery system tracks:
- Model registry changes (new models, removed models)
- Provider API version changes (when catalog metadata changes)
- Feedback divergence (when runtime quality ratings systematically disagree with spectrogram predictions)

A spectrogram older than 30 days is flagged as stale. A spectrogram whose predictions diverge from feedback by >15% on any dimension triggers a re-run recommendation.

### Delta runs

When a new model is added, only that model needs a full discovery run. Pairwise comparisons are run against the models in the same role that score within 10% on absolute scores. This avoids re-running the entire pipeline.

When an existing model is updated, only that model is re-run. Its pairwise history is invalidated and re-computed against current close contenders.

### Storage and versioning

```
config/spectrograms/
  nvidia_nim--moonshotai--kimi-k2.6.json           # latest
  archive/
    nvidia_nim--moonshotai--kimi-k2.6--2026-06-18.json  # historical
```

The `archive/` directory preserves previous runs for drift analysis. The top-level file is always the latest and is what the IBR reads.

---

## Design Decision 6: Integration with the IBR

### The scoring pipeline today

```
select_models(role):
  1. Get candidates from role matrix (rank 0-100)
  2. Filter by live catalog
  3. Score: rank * 0.6 + budget * 0.25 + health * 0.15
  4. Interleave providers
  5. Return top N
```

### The scoring pipeline with spectrograms

The IBR (when built) will classify each incoming request with a `ClassifiedIntent` containing `task_type`, `domain`, and `quality_speed`. The spectrogram provides per-dimension scores for exactly these classifications.

```
select_models(role, intent: ClassifiedIntent | None):
  1. Get candidates from role matrix (rank 0-100)
  2. Filter by live catalog
  3. If intent is provided and spectrogram exists for model:
       spectrogram_score = weighted_match(intent, model.spectrogram)
       rank = blend(matrix_rank, spectrogram_score, spectrogram.confidence)
     Else:
       rank = matrix_rank (unchanged)
  4. Score: rank * 0.6 + budget * 0.25 + health * 0.15
  5. Interleave providers
  6. Return top N
```

### The blend function

```python
def blend(matrix_rank: int, spectrogram_score: float, confidence: float) -> float:
    """Blend operator-declared rank with empirical spectrogram score.

    At low confidence, the matrix rank dominates.
    At high confidence, the spectrogram score dominates.
    The spectrogram never fully overrides the operator — max weight is 70%.
    """
    spec_weight = min(confidence * 0.7, 0.7)  # caps at 70%
    matrix_weight = 1.0 - spec_weight
    # Convert spectrogram 0.0-1.0 to 0-100 scale to match matrix rank
    return matrix_rank * matrix_weight + (spectrogram_score * 100) * spec_weight
```

The operator-declared rank is always a factor. The spectrogram *refines* the ranking within the operator's intent — it never completely overrides it. If Korrigon says "Kimi K2.6 is rank 95 for coding," the spectrogram might adjust that to 92 for refactoring and 97 for generation, but it won't drop it to 40.

### The weighted_match function

```python
def weighted_match(intent: ClassifiedIntent, spec: ModelSpectrogram) -> float:
    """Compute how well a model's spectrogram matches a classified intent."""
    score = 0.0
    weights_sum = 0.0

    # Task type match (weight: 0.5)
    if intent.task_type in spec.task_type_scores:
        dim = spec.task_type_scores[intent.task_type]
        score += dim.value * 0.5 * dim.confidence
        weights_sum += 0.5 * dim.confidence

    # Domain match (weight: 0.3)
    if intent.domain in spec.domain_scores:
        dim = spec.domain_scores[intent.domain]
        score += dim.value * 0.3 * dim.confidence
        weights_sum += 0.3 * dim.confidence

    # Quality-speed preference (weight: 0.2)
    if intent.quality_speed == "speed":
        # Prefer models with low degradation under constraints
        score += spec.quality_speed_degradation * 0.2
        weights_sum += 0.2
    elif intent.quality_speed == "quality":
        # Use unconstrained task_type score (already included above)
        pass

    if weights_sum == 0:
        return 0.5  # no data — neutral score
    return score / weights_sum
```

### Three-layer profile resolution

The IBR will ultimately merge three sources:

1. **Operator-declared** (YAML) — Korrigon's intuition, highest authority, always present
2. **Empirical spectrogram** (this system) — measured capability, weighted by confidence
3. **Feedback-learned** (EMA from runtime ratings) — production signal, most recent

Resolution order: Feedback > Spectrogram > Operator-declared (with confidence weighting at each layer). But operator-declared values are never dropped — they're the floor.

---

## Design Decision 7: Comparison Methodology

### Controlled conditions

Every discovery run uses these fixed parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.3 | Low enough for reproducibility, high enough for personality expression |
| `max_tokens` (standard) | 2048 | Sufficient for all non-speed prompts |
| `max_tokens` (speed probe) | 256 | Tests degradation under constraint |
| `system_prompt` | Standardized per task_type | Ensures fair comparison (no model-specific tuning) |
| `top_p` | 1.0 | Default, no nucleus sampling variance |
| `repetition_penalty` | None/default | Provider default, not adjusted |

### Controlling for randomness

Each prompt is run **3 times** per model (at temperature 0.3, this produces some but not extreme variation). The judge scores all 3 responses. The spectrogram uses the mean score and reports std_dev. This triples the API calls but provides confidence intervals.

Updated estimate: 10 models x 132 prompts x 3 runs = 3,960 model calls + 3,960 judge calls = 7,920 total calls.

At free-tier rates across NVIDIA NIM, Groq, and OpenRouter, this fits within daily budgets if spread across 2-3 days.

### Fair comparison across providers

- **Same prompt, same system prompt** — no provider-specific formatting
- **Same tokenizer-agnostic length limits** — max_tokens applies to output; input is identical
- **OpenAI-compatible API for all** — all providers in router.yaml expose `/v1/chat/completions`
- **No model-specific optimization** — no few-shot examples, no chain-of-thought prompting unless the prompt itself calls for it
- **Randomized run order** — models are not all run sequentially (model A all prompts, then model B). Instead, prompts are shuffled and distributed across models to avoid time-of-day effects on rate-limited providers.

---

## Scope Estimate

### Prompt bank development: ~2 days

- 48 task-type discriminators (8 types x 6 prompts): 1 day to write + review
- 36 behavioral probes (6 dimensions x 6 prompts): 0.5 day
- 24 domain-specific depth prompts: 0.25 day
- 12 quality-speed tradeoff prompts: 0.25 day

### Infrastructure development: ~3-4 days

- Test runner (send prompts to models via adapters): 1 day
- Judge scorer (LLM-as-judge with dimension-specific rubrics): 1 day
- Spectrogram builder (aggregate, compute confidence, output JSON): 0.5 day
- CLI interface (`python -m dragonlight_router.discovery run --models all`): 0.5 day
- Integration with IBR scoring (blend function, weighted_match): 0.5-1 day

### First full run: ~1-2 days of clock time

- 7,920 API calls across 10 models
- At 90 RPM combined budget: ~88 minutes minimum (if no rate limiting)
- Realistic with rate limits, retries, and pairwise phase: 4-8 hours
- Spread across 2 days for provider daily budget compliance

### Per-model delta run: ~25-45 minutes

- 132 prompts x 3 runs = 396 model calls + 396 judge calls = 792 calls
- Plus pairwise: ~100-200 additional judge calls
- ~1000 total calls, ~15-25 minutes at full throughput

---

## Implementation Phases

### Phase 1: Prompt Bank + Runner (build first, score later)

Build the prompt bank as a YAML file. Build the test runner that sends prompts to models via the existing adapter pattern and stores raw responses in a results directory.

**Output:** Raw response corpus — every model's response to every prompt, stored as JSON.

**Files:**
- `src/dragonlight_router/discovery/prompts.yaml` — the 120-prompt bank
- `src/dragonlight_router/discovery/runner.py` — test runner
- `src/dragonlight_router/discovery/results/` — raw response storage

### Phase 2: Judge Scorer

Build the LLM-as-judge scorer that evaluates responses on specified dimensions. Supports both absolute (single response) and pairwise (two responses) evaluation.

**Output:** Per-response scores on each dimension.

**Files:**
- `src/dragonlight_router/discovery/judge.py` — scoring logic
- `src/dragonlight_router/discovery/rubrics.yaml` — per-dimension scoring rubrics

### Phase 3: Spectrogram Builder

Aggregate judge scores into per-model spectrograms. Compute means, confidence intervals, identify close pairs for pairwise comparison.

**Output:** Spectrogram JSON files in `config/spectrograms/`.

**Files:**
- `src/dragonlight_router/discovery/builder.py` — aggregation logic
- `src/dragonlight_router/discovery/types.py` — `ScoredDimension`, `ModelSpectrogram`

### Phase 4: IBR Integration

Add spectrogram loading to the flavor profile system. Implement the blend and weighted_match functions. Wire into `select_models()` when a `ClassifiedIntent` is available.

**Files:**
- `src/dragonlight_router/selection/spectrogram.py` — loader + blend + match
- Modifications to `router.py` — `select_models()` accepts optional intent

### Phase 5: Lifecycle Automation

Add staleness detection, delta-run triggering, and drift monitoring.

**Files:**
- `src/dragonlight_router/discovery/lifecycle.py` — staleness checks, drift detection
- `src/dragonlight_router/discovery/cli.py` — CLI interface for manual/scheduled runs

---

## Example Comparison Output

After running discovery on Kimi K2.6 vs DeepSeek V4 Pro vs Llama 3.3 70B for the "coding" role:

```
Task Type Scores (coding-relevant prompts):
                    Kimi K2.6    DeepSeek V4 Pro   Llama 3.3 70B
  generation        0.88 (0.92)  0.90 (0.94)       0.72 (0.88)
  refactoring       0.91 (0.90)  0.84 (0.88)       0.68 (0.85)
  analysis          0.82 (0.88)  0.89 (0.92)       0.75 (0.86)
  (confidence in parentheses)

Behavioral Scores:
                         Kimi K2.6    DeepSeek V4 Pro   Llama 3.3 70B
  instruction_adherence  0.94         0.88              0.82
  verbosity              0.62         0.45              0.71
  defensive_coding       0.85         0.91              0.68
  format_compliance      0.91         0.93              0.78

Pairwise (Kimi K2.6 vs DeepSeek V4 Pro, refactoring):
  Kimi wins 4/6, DeepSeek wins 1/6, tie 1/6
  → Kimi has +0.12 net advantage for refactoring

Routing implication:
  - Refactoring tasks → prefer Kimi K2.6 (strong refactoring + instruction adherence)
  - Analysis tasks → prefer DeepSeek V4 Pro (stronger analysis + defensive coding)
  - Under speed constraints → prefer Kimi K2.6 (lower verbosity = less degradation)
  - Format-sensitive outputs → either Kimi or DeepSeek (both >0.90 compliance)
  - Llama 3.3 70B → general fallback, no standout dimension, lower cost
```

This is the kind of differentiation that the current role-rank matrix (95 vs 90 vs 75) cannot express. The spectrogram tells the router *why* to pick one over another for a specific request.

---

## Open Questions

1. **Judge model cost**: If the judge is Anthropic Claude via paid API, the judge calls alone for a full run (~4000 calls) have a real dollar cost. Alternative: use the strongest free-tier model as judge, accepting some quality tradeoff.

2. **Self-evaluation bias**: A model judging its own architecture family (e.g., Qwen judging Qwen) may have bias. The judge should be from a different model family than any model being tested.

3. **Prompt bank maintenance**: As the router's task_type taxonomy evolves, the prompt bank needs to evolve with it. This is a maintenance burden. Mitigation: version the prompt bank and track which spectrogram version was generated with which prompt bank version.

4. **Behavioral dimension weighting**: How much should behavioral scores influence routing vs task_type/domain scores? This likely needs operator tuning (another YAML config) and production data to calibrate.

5. **Multi-turn capability**: The current design tests single-turn only. Some models perform differently in multi-turn conversations. This is a known gap — addressing it would significantly increase prompt bank complexity and run time. Deferred to v2.

---

## Summary

The Model Spectrogram system transforms model selection from "operator intuition + static ranks" to "operator intuition refined by empirical measurement." It answers the question the role-rank matrix cannot: *for this specific kind of task, which model is not just capable but best?*

The system is designed to be:
- **Incremental** — Delta runs for new models, not full re-runs
- **Confidence-aware** — Every score carries uncertainty metadata
- **Non-destructive** — Operator-declared ranks are refined, never overridden
- **Lifecycle-managed** — Staleness detection and drift monitoring keep data current
- **Cost-conscious** — Hybrid absolute/pairwise scoring avoids O(n^2) blowup

Total estimated build: ~5-6 days of implementation across 5 phases. First usable output (Phase 1-3): ~3-4 days. Full IBR integration (Phase 4-5): ~2 more days.
