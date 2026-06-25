# Spectrography

Model spectrography is the router's empirical profiling system. It evaluates models directly through provider adapters to produce flavor fingerprints -- per-model scores across task type, domain, and quality/speed dimensions. These fingerprints feed into the Intent Based Router (IBR) to match models to request intent.

## What spectrography does

LLMs of similar capability still differ in important ways. Two models might both handle a coding prompt well, but one excels at refactoring while the other is stronger at analysis. One may be verbose by default; another may follow format constraints precisely.

Spectrography surfaces these differences. It sends ~80 discriminative probes to each model, scores the responses with an LLM judge, and produces a **flavor fingerprint** -- a normalized profile of relative strengths across all IBR dimensions.

The key distinction from general benchmarks: spectrography probes are designed to **discriminate** between models, not just measure absolute quality. A probe might ask for code with strict formatting constraints, testing whether a model follows exact instructions versus its instinct to add documentation.

## How profiles work

A spectrograph profile scores each model across three dimension types:

| Dimension | Values | Weight in IBR scoring |
|-----------|--------|----------------------|
| **Task type** | generation, analysis, refactoring, summarization, creative, reasoning, lookup, translation | 50% |
| **Domain** | code, technical, legal, business, creative_writing, general | 30% |
| **Quality/speed** | quality, balanced, speed | 20% |

Each dimension value gets a `SpectrographScore` with three fields:

- **score** (0.0--1.0): rank-normalized strength relative to other models. 1.0 = best model on this dimension, 0.0 = worst.
- **confidence** (0.0--1.0): derived from score standard deviation across probes. Higher confidence = more consistent performance.
- **sample_count**: number of probe evaluations contributing to this score.

Models without a profile default to neutral scores (0.5 / 0.0 confidence / 0 samples), so unknown models neither benefit nor suffer from spectrograph matching.

### Profile sources

Profiles come from three sources, merged at runtime in this priority order:

1. **Feedback-learned** (production quality ratings via EMA) -- highest priority
2. **Operator-declared** (`config/model_spectrograph_profiles.yaml`) -- hand-tuned estimates
3. **Neutral default** (0.5 across all dimensions) -- fallback for unknown models

Spectrography results can replace or validate operator-declared profiles. The `--write-profiles` flag writes empirical fingerprints directly to the config directory.

A floor enforcement rule (IBR-FLV-03) prevents feedback from dropping a score below 80% of the operator-declared value, protecting against feedback noise.

## Interpreting scores

### Reading a fingerprint

A model's fingerprint is a map of dimension values to scores. Example:

```yaml
"gemini/gemini-2.5-flash":
  task_scores:
    generation: 0.82
    analysis: 0.75
    refactoring: 0.55
    reasoning: 0.60
  domain_scores:
    code: 0.80
    technical: 0.70
    legal: 0.40
  qs_scores:
    speed: 0.90
    quality: 0.45
    balanced: 0.65
```

This tells you: relative to other models in the pool, Gemini Flash is strong at generation and code tasks, especially under speed-priority framing, but weaker at refactoring and legal domains.

### Score interpretation guide

| Score range | Meaning |
|-------------|---------|
| 0.80--1.00 | Top performer on this dimension |
| 0.60--0.79 | Above average |
| 0.40--0.59 | Average / neutral |
| 0.20--0.39 | Below average |
| 0.00--0.19 | Weakest on this dimension |

Scores are **relative**, not absolute. A score of 0.2 does not mean the model is bad at that task -- it means other models in the pool are better at it. If a new, stronger model joins the pool, all other models' scores on that dimension shift downward.

### Calibration deltas

After a spectrography run, calibration deltas compare empirical scores against operator-declared profiles:

| Delta | Recommendation | Meaning |
|-------|---------------|---------|
| 0.00--0.05 | `confirm` | Operator estimate matches empirical data |
| 0.06--0.15 | `review` | Moderate divergence -- worth investigating |
| > 0.15 | `update` | Significant divergence -- operator profile should be updated |

### Staleness and decay

Profiles older than 30 days decay toward 0.5 at a rate of 0.01 per day. This prevents stale profiles from making strong routing claims about models whose behavior may have changed. Re-run spectrography to refresh decaying profiles.

## Integration with IBR and cascade scoring

When IBR is enabled, the cascade scoring pipeline uses spectrograph profiles to bias model selection toward models that match the classified intent:

1. **Intent classification** -- the request is classified into task_type, domain, and quality_speed dimensions.
2. **Spectrograph match** -- for each candidate model, a weighted match score is computed: `0.50 * task_score + 0.30 * domain_score + 0.20 * qs_score`.
3. **Confidence gating** -- spectrograph matching is only applied when both intent classification confidence and average profile confidence exceed configurable thresholds (default: 0.6 and 0.3 respectively). Low-confidence classifications fall through to standard CBR/LBR scoring.
4. **CBR integration** -- the spectrograph match score is included as a weighted factor in the Cost-Based Ranking stage alongside budget and health scores.

The spectrograph match does not override other scoring factors. It is one signal among several (budget headroom, health, rate limits) that the cascade pipeline combines.

## Running spectrography

### CLI

```bash
python -m dragonlight_router.spectrography.runner [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--judge-model` | `gemini/gemini-2.5-pro` | Model used as the LLM judge |
| `--output-dir` | `spectrography_results/` | Output directory for reports |
| `--models` | all from role matrix | Subset of model IDs to evaluate |
| `--provider-delay K=V` | per-provider defaults | Override inter-request delays (e.g. `groq=2.0`) |
| `--write-profiles` | off | Write fingerprints to `config/model_spectrograph_profiles.yaml` |
| `--resume` | off | Resume from checkpoint (skip completed pairs) |
| `--resume-from RUN_ID` | none | Resume using checkpoint from a specific prior run |
| `--merge-checkpoints` | off | Merge all prior checkpoints before starting |
| `--dry-run` | off | Resolve targets and probes, then exit without evaluating |

### Examples

Run spectrography on all models in the role matrix:

```bash
python -m dragonlight_router.spectrography.runner
```

Profile specific models and write results to config:

```bash
python -m dragonlight_router.spectrography.runner \
  --models gemini/gemini-2.5-flash groq/llama-3.3-70b-versatile \
  --write-profiles
```

Dry run to see which models and probes would be evaluated:

```bash
python -m dragonlight_router.spectrography.runner --dry-run
```

Resume an interrupted run:

```bash
python -m dragonlight_router.spectrography.runner --resume
```

### Output files

Each run produces a directory under `spectrography_results/<run_id>/` containing:

| File | Contents |
|------|----------|
| `report.json` | Full structured report with per-probe results, fingerprints, calibration deltas, and rankings |
| `summary.md` | Human-readable markdown with rankings, proficiencies, and calibration tables |
| `fingerprints.yaml` | Drop-in replacement for `config/model_spectrograph_profiles.yaml` |
| `checkpoint.jsonl` | Completed (model, probe) pairs for resume support |

### Pacing

The runner enforces per-provider delays to avoid rate-limit exhaustion:

| Provider | Default delay |
|----------|--------------|
| gemini | 1.0s |
| groq | 1.5s |
| nvidia_nim | 1.0s |
| openrouter | 2.0s |

Override with `--provider-delay`: `--provider-delay groq=2.5 openrouter=3.0`.

The evaluation schedule interleaves across providers -- consecutive calls alternate between providers rather than hammering one provider for all probes before moving on.

### Graceful shutdown

On SIGINT or SIGTERM, the runner flushes the checkpoint file and writes a partial report from available data. Use `--resume` on the next run to continue from where it stopped.

## Probe design

The spectrography probe bank contains ~80 discriminative probes across six categories:

| Category | What it tests | Example |
|----------|--------------|---------|
| **Style** | Verbosity, formatting, code style | "Write binary search using exactly three variable names. No comments." |
| **Edge case** | Defensive thinking, corner-case awareness | "Find all bugs" with 3 obvious + 2 subtle bugs, weighting subtle detection |
| **Reasoning depth** | Chain-of-thought quality, step decomposition | "Explain WHY this is O(n^2), not just THAT it is" |
| **Domain cross** | Knowledge boundary identification | Technical question with legal implications |
| **Instruction following** | Exact constraint adherence | "Respond in valid JSON with exactly these keys" |
| **Speed/quality** | Response calibration under different framings | Same question asked as "quick answer" vs "think carefully" |

Each probe targets a specific `(task_type, domain, quality_speed)` triple and a discrimination axis. The judge scores responses on accuracy, completeness, clarity, and relevance (1-5 scale), normalized to 0.0-1.0.

Self-evaluation (when the judge model evaluates its own responses) is detected and flagged in results.
