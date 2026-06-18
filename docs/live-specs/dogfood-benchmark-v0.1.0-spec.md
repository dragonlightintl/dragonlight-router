# Dogfood Benchmark v0.1.0 Live Spec

**Version:** 0.1.0
**Effective:** 2026-06-18
**Status:** Design (pre-implementation)
**Depends on:** Model Pinning v0.1.0, Dragonlight Router v0.3.0, IBR v0.1.0

## 1. Problem Statement

The existing `benchmark/runner.py` calls model adapters directly, bypassing the router entirely. This means benchmarks never exercise the operational machinery that real requests hit: rate limiting, budget enforcement, health monitoring, cost tracking, provider interleaving, and now model pinning.

The dogfood benchmark closes this gap. It is an HTTP client that calls `POST /v1/dispatch` with the `model` field set -- using the pinning feature to isolate each model while still flowing through every router subsystem. The judge calls also route through the router. The result is a benchmark that tests the models *and* the router simultaneously: 1000 dispatch calls that exercise the full operational stack end-to-end.

Secondary value: the benchmark produces empirical flavor profiles to validate (or replace) the operator-declared profiles in `model_flavor_profiles.yaml`.

## 2. Architecture

```
dogfood_benchmark.py (HTTP client)
        |
        | POST /v1/dispatch { model: "...", messages: [...] }
        v
  Dragonlight Router (running instance)
        |
        +-- Pinned dispatch path (bypasses cascade)
        |     +-- Registry lookup
        |     +-- Health/circuit-breaker check
        |     +-- Budget enforcement (check_and_reserve)
        |     +-- Rate limit check (has_capacity)
        |     +-- Fresh adapter creation
        |     +-- Provider API call
        |     +-- Cost tracking (record_request)
        |     +-- Health recording (record_success / record_error)
        |
        v
  EngineResponse { dispatch_mode: "pinned", ... }
```

The benchmark is a standalone async Python script. It does not import router internals. It communicates exclusively over HTTP. This validates the public API contract and ensures the benchmark can run against any router instance (local dev, staging, remote).

## 3. Execution Flow

### 3.1 Pre-flight

1. **Health check.** `GET /v1/health` -- verify the router is running.
2. **Model reachability.** For each of the 10 models, send a single `POST /v1/dispatch` with `model` set and a trivial prompt ("Respond with OK"). Record which models are reachable. Any model that returns 400 (not found), 503 (circuit open), or a connection error is marked unreachable and excluded from the benchmark run. Log the exclusion.
3. **Resume check.** If `--resume` is set, load the checkpoint file. Skip any (model, prompt) pairs already scored.

### 3.2 Model Evaluation (500 calls)

For each model, for each of the 50 eval prompts from `benchmark/prompts.py`:

1. Build the dispatch request:
   ```json
   {
     "model": "nvidia_nim/moonshotai/kimi-k2.6",
     "messages": [{"role": "user", "content": "<prompt.prompt>"}],
     "max_tokens": 512,
     "temperature": 0.7,
     "stream": false,
     "metadata": {
       "benchmark_run_id": "<run_id>",
       "prompt_id": "<prompt.id>"
     }
   }
   ```
2. `POST /v1/dispatch`. Record: response content, backend_used, latency, tokens_in, tokens_out, estimated_cost_usd, dispatch_mode, HTTP status.
3. On success: store the response for judging.
4. On 429 (rate limited / budget exhausted): back off and retry (see section 4).
5. On 502/503/5xx: log the failure, record a null response. Do not retry -- the model failed.

### 3.3 Judge Evaluation (500 calls)

For each (model, prompt, response) triple:

1. Build the judge prompt using the same template from `benchmark/judge.py` (`_JUDGE_USER_TEMPLATE`).
2. Dispatch to the judge model via `POST /v1/dispatch` with `model` pinned to the judge.
3. Parse the JSON scoring response (accuracy, completeness, clarity, relevance on 1-5 scale).
4. Normalize to 0.0-1.0 using the same `_normalize_scores` logic.
5. On judge failure: assign 0.5 (neutral). Log the failure.

### 3.4 Aggregation

After all 1000 calls complete:

1. Aggregate per-model scores by task_type, domain, and quality_speed dimensions into `ModelFlavorProfile` format.
2. Compare empirical profiles against operator-declared profiles from `model_flavor_profiles.yaml`. Flag any dimension where the delta exceeds 0.15.
3. Produce the calibration report (see section 5).

## 4. Request Pacing Strategy

Five providers, different rate limit profiles. The pacing strategy has three layers:

### 4.1 Provider-Aware Round-Robin

Process models in provider-interleaved order, not sequentially. Instead of running all 50 prompts for one model before moving to the next, rotate across providers:

```
Round 1: gemini/flash prompt-1, groq/llama prompt-1, nvidia/kimi prompt-1, openrouter/qwen prompt-1, ...
Round 2: gemini/flash prompt-2, groq/llama prompt-2, nvidia/kimi prompt-2, openrouter/qwen prompt-2, ...
```

This distributes load across providers naturally and avoids slamming any single provider with 50 consecutive requests.

### 4.2 Per-Provider Delay

Insert a minimum inter-request delay per provider:

| Provider | Min delay between requests | Rationale |
|----------|---------------------------|-----------|
| gemini | 1.0s | 2 models, moderate RPM limits |
| groq | 1.5s | Aggressive rate limiting on free tier |
| nvidia_nim | 1.0s | 4 models sharing one key |
| openrouter | 2.0s | Free tier, strictest limits |

These are starting values. The script accepts `--provider-delay` overrides.

### 4.3 429 Backoff

On HTTP 429 from the router (which means the router's own budget/rate-limit enforcement fired):

1. Read `Retry-After` header if present.
2. Otherwise exponential backoff: 5s, 10s, 20s, 40s, max 60s.
3. Max 3 retries per request. After 3 failures, record the prompt as failed and move on.

## 5. Output Format

Two outputs: a machine-readable JSON report and a human-readable markdown summary.

### 5.1 JSON Report (`benchmark_results/<run_id>/report.json`)

Top-level keys: `run_id`, `started_at`, `completed_at`, `router_version`, `judge_model`, `models_benchmarked`, `prompts_per_model`, `total_dispatch_calls`, `total_cost_usd`.

Nested sections:
- **`profiles`**: keyed by model_id. Each contains `task_scores`, `domain_scores`, `qs_scores` in `FlavorScore` format (`{score, confidence, sample_count}`).
- **`calibration_deltas`**: keyed by model_id, then `dimension/key`. Each entry: `{declared, measured, delta}`. Only dimensions with delta > 0.15 are flagged.
- **`per_prompt_results`**: array of `{model, prompt_id, latency_ms, tokens_in, tokens_out, cost_usd, judge_scores, normalized_score, http_status, error}`.
- **`router_stats`**: `{rate_limit_hits, budget_exhaustions, circuit_breaker_trips, total_errors}`.

### 5.2 Markdown Summary (`benchmark_results/<run_id>/summary.md`)

Human-readable report containing:
- Run metadata (timestamp, duration, judge model, total cost)
- Per-model score table (sorted by overall average)
- Top proficiencies and deficiencies per model (dimensions where score deviates most from 0.5)
- Calibration delta table (declared vs measured, flagging deltas > 0.15)
- Router operational summary (429s hit, errors, circuit breaker events)

## 6. Error Handling and Resumption

### 6.1 Checkpoint File

After each completed (model, prompt) evaluation, append the result to `benchmark_results/<run_id>/checkpoint.jsonl`. One JSON line per result. On `--resume`, the script reads the checkpoint, builds the set of completed pairs, and skips them.

### 6.2 Failure Modes

| Failure | Handling |
|---------|----------|
| Router unreachable | Pre-flight fails, script exits with clear message |
| Model 400 (not found) | Exclude model from run, log warning |
| Model 503 (circuit open) | Exclude model from run, log warning |
| Model 502 (adapter failure) | Record null response, score 0.0 for that prompt |
| 429 (rate limit) | Backoff + retry up to 3 times (section 4.3) |
| Judge parse failure | Score 0.5 (neutral), log warning |
| Script crash | Resume from checkpoint |
| Partial completion | Report generates from whatever data exists |

### 6.3 Graceful Shutdown

On SIGINT/SIGTERM: flush the current checkpoint, write a partial report with whatever results exist, then exit. The partial report is clearly marked as incomplete.

## 7. Judge Model Selection

**Primary judge: `gemini/gemini-2.5-pro`.**

Rationale: Highest-ranked model for review (rank 88) and reasoning (rank 85) in the role matrix. Strong across all domains. Operator-declared quality score of 0.95. The judge needs to evaluate code, technical, business, legal, and creative responses -- Gemini 2.5 Pro has the broadest declared coverage.

**Conflict of interest:** Gemini 2.5 Pro is also a benchmarked model. Its self-evaluation scores should be flagged in the report as potentially biased. Future iterations could use a cross-judge approach (judge A evaluates model B, judge B evaluates model A).

**Fallback judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`.** If Gemini 2.5 Pro is unreachable during pre-flight, fall back to Qwen 3.5 (top-ranked for reasoning at 95, review at 95).

The `--judge-model` CLI flag overrides the default.

## 8. Prerequisites

- Router instance running and reachable at `--router-url` (default `http://localhost:8000`)
- All provider API keys configured in the router's environment
- Sufficient budget headroom for ~1000 dispatch calls (the benchmark should not exhaust daily budgets)
- The 10 models registered in the router's backend registry
- Model pinning implemented and deployed (model-pinning-v0.1.0-spec)

## 9. CLI Interface

```
python -m dragonlight_router.benchmark.dogfood \
  --router-url http://localhost:8000 \
  --judge-model gemini/gemini-2.5-pro \
  --output-dir benchmark_results/ \
  --resume \
  --provider-delay groq=2.0 openrouter=3.0 \
  --models gemini/gemini-2.5-flash groq/llama-3.3-70b-versatile  # optional subset
```

| Flag | Default | Description |
|------|---------|-------------|
| `--router-url` | `http://localhost:8000` | Router base URL |
| `--judge-model` | `gemini/gemini-2.5-pro` | Model to use for judging |
| `--output-dir` | `benchmark_results/` | Output directory |
| `--resume` | off | Resume from checkpoint |
| `--provider-delay` | see section 4.2 | Per-provider delay overrides (key=value) |
| `--models` | all 10 | Subset of models to benchmark |
| `--dry-run` | off | Pre-flight only, no benchmark calls |

## 10. What This Exercises in the Router

Every dispatch call through the dogfood benchmark validates these router subsystems:

- **Model pinning** (the core feature under test) -- `dispatch_mode: "pinned"` in every response
- **Budget enforcement** -- `check_and_reserve` runs per call, 1000 calls stress the sliding window
- **Rate limiting** -- provider RPM limits enforced, 429s expected under load
- **Health monitoring** -- `record_success`/`record_error` after every call, circuit breaker state evolves
- **Cost tracking** -- `record_request` after every call, daily cost accumulates
- **Request correlation** -- every call gets an `X-Request-ID`, traceable in router logs
- **Context filtering** -- trust tier mapping applied per pinned backend
- **Adapter lifecycle** -- fresh adapter per dispatch (HAZ-014)
- **Response validation** -- `_validate_llm_response` on every response
