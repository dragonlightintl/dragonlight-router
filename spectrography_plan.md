# Model Spectrography Execution Plan

**Generated:** 2026-06-20
**Target:** 84 models across 4 providers, 81 probes each = 6,804 evaluations
**Estimated total time:** 4-6 hours (parallelized across concurrent agents)

---

## Provider Health Assessment

| Provider | Status | RPM | RPD | Models | Judge-viable |
|----------|--------|-----|-----|--------|-------------|
| nvidia_nim | HEALTHY | 40 | unlimited | 52 | YES (primary) |
| groq | HEALTHY | 30 | 1,000 | 8 | YES (secondary) |
| mistral | HEALTHY | 2 | 66,000 | 13 | NO (too slow) |
| gemini | RATE-LIMITED | 10 | 250 | 11 | NO (deferred) |
| openrouter | BLOCKED | 20 | 1,000 | 0 | NO (privacy config) |
| cerebras | ADAPTER-ISSUE | 30 | 1,700 | 0 | NO (response format) |

## Judge Model Strategy

**Primary judge:** `nvidia_nim/qwen/qwen3.5-397b-a17b`
- 40 RPM, no daily cap — can sustain continuous judging
- Self-evaluation flagged automatically on 81 probes (acceptable)

**Fallback judge:** `groq/llama-3.3-70b-versatile`
- 30 RPM, 1000 RPD — viable for smaller batches
- Use when nvidia_nim is the test subject and self-eval must be avoided

## Wave Architecture

### Wave 1 — NVIDIA NIM (4 parallel agents, ~2.5 hours)

The 52 nvidia_nim models split into 4 batches of 13. Each agent targets one batch.
Rate limit: 40 RPM shared across all agents. With 4 agents pacing at 2.5s each, total ~24 RPM — within budget.

**Agent 1A — NIM Frontier (13 models)**
```
nvidia_nim/deepseek-ai/deepseek-v4-pro
nvidia_nim/deepseek-ai/deepseek-v4-flash
nvidia_nim/moonshotai/kimi-k2.6
nvidia_nim/qwen/qwen3.5-397b-a17b
nvidia_nim/qwen/qwen3.5-122b-a10b
nvidia_nim/qwen/qwen3-next-80b-a3b-instruct
nvidia_nim/openai/gpt-oss-120b
nvidia_nim/openai/gpt-oss-20b
nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
nvidia_nim/nvidia/nemotron-3-super-120b-a12b
nvidia_nim/nvidia/nemotron-3-nano-30b-a3b
nvidia_nim/nvidia/nemotron-4-340b-instruct
nvidia_nim/z-ai/glm-5.1
```
Judge: `groq/llama-3.3-70b-versatile` (avoids self-eval on qwen3.5)

**Agent 1B — NIM Llama/Nemotron (13 models)**
```
nvidia_nim/meta/llama-3.3-70b-instruct
nvidia_nim/meta/llama-3.1-70b-instruct
nvidia_nim/meta/llama-4-maverick-17b-128e-instruct
nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5
nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1
nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1
nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct
nvidia_nim/nvidia/llama-3.1-nemotron-51b-instruct
nvidia_nim/minimaxai/minimax-m3
nvidia_nim/minimaxai/minimax-m2.7
nvidia_nim/nvidia/nemotron-nano-3-30b-a3b
nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2
nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`

**Agent 1C — NIM Mistral/Google (13 models)**
```
nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512
nvidia_nim/mistralai/mistral-large-2-instruct
nvidia_nim/mistralai/mistral-medium-3.5-128b
nvidia_nim/mistralai/mistral-small-4-119b-2603
nvidia_nim/mistralai/mistral-nemotron
nvidia_nim/mistralai/codestral-22b-instruct-v0.1
nvidia_nim/mistralai/ministral-14b-instruct-2512
nvidia_nim/nv-mistralai/mistral-nemo-12b-instruct
nvidia_nim/nvidia/mistral-nemo-minitron-8b-8k-instruct
nvidia_nim/google/gemma-4-31b-it
nvidia_nim/google/gemma-3-12b-it
nvidia_nim/google/gemma-3-4b-it
nvidia_nim/google/gemma-3n-e4b-it
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`

**Agent 1D — NIM Others (13 models)**
```
nvidia_nim/google/gemma-2-2b-it
nvidia_nim/microsoft/phi-3.5-moe-instruct
nvidia_nim/microsoft/phi-4-mini-instruct
nvidia_nim/bytedance/seed-oss-36b-instruct
nvidia_nim/stepfun-ai/step-3.7-flash
nvidia_nim/stepfun-ai/step-3.5-flash
nvidia_nim/ai21labs/jamba-1.5-large-instruct
nvidia_nim/writer/palmyra-creative-122b
nvidia_nim/meta/llama-3.1-8b-instruct
nvidia_nim/meta/llama-3.2-3b-instruct
nvidia_nim/ibm/granite-3.0-8b-instruct
nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct
nvidia_nim/sarvamai/sarvam-m
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`

### Wave 2 — Groq + Mistral (2 parallel agents, ~2 hours)

Runs after Wave 1 completes (or concurrently if rate limits allow).

**Agent 2A — Groq (8 models)**
```
groq/llama-3.3-70b-versatile
groq/llama-3.1-8b-instant
groq/meta-llama/llama-4-scout-17b-16e-instruct
groq/qwen/qwen3-32b
groq/qwen/qwen3.6-27b
groq/openai/gpt-oss-120b
groq/openai/gpt-oss-20b
groq/allam-2-7b
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`
Note: 30 RPM, 1000 RPD. At 81 probes × 8 models = 648 calls + 648 judge calls.
Groq-side: 648 model calls at 30 RPM = ~22 min. Daily limit: within 1000 RPD.

**Agent 2B — Mistral (13 models)**
```
mistral/mistral-large-latest
mistral/mistral-medium-latest
mistral/mistral-small-latest
mistral/magistral-medium-latest
mistral/magistral-small-latest
mistral/codestral-latest
mistral/devstral-latest
mistral/devstral-medium-latest
mistral/ministral-14b-latest
mistral/ministral-8b-latest
mistral/ministral-3b-latest
mistral/open-mistral-nemo
mistral/mistral-tiny-latest
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`
Note: 2 RPM is brutal. 81 probes × 13 models = 1,053 model calls at 2 RPM = ~8.8 HOURS.
Mitigation: provider_delay override mistral=31.0 to respect 2 RPM.
This agent will be the long pole. Consider splitting into 2 sub-agents or running overnight.

### Wave 3 — Gemini (deferred, 1 agent, ~1 hour)

Run after Gemini rate limits reset (tomorrow). RPD budget: 250/day.

**Agent 3A — Gemini (11 models)**
```
gemini/gemini-3.5-flash
gemini/gemini-3.1-pro-preview
gemini/gemini-3-pro-preview
gemini/gemini-3-flash-preview
gemini/gemini-2.5-pro
gemini/gemini-2.5-flash
gemini/gemini-2.5-flash-lite
gemini/gemini-2.0-flash
gemini/gemini-2.0-flash-lite
gemini/gemma-4-31b-it
gemini/gemma-4-26b-a4b-it
```
Judge: `nvidia_nim/qwen/qwen3.5-397b-a17b`
Note: 10 RPM, 250 RPD. 81 probes × 11 models = 891 calls.
At 250 RPD, needs 4 days at full rate. Mitigation: run 3-4 models per day.

## Agent Execution Template

Each agent runs this command pattern:

```bash
cd ~/dragonlight-ops/dragonlight-router
python3 -m dragonlight_router.spectrography.runner \
  --models MODEL1 MODEL2 MODEL3 ... \
  --judge-model JUDGE_MODEL \
  --write-profiles \
  --output-dir spectrography_results \
  --provider-delay PROVIDER=DELAY
```

## Post-Execution Pipeline

After each wave completes:

1. **Merge profiles:** Results auto-merge to `config/model_spectrograph_profiles.yaml` via `--write-profiles`

2. **Update role matrix:**
   ```python
   from dragonlight_router.roles.matrix_updater import update_matrix_from_spectrography
   update_matrix_from_spectrography(Path('config'))
   ```

3. **Verify IBR integration:**
   ```python
   from dragonlight_router.selection.spectrograph import SpectrographProfileLoader
   loader = SpectrographProfileLoader(Path('config/model_spectrograph_profiles.yaml'))
   print(f'Loaded {len(loader.profiles)} empirical profiles')
   ```

## Estimated Resource Usage

| Wave | Models | Probes | Model Calls | Judge Calls | Est. Time |
|------|--------|--------|-------------|-------------|-----------|
| 1 (NIM) | 52 | 81 | 4,212 | 4,212 | ~2.5 hrs |
| 2A (Groq) | 8 | 81 | 648 | 648 | ~30 min |
| 2B (Mistral) | 13 | 81 | 1,053 | 1,053 | ~9 hrs |
| 3 (Gemini) | 11 | 81 | 891 | 891 | 4 days |
| **Total** | **84** | **81** | **6,804** | **6,804** | — |

## Known Issues to Address Before Execution

1. **Mistral 2 RPM bottleneck:** Consider splitting Agent 2B into 2 sub-agents targeting different models, but pacing must respect the shared 2 RPM across both.

2. **Gemini daily cap:** Must run in daily tranches of ~60 model calls (250 RPD minus judge overhead). Recommend 2-3 models per daily session.

3. **OpenRouter/Cerebras excluded:** OpenRouter needs privacy settings reconfigured at openrouter.ai/settings/privacy. Cerebras adapter needs to handle `reasoning` field in response. Both can be added in a future wave.

4. **Self-evaluation contamination:** When judge = qwen3.5-397b-a17b and test subject = same model, 81 probes are self-eval. Agent 1A uses groq judge to avoid this for the highest-priority frontier models. All self-eval results are flagged in the output.

5. **Rate limit sharing:** All agents in a wave share the same API key. The interleaved scheduler in the runner helps, but concurrent agents multiply effective RPM. The 4-agent NIM wave targets ~24 RPM against a 40 RPM limit — 60% utilization with headroom.
