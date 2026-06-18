# Intent Classification Layer — Design Brief

**Status:** Idea — awaiting discovery  
**Next step:** Exploratory conversation / design interview  
**Recorded:** 2026-06-18  

---

## Problem

The current MBR stage selects models based on **role** — a static mapping from logical task type ("code_review", "summarize") to a ranked list of model IDs. Every request with the same role gets the same candidate pool regardless of what the request actually asks for.

A request to `code_review` might be a quick syntax check or a deep architectural reasoning task. These have fundamentally different model requirements (speed vs depth), but the router treats them identically.

## Hypothesis

A lightweight LLM — Haiku-class hosted model or a small local model via Ollama — can classify incoming request intent in <100ms, adding a dynamic per-request signal that augments or refines model selection. This makes the router intent-aware, not just role-aware.

## Open Design Questions

### Pipeline placement
- **Before MBR:** Classification output becomes an input to MBR filtering (e.g., intent determines which tier of models to consider)
- **Parallel to MBR:** Classification runs concurrently while MBR filters by role, results merge at CBR scoring
- **As MBR input signal:** Intent classification augments the role → model mapping itself (role + intent → refined candidate list)
- **Relationship to ComplexityEstimator:** The existing complexity heuristic maps intent + context size to a tier (LOCAL/HAIKU/SONNET/OPUS). Intent classification could complement, replace, or layer on top of this.

### Model choice
- Local (Ollama small model) — zero cost, no network latency, but quality ceiling
- Hosted lightweight (Claude Haiku, Gemini Flash, Groq-served small model) — better quality, adds network hop
- Fine-tuned classifier — highest accuracy for the specific taxonomy, requires training data and maintenance
- Hybrid — local first, fall back to hosted if confidence is low

### Latency budget
- Current routing latency is dominated by cascade scoring, not network calls
- Classification must complete in <100ms to avoid becoming the bottleneck
- Timeout + fallback to static role behavior is essential

### Classification taxonomy
- What intent categories actually matter for routing decisions?
- Is this a fixed taxonomy or should it be operator-configurable?
- How granular? (binary "simple/complex" vs multi-class "lookup/analysis/generation/reasoning/creative")

### Fallback behavior
- Classification fails or times out → degrade gracefully to current static role behavior
- Low confidence classification → ignore classification signal, use role only
- This must never make routing worse than the current system

## Prior Art

- OpenRouter uses a "model suggestion" feature based on prompt analysis
- LiteLLM has a router with model group fallbacks but no intent classification
- Semantic routing (e.g., semantic-router library) uses embeddings for intent matching without LLM calls

## Next Steps

1. **Discovery conversation** — explore the design space, validate the hypothesis, identify the operator need
2. **Design sprint** — decide pipeline placement, model choice, taxonomy, and latency budget
3. **Prototype** — build a minimal classification step with the chosen model and measure latency impact
4. **Integration** — wire into the cascade pipeline with full fallback behavior
