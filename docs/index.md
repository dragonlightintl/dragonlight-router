# Dragonlight Router

**Multi-provider LLM routing engine — intelligent model selection and cascade dispatch across 11 providers.**

The Dragonlight Router picks the best available LLM for each request. You define roles (logical task types like `"code_review"` or `"summarize"`), the router consults a hot-reloadable role-to-model matrix, scores candidates on budget headroom and recent health, interleaves across providers to avoid thundering-herd concentration, and returns a ranked list of model IDs your application can call in order.

It exposes a dual interface: import `RouterEngine` directly in Python, or run it as an HTTP sidecar and call `/v1/select`.

## Why it exists

Every non-trivial LLM application accumulates the same ad-hoc pile: a giant if/else over provider names, manual rate-limit counters, a health-check spreadsheet, and a deployment that sends 100% of traffic to one provider until it breaks. The router replaces that pile with a single component that tracks budget windows, circuit-breaks unhealthy models, keeps its catalog fresh from provider APIs, and degrades gracefully across 11 providers.

Your application handles one ranked list instead of eleven provider SDKs.

## Feature highlights

- **11 providers** — Anthropic, Cerebras, Cohere, Gemini, Groq, Mistral, NVIDIA NIM, Ollama, OpenAI, OpenRouter, Together
- **Three-stage cascade** — MBR (capability filtering) + CBR (cost/health scoring) + LBR (rate-limit enforcement with provider interleaving)
- **Dual interface** — Python library or HTTP sidecar, same engine underneath
- **Hot-reloadable role matrix** — change role-to-model mappings without restart
- **Budget tracking** — sliding-window RPM, RPD, and TPM counters per provider
- **Circuit breaker** — CLOSED/OPEN/HALF_OPEN state machine per model, automatic recovery
- **Response caching** — SHA-256 exact-match cache and character n-gram Jaccard semantic cache
- **Complexity estimation** — heuristic tier mapping (LOCAL/HAIKU/SONNET/OPUS) from intent and context size
- **Streaming dispatch** — SSE streaming through the cascade with fallback on failure

## Quick links

- [Getting Started](getting-started.md) — install, configure one provider, first request in 60 seconds
- [Configuration](configuration.md) — full reference for `router.yaml`, role matrix, and environment variables
- [API Reference](api-reference.md) — HTTP endpoints, request/response schemas
- [Architecture](architecture.md) — subsystem breakdown, cascade pipeline, design decisions
- [Providers](providers.md) — supported providers, adding new ones, provider-specific notes
- [Security](security.md) — CORS, admin auth, SSRF validation, prompt sanitization, container hardening
