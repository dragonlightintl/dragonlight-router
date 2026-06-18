# Changelog

All notable changes to dragonlight-router are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.2.6] — 2026-06-17

### Added

- **Full cascade dispatch pipeline** — MBR (Model-Based Ranking) -> CBR (Cost-Based Ranking) -> LBR (Limit-Based Ranking) composition with fallback across the ranked candidate list
- **11 provider adapters** — Anthropic, Cerebras, Cohere, Google, Groq, Local (Ollama), Mistral, NVIDIA NIM, OpenAI, OpenRouter, Together (up from 1 in v0.1.0)
- **SSE streaming dispatch** — `dispatch_stream()` yields `StreamChunk` events (token, metadata, error) for real-time token streaming with fallback across backends
- **Context trust tier filtering** — HAZ-001 mitigation: `context_trust_tier` parameter on dispatch orders enforces minimum provider trust level (LOCAL > TRUSTED > SEMI_TRUSTED > UNTRUSTED), preventing sensitive context from reaching lower-trust providers
- **Cost governor** — Activates when daily or monthly spend exceeds configurable thresholds; reweights scoring to 70% cost / 10% latency / 10% priority / 5% queue / 5% health to aggressively prefer cheaper backends
- **Degraded backend deprioritization** — Backends in DEGRADED health status receive a 0.5x score penalty to route traffic toward healthy alternatives
- **Intent-based tier floor** — HAZ-013 mitigation: Maps intent categories (e.g., `complex_reasoning`, `code_review`, `debugging`) to minimum required backend tiers, preventing under-qualified model selection
- **Fallback policy control** — HAZ-004 mitigation: `fallback_policy` parameter (`allow` / `deny` / `same_tier`) restricts which backends are eligible for cascade fallback
- **State persistence** — Budget tracker state survives restarts via atomic file I/O (.tmp -> rename pattern)
- **Observability stack** — Prometheus-style metrics endpoint, correlation ID middleware (X-Request-ID propagation), structured logging via structlog, `/readiness` probe, OpenAPI spec generation
- **CORS middleware** — Configurable cross-origin resource sharing for browser-based clients
- **Admin auth** — Bearer token authentication for admin endpoints (metrics, health management)
- **Graceful shutdown** — Clean shutdown handler for the Starlette server
- **Docker + Makefile + CI** — Production-ready container image, `make test`, `make lint`, `make typecheck` targets
- **Property-based tests** — Hypothesis-driven invariant testing for scoring bounds, BudgetTracker monotonicity, MBR never-downgrade, LBR subset, and interleave permutation properties
- **`requirements.lock`** — Pinned dependency lockfile for reproducible production deploys

### Changed

- Provider adapter architecture refactored to use `GenerativeBackend` protocol with per-dispatch fresh adapter instantiation (HAZ-014 mitigation: prevents concurrent status mutation)
- Token estimation centralized via `_estimate_token_count()` with observability logging (HAZ-010 mitigation)
- Scoring weights extracted to `ScoringWeightsConfig` dataclass for cost governor override
- `DispatchOrder` extended with `context_trust_tier`, `fallback_policy`, `requires_long_context` fields
- `BackendConfig` extended with `capabilities`, `cost`, `rate_limits`, `priority` fields
- Test suite expanded from ~2,000 lines to 986 tests with 100% line coverage

### Fixed

- Unclosed SQLite connections in `SimpleCache` and `SemanticCache` — added `close()` method to both classes
- Coroutine-never-awaited warnings in fallback chain integration tests — mock failing backends now use async generators instead of plain async functions
- All 16 pytest warnings resolved (ResourceWarning: unclosed sqlite3.Connection, RuntimeWarning: coroutine never awaited)
- Strict warning mode (`pytest -W error`) now passes all 986 tests with zero warnings

### Security

- 14 hazard mitigations implemented (HAZ-001 through HAZ-014) covering context data egress, budget exhaustion, fallback control, model capability mismatch, concurrent adapter state, and token estimation accuracy
- 1 low-severity bandit finding: `random` module used for jitter in health check intervals — correct usage (not cryptographic)

### Quality

- 986 tests, 100% line coverage
- 0 mypy strict-mode errors
- 0 ruff lint errors
- All tests pass under `pytest -W error` strict warning mode

## [0.1.0] — 2025-05-18

### Added

- `RoleMatrix` — hot-reloadable JSON file mapping roles to ranked model IDs
- `BudgetTracker` — sliding-window RPM + daily RPD tracking per provider
- `HealthTracker` — per-model error counting and EWMA latency tracking
- `CircuitBreaker` — CLOSED→OPEN→HALF_OPEN state machine with configurable thresholds
- `CatalogCache` — file-backed TTL cache of live provider model lists
- `CatalogRefresher` — concurrent async fetch from provider `/v1/models` endpoints
- `RouterEngine` — orchestration layer with `select_models()` + `record_request()` interface
- `Server` — Starlette HTTP API: `/v1/select`, `/v1/record`, `/v1/health`, `/v1/catalog`
- `SimpleCache` — SHA-256 exact-match response cache backed by SQLite (WAL mode)
- `SemanticCache` — character n-gram Jaccard similarity cache for near-duplicate detection
- `ComplexityEstimator` — heuristic mapping intent + context size to tier (LOCAL/HAIKU/SONNET/OPUS)
- Full unit test suite (15 files, ~2,000 lines)
- `mypy --strict` typing throughout
- Provider support: NVIDIA NIM, Groq, OpenRouter, Cerebras, Gemini, Mistral, Anthropic, Ollama
