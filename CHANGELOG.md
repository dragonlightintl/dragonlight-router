# Changelog

All notable changes to dragonlight-router are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-06-18

### Added

- **Intent-Based Router (IBR)** — LLM-powered intent classification inserted between MBR and CBR in the cascade, with flavor-match scoring, 100ms hard timeout, graceful degradation, and feedback-loop learning
- **Model Pinning** — direct-dispatch escape hatch that bypasses the cascade, allowing operators and benchmarks to target specific provider/model pairs
- **Model Spectrography** — empirical model profiling via discriminative probes and LLM-as-judge scoring; produces multi-dimensional capability spectrograms consumed by IBR for flavor-matched routing
- **Calibration Audit** — self-calibration benchmark that routes through the router's own API, judges responses via LLM-as-judge, and produces calibration delta reports comparing empirical scores against declared profiles
- MkDocs Material documentation site with getting-started guide, architecture, provider reference, and ADRs
- GitHub Actions docs deployment workflow (GitHub Pages)
- ARCHITECTURE.md with cascade pipeline diagram and package structure
- ADR-001 (Result type pattern), ADR-002 (Provider adapter pattern), ADR-003 (Cascade dispatch design)
- OpenAPI specification and deployment runbook
- CONTRIBUTING.md with development setup, code style, and PR conventions
- SECURITY.md with vulnerability disclosure policy
- GitHub issue templates and pull request template

### Changed

- Cascade pipeline extended to MBR → IBR → CBR → LBR with IBR as an opt-in scoring stage
- "Dogfood Benchmark" renamed to "Calibration Audit" — clearer, no insider jargon
- "Model Flavor Discovery" renamed to "Model Spectrography" — accurately describes empirical profiling
- README rewritten with professional badges, feature list, cascade diagram, and streamlined quickstart
- CI workflow split into separate lint, typecheck, security, and test jobs with Python 3.11/3.12/3.13 matrix
- pyproject.toml enriched with full classifiers, keywords, and project URLs
- Makefile expanded with `format`, `docs`, `docs-serve`, and `all` targets
- Function decomposition across 22 functions to meet 40-line hard limit
- 5 read-only module dicts frozen with `types.MappingProxyType`
- 9 bare `except Exception` handlers narrowed to specific exception types

### Fixed

- Timing attack in admin auth — switched to `hmac.compare_digest`
- Log redaction — removed raw error bodies from structured log output
- Missing `httpx.Timeout` on benchmark HTTP client
- Mock anti-patterns in test suite (async generators instead of plain async functions for failing backends)
- Expanded property-based tests and added acceptance and security test suites

### Security

- CORS hardening, admin auth rate limiting, SSRF URL validation (SEC-003, SEC-005, SEC-006)
- Container image and supply-chain hardening (SEC-004, SEC-008)
- Timing-safe admin token comparison (hmac.compare_digest)
- 35 formal deviation records for standards exceptions, documented in delta-spec.md

## [0.2.6] — 2026-06-17

### Added

- **Full cascade dispatch pipeline** — MBR (Model-Based Ranking) → CBR (Cost-Based Ranking) → LBR (Limit-Based Ranking) composition with fallback across the ranked candidate list
- **11 provider adapters** — Anthropic, Cerebras, Cohere, Google, Groq, Local (Ollama), Mistral, NVIDIA NIM, OpenAI, OpenRouter, Together
- **SSE streaming dispatch** — `dispatch_stream()` yields `StreamChunk` events for real-time token streaming with fallback
- **Context trust tier filtering** — `context_trust_tier` parameter enforces minimum provider trust level, preventing sensitive context from reaching lower-trust providers
- **Cost governor** — activates when spend exceeds thresholds; reweights scoring to aggressively prefer cheaper backends
- **Degraded backend deprioritization** — backends in DEGRADED health receive a 0.5x score penalty
- **Intent-based tier floor** — maps intent categories to minimum required backend tiers
- **Fallback policy control** — `fallback_policy` parameter (`allow`/`deny`/`same_tier`) restricts cascade fallback eligibility
- **Budget state persistence** — atomic file I/O (.tmp → rename) survives restarts
- **Observability** — Prometheus-style metrics, correlation ID middleware (X-Request-ID), structlog, `/readiness` probe
- **CORS middleware** — configurable cross-origin resource sharing
- **Admin auth** — bearer token authentication for admin endpoints
- **Graceful shutdown** — clean shutdown handler for the Starlette server
- **Docker + Makefile + CI** — production-ready container image and development workflow
- **Property-based tests** — Hypothesis-driven invariant testing for scoring, budget, MBR, LBR, and interleave
- **`requirements.lock`** — pinned dependency lockfile for reproducible production deploys
- **Response caching** — exact-match (SHA-256) and semantic near-duplicate (n-gram Jaccard) caching via SQLite
- **Real cost tracking** — USD/Mtok cost data per model with cost-aware scoring
- **Retry with backoff** — exponential backoff + jitter on OpenAI-compatible adapters

### Changed

- Provider adapter architecture refactored to `GenerativeBackend` protocol with per-dispatch fresh instantiation
- Token estimation centralized via `_estimate_token_count()` with observability logging
- Scoring weights extracted to `ScoringWeightsConfig` dataclass for cost governor override
- `DispatchOrder` extended with `context_trust_tier`, `fallback_policy`, `requires_long_context`
- `BackendConfig` extended with `capabilities`, `cost`, `rate_limits`, `priority`
- OpenAI-compatible adapter base class extracted, consolidating 7 adapters
- Function decomposition across adapters, cascade, and router modules
- Server input validation, error sanitization, and rate limiting middleware
- All ruff lint errors resolved (228 fixes across source and tests)

### Fixed

- Unclosed SQLite connections in `SimpleCache` and `SemanticCache` — added `close()` methods
- Coroutine-never-awaited warnings in fallback chain integration tests
- Groq 404 from URL doubling — health check now delegates to adapter
- Gemini double-pathing in base URL
- Google/Local adapters now raise exceptions instead of yielding error strings
- All pytest warnings resolved (strict mode passes with zero warnings)

### Security

- 14 hazard mitigations implemented (HAZ-001 through HAZ-014) covering context data egress, budget exhaustion, fallback control, model capability mismatch, concurrent adapter state, and token estimation accuracy

## [0.1.0] — 2025-05-18

### Added

- `RoleMatrix` — hot-reloadable JSON file mapping roles to ranked model IDs
- `BudgetTracker` — sliding-window RPM + daily RPD tracking per provider
- `HealthTracker` — per-model error counting and EWMA latency tracking
- `CircuitBreaker` — CLOSED → OPEN → HALF_OPEN state machine
- `CatalogCache` — file-backed TTL cache of live provider model lists
- `CatalogRefresher` — concurrent async fetch from provider `/v1/models` endpoints
- `RouterEngine` — orchestration layer with `select_models()` + `record_request()` interface
- `Server` — Starlette HTTP API: `/v1/select`, `/v1/record`, `/v1/health`, `/v1/catalog`
- `SimpleCache` — SHA-256 exact-match response cache backed by SQLite (WAL mode)
- `SemanticCache` — character n-gram Jaccard similarity cache for near-duplicate detection
- `ComplexityEstimator` — heuristic tier mapping from intent + context size
- Full unit test suite (15 files)
- `mypy --strict` typing throughout
- Provider support: NVIDIA NIM, Groq, OpenRouter, Cerebras, Gemini, Mistral, Anthropic, Ollama

[Unreleased]: https://github.com/dragonlightintl/dragonlight-router/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/dragonlightintl/dragonlight-router/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/dragonlightintl/dragonlight-router/compare/v0.1.0...v0.2.6
[0.1.0]: https://github.com/dragonlightintl/dragonlight-router/releases/tag/v0.1.0
