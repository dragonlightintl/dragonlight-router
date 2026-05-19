# Changelog

All notable changes to dragonlight-router are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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
