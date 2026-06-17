# dragonlight-router

**Multi-provider LLM routing engine — intelligent model selection and cascade dispatch across 11 providers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## What it is

dragonlight-router is a Python library and HTTP service that picks the best available LLM for each request. Given a **role** (a logical task type such as `"code_review"` or `"summarize"`), the router consults a hot-reloadable role-to-model matrix, filters against live provider catalogs, scores candidates on budget headroom and recent health, interleaves across providers to avoid thundering-herd concentration, and returns a ranked list of model IDs your application can try in order. It exposes a clean dual interface: import `RouterEngine` directly in Python, or run it as an HTTP sidecar and call `/v1/select`.

---

## Why it exists

Every non-trivial LLM application ends up with the same ad-hoc pile: a giant if/else over provider names, manual rate-limit counters, a health-check spreadsheet updated by hand, and a deployment that sends 100% of traffic to one provider until it breaks. dragonlight-router replaces that pile with a single component that tracks budget windows, circuit-breaks unhealthy models, keeps its own catalog fresh from provider APIs, and degrades gracefully across 8 providers — so your application code handles one ranked list instead of eight provider SDKs.

---

## Install

```bash
# From local source (not yet published to PyPI)
pip install -e ".[all]"
```

**Extras:**

| Extra | Installs |
|---|---|
| `[server]` | Starlette + Uvicorn (HTTP API) |
| `[adapters]` | OpenAI SDK shim |
| `[cache]` | aiosqlite (SimpleCache + SemanticCache) |
| `[all]` | Everything above |
| `[dev]` | pytest, mypy, ruff |

**Requires Python ≥ 3.11.**

---

## Quickstart

### Option A — Python library

```python
from dragonlight_router import RouterEngine

router = RouterEngine()  # loads config/router.yaml by default

# Get ranked model IDs for a role
models = router.select_models("code_review", top_n=5)
# → ["groq/llama-3.3-70b-versatile", "cerebras/llama3.1-70b", ...]

# Call your preferred model here (router does not dispatch)
response = my_openai_client.chat.completions.create(
    model=models[0].split("/", 1)[1],  # strip provider prefix
    messages=[{"role": "user", "content": prompt}],
)

# Feed the outcome back so budget + health stay accurate
router.record_request(
    provider="groq",
    model_id=models[0],
    success=True,
    tokens_used=response.usage.total_tokens,
    latency_ms=elapsed_ms,
)
```

### Option B — HTTP sidecar

Start the server:

```bash
dragonlight-router
# → Uvicorn listening on http://127.0.0.1:8100
```

Select models for a role:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/select \
  -H "Content-Type: application/json" \
  -d '{"role": "summarize", "top_n": 3}' | jq .
```

```json
{
  "models": [
    "gemini/gemini-2.0-flash",
    "groq/llama-3.3-70b-versatile",
    "mistral/mistral-large-latest"
  ]
}
```

Record an outcome:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/record \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "model_id": "gemini/gemini-2.0-flash",
       "success": true, "tokens_used": 512, "latency_ms": 340}'
```

Check health:

```bash
curl -s http://127.0.0.1:8100/v1/health | jq .
```

Inspect live catalog:

```bash
curl -s http://127.0.0.1:8100/v1/catalog | jq .
```

Force catalog refresh:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/catalog/refresh | jq .
```

---

## Configuration

### router.yaml

The default config lives at `config/router.yaml` (override with `DRAGONLIGHT_ROUTER_CONFIG`).

```yaml
# Where to persist catalog, health, and budget state
state_dir: ./router_state

# Hours before the provider catalog is considered stale
catalog_ttl_hours: 24

# Seconds between budget flush to disk
budget_flush_interval_s: 5

# Default top_n when not specified per-request
default_top_n: 12

# Maximum consecutive models from the same provider in a ranked list
max_consecutive_same_provider: 2

providers:
  - name: groq
    base_url: https://api.groq.com/openai/v1
    catalog_url: https://api.groq.com/openai/v1/models
    env_key: GROQ_API_KEY
    model_prefix: "groq/"
    rate_limits:
      rpm: 30       # requests per minute (null = unlimited)
      rpd: 1000     # requests per day
      tpm: 6000     # tokens per minute

  # ... repeat for each provider
```

See [`config/router.yaml`](config/router.yaml) for the full default configuration including all 8 providers.

### Role matrix

The role-to-model mapping lives in `{state_dir}/model_role_matrix.json`. It is hot-reloaded on change — no restart required. The file maps role names to ordered lists of model IDs:

```json
{
  "code_review": [
    "groq/llama-3.3-70b-versatile",
    "cerebras/llama3.1-70b",
    "gemini/gemini-2.0-flash"
  ],
  "summarize": [
    "gemini/gemini-2.0-flash",
    "mistral/mistral-large-latest"
  ]
}
```

Bootstrap from the provided example: `cp config/model_role_matrix.json router_state/`.

### Environment variables

| Variable | Required for | Default |
|---|---|---|
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM provider | — |
| `GROQ_API_KEY` | Groq provider | — |
| `OPENROUTER_API_KEY` | OpenRouter provider | — |
| `CEREBRAS_API_KEY` | Cerebras provider | — |
| `GEMINI_API_KEY` | Gemini provider | — |
| `MISTRAL_API_KEY` | Mistral provider | — |
| `ANTHROPIC_API_KEY` | Anthropic provider | — |
| `OPENAI_API_KEY` | OpenAI provider | — |
| `COHERE_API_KEY` | Cohere provider | — |
| `TOGETHER_API_KEY` | Together provider | — |
| `DRAGONLIGHT_ROUTER_CONFIG` | Custom config path | `config/router.yaml` |
| `DRAGONLIGHT_HOST` | Server bind address | `127.0.0.1` |
| `DRAGONLIGHT_PORT` | Server port | `8100` |
| `DRAGONLIGHT_GRACEFUL_SHUTDOWN_TIMEOUT` | Graceful shutdown (seconds) | `10` |
| `DRAGONLIGHT_ADMIN_API_KEY` | Admin endpoint auth | — (open) |

Ollama is local and requires no API key. Copy `.env.example` to `.env` and fill in only the providers you intend to use.

---

## HTTP API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/select` | Return ranked model IDs for a role |
| `POST` | `/v1/dispatch` | Full cascade dispatch (MBR/CBR/LBR + adapter call) |
| `POST` | `/v1/record` | Record request outcome (budget + health) |
| `GET` | `/v1/health` | Live health snapshot of all tracked models |
| `GET` | `/v1/catalog` | Current in-memory provider model catalog |
| `POST` | `/v1/catalog/refresh` | Trigger immediate catalog refresh (admin) |
| `POST` | `/v1/retire` | Retire a backend from the active pool (admin) |
| `POST` | `/v1/reinstate` | Reinstate a previously retired backend (admin) |

**POST /v1/select** body:

```json
{
  "role": "code_review",
  "top_n": 5,
  "exclude_providers": ["openrouter"]
}
```

**POST /v1/record** body:

```json
{
  "provider": "groq",
  "model_id": "groq/llama-3.3-70b-versatile",
  "success": true,
  "tokens_used": 1024,
  "latency_ms": 620
}
```

---

## Architecture

dragonlight-router is composed of 11 subsystems. Each has a single responsibility.

| Subsystem | Package | Responsibility |
|---|---|---|
| **RouterEngine** | `dragonlight_router.router` | Central orchestrator — wires all subsystems, exposes `select_models()` and `record_request()` |
| **RoleMatrix** | `dragonlight_router.roles` | Hot-reloadable JSON mapping roles → ranked model ID lists |
| **BudgetTracker** | `dragonlight_router.budget` | Sliding-window RPM and daily RPD tracking per provider; emits a 0–100 budget score |
| **HealthTracker** | `dragonlight_router.health` | Per-model error counts and EWMA latency; emits a 0–100 health score |
| **CircuitBreaker** | `dragonlight_router.health` | CLOSED → OPEN → HALF_OPEN state machine; prevents requests to consistently failing models |
| **CatalogCache** | `dragonlight_router.catalog` | File-backed TTL cache of live provider model lists |
| **CatalogRefresher** | `dragonlight_router.catalog` | Concurrent async fetch from each provider's `/v1/models` endpoint |
| **Server** | `dragonlight_router.server` | Starlette HTTP API (`/v1/select`, `/v1/record`, `/v1/health`, `/v1/catalog`) |
| **SimpleCache** | `dragonlight_router.cache` | SHA-256 exact-match response cache backed by SQLite (WAL mode) |
| **SemanticCache** | `dragonlight_router.cache` | Character n-gram Jaccard similarity cache for near-duplicate prompt detection |
| **ComplexityEstimator** | `dragonlight_router.complexity` | Heuristic mapping intent + context size to tier (LOCAL / HAIKU / SONNET / OPUS) |

---

## Supported providers

| Provider | Catalog auto-refresh | Notes |
|---|---|---|
| NVIDIA NIM | ✅ | `NVIDIA_NIM_API_KEY` |
| Groq | ✅ | `GROQ_API_KEY` |
| OpenRouter | ✅ | `OPENROUTER_API_KEY` |
| Cerebras | ✅ | `CEREBRAS_API_KEY` |
| Gemini | ✅ | `GEMINI_API_KEY` |
| Mistral | ✅ | `MISTRAL_API_KEY` |
| Anthropic | ❌ (static) | `ANTHROPIC_API_KEY`; no public `/v1/models` endpoint |
| OpenAI | ✅ | `OPENAI_API_KEY` |
| Cohere | ✅ | `COHERE_API_KEY` |
| Together | ✅ | `TOGETHER_API_KEY` |
| Ollama | ✅ | No key needed; defaults to `localhost:11434` |

---

## Dual interface

dragonlight-router supports two modes of operation:

**Select mode** — returns a ranked model list, your app owns the API call:

```
Your app
  ├─ POST /v1/select → ["groq/llama-3.3-70b-versatile", ...]
  ├─ calls groq with its own SDK ← you own this
  └─ POST /v1/record (outcome) → budget + health updated
```

**Dispatch mode** — the router handles the full cascade (MBR/CBR/LBR), adapter call, and fallback:

```
Your app
  └─ POST /v1/dispatch → { content, backend_used, latency_ms, ... }
       Router handles: model selection → context filtering → adapter call → fallback
```

Both modes share the same budget tracking, health scoring, and circuit breaking.

---

## Running tests

```bash
make dev                   # install with dev dependencies
make test                  # full suite (939 tests)
make test-cov              # full suite with coverage report
make lint                  # ruff linter
make typecheck             # mypy strict mode
```

Or directly:

```bash
pip install -e ".[all,dev]"
python3 -m pytest --no-cov -q
```

---

## Docker

```bash
# Build and run
make docker-build
make docker-run            # reads API keys from .env

# Or with docker-compose
docker-compose up -d
```

The Docker image runs as a non-root user, persists state to a volume, and includes a health check.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, PR conventions, and linting requirements (`ruff`, `mypy --strict`).

One logical change per PR. All tests must pass. No new mypy or ruff errors.

---

## License

MIT © Korrigon @ Dragonlight International. See [LICENSE](LICENSE).
