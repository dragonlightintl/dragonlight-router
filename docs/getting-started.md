# Getting Started

Get the router running and make your first model selection request.

## Install

```bash
pip install -e ".[all]"
```

Requires **Python 3.11+**.

The `[all]` extra includes the HTTP server (Starlette + Uvicorn), OpenAI SDK shim, and SQLite cache. Install only what you need with individual extras:

| Extra | What it installs |
|---|---|
| `[server]` | Starlette + Uvicorn (HTTP API) |
| `[adapters]` | OpenAI SDK shim |
| `[cache]` | aiosqlite (SimpleCache + SemanticCache) |
| `[all]` | Everything above |
| `[dev]` | pytest, mypy, ruff, hypothesis, bandit |

For reproducible production deploys, use the lockfile:

```bash
pip install -r requirements.lock
```

## Configure a provider

Copy `.env.example` to `.env` and set at least one provider key:

```bash
cp .env.example .env
# Edit .env — fill in at least one API key
```

Bootstrap the role matrix:

```bash
cp config/model_role_matrix.json router_state/
```

The router loads `config/router.yaml` by default. No changes are needed to start — the default configuration includes all 11 providers.

## Option A — Python library

```python
from dragonlight_router import RouterEngine

router = RouterEngine()  # loads config/router.yaml

# Get ranked model IDs for a role
models = router.select_models("code_review", top_n=5)
# → ["groq/llama-3.3-70b-versatile", "cerebras/llama3.1-70b", ...]

# Call your preferred model with your own SDK client
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

## Option B — HTTP sidecar

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

## Dual interface

Both modes share the same budget tracking, health scoring, and circuit breaking.

**Select mode** returns a ranked model list — your app owns the API call:

```
Your app
  ├─ POST /v1/select → ["groq/llama-3.3-70b-versatile", ...]
  ├─ calls groq with its own SDK ← you own this
  └─ POST /v1/record (outcome) → budget + health updated
```

**Dispatch mode** handles the full cascade (MBR/CBR/LBR), adapter call, and fallback:

```
Your app
  └─ POST /v1/dispatch → { content, backend_used, latency_ms, ... }
       Router handles: model selection → context filtering → adapter call → fallback
```

## Next steps

- [Configuration](configuration.md) — full `router.yaml` schema, role matrix format, environment variables
- [API Reference](api-reference.md) — all HTTP endpoints with request/response details
- [Providers](providers.md) — supported providers and provider-specific setup
