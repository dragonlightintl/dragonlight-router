# Deployment Runbook

## Prerequisites

- Python 3.11+
- API keys for at least one provider (see Environment Variables below)
- `config/router.yaml` (ships with the repo; override path via `DRAGONLIGHT_ROUTER_CONFIG`)
- A bootstrapped role matrix: `cp config/model_role_matrix.json router_state/`

## Local Development

```bash
make dev                    # Install with all extras + dev tools
make run                    # Start server on 127.0.0.1:8100
make test                   # Run full test suite (no coverage)
make test-cov               # Run tests with coverage report
make lint                   # Ruff linter
make typecheck              # mypy strict mode
```

## Docker Deployment

```bash
# Build
make docker-build
# or: docker build -t dragonlight-router:latest .

# Run (reads API keys from .env)
make docker-run
# or: docker run --rm -p 8100:8100 --env-file .env dragonlight-router:latest

# Compose
docker-compose up -d
```

The image runs as a non-root user, persists state to a volume, and includes a built-in health check.

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq provider auth |
| `CEREBRAS_API_KEY` | Cerebras provider auth |
| `OPENAI_API_KEY` | OpenAI provider auth |
| `ANTHROPIC_API_KEY` | Anthropic provider auth |
| `GEMINI_API_KEY` | Google Gemini provider auth |
| `MISTRAL_API_KEY` | Mistral provider auth |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM provider auth |
| `OPENROUTER_API_KEY` | OpenRouter provider auth |
| `COHERE_API_KEY` | Cohere provider auth |
| `TOGETHER_API_KEY` | Together provider auth |
| `DRAGONLIGHT_ROUTER_CONFIG` | Config file path (default: `config/router.yaml`) |
| `DRAGONLIGHT_HOST` | Bind address (default: `127.0.0.1`) |
| `DRAGONLIGHT_PORT` | Bind port (default: `8100`) |
| `DRAGONLIGHT_ADMIN_API_KEY` | Bearer token for admin endpoints |
| `DRAGONLIGHT_GRACEFUL_SHUTDOWN_TIMEOUT` | Shutdown grace period in seconds (default: `10`) |

## Credential Rotation

1. Set the new API key value in your `.env` or environment.
2. Restart the router process (or container).
3. Trigger a catalog refresh to verify the new key works:
   ```bash
   curl -s -X POST http://127.0.0.1:8100/v1/catalog/refresh \
     -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY"
   ```
4. Check the response for `auth_failures` — any listed provider has an invalid key.
5. Verify via `/v1/health` that the provider shows as healthy (not `KEY_INVALID`).

## State Persistence

State lives in `router_state/` (configurable via `state_dir` in `router.yaml`):

| File | Contents | Rebuilt on restart? |
|---|---|---|
| `catalog.json` | Cached provider model lists | Yes (on first catalog refresh) |
| `budget.json` | RPM/RPD/TPM counters per provider | Yes (counters reset) |
| `model_role_matrix.json` | Role-to-model mapping | No (operator-managed) |

Budget counters flush to disk every 5 seconds (configurable). Health scores and circuit breaker states are in-memory only and reset on restart.

## Circuit Breaker Reset

When a backend trips its circuit breaker (too many consecutive failures), it enters OPEN state and is excluded from selection. To reinstate manually:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/reinstate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY" \
  -d '{"backend": "groq/llama-3.3-70b-versatile"}'
```

The backend returns to the AVAILABLE pool immediately.

## Monitoring

- **Liveness:** `GET /v1/health` — always returns 200 with `status` field (`healthy`, `degraded`, `unavailable`)
- **Readiness:** `GET /v1/ready` — returns 200 when catalog is loaded, 503 otherwise
- **Metrics:** `GET /metrics` — request counts, latency percentiles, dispatch stats, uptime, memory

**Alert on:**
- `/v1/health` status is `degraded` or `unavailable` for more than 5 minutes
- `/v1/ready` returns 503 after startup grace period
- `key_invalid_count > 0` in health response (credential issue)
- `circuit_breaker_trips` increasing in `/metrics` (provider instability)
- Memory usage trending upward without plateau (leak indicator)
