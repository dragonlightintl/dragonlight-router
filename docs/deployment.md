# Deployment

## Prerequisites

- Python 3.11+
- API keys for at least one provider (see [Configuration > Environment Variables](configuration.md#environment-variables))
- `config/router.yaml` (ships with the repo; override path via `DRAGONLIGHT_ROUTER_CONFIG`)
- A bootstrapped role matrix: `cp config/model_role_matrix.json router_state/`

## Local development

```bash
make dev                    # Install with all extras + dev tools
make run                    # Start server on 127.0.0.1:8100
make test                   # Run full test suite
make test-cov               # Run tests with coverage report
make lint                   # Ruff linter
make typecheck              # mypy strict mode
```

Or directly:

```bash
pip install -e ".[all,dev]"
python3 -m pytest --no-cov -q
```

## Docker

### Build and run

```bash
# Build
make docker-build
# or: docker build -t dragonlight-router:latest .

# Run (reads API keys from .env)
make docker-run
# or: docker run --rm -p 8100:8100 --env-file .env dragonlight-router:latest
```

### docker-compose

```bash
docker-compose up -d
```

The compose configuration:

- Sets `DRAGONLIGHT_HOST=0.0.0.0` for container networking
- Persists state to a named volume (`router-state`)
- Mounts `config/` as read-only
- Sets a 15-second graceful shutdown timeout
- Applies security hardening (see [Security > Container hardening](security.md#container-hardening))

### Image details

The Docker image uses a multi-stage build:

- **Build stage** installs dependencies (with hash verification when `requirements-hashed.txt` is present)
- **Runtime stage** copies only the installed packages and application code
- Runs as a non-root user (`router`)
- Includes a built-in health check against `/v1/health`
- State persists to `/app/state` volume

## State persistence

State lives in `router_state/` (configurable via `state_dir` in `router.yaml`):

| File | Contents | Rebuilt on restart? |
|---|---|---|
| `catalog.json` | Cached provider model lists | Yes (on first catalog refresh) |
| `budget.json` | RPM/RPD/TPM counters per provider | Yes (counters reset) |
| `model_role_matrix.json` | Role-to-model mapping | No (operator-managed) |

Budget counters flush to disk every 5 seconds (configurable). Health scores and circuit breaker states are in-memory only and reset on restart.

## Credential rotation

1. Set the new API key value in your `.env` or environment.
2. Restart the router process (or container).
3. Trigger a catalog refresh to verify the new key:
    ```bash
    curl -s -X POST http://127.0.0.1:8100/v1/catalog/refresh \
      -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY"
    ```
4. Check the response for `auth_failures` — any listed provider has an invalid key.
5. Verify via `/v1/health` that the provider shows as healthy (not `KEY_INVALID`).

## Circuit breaker reset

When a backend trips its circuit breaker (too many consecutive failures), it enters OPEN state and is excluded from selection. To reinstate manually:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/reinstate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY" \
  -d '{"backend": "groq/llama-3.3-70b-versatile"}'
```

The backend returns to the AVAILABLE pool immediately.

## Monitoring

| Endpoint | Purpose | Expected response |
|---|---|---|
| `GET /v1/health` | Liveness probe | Always 200; `status` is `healthy`, `degraded`, or `unavailable` |
| `GET /v1/ready` | Readiness probe | 200 when catalog is loaded, 503 otherwise |
| `GET /metrics` | Operational metrics | Request counts, latency percentiles, dispatch stats, uptime, memory |

### Alert conditions

- `/v1/health` status is `degraded` or `unavailable` for more than 5 minutes
- `/v1/ready` returns 503 after the startup grace period
- `key_invalid_count > 0` in the health response (credential issue)
- `circuit_breaker_trips` increasing in `/metrics` (provider instability)
- Memory usage trending upward without plateau (leak indicator)

## Graceful shutdown

The server handles SIGTERM/SIGINT via uvicorn's native signal handling. On shutdown:

1. In-flight requests drain for up to `DRAGONLIGHT_GRACEFUL_SHUTDOWN_TIMEOUT` seconds (default 10)
2. The health check background task is cancelled
3. Budget and health state is persisted to disk
