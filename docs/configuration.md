# Configuration

The router reads configuration from three sources: a YAML config file, a JSON role matrix, and environment variables.

## router.yaml

The default config lives at `config/router.yaml`. Override the path with the `DRAGONLIGHT_ROUTER_CONFIG` environment variable.

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

See `config/router.yaml` in the repository for the full default configuration including all providers.

### Config fields

| Field | Type | Default | Description |
|---|---|---|---|
| `state_dir` | string | `./router_state` | Directory for catalog, budget, and role matrix persistence |
| `catalog_ttl_hours` | int | `24` | Hours before the catalog is considered stale and triggers a refresh |
| `budget_flush_interval_s` | int | `5` | Seconds between budget counter flushes to disk |
| `default_top_n` | int | `12` | Number of ranked models returned when `top_n` is not specified |
| `max_consecutive_same_provider` | int | `2` | Maximum consecutive models from one provider in the ranked list |
| `admin_api_key` | string | *none* | Bearer token required for admin endpoints (retire, reinstate, catalog refresh) |

### Provider fields

Each entry in the `providers` list accepts:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Provider identifier (used in model prefixes and API responses) |
| `base_url` | string | yes | Base URL for the provider API |
| `catalog_url` | string | no | URL for fetching the model catalog (defaults to `{base_url}/v1/models`) |
| `env_key` | string | yes | Environment variable name for the API key |
| `model_prefix` | string | yes | Prefix prepended to model IDs (e.g., `groq/`) |
| `rate_limits.rpm` | int/null | no | Requests per minute limit (`null` for unlimited) |
| `rate_limits.rpd` | int/null | no | Requests per day limit |
| `rate_limits.tpm` | int/null | no | Tokens per minute limit |

## Role matrix

The role-to-model mapping lives at `{state_dir}/model_role_matrix.json`. The router hot-reloads this file on change — no restart required.

The file maps role names to ordered lists of model IDs:

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

Bootstrap from the provided example:

```bash
cp config/model_role_matrix.json router_state/
```

The order of models in each role list determines the preference ranking. The cascade pipeline uses this ordering as input, then applies budget, health, and rate-limit scoring to produce the final ranked output.

## Environment variables

### API keys

| Variable | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic |
| `CEREBRAS_API_KEY` | Cerebras |
| `COHERE_API_KEY` | Cohere |
| `GEMINI_API_KEY` | Gemini |
| `GROQ_API_KEY` | Groq |
| `MISTRAL_API_KEY` | Mistral |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM |
| `OPENAI_API_KEY` | OpenAI |
| `OPENROUTER_API_KEY` | OpenRouter |
| `TOGETHER_API_KEY` | Together |

Ollama is local and requires no API key. Set only the keys for providers you intend to use.

### Server settings

| Variable | Default | Description |
|---|---|---|
| `DRAGONLIGHT_ROUTER_CONFIG` | `config/router.yaml` | Path to the config file |
| `DRAGONLIGHT_HOST` | `127.0.0.1` | Server bind address |
| `DRAGONLIGHT_PORT` | `8100` | Server bind port |
| `DRAGONLIGHT_GRACEFUL_SHUTDOWN_TIMEOUT` | `10` | Seconds for in-flight request draining on shutdown |
| `DRAGONLIGHT_ADMIN_API_KEY` | *none* | Bearer token for admin endpoints |

### CORS settings

| Variable | Default | Description |
|---|---|---|
| `DRAGONLIGHT_CORS_ORIGINS` | *empty* (CORS disabled) | Comma-separated allowed origins. Set to `*` for development |
| `DRAGONLIGHT_CORS_METHODS` | `GET,POST,OPTIONS` | Comma-separated allowed HTTP methods |
| `DRAGONLIGHT_CORS_HEADERS` | `*` | Comma-separated allowed request headers |
| `DRAGONLIGHT_CORS_CREDENTIALS` | `false` | Set to `true` to allow cookies and auth headers |

When `DRAGONLIGHT_CORS_ORIGINS` is empty, CORS middleware is not applied.
