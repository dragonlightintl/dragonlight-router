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

### Intent classification (IBR)

The Intent Based Router (IBR) subsystem classifies incoming requests by task type, domain, and quality-speed tradeoff, then uses model spectrograph profiles to bias candidate scoring toward models that excel at the detected intent. When disabled (the default), the pipeline behaves identically to v0.3.0.

```yaml
intent_classification:
  enabled: false
  timeout_ms: 100
  cache_ttl_s: 300
  cache_max_entries: 5000
  confidence_threshold: 0.6
  profile_confidence_threshold: 0.3
  spectrograph_match_weight: 0.15
  spectrograph_match_weight_governor: 0.05
```

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable IBR intent classification. When `false`, the cascade operates without intent-aware scoring |
| `timeout_ms` | int | `100` | Maximum wall-clock time (ms) for the classification call. Requests exceeding this fall back to unclassified dispatch |
| `cache_ttl_s` | int | `300` | Seconds before a cached classification result expires |
| `cache_max_entries` | int | `5000` | Maximum number of classification results held in the LRU cache |
| `confidence_threshold` | float | `0.6` | Minimum classifier confidence (0.0-1.0) required to use the classification result. Below this, the request is treated as unclassified |
| `profile_confidence_threshold` | float | `0.3` | Minimum spectrograph profile confidence (0.0-1.0) required to apply a model's profile score. Profiles below this threshold contribute zero signal |
| `spectrograph_match_weight` | float | `0.15` | Weight of the spectrograph match score in the composite candidate score (0.0-1.0) |
| `spectrograph_match_weight_governor` | float | `0.05` | Maximum per-cycle adjustment to the spectrograph match weight during adaptive tuning |

### Pinned dispatch

Pinned dispatch allows a caller to bypass the cascade pipeline and route directly to a specific backend by setting `DispatchOrder.model` to a registered backend name (e.g. `"anthropic/claude-sonnet-4-20250514"`). This section controls operational guardrails for that bypass.

```yaml
pinned_dispatch:
  honor_health: true
```

| Field | Type | Default | Description |
|---|---|---|---|
| `honor_health` | bool | `true` | When `true`, pinned dispatch respects backend health status (circuit-open and retired backends are rejected). When `false`, the pinned model is used regardless of health state |

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
