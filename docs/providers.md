# Providers

The router supports 11 LLM providers. Each provider is implemented as an adapter that conforms to the `GenerativeBackend` protocol.

## Supported providers

| Provider | Adapter | Catalog auto-refresh | Auth method | Notes |
|---|---|---|---|---|
| Anthropic | `AnthropicBackend` | Static | `x-api-key` header | No public `/v1/models` endpoint; catalog is hardcoded |
| Cerebras | `CerebrasBackend` | Dynamic | Bearer token | OpenAI-compatible |
| Cohere | `CohereBackend` | Dynamic | Bearer token | OpenAI-compatible |
| Gemini | `GoogleBackend` | Dynamic | `x-goog-api-key` header | Uses Gemini REST API, not OpenAI-compatible wire format |
| Groq | `GroqBackend` | Dynamic | Bearer token | OpenAI-compatible |
| Mistral | `MistralBackend` | Dynamic | Bearer token | OpenAI-compatible |
| NVIDIA NIM | `NvidiaBackend` | Dynamic | Bearer token | OpenAI-compatible |
| Ollama | `LocalBackend` | Dynamic | None | Local-only, no API key, defaults to `localhost:11434` |
| OpenAI | `OpenAIBackend` | Dynamic | Bearer token | OpenAI-compatible (canonical implementation) |
| OpenRouter | `OpenRouterBackend` | Dynamic | Bearer token | OpenAI-compatible |
| Together | `TogetherBackend` | Dynamic | Bearer token | OpenAI-compatible |

"Dynamic" catalog refresh means the router fetches the provider's `/v1/models` endpoint (or equivalent) to discover available models. The refresh runs at startup and can be triggered on demand via `POST /v1/catalog/refresh`.

## OpenAI-compatible base class

Most providers speak the OpenAI chat completions protocol. The `OpenAICompatibleBackend` base class handles the shared logic:

- Chat completion requests (`/v1/chat/completions`)
- SSE stream parsing
- Bearer token auth headers
- Health checks via the models endpoint
- Retry with exponential backoff + jitter (3 attempts, 0.5s base delay, 8s max delay)
- Retryable status codes: 429, 500, 502, 503, 504

Provider-specific adapters inherit from this base and override only what differs: base URL defaults, auth header format, or endpoint path construction.

## Provider-specific notes

### Anthropic

Anthropic does not expose a public `/v1/models` endpoint for catalog discovery. The adapter uses a static model catalog defined in the codebase. Health checks send a minimal messages request to verify connectivity and auth.

The adapter uses Anthropic's native Messages API wire format (not OpenAI-compatible). System messages are extracted from the message list and passed as the top-level `system` field. Auth uses the `x-api-key` header with the `anthropic-version` header set to `2023-06-01`.

### Gemini

The Gemini adapter uses Google's Generative Language REST API, which has its own wire format distinct from OpenAI's. Messages are converted from OpenAI format (`role: "user"/"assistant"/"system"`) to Gemini format (`role: "user"/"model"` with `parts` arrays). System instructions use the `systemInstruction` field.

Auth uses the `x-goog-api-key` header. Bearer token auth is also supported for service account credentials.

The streaming endpoint is `v1beta/models/{model}:streamGenerateContent?alt=sse`.

### Ollama

Ollama runs locally and requires no API key or rate limits. The adapter connects to `http://localhost:11434` by default (configurable via `base_url` in `router.yaml`).

The adapter uses Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`). Health checks hit `GET /api/tags`. Connection timeouts are set to 120 seconds to accommodate slower initial model loads.

When Ollama is unreachable, the backend is marked as OFFLINE (not ERROR) since it is expected to be unavailable on machines without a local Ollama instance.

## Adding a new provider

To add a provider that speaks the OpenAI chat completions protocol:

1. **Create the adapter module** at `src/dragonlight_router/adapters/{name}.py`.

2. **Inherit from `OpenAICompatibleBackend`** and set the required class attributes:

    ```python
    from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend

    class NewProviderBackend(OpenAICompatibleBackend):
        _provider_name = "NewProvider"
        _default_base_url = "https://api.newprovider.com"
    ```

3. **Override methods** only if the provider's API diverges from the standard:
    - `_build_auth_headers()` for non-Bearer auth
    - `_completions_path` if the endpoint path differs from `/v1/chat/completions`
    - `_models_path` if the models endpoint differs from `/v1/models`

4. **Add the provider** to `config/router.yaml` with its `name`, `base_url`, `env_key`, `model_prefix`, and optional `rate_limits`.

5. **Register the adapter** in the adapter registry so the engine discovers it at startup.

For providers with non-OpenAI wire formats (like Anthropic or Gemini), implement the `GenerativeBackend` protocol directly instead of inheriting from the base class.
