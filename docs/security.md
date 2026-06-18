# Security

Security posture of the Dragonlight Router, covering network, authentication, input validation, and container hardening.

## CORS configuration

CORS is **disabled by default** — when `DRAGONLIGHT_CORS_ORIGINS` is empty, the CORS middleware is not applied and no `Access-Control-Allow-Origin` header is set.

To enable CORS, set the environment variables:

| Variable | Default | Description |
|---|---|---|
| `DRAGONLIGHT_CORS_ORIGINS` | *empty* | Comma-separated allowed origins (`*` for development) |
| `DRAGONLIGHT_CORS_METHODS` | `GET,POST,OPTIONS` | Allowed HTTP methods |
| `DRAGONLIGHT_CORS_HEADERS` | `*` | Allowed request headers |
| `DRAGONLIGHT_CORS_CREDENTIALS` | `false` | Allow cookies and auth headers |

For production, set specific origins rather than `*`.

## Admin endpoint authentication

Three endpoints require admin auth when `DRAGONLIGHT_ADMIN_API_KEY` is set:

- `POST /v1/catalog/refresh`
- `POST /v1/retire`
- `POST /v1/reinstate`

Auth uses a Bearer token in the `Authorization` header:

```bash
curl -X POST http://127.0.0.1:8100/v1/retire \
  -H "Authorization: Bearer $DRAGONLIGHT_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"backend": "groq/llama-3.3-70b-versatile"}'
```

When no admin key is configured, these endpoints are open. The server logs a warning at startup when admin endpoints are unprotected.

### Brute-force protection

Failed admin auth attempts are tracked per IP address. After 5 failures within 60 seconds from the same IP, the server returns 429 (Too Many Requests) before checking credentials. The failure window resets automatically.

## Rate limiting

The server applies per-IP rate limiting via a token-bucket algorithm. The default configuration allows 60 requests per minute per IP.

The rate limiter sits inside the middleware stack (innermost layer). Requests that exceed the limit receive a 429 response with a JSON error body.

## SSRF validation

Provider URLs are validated before use to prevent Server-Side Request Forgery attacks. The validation (`core/validation.py`, SEC-003) rejects URLs that:

- Use a scheme other than `https` (except `http` for localhost, to support Ollama)
- Resolve to private IP ranges (`10.x`, `172.16-31.x`, `192.168.x`, `127.x`, `169.254.x`)
- Point to cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`, `metadata.goog`)
- Have no hostname

DNS resolution is performed to catch hostnames that resolve to private addresses. The `http` scheme is allowed only for localhost to support local Ollama instances.

## Prompt sanitization

Operator prompt text is sanitized before dispatch to the LLM (`_sanitize_prompt` in `server/routes.py`):

- **Null bytes and control characters** are stripped (newlines, carriage returns, and tabs are preserved)
- **Truncation** to 100,000 characters maximum
- A warning is logged when input is modified

The same sanitization applies to both `operator_message` and `system_prompt` fields in dispatch requests.

## Output validation

LLM response content is validated before returning to the client (`_validate_llm_response` in `server/routes.py`):

- **Empty responses** are caught and logged
- **Null bytes** are stripped
- **Excessive length** (over 500,000 characters) is truncated with a warning

Output validation applies to both non-streaming JSON responses and individual SSE stream chunks.

## Input validation

Request bodies are validated at the route handler level:

- **String fields** are type-checked and length-limited (100K characters)
- **Intent categories** are validated against a fixed allowlist (HAZ-007) to prevent adversarial intent injection affecting routing decisions
- **Fallback policies** must be one of `allow`, `deny`, or `same_tier` (HAZ-004)
- **Numeric fields** are range-checked (e.g., `top_n` between 1 and 500, `context_tokens` non-negative)
- Malformed JSON returns 400

## Request correlation

Every request receives an `X-Request-ID` header. The middleware reads the ID from the incoming request or generates a UUID4. This ID is bound to structlog context variables, so all log lines within a request automatically include it. The ID is also set on the response for client-side correlation.

## Container hardening

The Docker and docker-compose configurations implement defense-in-depth:

| Control | Implementation |
|---|---|
| **Multi-stage build** | Build dependencies never ship in the runtime image |
| **Hash-verified dependencies** | `requirements-hashed.txt` with `pip --require-hashes` ensures supply-chain integrity |
| **Non-root user** | Runtime runs as `router` user (never root) |
| **Read-only root filesystem** | Enforced via `read_only: true` in compose |
| **tmpfs for /tmp** | Writable temporary storage without persisting to the root filesystem |
| **Capability drops** | All Linux capabilities dropped (`cap_drop: ALL` in compose) |
| **No new privileges** | `no-new-privileges:true` prevents setuid escalation |

To regenerate the hash-verified lockfile:

```bash
pip-compile --generate-hashes --output-file=requirements-hashed.txt pyproject.toml
```

## Middleware stack

The middleware layers apply in this order (outermost to innermost):

1. **CORS** (only when configured via `DRAGONLIGHT_CORS_ORIGINS`)
2. **RequestCorrelationMiddleware** — attaches request IDs, logs request summaries, records metrics
3. **RateLimitMiddleware** — per-IP token-bucket rate limiting
