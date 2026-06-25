# Router as Transparent LLM Proxy

**Status:** Draft
**Author:** Korrigon / Claude
**Date:** 2026-06-25
**Version:** 0.1.0

---

## 1. Vision

The Dragonlight Router presents a standard **OpenAI-compatible `/v1/chat/completions` endpoint** that pydantic-ai connects to directly. The factory sets ONE URL and never thinks about models again:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "auto",
    provider=OpenAIProvider(base_url="http://localhost:8100/v1"),
)
```

The router receives every request, selects the best model, forwards the call to the real provider, and returns the response as if it were the provider. The factory has **zero model selection logic**.

---

## 2. Architecture

### 2.1 Request Flow

```
Factory (pydantic-ai)                  Router (:8100)                      Provider APIs
        |                                   |                                   |
        |  POST /v1/chat/completions        |                                   |
        |  model: "auto"                    |                                   |
        |  messages: [...]                  |                                   |
        |  tools: [...]                     |                                   |
        |  stream: true/false               |                                   |
        |  X-Dragonlight-Intent: ...        |                                   |
        |  X-Dragonlight-Stakes: ...        |                                   |
        | --------------------------------> |                                   |
        |                                   |                                   |
        |                          1. Parse OpenAI request                      |
        |                          2. Extract intent from                       |
        |                             headers / model name                      |
        |                          3. Build DispatchOrder                       |
        |                          4. Run MBR->IBR->CBR->LBR                   |
        |                             cascade (select model)                   |
        |                          5. Normalize request for                     |
        |                             selected provider                        |
        |                                   |                                   |
        |                                   |  POST /v1/chat/completions        |
        |                                   |  model: "deepseek-v4-pro"         |
        |                                   |  Authorization: Bearer <key>      |
        |                                   | --------------------------------> |
        |                                   |                                   |
        |                                   |  200 OK / SSE stream              |
        |                                   | <-------------------------------- |
        |                                   |                                   |
        |                          6. Record health/budget/latency             |
        |                          7. Normalize response to                     |
        |                             OpenAI format                            |
        |                                   |                                   |
        |  200 OK / SSE stream              |                                   |
        | <-------------------------------- |                                   |
        |                                   |                                   |
        |                         On failure:                                  |
        |                          8. Retry with next model                     |
        |                             in cascade (transparent)                 |
        |                                   |                                   |
```

### 2.2 Two Paths Coexist

The router retains its existing `/v1/dispatch` and `/v1/select` endpoints unchanged. The new `/v1/chat/completions` endpoint is an additional surface that wraps the same internal dispatch pipeline.

```
/v1/chat/completions   <-- NEW: OpenAI-compatible proxy (this design)
/v1/dispatch           <-- EXISTING: router-native dispatch
/v1/select             <-- EXISTING: model selection API
/v1/record             <-- EXISTING: outcome recording (not needed by proxy callers)
```

The proxy endpoint builds a `DispatchOrder` from the OpenAI request body and passes it through the same `engine.dispatch()` / `engine.dispatch_stream()` that `/v1/dispatch` already uses. All cascade logic, health tracking, budget tracking, and fallback are reused without duplication.

---

## 3. Intent Communication Mechanism

### 3.1 Model Name Encoding (Primary)

pydantic-ai sends the model name in the request body. The router interprets it as a routing hint:

| Model Name | Router Behavior |
|---|---|
| `"auto"` | Full cascade, general-purpose routing (intent_category="general") |
| `"auto-test"` | Cascade optimized for test generation (intent_category="test_generation") |
| `"auto-code"` | Cascade optimized for implementation (intent_category="coding") |
| `"auto-reason"` | Cascade for complex reasoning (intent_category="complex_reasoning") |
| `"auto-review"` | Cascade for code review (intent_category="code_review") |
| `"nvidia_nim/deepseek-v4-pro"` | Pinned dispatch to exact model (existing pinned path) |

Parsing rule: if the model name starts with `"auto"`, it is a routing hint. Everything after `"auto-"` maps to an intent_category via a configurable lookup table. Any model name that does NOT start with `"auto"` is treated as a pinned model and dispatched directly (existing pinned dispatch path).

### 3.2 Custom Headers (Secondary, Optional)

For callers that need finer control without changing the model name:

```
X-Dragonlight-Intent: test_generation
X-Dragonlight-Stakes: low
X-Dragonlight-Complexity: trivial
X-Dragonlight-Trust-Tier: trusted
X-Dragonlight-Fallback-Policy: same_tier
```

Headers override the model-name-derived values. This allows the factory to use `model="auto"` everywhere while varying intent per call via headers.

### 3.3 Why This Approach

- **Model name encoding** is the only mechanism guaranteed to survive pydantic-ai's HTTP layer unchanged. pydantic-ai constructs the request body and makes the HTTP call itself -- we cannot inject custom body fields.
- **Custom headers** work because pydantic-ai's `OpenAIProvider` accepts an `http_client` parameter (httpx.AsyncClient), which supports default headers via `httpx.AsyncClient(headers={...})`.
- **Extra body fields** are rejected because pydantic-ai strictly constructs the request body; there is no mechanism to inject additional fields.

### 3.4 Factory-Side Header Injection

```python
import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Per-task headers via httpx event hooks or custom transport
client = httpx.AsyncClient(
    headers={
        "X-Dragonlight-Intent": "test_generation",
        "X-Dragonlight-Stakes": "low",
    },
    timeout=httpx.Timeout(300.0),
)

model = OpenAIChatModel(
    "auto",
    provider=OpenAIProvider(
        base_url="http://localhost:8100/v1",
        api_key="not-used",  # router does not require auth for proxy
        http_client=client,
    ),
)
```

For varying intent per call, the factory creates a fresh `httpx.AsyncClient` per agent invocation with the appropriate headers set. This is lightweight (httpx clients are cheap to construct).

---

## 4. Router Changes

### 4.1 New Endpoint: `/v1/chat/completions`

**File:** `src/dragonlight_router/server/routes.py` (new handler)
**File:** `src/dragonlight_router/server/openai_proxy.py` (new module)

The handler:

1. Parses the standard OpenAI request body (messages, model, tools, tool_choice, stream, temperature, max_tokens, etc.)
2. Extracts routing hints from the `model` field and `X-Dragonlight-*` headers
3. Builds a `DispatchOrder` from the parsed request
4. Calls `engine.dispatch()` (non-streaming) or `engine.dispatch_stream()` (streaming)
5. Converts the `EngineResponse` / `StreamChunk` sequence back to OpenAI response format
6. Returns the response with standard OpenAI response structure

#### 4.1.1 Non-Streaming Response Format

The router returns a response that matches the OpenAI chat completions format exactly:

```json
{
  "id": "chatcmpl-dragonlight-<uuid>",
  "object": "chat.completion",
  "created": 1719300000,
  "model": "nvidia_nim/deepseek-v4-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is the implementation...",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 800,
    "total_tokens": 2300
  },
  "system_fingerprint": "dragonlight-router-v0.5.0"
}
```

When tool_calls are present in the EngineResponse:

```json
{
  "id": "chatcmpl-dragonlight-<uuid>",
  "object": "chat.completion",
  "created": 1719300000,
  "model": "nvidia_nim/deepseek-v4-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "read_file",
              "arguments": "{\"path\": \"src/main.py\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 50,
    "total_tokens": 1550
  }
}
```

#### 4.1.2 Streaming Response Format (SSE)

When `stream: true`, the router returns `text/event-stream` with standard OpenAI SSE chunks:

```
data: {"id":"chatcmpl-dragonlight-xxx","object":"chat.completion.chunk","created":1719300000,"model":"nvidia_nim/deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-dragonlight-xxx","object":"chat.completion.chunk","created":1719300000,"model":"nvidia_nim/deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}]}

data: {"id":"chatcmpl-dragonlight-xxx","object":"chat.completion.chunk","created":1719300000,"model":"nvidia_nim/deepseek-v4-pro","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-dragonlight-xxx","object":"chat.completion.chunk","created":1719300000,"model":"nvidia_nim/deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The router translates its internal `StreamChunk(event_type="token", content="...")` events to the OpenAI delta format. The final `metadata` StreamChunk becomes the `usage` object on the last chunk (or a separate final chunk before `[DONE]`).

#### 4.1.3 Router-Specific Response Extensions

The router includes additional metadata in a non-standard `dragonlight` field that OpenAI-compatible clients will ignore but the factory can optionally read:

```json
{
  "id": "chatcmpl-dragonlight-xxx",
  "object": "chat.completion",
  "...": "...",
  "dragonlight": {
    "backend_used": "nvidia_nim/deepseek-v4-pro",
    "backend_tier": "complex",
    "dispatch_mode": "cascade",
    "was_fallback": false,
    "fallback_chain": [],
    "estimated_cost_usd": 0.0023,
    "latency_ms": 1234.5
  }
}
```

### 4.2 Model Name Parsing

**File:** `src/dragonlight_router/server/openai_proxy.py`

```python
# Model name -> routing config
AUTO_INTENT_MAP: dict[str, str] = {
    "auto": "general",
    "auto-test": "test_generation",
    "auto-code": "coding",
    "auto-impl": "coding",
    "auto-review": "code_review",
    "auto-reason": "complex_reasoning",
    "auto-debug": "debugging",
    "auto-arch": "architecture",
    "auto-doc": "documentation",
}

def parse_model_routing(
    model_name: str,
    headers: dict[str, str],
) -> tuple[str | None, str, str, str]:
    """Parse model name and headers into routing parameters.

    Returns:
        (pinned_model, intent_category, stakes, complexity)
        pinned_model is None for auto-routing, or the exact model name for pinned dispatch.
    """
    # Check for auto-routing
    if model_name in AUTO_INTENT_MAP:
        intent = headers.get("x-dragonlight-intent", AUTO_INTENT_MAP[model_name])
        stakes = headers.get("x-dragonlight-stakes", "mid")
        complexity = headers.get("x-dragonlight-complexity", "standard")
        return None, intent, stakes, complexity

    if model_name.startswith("auto"):
        # Unknown auto-* variant, default to general
        intent = headers.get("x-dragonlight-intent", "general")
        stakes = headers.get("x-dragonlight-stakes", "mid")
        complexity = headers.get("x-dragonlight-complexity", "standard")
        return None, intent, stakes, complexity

    # Pinned model — dispatch directly
    return model_name, "general", "mid", "standard"
```

### 4.3 Request-to-DispatchOrder Translation

The proxy translates an OpenAI chat completions request into the existing `DispatchOrder` type:

```python
def openai_request_to_dispatch_order(
    body: dict[str, Any],
    headers: dict[str, str],
) -> DispatchOrder:
    """Convert an OpenAI-format chat completions request to a DispatchOrder."""
    model_name = body.get("model", "auto")
    pinned_model, intent_category, stakes, complexity = parse_model_routing(
        model_name, headers,
    )

    messages = body.get("messages", [])
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    # Extract the operator message from the last user message
    operator_message = ""
    system_prompt = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                operator_message = content
            elif isinstance(content, list):
                # Multi-modal content blocks — extract text
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                operator_message = "\n".join(text_parts)

    # Estimate context tokens from all message content
    total_chars = sum(
        len(m.get("content", "") or "")
        for m in messages
        if isinstance(m.get("content"), str)
    )
    context_tokens = max(1, total_chars // 4)

    return DispatchOrder(
        intent_category=intent_category,
        specific_intent=intent_category,
        operator_message=operator_message,
        system_prompt=system_prompt,
        context_tokens=context_tokens,
        requires_tool_use=tools is not None and len(tools) > 0,
        requires_long_context=context_tokens > 32_000,
        model=pinned_model,
        tools=tuple(tools) if tools else None,
        tool_choice=tool_choice,
        messages=tuple(messages),
        fallback_policy=headers.get("x-dragonlight-fallback-policy", "allow"),
        context_trust_tier=headers.get("x-dragonlight-trust-tier"),
    )
```

### 4.4 EngineResponse-to-OpenAI Translation

```python
def engine_response_to_openai(
    engine_resp: EngineResponse,
    request_model: str,
) -> dict[str, Any]:
    """Convert an EngineResponse to an OpenAI chat completions response."""
    response_id = f"chatcmpl-dragonlight-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Build the message
    message: dict[str, Any] = {"role": "assistant"}

    if engine_resp.tool_calls:
        message["content"] = engine_resp.content if engine_resp.content else None
        message["tool_calls"] = engine_resp.tool_calls
        finish_reason = engine_resp.finish_reason or "tool_calls"
    else:
        message["content"] = engine_resp.content
        finish_reason = engine_resp.finish_reason or "stop"

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": engine_resp.backend_used,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": engine_resp.tokens_in,
            "completion_tokens": engine_resp.tokens_out,
            "total_tokens": engine_resp.tokens_in + engine_resp.tokens_out,
        },
        "system_fingerprint": f"dragonlight-router-v{__version__}",
        "dragonlight": {
            "backend_used": engine_resp.backend_used,
            "backend_tier": engine_resp.backend_tier.value,
            "dispatch_mode": engine_resp.dispatch_mode,
            "was_fallback": engine_resp.was_fallback,
            "fallback_chain": engine_resp.fallback_chain,
            "estimated_cost_usd": engine_resp.estimated_cost_usd,
            "latency_ms": engine_resp.latency_ms,
        },
    }
```

### 4.5 Streaming Translation

The streaming path converts the router's `StreamChunk` sequence to OpenAI SSE format:

```python
async def stream_openai_response(
    engine: RouterEngine,
    order: DispatchOrder,
    request_model: str,
) -> AsyncIterator[str]:
    """Yield OpenAI-format SSE chunks from router dispatch_stream."""
    response_id = f"chatcmpl-dragonlight-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    actual_model = ""

    # Initial role chunk
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    async for chunk in engine.dispatch_stream(order):
        if chunk.event_type == "token" and chunk.content:
            delta_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": actual_model or request_model,
                "choices": [
                    {"index": 0, "delta": {"content": chunk.content}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(delta_chunk)}\n\n"

        elif chunk.event_type == "metadata":
            actual_model = chunk.backend_used
            # Final chunk with finish_reason and usage
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": actual_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": chunk.tokens_in,
                    "completion_tokens": chunk.tokens_out,
                    "total_tokens": chunk.tokens_in + chunk.tokens_out,
                },
                "dragonlight": {
                    "backend_used": chunk.backend_used,
                    "backend_tier": chunk.backend_tier,
                    "dispatch_mode": chunk.dispatch_mode,
                    "was_fallback": chunk.was_fallback,
                    "fallback_chain": chunk.fallback_chain or [],
                    "estimated_cost_usd": chunk.estimated_cost_usd,
                    "latency_ms": chunk.latency_ms,
                },
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

        elif chunk.event_type == "error":
            # Emit an error as a final chunk then terminate
            error_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"
```

### 4.6 Route Registration

In `app.py`, add the new route alongside existing ones:

```python
from dragonlight_router.server.routes import (
    # ... existing imports ...
    chat_completions_handler,
)

routes = [
    # ... existing routes ...
    Route("/v1/chat/completions", chat_completions_handler, methods=["POST"]),
]
```

### 4.7 Models Endpoint (Optional but Valuable)

pydantic-ai and other OpenAI-compatible clients may call `GET /v1/models` to list available models. The router should respond:

```python
async def models_handler(request: Request) -> JSONResponse:
    """GET /v1/models -- list available models (OpenAI-compatible)."""
    engine: RouterEngine = request.app.state.engine
    models = [
        {"id": "auto", "object": "model", "created": 0, "owned_by": "dragonlight-router"},
        {"id": "auto-test", "object": "model", "created": 0, "owned_by": "dragonlight-router"},
        {"id": "auto-code", "object": "model", "created": 0, "owned_by": "dragonlight-router"},
        {"id": "auto-reason", "object": "model", "created": 0, "owned_by": "dragonlight-router"},
        {"id": "auto-review", "object": "model", "created": 0, "owned_by": "dragonlight-router"},
    ]
    # Also list all registered backends as available pinned models
    for name, _backend, state in engine._registry.all_backends():
        if state.status not in (BackendStatus.RETIRED, BackendStatus.KEY_INVALID):
            models.append({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "dragonlight-router",
            })
    return JSONResponse({"object": "list", "data": models})
```

Register as:
```python
Route("/v1/models", models_handler, methods=["GET"]),
```

---

## 5. Streaming Support

### 5.1 How It Works

1. pydantic-ai sends `{"stream": true, ...}` in the request body
2. The proxy handler detects `stream=true` and returns a `StreamingResponse` with `media_type="text/event-stream"`
3. The response generator calls `engine.dispatch_stream(order)` which yields `StreamChunk` objects
4. Each `StreamChunk` is translated to an OpenAI-format SSE delta chunk (see Section 4.5)
5. The final `[DONE]` sentinel is sent after the metadata chunk

### 5.2 Cascade Fallback During Streaming

The existing `dispatch_stream()` already handles cascade fallback: if backend A fails mid-stream, it falls back to backend B and starts streaming from scratch. The proxy is transparent to this -- it simply converts whatever `StreamChunk` events arrive.

There is one edge case: if backend A yields some tokens then fails, those tokens have already been sent to the client. The stream fallback in `_stream_with_fallback()` currently starts the next backend from scratch, which means the client may see partial content from A followed by complete content from B. This is an existing behavior in the router's streaming dispatch and is not introduced by the proxy.

### 5.3 Tool-Use and Streaming

The current router architecture uses non-streaming dispatch for tool-use requests (when `tools` is present in the DispatchOrder). The proxy should follow the same pattern:

- If `stream=true` AND `tools` is present in the request body: use non-streaming dispatch internally, then return the result as if it were a single-chunk stream (matching what OpenAI does for tool_calls responses).
- If `stream=true` AND no `tools`: use streaming dispatch, translate to SSE chunks.

This matches the existing behavior in `router_model.py` where `request_stream()` uses the text-only path and `request()` handles tool-use.

---

## 6. Error Handling

### 6.1 All Models Exhausted

When the cascade exhausts all backends, the router returns an OpenAI-compatible error response:

```json
{
  "error": {
    "message": "All 5 backends exhausted. Fallback chain: nvidia_nim/deepseek-v4-pro -> groq/llama-3.3-70b-versatile -> ...",
    "type": "server_error",
    "param": null,
    "code": "all_backends_exhausted"
  }
}
```

HTTP status: **503 Service Unavailable**

pydantic-ai handles 5xx errors with its built-in retry logic, so the factory gets transparent retry behavior.

### 6.2 Invalid Model Name (Pinned)

When a pinned model name is not found in the registry:

```json
{
  "error": {
    "message": "Model 'nvidia_nim/nonexistent' not found in registry",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

HTTP status: **400 Bad Request**

### 6.3 Budget Exhausted

```json
{
  "error": {
    "message": "Budget exhausted for provider 'nvidia_nim'",
    "type": "rate_limit_error",
    "param": null,
    "code": "budget_exhausted"
  }
}
```

HTTP status: **429 Too Many Requests**

### 6.4 Validation Errors

Missing required fields, invalid JSON, etc.:

```json
{
  "error": {
    "message": "Missing required field: messages",
    "type": "invalid_request_error",
    "param": "messages",
    "code": null
  }
}
```

HTTP status: **400 Bad Request**

---

## 7. Provider-Specific Normalization

### 7.1 The Problem

Different providers have different quirks:
- Some do not support `temperature` (or only accept certain values)
- Some have different tool call response formats
- Some have different max token limits
- Anthropic uses a different API format entirely (Messages API vs Chat Completions)

### 7.2 The Solution: Existing Adapter Layer

The router already solves this problem. Each provider has an adapter in `src/dragonlight_router/adapters/` that inherits from `OpenAICompatibleBackend`. These adapters:

- Know how to construct the correct request payload for their provider
- Handle auth header differences
- Handle endpoint path differences
- Handle streaming response format differences
- Handle tool call response normalization

The proxy does NOT need to normalize requests per-provider. It passes the OpenAI-format messages and tools through the `DispatchOrder`, and the cascade's `_try_adapter_dispatch()` uses the adapter's `generate()` / `generate_with_tools()` methods which handle all provider-specific translation.

The one new responsibility is **response normalization**: the proxy must convert the `EngineResponse` (which is already provider-normalized by the adapter layer) back to OpenAI format. This is a single, clean translation (Section 4.4).

### 7.3 Anthropic Specifics

The Anthropic adapter already translates between OpenAI-format messages and the Anthropic Messages API format internally. The proxy treats Anthropic the same as any other provider -- it sends an OpenAI-format request, the cascade selects the Anthropic backend, the Anthropic adapter translates to Messages API format, calls the API, and translates the response back. The proxy never sees Anthropic-specific details.

---

## 8. Multi-Turn Consistency

### 8.1 The Concern

In a multi-turn tool-use conversation, turn 1 might use Model A and turn 2 might use Model B. Does this cause coherence issues?

### 8.2 Why It Is Not a Problem

Each turn of a pydantic-ai agent loop sends the **full conversation history** in every request. Turn 2 includes all messages from turn 1 (user message, assistant response with tool_calls, tool results). The new model sees the complete context and can continue coherently.

This is the same behavior as OpenAI's own API -- there is no server-side session state. Every request is self-contained.

### 8.3 Optional: Model Pinning for Consistency

If a caller wants consistency within a single agent run, they can use a pinned model name:

```python
model = OpenAIChatModel("nvidia_nim/deepseek-v4-pro", provider=...)
```

This bypasses the cascade and uses the same model for every turn. The factory could set this per-agent-run if coherence issues are observed in practice.

---

## 9. Automatic Outcome Recording

### 9.1 How It Works Today

With the current architecture (factory calls router_bridge), the factory must explicitly call `router_bridge.record_outcome()` after every model attempt. This is error-prone and creates coupling.

### 9.2 How It Works With the Proxy

The proxy endpoint records outcomes automatically as part of the dispatch pipeline. The existing `_record_dispatch_success()` and `_record_adapter_failure()` functions in `cascade.py` already handle this. The factory never calls a separate record endpoint.

Health tracking, budget tracking, latency recording, and token counting all happen transparently inside `dispatch()` / `dispatch_stream()`.

The `/v1/record` endpoint remains available for external callers or manual corrections, but the factory will not use it.

---

## 10. Factory Changes

### 10.1 What to Delete

The following files, functions, constants, imports, and env vars become unnecessary once the proxy is the sole communication mechanism:

#### Files to Delete

| File | Reason |
|---|---|
| `scripts/router_model.py` | Entire file. The RouterModel pydantic-ai adapter is replaced by `OpenAIChatModel("auto", provider=OpenAIProvider(base_url=...))`. |
| `scripts/router_bridge.py` | Entire file. The factory no longer needs a bridge to the router -- it just speaks HTTP to the proxy endpoint. |

#### Functions to Delete from `scripts/factory.py`

| Function | Lines (approx) | Reason |
|---|---|---|
| `_is_router_dispatch_enabled()` | 101-111 | No feature flag needed -- proxy is always-on |
| `_run_via_router()` | 3964-4050 | Replaced by standard pydantic-ai agent invocation with proxy URL |
| `_select_next_model()` | 2714-2748 | Router handles model selection transparently |
| `select_model_for_ticket()` | 995-1025 | Router handles model selection transparently |
| `_resolve_model_chain()` | 2776-2809 | Router handles fallback chain internally |
| `_record_model_outcome()` | 2751-2772 | Router records outcomes automatically |
| `_ticket_to_intent()` | 2703-2711 | Intent communicated via model name or headers |
| `_ticket_to_role()` | 2685-2700 | Role communicated via model name or headers |
| `parse_model_chain()` | 986-992 | No more AGENT_MODELS env var chain |

#### Functions to Delete from `scripts/coding_agent.py`

| Function | Lines (approx) | Reason |
|---|---|---|
| `_create_model()` | 1267-1386 | Giant if/elif chain for provider-specific wiring. Replaced by one-line OpenAIChatModel construction. |

#### Imports to Remove from `scripts/factory.py`

```python
# Remove these:
from router_model import create_router_model  # lines 93-95
import router_bridge  # line 98
```

#### Imports to Remove from `scripts/coding_agent.py`

```python
# Remove or reduce these:
from pydantic_ai.models.anthropic import AnthropicModel  # line 29
from pydantic_ai.providers.openai import OpenAIProvider  # line 33 (keep for proxy)
# Remove: all provider-specific imports (OpenRouterModel, AnthropicProvider, etc.)
```

#### Environment Variables to Deprecate

| Env Var | Reason |
|---|---|
| `ROUTER_DISPATCH` | No feature flag needed -- proxy is always-on |
| `AGENT_MODEL` | Model selection is router's job |
| `AGENT_MODELS` | Model selection is router's job |
| `DRAGONLIGHT_ROUTER_CONFIG` | Factory no longer loads router config (communicates via HTTP) |
| `DRAGONLIGHT_ROUTER_STATE_DIR` | Factory no longer manages router state |
| Individual provider API keys in factory | API keys live on the router only |
| `GROQ_RPM_LIMIT`, `NIM_RPM_LIMIT` | Rate limiting is router's job |

#### Environment Variables to Add

| Env Var | Default | Purpose |
|---|---|---|
| `DRAGONLIGHT_ROUTER_URL` | `http://localhost:8100` | The one URL the factory needs |

### 10.2 What the Factory Looks Like After

The entire model construction in the factory reduces to:

```python
import os
import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROUTER_URL = os.environ.get("DRAGONLIGHT_ROUTER_URL", "http://localhost:8100")

def _create_model_for_task(
    intent: str = "coding",
    stakes: str = "mid",
    complexity: str = "standard",
) -> OpenAIChatModel:
    """Create a pydantic-ai model that routes through the Dragonlight router proxy."""
    return OpenAIChatModel(
        "auto",
        provider=OpenAIProvider(
            base_url=f"{ROUTER_URL}/v1",
            api_key="dragonlight",  # not validated, placeholder for SDK requirement
            http_client=httpx.AsyncClient(
                headers={
                    "X-Dragonlight-Intent": intent,
                    "X-Dragonlight-Stakes": stakes,
                    "X-Dragonlight-Complexity": complexity,
                },
                timeout=httpx.Timeout(300.0),
            ),
        ),
    )
```

Every call site that currently uses `_create_model()`, `_run_via_router()`, or `_select_next_model()` becomes:

```python
model = _create_model_for_task(intent="test_generation", stakes="low")
result = run_coding_agent(prompt, target_dir, model=model, ...)
```

No fallback chain management. No outcome recording. No model probing. No provider-specific wiring.

---

## 11. Migration Path

### Phase 1: Build the Proxy Endpoint (Router Side)

**Goal:** `/v1/chat/completions` works end-to-end for non-streaming, text-only requests.

1. Create `src/dragonlight_router/server/openai_proxy.py` with:
   - `parse_model_routing()` -- model name and header parsing
   - `openai_request_to_dispatch_order()` -- request translation
   - `engine_response_to_openai()` -- response translation
   - `chat_completions_handler()` -- the Starlette route handler
2. Register the route in `app.py`
3. Add `/v1/models` endpoint
4. Write tests:
   - Unit tests for request/response translation
   - Integration test: send an OpenAI-format request, verify OpenAI-format response
   - Pinned model dispatch test
   - Auto-routing intent parsing test

**No factory changes. Existing dispatch/select paths unaffected.**

### Phase 2: Add Streaming and Tool-Use

**Goal:** Full feature parity with the existing `router_model.py` adapter.

1. Add `stream_openai_response()` SSE generator
2. Handle `stream=true` in the handler
3. Handle tool-use requests (forward tools/tool_choice through DispatchOrder)
4. Handle tool-use responses (convert tool_calls in EngineResponse to OpenAI format)
5. Write tests:
   - Streaming SSE format test
   - Tool-use round-trip test
   - Streaming + tool-use behavior test (non-streaming internally, chunked response externally)

**No factory changes yet.**

### Phase 3: Factory Integration (Parallel Path)

**Goal:** Factory can use the proxy alongside the existing path.

1. Add `DRAGONLIGHT_ROUTER_URL` env var to factory
2. Create `_create_model_for_task()` function using OpenAIChatModel + proxy URL
3. Add a feature flag: `PROXY_DISPATCH=1` to opt in
4. When `PROXY_DISPATCH=1`, all agent invocations use the proxy path
5. Keep existing `router_model.py` / `router_bridge.py` paths as fallback
6. Run factory builds with both paths, compare results

### Phase 4: Factory Cleanup

**Goal:** Delete all model selection logic from the factory.

Only after Phase 3 is validated in production:

1. Delete `scripts/router_model.py`
2. Delete `scripts/router_bridge.py`
3. Remove `_run_via_router()`, `_select_next_model()`, `_resolve_model_chain()`, `_record_model_outcome()`, `_ticket_to_intent()`, `_ticket_to_role()`, `select_model_for_ticket()`, `parse_model_chain()`
4. Remove `_create_model()` from `coding_agent.py`
5. Simplify all call sites to use `_create_model_for_task()`
6. Remove deprecated env vars from Dockerfile / docker-compose
7. Move all provider API keys from factory env to router env only
8. Remove `ROUTER_DISPATCH` feature flag
9. Remove `router_bridge` import

### Phase 5: Remove In-Process Router Dependency

**Goal:** Factory no longer imports `dragonlight_router` as a Python package.

1. Remove `dragonlight-router` from factory's `pyproject.toml` / `requirements.txt`
2. The factory and router are now fully decoupled processes communicating over HTTP
3. They can be deployed, scaled, and versioned independently

---

## 12. Risk Assessment

### 12.1 Network Latency

**Risk:** Adding an HTTP hop between factory and router adds latency.
**Mitigation:** The router runs on localhost (same machine). HTTP overhead is ~0.1-0.5ms per request, negligible compared to LLM call latency (1-60 seconds). If deployed across machines, the latency is still far below LLM response time.
**Severity:** Low.

### 12.2 Single Point of Failure

**Risk:** If the router process crashes, the factory cannot make any LLM calls.
**Mitigation:** The router is already a dependency in the current architecture (imported as a Python package). The HTTP boundary makes failure modes more explicit and easier to monitor (health checks, readiness probes). Process supervision (systemd, Docker restart policy) handles crashes.
**Severity:** Medium. Mitigated by process supervision and `/v1/health` monitoring.

### 12.3 Request Body Size Limits

**Risk:** Large context windows (100K+ tokens) produce very large request bodies. The `RequestBodySizeLimitMiddleware` could reject them.
**Mitigation:** Configure the body size limit to accommodate the largest expected request (e.g., 50MB). The middleware already exists and is configurable.
**Severity:** Low.

### 12.4 Streaming Connection Drops

**Risk:** Long-running streaming connections could drop due to proxy timeouts, network interrupts, etc.
**Mitigation:** This is an existing risk with direct provider calls. The proxy does not add new timeout surfaces. The `X-Accel-Buffering: no` header is already set in the existing streaming handler. httpx client timeout is configurable.
**Severity:** Low.

### 12.5 pydantic-ai Version Coupling

**Risk:** pydantic-ai changes its OpenAI client behavior (request format, header handling, etc.) and breaks compatibility.
**Mitigation:** The proxy speaks standard OpenAI protocol, not pydantic-ai-specific protocol. Any OpenAI-compatible client should work. If pydantic-ai changes its OpenAI implementation, it will also break against real OpenAI, so it would be caught upstream.
**Severity:** Very Low.

### 12.6 API Key Placeholder

**Risk:** pydantic-ai's OpenAIProvider requires an `api_key` parameter. Passing a placeholder string works today but could break if pydantic-ai adds key validation.
**Mitigation:** The router can accept and ignore any `Authorization: Bearer <token>` header. If pydantic-ai starts validating keys, we can set a known dummy key that the router recognizes. Alternatively, the router could implement a simple API key for the proxy endpoint (which would also add security).
**Severity:** Low.

### 12.7 Partial Stream on Fallback

**Risk:** If backend A streams 500 tokens then fails, those tokens are already sent to the client. Backend B starts from scratch, so the client sees partial content from A followed by complete content from B.
**Mitigation:** This is an existing behavior in `dispatch_stream()`. The proxy does not change it. In practice, mid-stream failures are rare (most failures happen at connection time, before tokens flow). A future improvement could buffer the first few tokens before committing to a backend.
**Severity:** Low (existing behavior, not introduced by this change).

### 12.8 Concurrent Request Isolation

**Risk:** Multiple factory workers hitting the proxy simultaneously could cause shared-state issues.
**Mitigation:** The router already handles concurrency. Each request gets a fresh adapter instance (HAZ-014). Budget and health trackers use thread-safe SQLite databases. The proxy handler is stateless.
**Severity:** Very Low.

---

## 13. Testing Strategy

### 13.1 Unit Tests

| Test | What It Verifies |
|---|---|
| `test_parse_model_routing_auto` | "auto" maps to intent_category="general" |
| `test_parse_model_routing_auto_test` | "auto-test" maps to intent_category="test_generation" |
| `test_parse_model_routing_pinned` | "nvidia_nim/deepseek-v4-pro" passes through as pinned |
| `test_parse_model_routing_header_override` | X-Dragonlight-Intent header overrides model-name-derived intent |
| `test_openai_request_to_dispatch_order` | Full request body translates to correct DispatchOrder fields |
| `test_openai_request_with_tools` | Tools and tool_choice are forwarded correctly |
| `test_engine_response_to_openai_text` | Text-only EngineResponse maps to OpenAI format |
| `test_engine_response_to_openai_tool_calls` | Tool-call EngineResponse maps to OpenAI format with tool_calls |
| `test_stream_chunk_to_openai_delta` | StreamChunk(event_type="token") maps to delta chunk |
| `test_stream_metadata_to_usage` | StreamChunk(event_type="metadata") maps to usage+finish |
| `test_error_response_format` | DispatchFailure maps to OpenAI error format |
| `test_budget_exhausted_error_format` | BudgetExhaustedError maps to 429 with OpenAI error format |

### 13.2 Integration Tests

| Test | What It Verifies |
|---|---|
| `test_chat_completions_round_trip` | POST to /v1/chat/completions returns a valid OpenAI response (using mock adapter) |
| `test_chat_completions_streaming` | POST with stream=true returns valid SSE chunks ending in [DONE] |
| `test_chat_completions_tool_use` | Tools in request body produce tool_calls in response |
| `test_chat_completions_auto_model` | "auto" triggers cascade selection (not pinned) |
| `test_chat_completions_pinned_model` | Exact model name triggers pinned dispatch |
| `test_chat_completions_cascade_fallback` | Primary backend fails, fallback succeeds, client gets response |
| `test_chat_completions_all_backends_fail` | All backends fail, client gets 503 with error body |
| `test_models_endpoint` | GET /v1/models returns list including auto-* and registered backends |

### 13.3 End-to-End Tests

| Test | What It Verifies |
|---|---|
| `test_pydantic_ai_via_proxy` | pydantic-ai Agent with OpenAIChatModel("auto", ...) completes a request through the proxy |
| `test_pydantic_ai_tool_use_via_proxy` | pydantic-ai Agent with tools completes a multi-turn tool-use conversation through the proxy |
| `test_factory_build_via_proxy` | Full factory build with `PROXY_DISPATCH=1` succeeds end-to-end |

### 13.4 Compatibility Tests

| Test | What It Verifies |
|---|---|
| `test_openai_python_client` | Standard `openai` Python package works against the proxy |
| `test_curl_request` | Manual curl request produces valid response (smoke test) |

### 13.5 Regression Tests

All existing `/v1/dispatch` and `/v1/select` tests must continue to pass unchanged. The proxy is additive -- it does not modify any existing endpoint behavior.

---

## 14. Implementation Sizing

| Component | Estimated Lines | Complexity |
|---|---|---|
| `openai_proxy.py` (new module) | ~350 | Medium -- mostly translation logic |
| Route registration in `app.py` | ~5 | Trivial |
| `/v1/models` handler | ~30 | Low |
| Unit tests | ~400 | Medium |
| Integration tests | ~300 | Medium |
| Factory `_create_model_for_task()` | ~30 | Low |
| Factory cleanup (deletions) | -800 (net reduction) | Low but tedious |
| **Total new code** | **~1100** | |
| **Total code removed from factory** | **~800** | |

---

## 15. Open Questions

1. **Authentication on the proxy endpoint:** Should the proxy require an API key? Currently the router's dispatch endpoint does not require auth. Adding a key would prevent unauthorized LLM usage if the router is network-accessible. For localhost-only deployment, no auth is fine.

2. **Rate limiting the proxy:** The existing `RateLimitMiddleware` applies to all endpoints. Should the proxy have its own rate limit, or is the per-provider rate limiting in the cascade sufficient?

3. **Caching through the proxy:** The existing `_dispatch_cache` in cascade.py handles caching transparently. Should the proxy add an HTTP-level cache (e.g., `Cache-Control` headers) or is the internal cache sufficient?

4. **Model name in streaming response:** The first SSE chunk uses the requested model name ("auto"), but subsequent chunks and the final chunk use the actual backend model name. Is this acceptable, or should all chunks use the same model name? OpenAI uses the actual model name in all chunks.

5. **Maximum auto-* variants:** How many auto-* variants do we need? The initial set (auto, auto-test, auto-code, auto-reason, auto-review) covers the factory's current intent categories. Adding more is trivial (one line in the lookup table).

---

## 16. Decision Record

This section will be filled in as decisions are made during implementation.

| # | Decision | Date | Rationale |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
