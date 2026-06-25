"""OpenAI-compatible proxy layer for the Dragonlight Router.

Translates between OpenAI chat completions API format and the router's
internal DispatchOrder/EngineResponse types. Enables any OpenAI SDK
client to use the router as a drop-in backend.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from dragonlight_router import __version__
from dragonlight_router.core.types import (
    BudgetExhaustedError,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    ModelNotFoundError,
    ModelUnhealthyError,
    StreamChunk,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.router import RouterEngine

logger = structlog.get_logger()

# --- Auto-intent routing map ---

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
    """Determine routing parameters from model name and request headers.

    Returns:
        (pinned_model, intent, stakes, complexity) — pinned_model is None
        for auto-routed requests.
    """
    assert isinstance(model_name, str) and model_name, "model_name must be non-empty"

    if model_name in AUTO_INTENT_MAP:
        intent = headers.get("x-dragonlight-intent", AUTO_INTENT_MAP[model_name])
        stakes = headers.get("x-dragonlight-stakes", "mid")
        complexity = headers.get("x-dragonlight-complexity", "standard")
        return None, intent, stakes, complexity

    if model_name.startswith("auto"):
        intent = headers.get("x-dragonlight-intent", "general")
        stakes = headers.get("x-dragonlight-stakes", "mid")
        complexity = headers.get("x-dragonlight-complexity", "standard")
        return None, intent, stakes, complexity

    return model_name, "general", "mid", "standard"


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """Concatenate all system messages into a single system prompt."""
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
    return "\n\n".join(parts)


def _extract_operator_message(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message as the operator message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(text_parts)
    return ""


def _estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: total chars / 4."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
        role = msg.get("role", "")
        total_chars += len(role)
    return max(1, total_chars // 4)


def openai_request_to_dispatch_order(
    body: dict[str, Any],
    headers: dict[str, str],
) -> DispatchOrder:
    """Convert an OpenAI chat completions request body to a DispatchOrder."""
    messages = body.get("messages", [])
    assert isinstance(messages, list) and len(messages) > 0, "messages must be non-empty list"

    model_name = body.get("model", "auto")
    pinned_model, intent, _stakes, _complexity = parse_model_routing(model_name, headers)

    system_prompt = _extract_system_prompt(messages)
    operator_message = _extract_operator_message(messages)
    context_tokens = _estimate_context_tokens(messages)

    tools_raw = body.get("tools")
    tools = tuple(tools_raw) if tools_raw else None
    tool_choice = body.get("tool_choice")

    fallback_policy = headers.get("x-dragonlight-fallback-policy", "allow")

    return DispatchOrder(
        intent_category=intent,
        specific_intent=intent,
        operator_message=operator_message,
        system_prompt=system_prompt,
        context_tokens=context_tokens,
        requires_tool_use=tools is not None,
        model=pinned_model,
        tools=tools,
        tool_choice=tool_choice,
        messages=tuple(messages),
        fallback_policy=fallback_policy,
        min_output_tokens=0,
    )


def _generate_completion_id() -> str:
    """Generate an OpenAI-style completion ID."""
    return f"chatcmpl-dragonlight-{uuid.uuid4().hex[:12]}"


def engine_response_to_openai(
    engine_resp: EngineResponse,
    request_model: str,
) -> dict[str, Any]:
    """Convert an EngineResponse to OpenAI chat completions response format."""
    assert isinstance(engine_resp, EngineResponse), "engine_resp must be EngineResponse"

    message: dict[str, Any] = {"role": "assistant"}

    if engine_resp.tool_calls:
        message["content"] = engine_resp.content or None
        message["tool_calls"] = engine_resp.tool_calls
    else:
        message["content"] = engine_resp.content

    finish_reason = engine_resp.finish_reason or (
        "tool_calls" if engine_resp.tool_calls else "stop"
    )

    return {
        "id": _generate_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine_resp.backend_used,
        "system_fingerprint": f"dragonlight-router-v{__version__}",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            },
        ],
        "usage": {
            "prompt_tokens": engine_resp.tokens_in,
            "completion_tokens": engine_resp.tokens_out,
            "total_tokens": engine_resp.tokens_in + engine_resp.tokens_out,
        },
        "dragonlight": {
            "backend_used": engine_resp.backend_used,
            "backend_tier": engine_resp.backend_tier.value,
            "dispatch_mode": engine_resp.dispatch_mode,
            "was_fallback": engine_resp.was_fallback,
            "fallback_chain": engine_resp.fallback_chain,
            "estimated_cost_usd": engine_resp.estimated_cost_usd,
            "latency_ms": engine_resp.latency_ms,
            "router_version": __version__,
            "request_model": request_model,
        },
    }


def _format_sse_chunk(data: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# DEVIATION CS-004: stream_openai_response is 72 lines.
# Justification: Streaming generator with three phases (initial role chunk,
# content/tool_call deltas, final metadata chunk) plus error handling. Splitting
# would fragment the SSE contract and scatter the chunk-type state machine.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
async def stream_openai_response(
    engine: RouterEngine,
    order: DispatchOrder,
    request_model: str,
) -> AsyncIterator[str]:
    """Yield OpenAI-format SSE chunks from a streaming dispatch."""
    completion_id = _generate_completion_id()
    created = int(time.time())

    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
    }

    # Initial chunk: role declaration
    initial = {
        **base,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield _format_sse_chunk(initial)

    final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_backend = request_model
    final_metadata: dict[str, Any] = {}

    try:
        async for chunk in engine.dispatch_stream(order):
            if chunk.event_type == "token" and chunk.content:
                delta_chunk = {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {"content": chunk.content}, "finish_reason": None}
                    ],
                }
                yield _format_sse_chunk(delta_chunk)

            elif chunk.event_type == "metadata":
                final_usage = {
                    "prompt_tokens": chunk.tokens_in,
                    "completion_tokens": chunk.tokens_out,
                    "total_tokens": chunk.tokens_in + chunk.tokens_out,
                }
                final_backend = chunk.backend_used or request_model
                final_metadata = {
                    "backend_tier": chunk.backend_tier,
                    "dispatch_mode": chunk.dispatch_mode,
                    "was_fallback": chunk.was_fallback,
                    "fallback_chain": chunk.fallback_chain or [],
                    "estimated_cost_usd": chunk.estimated_cost_usd,
                    "latency_ms": chunk.latency_ms,
                    "router_version": __version__,
                    "request_model": request_model,
                }

                finish_reason = chunk.finish_reason or (
                    "tool_calls" if chunk.tool_calls else "stop"
                )

                if chunk.tool_calls:
                    tool_delta_chunk = {
                        **base,
                        "model": final_backend,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": chunk.tool_calls},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield _format_sse_chunk(tool_delta_chunk)

                final_chunk = {
                    **base,
                    "model": final_backend,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": final_usage,
                    "dragonlight": final_metadata,
                }
                yield _format_sse_chunk(final_chunk)

            elif chunk.event_type == "error":
                error_chunk = {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "error": {"message": chunk.error_message, "type": "server_error"},
                }
                yield _format_sse_chunk(error_chunk)

    except (RuntimeError, ConnectionError, ValueError, TypeError, OSError) as exc:
        logger.error("openai_stream_failed", error=str(exc), exc_info=True)
        error_chunk = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "error": {"message": "Internal server error", "type": "server_error"},
        }
        yield _format_sse_chunk(error_chunk)

    yield "data: [DONE]\n\n"


def format_openai_error(
    message: str,
    error_type: str,
    status_code: int,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Return an OpenAI-compatible error response."""
    error_body: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    return JSONResponse({"error": error_body}, status_code=status_code)


def _dispatch_failure_to_openai_error(error: object) -> JSONResponse:
    """Map router error types to OpenAI-compatible error responses."""
    if isinstance(error, ModelNotFoundError):
        return format_openai_error(
            message=error.message,
            error_type="invalid_request_error",
            status_code=400,
            param="model",
            code="model_not_found",
        )
    if isinstance(error, ModelUnhealthyError):
        return format_openai_error(
            message=error.message,
            error_type="server_error",
            status_code=503,
            code="model_unavailable",
        )
    if isinstance(error, BudgetExhaustedError):
        return format_openai_error(
            message=error.message,
            error_type="rate_limit_error",
            status_code=429,
            code="budget_exhausted",
        )
    if isinstance(error, DispatchFailure):
        return format_openai_error(
            message=error.message,
            error_type="server_error",
            status_code=500,
            code="dispatch_failed",
        )
    return format_openai_error(
        message="Internal server error",
        error_type="server_error",
        status_code=500,
    )


async def chat_completions_handler(request: Request) -> JSONResponse | StreamingResponse:
    """POST /v1/chat/completions — OpenAI-compatible chat completions endpoint."""
    engine: RouterEngine = request.app.state.engine

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return format_openai_error(
            message="Invalid JSON body",
            error_type="invalid_request_error",
            status_code=400,
        )

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return format_openai_error(
            message="'messages' is required and must be a non-empty array",
            error_type="invalid_request_error",
            status_code=400,
            param="messages",
        )

    headers = dict(request.headers)
    request_model = body.get("model", "auto")

    try:
        order = openai_request_to_dispatch_order(body, headers)
    except (AssertionError, KeyError, TypeError) as exc:
        return format_openai_error(
            message=str(exc),
            error_type="invalid_request_error",
            status_code=400,
        )

    if body.get("stream", False):
        return StreamingResponse(
            stream_openai_response(engine, order, request_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
            },
        )

    try:
        dispatch_result = await engine.dispatch(order)
    except (RuntimeError, ConnectionError, ValueError, TypeError, OSError) as exc:
        logger.error("openai_dispatch_failed", error=str(exc), exc_info=True)
        return format_openai_error(
            message="Internal server error",
            error_type="server_error",
            status_code=500,
        )

    if isinstance(dispatch_result, Ok):
        response_body = engine_response_to_openai(dispatch_result.value, request_model)
        return JSONResponse(response_body)

    return _dispatch_failure_to_openai_error(dispatch_result.error)


def _build_auto_models() -> list[dict[str, Any]]:
    """Build the list of auto-* virtual models."""
    created = int(time.time())
    models = []
    for model_id in sorted(AUTO_INTENT_MAP.keys()):
        models.append({
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "dragonlight",
        })
    return models


async def models_handler(request: Request) -> JSONResponse:
    """GET /v1/models — list available models in OpenAI format."""
    engine: RouterEngine = request.app.state.engine

    models = _build_auto_models()

    for name, backend, state in engine._registry.all_backends():
        models.append({
            "id": backend.config.name,
            "object": "model",
            "created": 0,
            "owned_by": backend.config.provider,
        })

    return JSONResponse({
        "object": "list",
        "data": models,
    })
