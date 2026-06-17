"""Tests for the Cerebras backend adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from dragonlight_router.adapters.cerebras import CerebrasBackend, _DEFAULT_BASE_URL
from dragonlight_router.core.types import BackendStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_lines(*chunks: str) -> list[str]:
    """Build raw SSE lines from content strings (OpenAI-compatible format)."""
    lines: list[str] = []
    for i, text in enumerate(chunks):
        data = {
            "id": f"chatcmpl-{i}",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return lines


def _stream_response(status_code: int, sse_lines: list[str]) -> httpx.Response:
    """Create a streaming-capable httpx.Response backed by raw SSE text."""
    body = "\n".join(sse_lines).encode()
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream(body),
    )


def _make_cerebras_backend(
    make_backend_config,
    transport: httpx.MockTransport | None = None,
    *,
    env_key: str = "CEREBRAS_API_KEY",
    env_value: str | None = "csk-test-key",
) -> CerebrasBackend:
    """Build a CerebrasBackend with optional transport injection."""
    config = make_backend_config(
        name="cerebras-test",
        provider="cerebras",
        model="llama3.1-8b",
        base_url="https://api.cerebras.ai",
        env_key=env_key,
    )
    env = {env_key: env_value} if env_value else {}
    with patch.dict("os.environ", env, clear=True):
        return CerebrasBackend(config, _transport=transport)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(make_backend_config):
    """CerebrasBackend wired to a fake API key via env (no transport)."""
    config = make_backend_config(
        name="cerebras-test",
        provider="cerebras",
        model="llama3.1-8b",
        base_url="https://api.cerebras.ai",
        env_key="CEREBRAS_API_KEY",
    )
    with patch.dict("os.environ", {"CEREBRAS_API_KEY": "csk-test-key"}):
        yield CerebrasBackend(config)


@pytest.fixture
def backend_no_key(make_backend_config):
    """CerebrasBackend with no API key configured."""
    config = make_backend_config(
        name="cerebras-test",
        provider="cerebras",
        model="llama3.1-8b",
        base_url="https://api.cerebras.ai",
        env_key="CEREBRAS_API_KEY",
    )
    with patch.dict("os.environ", {}, clear=True):
        yield CerebrasBackend(config)


# ---------------------------------------------------------------------------
# generate() -- streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_streaming_success(make_backend_config):
    """Streaming generate yields text chunks from SSE delta events."""
    sse = _sse_lines("Hello", " world")

    transport = httpx.MockTransport(
        lambda request: _stream_response(200, sse)
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    chunks: list[str] = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


# ---------------------------------------------------------------------------
# generate() -- non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_non_streaming(make_backend_config):
    """Non-streaming generate yields full message content."""
    response_body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Cerebras"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    chunks: list[str] = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=False,
    ):
        chunks.append(chunk)

    assert chunks == ["Hello from Cerebras"]


# ---------------------------------------------------------------------------
# generate() -- error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_no_api_key(backend_no_key):
    """generate raises ValueError when no API key is configured."""
    with pytest.raises(ValueError, match="API key not configured"):
        async for _ in backend_no_key.generate(
            [{"role": "user", "content": "Hi"}],
        ):
            pass
    assert backend_no_key.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_401_unauthorized(make_backend_config):
    """401 from the API raises RuntimeError and sets ERROR status."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    with pytest.raises(RuntimeError, match="Cerebras API error"):
        async for _ in backend.generate(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_429_rate_limited(make_backend_config):
    """429 from the API raises RuntimeError and sets ERROR status."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            429, json={"error": {"message": "rate limited"}}
        )
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    with pytest.raises(RuntimeError, match="Cerebras API error"):
        async for _ in backend.generate(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_500_server_error(make_backend_config):
    """500 from the API raises RuntimeError and sets ERROR status."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            500, json={"error": {"message": "internal error"}}
        )
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    with pytest.raises(RuntimeError, match="Cerebras API error"):
        async for _ in backend.generate(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass
    assert backend.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_success(make_backend_config):
    """health_check returns True on 200 and sets AVAILABLE status."""
    response_body = {
        "object": "list",
        "data": [{"id": "llama3.1-8b", "object": "model"}],
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is True
    assert backend.status == BackendStatus.AVAILABLE


@pytest.mark.asyncio
async def test_health_check_failure(make_backend_config):
    """health_check returns False on 401."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    backend = _make_cerebras_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is False
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_no_api_key(backend_no_key):
    """health_check returns False when no API key is set."""
    result = await backend_no_key.health_check()
    assert result is False
    assert backend_no_key.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_connection_error(make_backend_config):
    """health_check returns False on connection failure."""

    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(_raise)
    backend = _make_cerebras_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is False
    assert backend.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# Default base URL
# ---------------------------------------------------------------------------


def test_default_base_url():
    """The default Cerebras base URL is correct."""
    assert _DEFAULT_BASE_URL == "https://api.cerebras.ai"


# ---------------------------------------------------------------------------
# Auth header format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_header_format(make_backend_config):
    """Requests use Bearer token auth with the configured API key."""
    sse = _sse_lines("ok")
    captured_request: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_request.append(request)
        body = "\n".join(sse).encode()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(body),
        )

    transport = httpx.MockTransport(_handler)
    backend = _make_cerebras_backend(make_backend_config, transport)

    chunks = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks == ["ok"]
    assert captured_request[0].headers["authorization"] == "Bearer csk-test-key"


# ---------------------------------------------------------------------------
# record_usage() / properties
# ---------------------------------------------------------------------------


def test_record_usage_noop(backend):
    """record_usage is a no-op but does not raise."""
    backend.record_usage(100, 50)  # should not raise


def test_properties(backend):
    """config and status properties return expected values."""
    assert backend.config.name == "cerebras-test"
    assert backend.config.provider == "cerebras"
    assert backend.config.model == "llama3.1-8b"
    assert backend.status == BackendStatus.AVAILABLE
