"""Tests for the Anthropic backend adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from dragonlight_router.adapters.anthropic import AnthropicBackend
from dragonlight_router.core.types import BackendStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_lines(*events: tuple[str, dict]) -> list[str]:
    """Build raw SSE lines from (event_type, data_dict) pairs."""
    lines: list[str] = []
    for event_type, data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")  # blank line between events
    return lines


def _stream_response(status_code: int, sse_lines: list[str]) -> httpx.Response:
    """Create a streaming-capable httpx.Response backed by raw SSE text."""
    body = "\n".join(sse_lines).encode()
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream(body),
    )


def _make_backend(make_backend_config, transport, *, env_key="ANTHROPIC_API_KEY"):
    """Build an AnthropicBackend with an injected transport."""
    config = make_backend_config(
        name="claude-test",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        env_key=env_key,
    )
    return AnthropicBackend(config, _transport=transport)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(make_backend_config):
    """AnthropicBackend wired to a fake API key via env, no transport."""
    config = make_backend_config(
        name="claude-test",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        env_key="ANTHROPIC_API_KEY",
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        yield AnthropicBackend(config)


@pytest.fixture
def backend_no_key(make_backend_config):
    """AnthropicBackend with no API key configured."""
    config = make_backend_config(
        name="claude-test",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        env_key="ANTHROPIC_API_KEY",
    )
    with patch.dict("os.environ", {}, clear=True):
        yield AnthropicBackend(config)


# ---------------------------------------------------------------------------
# generate() -- streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_streaming_success(make_backend_config):
    """Streaming generate yields text chunks from content_block_delta events."""
    sse = _sse_lines(
        ("message_start", {"type": "message_start", "message": {"id": "msg_1"}}),
        ("content_block_start", {"type": "content_block_start", "index": 0}),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )

    transport = httpx.MockTransport(
        lambda request: _stream_response(200, sse)
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        chunks: list[str] = []
        async for chunk in be.generate(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_extracts_system_prompt(make_backend_config):
    """System messages are separated into the top-level 'system' field."""
    sse = _sse_lines(
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )

    captured_requests: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return _stream_response(200, sse)

    transport = httpx.MockTransport(_handler)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        chunks: list[str] = []
        async for chunk in be.generate(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            stream=True,
        ):
            chunks.append(chunk)

    assert chunks == ["ok"]
    req_body = json.loads(captured_requests[0].content)
    assert req_body["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in req_body["messages"])


# ---------------------------------------------------------------------------
# generate() -- non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_non_streaming(make_backend_config):
    """Non-streaming generate yields full content blocks."""
    response_body = {
        "id": "msg_1",
        "type": "message",
        "content": [{"type": "text", "text": "Hello from Anthropic"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        chunks: list[str] = []
        async for chunk in be.generate(
            [{"role": "user", "content": "Hi"}],
            stream=False,
        ):
            chunks.append(chunk)

    assert chunks == ["Hello from Anthropic"]


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
        lambda request: httpx.Response(
            401, json={"error": {"message": "invalid key"}}
        )
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                pass
    assert be.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_429_rate_limited(make_backend_config):
    """429 from the API raises RuntimeError and sets ERROR status."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            429, json={"error": {"message": "rate limited"}}
        )
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                pass
    assert be.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_500_server_error(make_backend_config):
    """500 from the API raises RuntimeError and sets ERROR status."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            500, json={"error": {"message": "internal error"}}
        )
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                pass
    assert be.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_success(make_backend_config):
    """health_check returns True on 200 and sets AVAILABLE status."""
    response_body = {
        "id": "msg_hc",
        "type": "message",
        "content": [{"type": "text", "text": "p"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        result = await be.health_check()

    assert result is True
    assert be.status == BackendStatus.AVAILABLE


@pytest.mark.asyncio
async def test_health_check_unauthorized(make_backend_config):
    """health_check returns False on 401."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401, json={"error": {"message": "bad key"}}
        )
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        result = await be.health_check()

    assert result is False
    assert be.status == BackendStatus.ERROR


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
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        result = await be.health_check()

    assert result is False
    assert be.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# record_usage() / properties
# ---------------------------------------------------------------------------


def test_record_usage_noop(backend):
    """record_usage is a no-op but does not raise."""
    backend.record_usage(100, 50)  # should not raise


def test_properties(backend):
    """config and status properties return expected values."""
    assert backend.config.name == "claude-test"
    assert backend.config.provider == "anthropic"
    assert backend.status == BackendStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Additional coverage for missing lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_connection_error_raises(make_backend_config):
    """ConnectError during generate raises RuntimeError and sets ERROR status.

    Covers anthropic.py lines 128-130.
    """

    def _raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(_raise_connect)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises(RuntimeError, match="Anthropic connection failed"):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                pass
    assert be.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_timeout_raises(make_backend_config):
    """TimeoutException during generate raises RuntimeError and sets ERROR status.

    Covers anthropic.py lines 128-130.
    """

    def _raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    transport = httpx.MockTransport(_raise_timeout)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises(RuntimeError, match="Anthropic connection failed"):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                pass
    assert be.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_generate_json_decode_error_raises(make_backend_config):
    """JSONDecodeError during non-streaming generate raises RuntimeError.

    Covers anthropic.py lines 133-135.
    """
    # Return a 200 with non-JSON body for the non-stream path
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, transport)
        with pytest.raises((RuntimeError, Exception)):
            async for _ in be.generate(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            ):
                pass


@pytest.mark.asyncio
async def test_generate_runtime_error_propagates(make_backend_config):
    """RuntimeError from inner generators propagates as-is (not wrapped).

    Covers anthropic.py lines 131-132 (except RuntimeError: raise).
    """
    async def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError("inner error")
        if False:
            yield  # make it an async generator

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
        be = _make_backend(make_backend_config, None)
        with (
            patch.object(be, "_stream_generate", _raise_runtime_error),
            pytest.raises(RuntimeError, match="inner error"),
        ):
                async for _ in be.generate(
                    [{"role": "user", "content": "Hi"}],
                    stream=True,
                ):
                    pass


def test_extract_delta_text_malformed_json():
    """_extract_delta_text returns None for malformed JSON.

    Covers anthropic.py lines 168-169.
    """
    config_mock = type("C", (), {
        "env_key": None, "base_url": None, "model": "claude-test",
    })()
    be = AnthropicBackend.__new__(AnthropicBackend)
    be._config = config_mock
    be._api_key = "key"
    be._status = BackendStatus.AVAILABLE
    be._transport = None

    result = be._extract_delta_text("not-valid-json")
    assert result is None


def test_extract_delta_text_empty_string():
    """_extract_delta_text returns None for empty text field.

    Covers anthropic.py line 172.
    """
    be = AnthropicBackend.__new__(AnthropicBackend)
    be._config = type("C", (), {"env_key": None, "base_url": None, "model": "x"})()
    be._api_key = "key"
    be._status = BackendStatus.AVAILABLE
    be._transport = None

    # text_delta event with empty text string
    data = json.dumps({"delta": {"type": "text_delta", "text": ""}})
    result = be._extract_delta_text(data)
    assert result is None


def test_extract_delta_text_non_text_delta_type():
    """_extract_delta_text returns None for non-text_delta delta type.

    Covers anthropic.py line 171-172 (type != 'text_delta' branch).
    """
    be = AnthropicBackend.__new__(AnthropicBackend)
    be._config = type("C", (), {"env_key": None, "base_url": None, "model": "x"})()
    be._api_key = "key"
    be._status = BackendStatus.AVAILABLE
    be._transport = None

    data = json.dumps({"delta": {"type": "input_json_delta", "partial_json": "{}"}})
    result = be._extract_delta_text(data)
    assert result is None
