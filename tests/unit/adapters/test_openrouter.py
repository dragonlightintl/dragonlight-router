"""Tests for the OpenRouter backend adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from dragonlight_router.adapters.openrouter import OpenRouterBackend
from dragonlight_router.core.types import BackendStatus

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_lines(*chunks: str) -> list[str]:
    """Build raw SSE lines from content strings (OpenAI-compatible format)."""
    lines: list[str] = []
    for i, text in enumerate(chunks):
        data = {
            "id": f"chatcmpl-{i}",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
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


def _make_backend(
    make_backend_config,
    transport: httpx.MockTransport | None = None,
    *,
    env_key: str = "OPENROUTER_API_KEY",
    env_value: str | None = "sk-or-test-key",
) -> OpenRouterBackend:
    """Build an OpenRouterBackend with optional transport injection."""
    config = make_backend_config(
        name="openrouter-test",
        provider="openrouter",
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api",
        env_key=env_key,
    )
    env = {env_key: env_value} if env_value else {}
    with patch.dict("os.environ", env, clear=True):
        return OpenRouterBackend(config, _transport=transport)


# ---------------------------------------------------------------------------
# __init__ — env_key branch (line 33)
# ---------------------------------------------------------------------------


def test_init_sets_api_key_from_env(make_backend_config):
    """OpenRouterBackend resolves the API key from env_key on init."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api",
        env_key="OPENROUTER_API_KEY",
    )
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-abc123"}):
        backend = OpenRouterBackend(config)
    assert backend._api_key == "sk-or-abc123"


def test_init_no_env_key_leaves_empty_api_key(make_backend_config):
    """When env_key is None, _api_key stays empty (base class behaviour)."""
    config = make_backend_config(
        name="or-no-key",
        provider="openrouter",
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
    assert backend._api_key == ""


# ---------------------------------------------------------------------------
# _validate_api_key (lines 37–42)
# ---------------------------------------------------------------------------


def test_validate_api_key_success(make_backend_config):
    """_validate_api_key returns the key when env var is set."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key="OPENROUTER_API_KEY",
    )
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-valid"}):
        backend = OpenRouterBackend(config)
        key = backend._validate_api_key()
    assert key == "sk-or-valid"


def test_validate_api_key_no_env_key_raises(make_backend_config):
    """_validate_api_key raises ValueError when env_key is not configured."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
        with pytest.raises(ValueError, match="API key not configured for OpenRouter backend"):
            backend._validate_api_key()


def test_validate_api_key_missing_env_var_raises(make_backend_config):
    """_validate_api_key raises ValueError when the env var is not set."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key="OPENROUTER_API_KEY",
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            backend._validate_api_key()


# ---------------------------------------------------------------------------
# _build_auth_headers (lines 46–47)
# ---------------------------------------------------------------------------


def test_build_auth_headers(make_backend_config):
    """_build_auth_headers returns Bearer token headers."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key="OPENROUTER_API_KEY",
    )
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-header-test"}):
        backend = OpenRouterBackend(config)
        headers = backend._build_auth_headers()
    assert headers["Authorization"] == "Bearer sk-or-header-test"
    assert headers["Content-Type"] == "application/json"


def test_build_auth_headers_raises_without_key(make_backend_config):
    """_build_auth_headers propagates ValueError from _validate_api_key."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
        with pytest.raises(ValueError, match="API key not configured"):
            backend._build_auth_headers()


# ---------------------------------------------------------------------------
# health_check (lines 54–65)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_no_env_key_returns_false(make_backend_config):
    """health_check returns False and sets ERROR when env_key is not configured."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
    result = await backend.health_check()
    assert result is False
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_missing_env_var_returns_false(make_backend_config):
    """health_check returns False and sets ERROR when the env var is unset."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key="OPENROUTER_API_KEY",
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
        result = await backend.health_check()
    assert result is False
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_success(make_backend_config):
    """health_check returns True on 200 from the models endpoint."""
    response_body = {"object": "list", "data": [{"id": "openai/gpt-4o", "object": "model"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response_body))
    backend = _make_backend(make_backend_config, transport)
    result = await backend.health_check()
    assert result is True
    assert backend.status == BackendStatus.AVAILABLE


@pytest.mark.asyncio
async def test_health_check_unauthorized(make_backend_config):
    """health_check returns False on 401."""
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "bad key"}))
    backend = _make_backend(make_backend_config, transport)
    result = await backend.health_check()
    assert result is False
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_connection_error(make_backend_config):
    """health_check returns False on connection failure."""

    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(_raise)
    backend = _make_backend(make_backend_config, transport)
    result = await backend.health_check()
    assert result is False
    assert backend.status == BackendStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_404_sets_offline(make_backend_config):
    """health_check sets OFFLINE status on 404."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": "not found"})
    )
    backend = _make_backend(make_backend_config, transport)
    result = await backend.health_check()
    assert result is False
    assert backend.status == BackendStatus.OFFLINE


# ---------------------------------------------------------------------------
# generate() — streaming (inherited, but exercises validate path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_streaming_success(make_backend_config):
    """Streaming generate yields text chunks through OpenRouter."""
    sse = _sse_lines("Hello", " world")
    transport = httpx.MockTransport(lambda request: _stream_response(200, sse))
    backend = _make_backend(make_backend_config, transport)

    chunks: list[str] = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_no_api_key_raises(make_backend_config):
    """generate raises ValueError when env_key is missing."""
    config = make_backend_config(
        name="or-test",
        provider="openrouter",
        model="openai/gpt-4o",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        backend = OpenRouterBackend(config)
    with pytest.raises(ValueError):
        async for _ in backend.generate([{"role": "user", "content": "Hi"}]):
            pass
