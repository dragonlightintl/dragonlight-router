"""Tests for the Groq backend adapter.

Spec traceability: TM-005 (Adapter implementations)

Key concern: Groq's base URL is https://api.groq.com/openai/v1, which already
includes /v1. The adapter must use _models_path = "/models" (not "/v1/models")
to avoid constructing a double-prefixed URL like /openai/v1/v1/models.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from dragonlight_router.adapters.groq import _GROQ_BASE_URL, GroqBackend
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


def _make_groq_backend(
    make_backend_config,
    transport: httpx.MockTransport | None = None,
    *,
    base_url: str = _GROQ_BASE_URL,
    env_key: str = "GROQ_API_KEY",
    env_value: str | None = "gsk-test-key",
) -> GroqBackend:
    """Build a GroqBackend with optional transport injection."""
    config = make_backend_config(
        name="groq-test",
        provider="groq",
        model="llama-3.3-70b-versatile",
        base_url=base_url,
        env_key=env_key,
    )
    env = {env_key: env_value} if env_value else {}
    with patch.dict("os.environ", env, clear=True):
        return GroqBackend(config, _transport=transport)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(make_backend_config):
    """GroqBackend wired to a fake API key via env (no transport)."""
    config = make_backend_config(
        name="groq-test",
        provider="groq",
        model="llama-3.3-70b-versatile",
        base_url=_GROQ_BASE_URL,
        env_key="GROQ_API_KEY",
    )
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test-key"}):
        yield GroqBackend(config)


@pytest.fixture
def backend_no_key(make_backend_config):
    """GroqBackend with no API key configured."""
    config = make_backend_config(
        name="groq-test",
        provider="groq",
        model="llama-3.3-70b-versatile",
        base_url=_GROQ_BASE_URL,
        env_key="GROQ_API_KEY",
    )
    with patch.dict("os.environ", {}, clear=True):
        yield GroqBackend(config)


# ---------------------------------------------------------------------------
# URL construction — the core regression guard
# ---------------------------------------------------------------------------


def test_health_check_url_does_not_double_prefix_v1(make_backend_config):
    """Health check URL must be .../openai/v1/models, NOT .../openai/v1/v1/models.

    Groq's base_url already includes /v1. The adapter overrides _models_path
    to "/models" so the resolved URL is correct. This test guards against
    regression where _models_path is changed back to "/v1/models".
    """
    backend = _make_groq_backend(make_backend_config)
    base_url = backend._resolve_base_url()
    health_url = f"{base_url}{backend._models_path}"
    assert health_url == "https://api.groq.com/openai/v1/models", (
        f"Wrong health check URL: {health_url!r}. "
        "Groq base_url already contains /v1 — _models_path must be '/models', not '/v1/models'."
    )


def test_completions_url_does_not_double_prefix(make_backend_config):
    """Completions URL must be .../openai/v1/chat/completions."""
    backend = _make_groq_backend(make_backend_config)
    base_url = backend._resolve_base_url()
    completions_url = f"{base_url}{backend._completions_path}"
    assert completions_url == "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# health_check() — HTTP behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_success(make_backend_config):
    """health_check returns True on 200 from /openai/v1/models."""
    captured_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "llama-3.3-70b-versatile"}]},
        )

    transport = httpx.MockTransport(_handler)
    backend = _make_groq_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is True
    assert backend.status == BackendStatus.AVAILABLE
    # Verify the URL hit is correct (no double /v1/)
    assert len(captured_urls) == 1
    assert captured_urls[0] == "https://api.groq.com/openai/v1/models"
    assert "/v1/v1/" not in captured_urls[0]


@pytest.mark.asyncio
async def test_health_check_404_sets_offline(make_backend_config):
    """health_check returns False and sets OFFLINE on 404."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": {"message": "not found"}})
    )
    backend = _make_groq_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is False
    assert backend.status == BackendStatus.OFFLINE


@pytest.mark.asyncio
async def test_health_check_401_sets_error(make_backend_config):
    """health_check returns False and sets ERROR on 401 (not 404)."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    backend = _make_groq_backend(make_backend_config, transport)

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
    backend = _make_groq_backend(make_backend_config, transport)

    result = await backend.health_check()

    assert result is False
    assert backend.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# generate() — streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_streaming_success(make_backend_config):
    """Streaming generate yields text chunks from SSE delta events."""
    sse = _sse_lines("Hello", " world")

    transport = httpx.MockTransport(lambda request: _stream_response(200, sse))
    backend = _make_groq_backend(make_backend_config, transport)

    chunks: list[str] = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_no_api_key(backend_no_key):
    """generate raises ValueError when no API key is configured."""
    with pytest.raises(ValueError, match="API key not configured"):
        async for _ in backend_no_key.generate(
            [{"role": "user", "content": "Hi"}],
        ):
            pass
    assert backend_no_key.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# Default base URL and path config
# ---------------------------------------------------------------------------


def test_default_base_url():
    """The Groq base URL is the OpenAI-compatible v1 endpoint."""
    assert _GROQ_BASE_URL == "https://api.groq.com/openai/v1"


def test_models_path_excludes_v1_prefix(make_backend_config):
    """_models_path must be '/models' since base_url already includes /v1."""
    backend = _make_groq_backend(make_backend_config)
    assert backend._models_path == "/models"


def test_completions_path_excludes_v1_prefix(make_backend_config):
    """_completions_path must be '/chat/completions' since base_url includes /v1."""
    backend = _make_groq_backend(make_backend_config)
    assert backend._completions_path == "/chat/completions"


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
    backend = _make_groq_backend(make_backend_config, transport)

    chunks = []
    async for chunk in backend.generate(
        [{"role": "user", "content": "Hi"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks == ["ok"]
    assert captured_request[0].headers["authorization"] == "Bearer gsk-test-key"


# ---------------------------------------------------------------------------
# record_usage() / properties
# ---------------------------------------------------------------------------


def test_record_usage_noop(backend):
    """record_usage is a no-op but does not raise."""
    backend.record_usage(100, 50)


def test_properties(backend):
    """config and status properties return expected values."""
    assert backend.config.name == "groq-test"
    assert backend.config.provider == "groq"
    assert backend.config.model == "llama-3.3-70b-versatile"
    assert backend.status == BackendStatus.AVAILABLE
