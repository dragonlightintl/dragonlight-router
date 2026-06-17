"""Tests for the Local (Ollama) backend adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.adapters.local import LocalBackend
from dragonlight_router.core.types import BackendStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oai_stream_chunk(content: str, finish: bool = False) -> dict:
    """Build a minimal OpenAI-compatible streaming chunk."""
    delta = {"content": content} if content else {}
    choice: dict = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = "stop"
    return {"choices": [choice]}


def _make_sse_lines(chunks: list[dict], done: bool = True) -> list[str]:
    """Build SSE-formatted lines from a list of OpenAI-compatible chunks."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    if done:
        lines.append("data: [DONE]")
    return lines


class _FakeStreamResponse:
    """Fake httpx streaming response that yields SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b'{"error": "test error"}'


# ---------------------------------------------------------------------------
# Streaming generation tests
# ---------------------------------------------------------------------------

class TestLocalGenerate:
    @pytest.fixture
    def backend(self, make_backend_config):
        config = make_backend_config(
            name="ollama-llama",
            provider="local",
            model="llama3.2",
            base_url="http://localhost:11434",
            env_key=None,
        )
        return LocalBackend(config)

    async def test_streaming_generation(self, backend):
        sse_lines = _make_sse_lines([
            _oai_stream_chunk("Hello"),
            _oai_stream_chunk(", world!"),
            _oai_stream_chunk("", finish=True),
        ])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [{"role": "user", "content": "Say hello"}]
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", ", world!"]

    async def test_non_streaming(self, backend):
        response_body = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, world!"},
                "finish_reason": "stop",
            }]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_body

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [{"role": "user", "content": "Say hello"}],
                stream=False,
            ):
                chunks.append(chunk)

        assert chunks == ["Hello, world!"]

    async def test_messages_passed_through(self, backend):
        """Verify messages are forwarded as-is (OpenAI-compatible format)."""
        captured_kwargs = {}

        sse_lines = _make_sse_lines([_oai_stream_chunk("ok")])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            captured_kwargs.update(kwargs)
            yield fake_response

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in backend.generate(messages):
                pass

        body = captured_kwargs.get("json", {})
        assert body["messages"] == messages

    async def test_api_error(self, backend):
        fake_response = _FakeStreamResponse([], status_code=500)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "500" in chunks[0]
        assert backend.status == BackendStatus.ERROR

    async def test_connection_refused(self, backend):
        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            raise httpx.ConnectError("connection refused")
            yield  # noqa: unreachable

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "Cannot connect" in chunks[0]
        assert backend.status == BackendStatus.OFFLINE

    async def test_timeout(self, backend):
        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            raise httpx.ReadTimeout("timed out")
            yield  # noqa: unreachable

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "timed out" in chunks[0].lower()
        assert backend.status == BackendStatus.ERROR

    async def test_generation_params_forwarded(self, backend):
        """Verify max_tokens and temperature are forwarded."""
        captured_kwargs = {}

        sse_lines = _make_sse_lines([_oai_stream_chunk("ok")])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            captured_kwargs.update(kwargs)
            yield fake_response

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in backend.generate(
                [{"role": "user", "content": "test"}],
                max_tokens=2048,
                temperature=0.3,
            ):
                pass

        body = captured_kwargs.get("json", {})
        assert body["max_tokens"] == 2048
        assert body["temperature"] == 0.3

    async def test_uses_correct_url(self, backend):
        """Generate posts to the Ollama OpenAI-compatible endpoint."""
        sse_lines = _make_sse_lines([_oai_stream_chunk("ok")])
        fake_response = _FakeStreamResponse(sse_lines)
        captured_url = None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            yield fake_response

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in backend.generate(
                [{"role": "user", "content": "test"}]
            ):
                pass

        assert captured_url == "http://localhost:11434/v1/chat/completions"


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestLocalHealthCheck:
    @pytest.fixture
    def backend(self, make_backend_config):
        config = make_backend_config(
            name="ollama-llama",
            provider="local",
            model="llama3.2",
            base_url="http://localhost:11434",
            env_key=None,
        )
        return LocalBackend(config)

    async def test_healthy(self, backend):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is True
        assert backend.status == BackendStatus.AVAILABLE

    async def test_unhealthy_not_running(self, backend):
        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is False
        assert backend.status == BackendStatus.OFFLINE

    async def test_unhealthy_error_response(self, backend):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is False
        assert backend.status == BackendStatus.OFFLINE

    async def test_health_check_hits_tags_endpoint(self, backend):
        """Health check queries /api/tags."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        captured_url = None

        async def fake_get(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_response

        with patch("dragonlight_router.adapters.local.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await backend.health_check()

        assert captured_url == "http://localhost:11434/api/tags"


# ---------------------------------------------------------------------------
# Usage recording
# ---------------------------------------------------------------------------

class TestLocalUsage:
    def test_record_usage(self, make_backend_config):
        config = make_backend_config(name="ollama", provider="local", env_key=None)
        backend = LocalBackend(config)

        backend.record_usage(100, 200)
        assert backend._tokens_in == 100
        assert backend._tokens_out == 200

        backend.record_usage(50, 75)
        assert backend._tokens_in == 150
        assert backend._tokens_out == 275

    def test_no_api_key_needed(self, make_backend_config):
        """Local backend should work without any env_key."""
        config = make_backend_config(name="ollama", provider="local", env_key=None)
        backend = LocalBackend(config)
        assert backend.status == BackendStatus.AVAILABLE
