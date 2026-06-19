"""Tests for the Together adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.adapters.together import TogetherBackend
from dragonlight_router.core.types import BackendStatus

pytestmark = pytest.mark.unit


@pytest.fixture
def together_config(make_backend_config):
    return make_backend_config(
        name="together-llama",
        provider="together",
        model="meta-llama/Llama-3-70b-chat-hf",
        base_url="https://api.together.xyz",
        env_key="TOGETHER_API_KEY",
    )


@pytest.fixture
def together_backend(together_config, monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-456")
    return TogetherBackend(together_config)


def _make_sse_lines(chunks: list[str], done: bool = True) -> list[str]:
    """Build SSE lines for OpenAI-compatible streaming."""
    lines = []
    for text in chunks:
        payload = {"choices": [{"delta": {"content": text}, "index": 0}]}
        lines.append(f"data: {json.dumps(payload)}")
    if done:
        lines.append("data: [DONE]")
    return lines


class _FakeStreamResponse:
    """Fake httpx streaming response that yields SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines
        self.is_success = 200 <= status_code < 300

    def raise_for_status(self):
        if not self.is_success:
            raise httpx.HTTPStatusError(
                "error",
                request=MagicMock(),
                response=self,
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        return json.loads(self._lines[0]) if self._lines else {}


class TestTogetherGenerate:
    @pytest.mark.asyncio
    async def test_streaming_generation(self, together_backend):
        """Streaming generate yields content chunks from SSE."""
        sse_lines = _make_sse_lines(["Hello", ", ", "world!"])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in together_backend.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world!"]
        assert together_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_non_streaming_generation(self, together_backend):
        """Non-streaming generate yields the full message content."""
        response_body = {"choices": [{"message": {"content": "Full response here."}}]}
        fake_response = _FakeStreamResponse([json.dumps(response_body)])
        fake_response.json = lambda: response_body

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in together_backend.generate(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            ):
                chunks.append(chunk)

        assert chunks == ["Full response here."]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, together_config, monkeypatch):
        """Generate raises ValueError when API key is missing."""
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        backend = TogetherBackend(together_config)

        with pytest.raises(ValueError, match="API key not configured"):
            async for _ in backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_http_error_sets_error_status(self, together_backend):
        """HTTP errors set status to ERROR and raise RuntimeError."""
        fake_response = _FakeStreamResponse([], status_code=429)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Together API error"):
                async for _ in together_backend.generate(
                    [{"role": "user", "content": "Hi"}],
                ):
                    pass

        assert together_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_uses_correct_url(self, together_backend):
        """Generate posts to the Together base URL."""
        sse_lines = _make_sse_lines(["ok"])
        fake_response = _FakeStreamResponse(sse_lines)
        captured_url = None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            yield fake_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in together_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert captured_url == "https://api.together.xyz/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_skips_malformed_json(self, together_backend):
        """Malformed JSON lines in SSE are silently skipped."""
        lines = [
            "data: not-valid-json",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}",
            "data: [DONE]",
        ]
        fake_response = _FakeStreamResponse(lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in together_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                chunks.append(chunk)

        assert chunks == ["ok"]


class TestTogetherHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_returns_true(self, together_backend):
        """Health check returns True on successful response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await together_backend.health_check()

        assert result is True
        assert together_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false(self, together_backend):
        """Health check returns False on failed response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = False
        mock_response.status_code = 500

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await together_backend.health_check()

        assert result is False
        assert together_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self, together_config, monkeypatch):
        """Health check returns False when no API key is set."""
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        backend = TogetherBackend(together_config)
        result = await backend.health_check()
        assert result is False
        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self, together_backend):
        """Health check returns False on connection error."""
        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await together_backend.health_check()

        assert result is False
        assert together_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_health_check_hits_models_endpoint(self, together_backend):
        """Health check queries /v1/models."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True
        captured_url = None

        async def fake_get(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_response

        with patch("dragonlight_router.adapters.together.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await together_backend.health_check()

        assert captured_url == "https://api.together.xyz/v1/models"
