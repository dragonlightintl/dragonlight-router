"""Tests for the Mistral adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.adapters.mistral import MistralBackend
from dragonlight_router.core.types import BackendStatus


@pytest.fixture
def mistral_config(make_backend_config):
    return make_backend_config(
        name="mistral-large",
        provider="mistral",
        model="mistral-large-latest",
        base_url="https://api.mistral.ai",
        env_key="MISTRAL_API_KEY",
    )


@pytest.fixture
def mistral_backend(mistral_config, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")
    return MistralBackend(mistral_config)


def _make_sse_lines(chunks: list[str], done: bool = True) -> list[str]:
    """Build SSE lines for OpenAI-compatible streaming."""
    lines = []
    for text in chunks:
        payload = {
            "choices": [{"delta": {"content": text}, "index": 0}]
        }
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
                "error", request=MagicMock(), response=self,
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        return json.loads(self._lines[0]) if self._lines else {}


class TestMistralGenerate:
    @pytest.mark.asyncio
    async def test_streaming_generation(self, mistral_backend):
        """Streaming generate yields content chunks from SSE."""
        sse_lines = _make_sse_lines(["Hello", ", ", "world!"])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in mistral_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world!"]
        assert mistral_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_non_streaming_generation(self, mistral_backend):
        """Non-streaming generate yields the full message content."""
        response_body = {
            "choices": [{"message": {"content": "Full response here."}}]
        }
        fake_response = _FakeStreamResponse([json.dumps(response_body)])
        fake_response.json = lambda: response_body

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in mistral_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=False,
            ):
                chunks.append(chunk)

        assert chunks == ["Full response here."]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, mistral_config, monkeypatch):
        """Generate raises ValueError when API key is missing."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        backend = MistralBackend(mistral_config)

        with pytest.raises(ValueError, match="API key not configured"):
            async for _ in backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_http_error_sets_error_status(self, mistral_backend):
        """HTTP errors set status to ERROR and raise RuntimeError."""
        fake_response = _FakeStreamResponse([], status_code=429)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Mistral API error"):
                async for _ in mistral_backend.generate(
                    [{"role": "user", "content": "Hi"}],
                ):
                    pass

        assert mistral_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_uses_correct_url(self, mistral_backend):
        """Generate posts to the Mistral base URL."""
        sse_lines = _make_sse_lines(["ok"])
        fake_response = _FakeStreamResponse(sse_lines)
        captured_url = None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            yield fake_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in mistral_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert captured_url == "https://api.mistral.ai/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_skips_malformed_json(self, mistral_backend):
        """Malformed JSON lines in SSE are silently skipped."""
        lines = [
            'data: not-valid-json',
            f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}',
            'data: [DONE]',
        ]
        fake_response = _FakeStreamResponse(lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in mistral_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                chunks.append(chunk)

        assert chunks == ["ok"]


class TestMistralHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_returns_true(self, mistral_backend):
        """Health check returns True on successful response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await mistral_backend.health_check()

        assert result is True
        assert mistral_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false(self, mistral_backend):
        """Health check returns False on failed response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = False
        mock_response.status_code = 500

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await mistral_backend.health_check()

        assert result is False
        assert mistral_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self, mistral_config, monkeypatch):
        """Health check returns False when no API key is set."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        backend = MistralBackend(mistral_config)
        result = await backend.health_check()
        assert result is False
        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self, mistral_backend):
        """Health check returns False on connection error."""
        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await mistral_backend.health_check()

        assert result is False
        assert mistral_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_health_check_hits_models_endpoint(self, mistral_backend):
        """Health check queries /v1/models."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True
        captured_url = None

        async def fake_get(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_response

        with patch("dragonlight_router.adapters.mistral.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await mistral_backend.health_check()

        assert captured_url == "https://api.mistral.ai/v1/models"
