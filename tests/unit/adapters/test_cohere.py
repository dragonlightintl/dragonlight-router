"""Tests for the Cohere adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.adapters.cohere import CohereBackend
from dragonlight_router.core.types import BackendStatus


@pytest.fixture
def cohere_config(make_backend_config):
    return make_backend_config(
        name="cohere-command-r-plus",
        provider="cohere",
        model="command-r-plus",
        base_url="https://api.cohere.com",
        env_key="COHERE_API_KEY",
    )


@pytest.fixture
def cohere_backend(cohere_config, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key-789")
    return CohereBackend(cohere_config)


def _make_cohere_sse_lines(chunks: list[str], finish: bool = True) -> list[str]:
    """Build SSE lines for Cohere v2 streaming format."""
    lines = []
    for text in chunks:
        payload = {
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {"text": text}
                }
            },
        }
        lines.append(f"data: {json.dumps(payload)}")
    if finish:
        end_payload = {
            "type": "message-end",
            "delta": {
                "finish_reason": "COMPLETE",
                "usage": {
                    "billed_units": {"input_tokens": 10, "output_tokens": 20}
                },
            },
        }
        lines.append(f"data: {json.dumps(end_payload)}")
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


class TestCohereGenerate:
    @pytest.mark.asyncio
    async def test_streaming_generation(self, cohere_backend):
        """Streaming generate yields content chunks from Cohere SSE."""
        sse_lines = _make_cohere_sse_lines(["Hello", ", ", "world!"])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world!"]
        assert cohere_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_streaming_stops_on_message_end(self, cohere_backend):
        """Streaming stops when message-end event is received."""
        # Add content after message-end -- it should be ignored
        lines = _make_cohere_sse_lines(["partial"], finish=True)
        # Append another content-delta after the message-end
        extra = {
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "SHOULD NOT APPEAR"}}},
        }
        lines.append(f"data: {json.dumps(extra)}")
        fake_response = _FakeStreamResponse(lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["partial"]
        assert "SHOULD NOT APPEAR" not in chunks

    @pytest.mark.asyncio
    async def test_non_streaming_generation(self, cohere_backend):
        """Non-streaming generate yields content from Cohere v2 response."""
        response_body = {
            "message": {
                "content": [{"type": "text", "text": "Full response."}]
            }
        }
        fake_response = _FakeStreamResponse([json.dumps(response_body)])
        fake_response.json = lambda: response_body

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=False,
            ):
                chunks.append(chunk)

        assert chunks == ["Full response."]

    @pytest.mark.asyncio
    async def test_non_streaming_multiple_content_parts(self, cohere_backend):
        """Non-streaming concatenates multiple content parts."""
        response_body = {
            "message": {
                "content": [
                    {"type": "text", "text": "Part 1. "},
                    {"type": "text", "text": "Part 2."},
                ]
            }
        }
        fake_response = _FakeStreamResponse([json.dumps(response_body)])
        fake_response.json = lambda: response_body

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}], stream=False,
            ):
                chunks.append(chunk)

        assert chunks == ["Part 1. Part 2."]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, cohere_config, monkeypatch):
        """Generate raises ValueError when API key is missing."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        backend = CohereBackend(cohere_config)

        with pytest.raises(ValueError, match="API key not configured"):
            async for _ in backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_http_error_sets_error_status(self, cohere_backend):
        """HTTP errors set status to ERROR and raise RuntimeError."""
        fake_response = _FakeStreamResponse([], status_code=500)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cohere API error"):
                async for _ in cohere_backend.generate(
                    [{"role": "user", "content": "Hi"}],
                ):
                    pass

        assert cohere_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_uses_correct_url(self, cohere_backend):
        """Generate posts to the Cohere v2 chat endpoint."""
        sse_lines = _make_cohere_sse_lines(["ok"])
        fake_response = _FakeStreamResponse(sse_lines)
        captured_url = None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                pass

        assert captured_url == "https://api.cohere.com/v2/chat"

    @pytest.mark.asyncio
    async def test_skips_malformed_json(self, cohere_backend):
        """Malformed JSON lines in SSE are silently skipped."""
        good_payload = {
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "ok"}}},
        }
        lines = [
            'data: not-valid-json',
            f'data: {json.dumps(good_payload)}',
            'data: [DONE]',
        ]
        fake_response = _FakeStreamResponse(lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in cohere_backend.generate(
                [{"role": "user", "content": "Hi"}],
            ):
                chunks.append(chunk)

        assert chunks == ["ok"]


class TestCohereHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_returns_true(self, cohere_backend):
        """Health check returns True on successful response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await cohere_backend.health_check()

        assert result is True
        assert cohere_backend.status == BackendStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false(self, cohere_backend):
        """Health check returns False on failed response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = False

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await cohere_backend.health_check()

        assert result is False
        assert cohere_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self, cohere_config, monkeypatch):
        """Health check returns False when no API key is set."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        backend = CohereBackend(cohere_config)
        result = await backend.health_check()
        assert result is False
        assert backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self, cohere_backend):
        """Health check returns False on connection error."""
        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await cohere_backend.health_check()

        assert result is False
        assert cohere_backend.status == BackendStatus.ERROR

    @pytest.mark.asyncio
    async def test_health_check_hits_v2_models(self, cohere_backend):
        """Health check queries /v2/models."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True
        captured_url = None

        async def fake_get(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await cohere_backend.health_check()

        assert captured_url == "https://api.cohere.com/v2/models"
