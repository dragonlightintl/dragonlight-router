"""Tests for the Google (Gemini) backend adapter.

Spec traceability: TM-005 (Adapter implementations)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.adapters.google import (
    GoogleBackend,
    _convert_messages,
    _extract_text_from_chunk,
)
from dragonlight_router.core.types import BackendStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gemini_chunk(text: str) -> dict:
    """Build a minimal Gemini streaming chunk."""
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}}
        ]
    }


def _make_sse_lines(chunks: list[dict], done: bool = True) -> list[str]:
    """Build SSE-formatted lines from a list of Gemini chunks."""
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
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def test_user_message(self):
        contents, sys = _convert_messages([{"role": "user", "content": "hello"}])
        assert contents == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert sys is None

    def test_system_message(self):
        contents, sys = _convert_messages([
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ])
        assert sys == {"parts": [{"text": "be helpful"}]}
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_maps_to_model(self):
        contents, _ = _convert_messages([
            {"role": "assistant", "content": "sure"},
        ])
        assert contents[0]["role"] == "model"


class TestExtractText:
    def test_normal_chunk(self):
        assert _extract_text_from_chunk(_gemini_chunk("hello")) == "hello"

    def test_empty_candidates(self):
        assert _extract_text_from_chunk({"candidates": []}) is None

    def test_missing_candidates(self):
        assert _extract_text_from_chunk({}) is None

    def test_empty_parts(self):
        chunk = {"candidates": [{"content": {"parts": []}}]}
        assert _extract_text_from_chunk(chunk) is None


# ---------------------------------------------------------------------------
# Streaming generation tests
# ---------------------------------------------------------------------------

class TestGoogleGenerate:
    @pytest.fixture
    def backend(self, make_backend_config, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
        config = make_backend_config(
            name="gemini-pro",
            provider="google",
            model="gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com",
            env_key="GOOGLE_API_KEY",
        )
        return GoogleBackend(config)

    async def test_streaming_generation(self, backend):
        sse_lines = _make_sse_lines([
            _gemini_chunk("Hello"),
            _gemini_chunk(", world!"),
        ])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
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

    async def test_missing_env_key(self, make_backend_config, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = make_backend_config(
            name="gemini-pro",
            provider="google",
            model="gemini-2.0-flash",
            env_key="GOOGLE_API_KEY",
        )
        backend = GoogleBackend(config)

        chunks = []
        async for chunk in backend.generate(
            [{"role": "user", "content": "test"}]
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "not set" in chunks[0]
        assert backend.status == BackendStatus.ERROR

    async def test_no_env_key_configured(self, make_backend_config):
        config = make_backend_config(
            name="gemini-no-key",
            provider="google",
            model="gemini-2.0-flash",
            env_key=None,
        )
        backend = GoogleBackend(config)

        chunks = []
        async for chunk in backend.generate(
            [{"role": "user", "content": "test"}]
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "No API key configured" in chunks[0]

    async def test_api_error_status(self, backend):
        fake_response = _FakeStreamResponse([], status_code=429)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
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
        assert "429" in chunks[0]
        assert backend.status == BackendStatus.ERROR

    async def test_system_instruction_included(self, backend):
        captured_kwargs = {}

        sse_lines = _make_sse_lines([_gemini_chunk("ok")])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            captured_kwargs.update(kwargs)
            yield fake_response

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            chunks = []
            async for chunk in backend.generate(
                [
                    {"role": "system", "content": "You are a pirate"},
                    {"role": "user", "content": "Ahoy"},
                ]
            ):
                chunks.append(chunk)

        body = captured_kwargs.get("json", {})
        assert "systemInstruction" in body
        assert body["systemInstruction"]["parts"][0]["text"] == "You are a pirate"
        # System message should not appear in contents
        for content in body["contents"]:
            assert content["role"] != "system"

    async def test_timeout_handling(self, backend):
        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            raise httpx.ReadTimeout("timed out")
            yield  # noqa: unreachable — required for generator syntax

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
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

    async def test_url_contains_model_and_key_in_header(self, backend):
        """Generate posts to the correct Gemini URL with model; API key in header, not URL."""
        sse_lines = _make_sse_lines([_gemini_chunk("ok")])
        fake_response = _FakeStreamResponse(sse_lines)
        captured_url = None
        captured_headers = None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            nonlocal captured_url, captured_headers
            captured_url = url
            captured_headers = kwargs.get("headers", {})
            yield fake_response

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            async for _ in backend.generate(
                [{"role": "user", "content": "test"}]
            ):
                pass

        assert "gemini-2.0-flash:streamGenerateContent" in captured_url
        assert "key=" not in captured_url
        assert captured_headers["x-goog-api-key"] == "test-key-123"

    async def test_malformed_json_skipped(self, backend):
        """Malformed JSON in SSE lines is silently skipped."""
        lines = [
            "data: not-valid-json",
            f'data: {json.dumps(_gemini_chunk("ok"))}',
            "data: [DONE]",
        ]
        fake_response = _FakeStreamResponse(lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
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

        assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestGoogleHealthCheck:
    @pytest.fixture
    def backend(self, make_backend_config, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
        config = make_backend_config(
            name="gemini-pro",
            provider="google",
            model="gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com",
            env_key="GOOGLE_API_KEY",
        )
        return GoogleBackend(config)

    async def test_healthy(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is True
        assert backend.status == BackendStatus.AVAILABLE

    async def test_unhealthy_no_key(self, make_backend_config, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = make_backend_config(
            name="gemini-pro",
            provider="google",
            model="gemini-2.0-flash",
            env_key="GOOGLE_API_KEY",
        )
        backend = GoogleBackend(config)

        result = await backend.health_check()

        assert result is False
        assert backend.status == BackendStatus.ERROR

    async def test_unhealthy_api_error(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is False
        assert backend.status == BackendStatus.ERROR

    async def test_unhealthy_connection_error(self, backend):
        with patch("dragonlight_router.adapters.google.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await backend.health_check()

        assert result is False
        assert backend.status == BackendStatus.ERROR


# ---------------------------------------------------------------------------
# Usage recording
# ---------------------------------------------------------------------------

class TestGoogleUsage:
    def test_record_usage(self, make_backend_config):
        config = make_backend_config(name="gemini", provider="google", env_key="GOOGLE_API_KEY")
        backend = GoogleBackend(config)

        backend.record_usage(100, 200)
        assert backend._tokens_in == 100
        assert backend._tokens_out == 200

        backend.record_usage(50, 75)
        assert backend._tokens_in == 150
        assert backend._tokens_out == 275
