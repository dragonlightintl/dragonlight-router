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

pytestmark = pytest.mark.unit


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
            "delta": {"message": {"content": {"text": text}}},
        }
        lines.append(f"data: {json.dumps(payload)}")
    if finish:
        end_payload = {
            "type": "message-end",
            "delta": {
                "finish_reason": "COMPLETE",
                "usage": {"billed_units": {"input_tokens": 10, "output_tokens": 20}},
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
                "error",
                request=MagicMock(),
                response=self,
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
                [{"role": "user", "content": "Hi"}],
                stream=True,
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
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["partial"]
        assert "SHOULD NOT APPEAR" not in chunks

    @pytest.mark.asyncio
    async def test_non_streaming_generation(self, cohere_backend):
        """Non-streaming generate yields content from Cohere v2 response."""
        response_body = {"message": {"content": [{"type": "text", "text": "Full response."}]}}
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
                [{"role": "user", "content": "Hi"}],
                stream=False,
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
                [{"role": "user", "content": "Hi"}],
                stream=False,
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
            "data: not-valid-json",
            f"data: {json.dumps(good_payload)}",
            "data: [DONE]",
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


# ---------------------------------------------------------------------------
# Additional coverage for missing lines
# ---------------------------------------------------------------------------


class TestCohereAdditionalCoverage:
    """Tests targeting uncovered code paths in cohere.py."""

    # line 37 — no env_key configured: _api_key stays empty
    def test_no_env_key_api_key_empty(self, make_backend_config):
        """When env_key is None, _api_key is an empty string."""
        config = make_backend_config(
            name="cohere-no-key",
            provider="cohere",
            model="command-r-plus",
            env_key=None,
        )
        backend = CohereBackend(config)
        assert backend._api_key == ""

    # line 37 — config property
    def test_config_property(self, make_backend_config, monkeypatch):
        """config property returns the BackendConfig instance."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-prop",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)
        assert backend.config.name == "cohere-prop"
        assert backend.config.provider == "cohere"

    # line 47 — _resolve_base_url default (no base_url)
    def test_resolve_base_url_default(self, make_backend_config, monkeypatch):
        """_resolve_base_url returns default URL when config.base_url is not set."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-default-url",
            provider="cohere",
            model="command-r-plus",
            base_url="",  # empty triggers default
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)
        assert backend._resolve_base_url() == "https://api.cohere.com"

    # line 47 — _resolve_base_url with custom base_url (line 46 — rstrip path)
    def test_resolve_base_url_custom(self, make_backend_config, monkeypatch):
        """_resolve_base_url returns config.base_url when set."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-custom",
            provider="cohere",
            model="command-r-plus",
            base_url="https://custom.cohere.example.com",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)
        assert backend._resolve_base_url() == "https://custom.cohere.example.com"

    # lines 101-103 — ConnectError raises RuntimeError
    @pytest.mark.asyncio
    async def test_generate_connect_error_raises(self, make_backend_config, monkeypatch):
        """ConnectError during generate raises RuntimeError and sets ERROR status."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-conn",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            raise httpx.ConnectError("refused")
            yield  # noqa: F841

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cohere connection failed"):
                async for _ in backend.generate(
                    [{"role": "user", "content": "Hi"}],
                ):
                    pass

        assert backend.status == BackendStatus.ERROR

    # lines 104-108 — TimeoutException raises RuntimeError
    @pytest.mark.asyncio
    async def test_generate_timeout_raises(self, make_backend_config, monkeypatch):
        """TimeoutException during generate raises RuntimeError and sets ERROR status."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-timeout",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            raise httpx.ReadTimeout("timed out")
            yield  # noqa: F841

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cohere connection failed"):
                async for _ in backend.generate(
                    [{"role": "user", "content": "Hi"}],
                ):
                    pass

        assert backend.status == BackendStatus.ERROR

    # lines 104-105 — RuntimeError propagates unchanged from inner generators
    @pytest.mark.asyncio
    async def test_generate_runtime_error_propagates(self, make_backend_config, monkeypatch):
        """RuntimeError from _parse_stream is re-raised as-is.

        Covers cohere.py lines 104-105 (except RuntimeError: raise).
        """
        from unittest.mock import patch as _patch

        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-rt-err",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        sse_lines = _make_cohere_sse_lines(["ok"])
        fake_response = _FakeStreamResponse(sse_lines)

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        async def _raise_runtime(*args, **kwargs):
            raise RuntimeError("inner cohere error")
            if False:
                yield

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with (
                _patch.object(backend, "_parse_stream", _raise_runtime),
                pytest.raises(RuntimeError, match="inner cohere error"),
            ):
                async for _ in backend.generate(
                    [{"role": "user", "content": "Hi"}],
                    stream=True,
                ):
                    pass

    # line 114 — non-data lines in _parse_stream are skipped (continue)
    @pytest.mark.asyncio
    async def test_parse_stream_skips_non_data_lines(self, make_backend_config, monkeypatch):
        """Non-data SSE lines (event:, empty lines) are skipped in _parse_stream.

        Covers cohere.py line 114 (continue when line doesn't start with 'data: ').
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-skip",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        good_payload = {
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "hello"}}},
        }
        lines = [
            "",  # blank line
            "event: content-delta",  # event prefix — no data
            f"data: {json.dumps(good_payload)}",
            "data: [DONE]",
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
            async for chunk in backend.generate(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            ):
                chunks.append(chunk)

        assert chunks == ["hello"]

    # line 114 — non-streaming path when content is empty
    @pytest.mark.asyncio
    async def test_non_streaming_empty_content(self, make_backend_config, monkeypatch):
        """Non-streaming generate yields nothing when content is empty."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-empty",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        response_body = {"message": {"content": []}}
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
            async for chunk in backend.generate(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            ):
                chunks.append(chunk)

        assert chunks == []

    # lines 106-108 — JSONDecodeError in non-streaming path raises RuntimeError
    @pytest.mark.asyncio
    async def test_non_streaming_json_decode_error_raises(self, make_backend_config, monkeypatch):
        """JSONDecodeError from response.json() raises RuntimeError in non-stream path.

        Covers cohere.py lines 106-108.
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-json-err",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)

        class _BadJsonStreamResponse(_FakeStreamResponse):
            def json(self):
                raise json.JSONDecodeError("bad json", "", 0)

        fake_response = _BadJsonStreamResponse([], status_code=200)
        fake_response.raise_for_status = lambda: None

        @asynccontextmanager
        async def fake_stream(method, url, **kwargs):
            yield fake_response

        with patch("dragonlight_router.adapters.cohere.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.stream = fake_stream
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cohere request failed"):
                async for _ in backend.generate(
                    [{"role": "user", "content": "Hi"}],
                    stream=False,
                ):
                    pass

        assert backend.status == BackendStatus.ERROR

    # lines 150-152 — _extract_non_stream_content with string content_parts
    def test_extract_non_stream_content_string(self, make_backend_config, monkeypatch):
        """_extract_non_stream_content handles string content_parts directly."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-str",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)
        response_data = {"message": {"content": "plain string response"}}
        result = backend._extract_non_stream_content(response_data)
        assert result == "plain string response"

    def test_extract_non_stream_content_empty(self, make_backend_config, monkeypatch):
        """_extract_non_stream_content returns empty string for unknown content type."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        config = make_backend_config(
            name="cohere-empty2",
            provider="cohere",
            model="command-r-plus",
            env_key="COHERE_API_KEY",
        )
        backend = CohereBackend(config)
        response_data = {"message": {"content": 12345}}  # unexpected type
        result = backend._extract_non_stream_content(response_data)
        assert result == ""
