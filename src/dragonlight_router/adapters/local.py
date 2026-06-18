"""Local (Ollama) adapter implementing GenerativeBackend protocol.

Uses Ollama's OpenAI-compatible endpoint (POST /v1/chat/completions)
for simplicity. No API key required. No rate limits or billing.
Health check via GET /api/tags.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from dragonlight_router.core.types import (
    BackendConfig,
    BackendStatus,
    GenerativeBackend,
)

logger = structlog.get_logger()

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0  # Local models can be slower on first load


class LocalBackend(GenerativeBackend):
    """Local Ollama backend adapter.

    Connects to a locally running Ollama instance via its OpenAI-compatible
    chat completions endpoint. No API key required.
    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._status = BackendStatus.AVAILABLE
        self._tokens_in = 0
        self._tokens_out = 0
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using Ollama's OpenAI-compatible endpoint."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        url = f"{self._base_url}/v1/chat/completions"
        body = self._build_request_body(messages, max_tokens, temperature, stream)
        headers = {"Content-Type": "application/json"}

        try:
            if stream:
                async for chunk in self._stream_response(url, body, headers):
                    yield chunk
            else:
                async for chunk in self._non_stream_response(url, body, headers):
                    yield chunk
        except httpx.ConnectError:
            self._status = BackendStatus.OFFLINE
            logger.error("ollama_connect_error", base_url=self._base_url)
            yield f"[Local] Cannot connect to Ollama at {self._base_url}"
        except httpx.TimeoutException:
            self._status = BackendStatus.ERROR
            logger.error("ollama_timeout", model=self._config.model)
            yield "[Local] Request timed out"
        except httpx.HTTPError as exc:
            self._status = BackendStatus.ERROR
            logger.error("ollama_http_error", error=str(exc))
            yield f"[Local] HTTP error: {exc}"

    def _build_request_body(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        """Construct the request payload for Ollama."""
        return {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    async def _stream_response(
        self, url: str, body: dict[str, Any], headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Handle SSE streaming from the OpenAI-compatible endpoint."""
        async with (
            httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client,
            client.stream("POST", url, json=body, headers=headers) as response,
        ):
            if response.status_code != 200:
                error_body = await response.aread()
                self._status = BackendStatus.ERROR
                logger.error(
                    "ollama_api_error",
                    status=response.status_code,
                    body=error_body[:500],
                )
                yield f"[Local] API error {response.status_code}"
                return

            async for line in response.aiter_lines():
                text = self._parse_sse_line(line.strip())
                if text == "__DONE__":
                    break
                if text is not None:
                    yield text

    def _parse_sse_line(self, line: str) -> str | None:
        """Parse a single SSE line, returning content, __DONE__, or None."""
        if not line or not line.startswith("data: "):
            return None
        json_str = line[len("data: "):]
        if json_str == "[DONE]":
            return "__DONE__"
        try:
            chunk = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("ollama_sse_parse_failed", chunk_preview=json_str[:200])
            return None
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            assert isinstance(delta, str), "streamed content must be a string"
        return delta if delta else None

    async def _non_stream_response(
        self, url: str, body: dict[str, Any], headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Handle non-streaming response."""
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code != 200:
                self._status = BackendStatus.ERROR
                logger.error(
                    "ollama_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                yield f"[Local] API error {response.status_code}"
                return

            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if content:
                assert isinstance(content, str), "response content must be a string"
                yield content

    async def health_check(self) -> bool:
        """Check if Ollama is running by hitting GET /api/tags."""
        url = f"{self._base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                healthy = resp.status_code == 200
                self._status = BackendStatus.AVAILABLE if healthy else BackendStatus.OFFLINE
                return healthy
        except httpx.ConnectError:
            self._status = BackendStatus.OFFLINE
            return False
        except httpx.HTTPError:
            self._status = BackendStatus.ERROR
            return False

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage for tracking (no billing for local models)."""
        assert tokens_in >= 0, "tokens_in must be non-negative"
        assert tokens_out >= 0, "tokens_out must be non-negative"
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
