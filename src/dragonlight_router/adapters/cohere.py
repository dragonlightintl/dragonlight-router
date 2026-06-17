"""Cohere adapter implementing GenerativeBackend protocol.

Cohere v2 API uses a different format from OpenAI:
- Endpoint: POST /v2/chat
- Streaming: SSE with type-based events (content-delta, message-end)
- Messages format is the same as OpenAI (role/content dicts)
"""

from __future__ import annotations

import json
import os
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


class CohereBackend(GenerativeBackend):
    """Cohere backend adapter using the Cohere v2 chat API."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._api_key = os.environ.get(config.env_key, "") if config.env_key else ""
        self._status = BackendStatus.AVAILABLE

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from config or default."""
        if self._config.base_url:
            return self._config.base_url.rstrip("/")
        return "https://api.cohere.com"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with Bearer auth."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using Cohere's v2 chat API."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        if not self._api_key:
            self._status = BackendStatus.ERROR
            raise ValueError(
                f"cohere: API key not configured (env: {self._config.env_key})"
            )

        url = f"{self._resolve_base_url()}/v2/chat"
        headers = self._build_headers()
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST", url, headers=headers, json=payload, timeout=60.0,
                ) as response:
                    response.raise_for_status()
                    if stream:
                        async for chunk in self._parse_stream(response):
                            yield chunk
                    else:
                        content = self._extract_non_stream_content(response.json())
                        if content:
                            assert isinstance(content, str), "response content must be a string"
                            yield content
            except httpx.HTTPStatusError as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Cohere API error: {e}") from e
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Cohere connection failed: {e}") from e
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Cohere request failed: {e}") from e

    async def _parse_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        """Parse Cohere v2 SSE stream and yield text chunks."""
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_type = chunk.get("type", "")
            if event_type == "content-delta":
                text = self._extract_delta_text(chunk)
                if text:
                    assert isinstance(text, str), "streamed text must be a string"
                    yield text
            elif event_type == "message-end":
                break

    def _extract_delta_text(self, chunk: dict[str, Any]) -> str:
        """Extract text from a Cohere content-delta event."""
        text: str = (
            chunk.get("delta", {})
            .get("message", {})
            .get("content", {})
            .get("text", "")
        )
        return text

    def _extract_non_stream_content(self, response_data: dict[str, Any]) -> str:
        """Extract content from a non-streaming Cohere v2 response."""
        message = response_data.get("message", {})
        content_parts = message.get("content", [])
        if isinstance(content_parts, list):
            return "".join(
                part.get("text", "")
                for part in content_parts
                if isinstance(part, dict)
            )
        if isinstance(content_parts, str):
            return content_parts
        return ""

    async def health_check(self) -> bool:
        """Check if the Cohere API is reachable."""
        if not self._api_key:
            self._status = BackendStatus.ERROR
            return False

        url = f"{self._resolve_base_url()}/v2/models"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.is_success:
                    self._status = BackendStatus.AVAILABLE
                    return True
                self._status = BackendStatus.ERROR
                return False
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
                self._status = BackendStatus.ERROR
                return False

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage -- no-op until usage tracking is wired."""
