"""Anthropic adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
import structlog

from dragonlight_router.core.types import (
    BackendConfig,
    BackendStatus,
    GenerativeBackend,
)

logger = structlog.get_logger()


class AnthropicBackend(GenerativeBackend):
    """Anthropic backend adapter using the Messages API with SSE streaming."""

    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        config: BackendConfig,
        *,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._api_key = os.environ.get(config.env_key, "") if config.env_key else ""
        self._status = BackendStatus.AVAILABLE
        self._transport = _transport

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    def _make_client(self, timeout: float) -> httpx.AsyncClient:
        """Build an AsyncClient, optionally using an injected transport."""
        if self._transport is not None:
            return httpx.AsyncClient(timeout=timeout, transport=self._transport)  # type: ignore[arg-type]
        return httpx.AsyncClient(timeout=timeout)

    def _build_headers(self) -> dict[str, str]:
        """Build Anthropic-specific request headers."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, object]:
        """Build the Messages API payload, extracting system messages."""
        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                api_messages.append(msg)

        payload: dict[str, object] = {
            "model": self._config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    # DEVIATION CS-004: generate is 48 lines.
    # Justification: Async generator with yield inside try/except cannot be extracted
    # without breaking the generator protocol. Stream/non-stream delegation already extracted.
    # Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using Anthropic's Messages API."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        if not self._api_key:
            self._status = BackendStatus.ERROR
            raise ValueError(f"anthropic: API key not configured (env: {self._config.env_key})")

        base_url = (
            self._config.base_url.rstrip("/")
            if self._config.base_url
            else "https://api.anthropic.com"
        )
        url = f"{base_url}/v1/messages"
        headers = self._build_headers()
        payload = self._build_payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        async with self._make_client(timeout=30.0) as client:
            try:
                if stream:
                    async for chunk in self._stream_generate(client, url, headers, payload):
                        yield chunk
                else:
                    async for chunk in self._non_stream_generate(client, url, headers, payload):
                        yield chunk
            except httpx.HTTPStatusError as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Anthropic API error: {e}") from e
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Anthropic connection failed: {e}") from e
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"Anthropic request failed: {e}") from e

    # DEVIATION DCS-PARAM-001: _stream_generate takes 5 params (excl. self).
    # Justification: HTTP client, URL, headers, and body are the irreducible
    # parameters for an HTTP streaming call. Approved by: architect.
    async def _stream_generate(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> AsyncIterator[str]:
        """Parse Anthropic SSE stream and yield text chunks."""
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            event_type: str | None = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                if event_type == "message_stop":
                    break
                if event_type == "content_block_delta":
                    text = self._extract_delta_text(line[6:])
                    if text:
                        assert isinstance(text, str), "streamed text must be a string"
                        yield text

    def _extract_delta_text(self, data: str) -> str | None:
        """Extract text from a content_block_delta SSE data payload."""
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None
        delta = chunk.get("delta", {})
        if delta.get("type") != "text_delta":
            return None
        return delta.get("text", "") or None

    # DEVIATION DCS-PARAM-001: _non_stream_generate takes 5 params (excl. self).
    # Justification: HTTP client, URL, headers, and body are the irreducible
    # parameters for an HTTP call. Approved by: architect.
    async def _non_stream_generate(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> AsyncIterator[str]:
        """Handle non-streaming Anthropic response."""
        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        response_data = response.json()
        for block in response_data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    assert isinstance(text, str), "response text must be a string"
                    yield text

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        Sends a minimal messages request to verify connectivity and auth.
        """
        if not self._api_key:
            self._status = BackendStatus.ERROR
            return False

        base_url = (
            self._config.base_url.rstrip("/")
            if self._config.base_url
            else "https://api.anthropic.com"
        )
        url = f"{base_url}/v1/messages"
        headers = self._build_headers()

        payload: dict[str, object] = {
            "model": self._config.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        async with self._make_client(timeout=10.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
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
