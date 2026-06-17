"""Base class for OpenAI-compatible backend adapters.

Extracts the shared generate(), health_check(), and record_usage()
logic used by cerebras, groq, mistral, nvidia, openai, openrouter,
and together adapters. Provider-specific adapters inherit from
OpenAICompatibleBackend and override only what differs (base URL
defaults, auth header format, endpoint path construction).
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


class OpenAICompatibleBackend(GenerativeBackend):
    """Base class for adapters that speak the OpenAI chat completions protocol.

    Subclasses MUST set:
        _provider_name: str  — human-readable provider name for error messages
        _default_base_url: str — fallback base URL when config.base_url is empty

    Subclasses MAY override:
        _completions_path — default "/v1/chat/completions"
        _models_path — default "/v1/models"
        _build_auth_headers — default Bearer token auth
        _make_client — default httpx.AsyncClient (override to inject transport)
    """

    _provider_name: str = "OpenAI-compatible"
    _default_base_url: str = "https://api.openai.com"
    _completions_path: str = "/v1/chat/completions"
    _models_path: str = "/v1/models"

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

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from config or fall back to the provider default."""
        if self._config.base_url:
            return self._config.base_url.rstrip("/")
        return self._default_base_url

    def _build_auth_headers(self) -> dict[str, str]:
        """Build auth headers. Override for non-Bearer auth schemes."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _make_client(self, timeout: float) -> httpx.AsyncClient:
        """Build an AsyncClient, optionally using an injected transport."""
        if self._transport is not None:
            return httpx.AsyncClient(timeout=timeout, transport=self._transport)  # type: ignore[arg-type]
        return httpx.AsyncClient(timeout=timeout)

    def _build_request_payload(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, object]:
        """Construct the JSON body for chat completions."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"
        return {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using an OpenAI-compatible chat completions API."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        if not self._api_key:
            self._status = BackendStatus.ERROR
            raise ValueError(
                f"{self._provider_name.lower()}: API key not configured "
                f"(env: {self._config.env_key})"
            )

        base_url = self._resolve_base_url()
        url = f"{base_url}{self._completions_path}"
        headers = self._build_auth_headers()
        payload = self._build_request_payload(
            messages, max_tokens=max_tokens, temperature=temperature, stream=stream,
        )

        async with self._make_client(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST", url, headers=headers, json=payload, timeout=60.0,
                ) as response:
                    response.raise_for_status()
                    if stream:
                        async for chunk in self._parse_sse_stream(response):
                            yield chunk
                    else:
                        content = self._extract_non_stream_content(response.json())
                        if content:
                            assert isinstance(content, str), "response content must be a string"
                            yield content
            except httpx.HTTPStatusError as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"{self._provider_name} API error: {e}") from e
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"{self._provider_name} connection failed: {e}") from e
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"{self._provider_name} request failed: {e}") from e

    async def _parse_sse_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        """Parse SSE stream from response and yield content chunks."""
        async for line in response.aiter_lines():
            content = self._parse_sse_line(line)
            if content is None:
                continue
            if content == "__DONE__":
                break
            yield content

    def _parse_sse_line(self, line: str) -> str | None:
        """Parse a single SSE line, returning content, __DONE__, or None.

        Returns None for non-data lines and malformed JSON.
        Returns "__DONE__" for the [DONE] sentinel.
        Returns the content string otherwise.
        """
        if not line.startswith("data: "):
            return None
        data = line[6:]
        if data.strip() == "[DONE]":
            return "__DONE__"
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None
        return self._extract_stream_content(chunk)

    def _extract_stream_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract content from a streaming SSE JSON chunk."""
        choices = chunk.get("choices")
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        content: str | None = delta.get("content")
        if content is not None:
            assert isinstance(content, str), "streamed content must be a string"
        return content

    def _extract_non_stream_content(self, response_data: dict[str, Any]) -> str:
        """Extract content from a non-streaming response."""
        choices = response_data.get("choices")
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        assert isinstance(content, str), "response content must be a string"
        return content

    async def health_check(self) -> bool:
        """Check if the provider API is reachable via its models endpoint.

        Sets status to OFFLINE on 404 (model retired / endpoint gone),
        ERROR on other failures, and AVAILABLE on success.
        """
        if not self._api_key:
            self._status = BackendStatus.ERROR
            return False

        base_url = self._resolve_base_url()
        url = f"{base_url}{self._models_path}"
        headers = self._build_auth_headers()

        async with self._make_client(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.is_success:
                    self._status = BackendStatus.AVAILABLE
                    return True
                if response.status_code == 404:
                    self._status = BackendStatus.OFFLINE
                else:
                    self._status = BackendStatus.ERROR
                return False
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
                self._status = BackendStatus.ERROR
                return False

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage -- no-op until usage tracking is wired."""
