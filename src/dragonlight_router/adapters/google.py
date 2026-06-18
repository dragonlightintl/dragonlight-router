"""Google (Gemini) adapter implementing GenerativeBackend protocol.

Uses the Gemini REST API with streaming via server-sent events.
Supports both API-key auth (x-goog-api-key header) and Bearer token auth.
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

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_DEFAULT_TIMEOUT = 60.0


def _convert_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Convert OpenAI-style messages to Gemini format.

    Returns (contents, system_instruction | None).
    """
    assert isinstance(messages, list), "messages must be a list"
    contents: list[dict[str, Any]] = []
    system_instruction: dict[str, Any] | None = None

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")

        if role == "system":
            system_instruction = {"parts": [{"text": text}]}
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    return contents, system_instruction


def _extract_text_from_chunk(data: dict[str, Any]) -> str | None:
    """Extract generated text from a Gemini streaming JSON chunk."""
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        text: str | None = parts[0].get("text")
        return text
    except (IndexError, KeyError, TypeError):
        return None


class GoogleBackend(GenerativeBackend):
    """Google Gemini backend adapter.

    Streams responses from the Gemini REST API using httpx.
    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._status = BackendStatus.AVAILABLE
        self._tokens_in = 0
        self._tokens_out = 0
        self._base_url = config.base_url or _DEFAULT_BASE_URL

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    def _resolve_api_key(self) -> str | None:
        """Resolve the API key from the environment."""
        if not self._config.env_key:
            return None
        key: str | None = os.getenv(self._config.env_key)
        return key

    def _build_url(self, model: str) -> str:
        """Build the streaming endpoint URL."""
        base = self._base_url.rstrip("/")
        return f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"

    def _build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build request headers.

        Uses x-goog-api-key header for API key auth, or Bearer auth
        for service account tokens.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key
        else:
            token = self._resolve_api_key()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using Google Gemini's streaming API."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        api_key = self._resolve_api_key()

        if not api_key and self._config.env_key:
            self._status = BackendStatus.ERROR
            raise ValueError(
                f"google: API key not configured "
                f"(env: {self._config.env_key})"
            )

        if not api_key and not self._config.env_key:
            self._status = BackendStatus.ERROR
            raise ValueError(
                f"google: No API key configured for backend {self._config.name}"
            )

        body = self._build_request_body(messages, max_tokens=max_tokens, temperature=temperature)
        url = self._build_url(self._config.model)
        headers = self._build_headers(api_key)

        try:
            async for chunk in self._execute_stream(url, headers, body):
                yield chunk
        except httpx.TimeoutException as exc:
            self._status = BackendStatus.ERROR
            logger.error("google_api_timeout", model=self._config.model)
            raise RuntimeError(f"Google connection failed: {exc}") from exc
        except httpx.HTTPError as exc:
            self._status = BackendStatus.ERROR
            logger.error("google_api_http_error", error=str(exc))
            raise RuntimeError(f"Google API error: {exc}") from exc

    def _build_request_body(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Construct the Gemini API request body."""
        contents, system_instruction = _convert_messages(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction
        return body

    async def _execute_stream(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Execute the streaming request and yield text chunks."""
        async with (
            httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client,
            client.stream("POST", url, json=body, headers=headers) as response,
        ):
            if response.status_code != 200:
                error_body = await response.aread()
                self._status = BackendStatus.ERROR
                logger.error(
                    "google_api_error",
                    status=response.status_code,
                    body=error_body[:500],
                )
                raise RuntimeError(
                    f"Google API error {response.status_code}"
                )

            async for line in response.aiter_lines():
                text = self._parse_sse_line(line.strip())
                if text == "__DONE__":
                    break
                if text is not None:
                    yield text

    def _parse_sse_line(self, line: str) -> str | None:
        """Parse a Gemini SSE line, returning text, __DONE__, or None."""
        if not line or not line.startswith("data: "):
            return None
        json_str = line[len("data: "):]
        if json_str == "[DONE]":
            return "__DONE__"
        try:
            chunk = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("google_sse_parse_failed", chunk_preview=json_str[:200])
            return None
        text = _extract_text_from_chunk(chunk)
        if text:
            assert isinstance(text, str), "extracted text must be a string"
        return text

    async def health_check(self) -> bool:
        """Check health by verifying the API key is set and the endpoint is reachable."""
        api_key = self._resolve_api_key()
        if not api_key:
            self._status = BackendStatus.ERROR
            return False

        base = self._base_url.rstrip("/")
        url = f"{base}/v1beta/models"
        headers = {"x-goog-api-key": api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                healthy = resp.status_code == 200
                self._status = BackendStatus.AVAILABLE if healthy else BackendStatus.ERROR
                return healthy
        except httpx.HTTPError:
            self._status = BackendStatus.ERROR
            return False

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage for budget tracking."""
        assert tokens_in >= 0, "tokens_in must be non-negative"
        assert tokens_out >= 0, "tokens_out must be non-negative"
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
