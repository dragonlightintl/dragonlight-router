"""Base class for OpenAI-compatible backend adapters.

Extracts the shared generate(), health_check(), and record_usage()
logic used by cerebras, groq, mistral, nvidia, openai, openrouter,
and together adapters. Provider-specific adapters inherit from
OpenAICompatibleBackend and override only what differs (base URL
defaults, auth header format, endpoint path construction).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
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

    # Retry configuration — exponential backoff with jitter before circuit breaker.
    _max_retries: int = 3
    _base_delay_s: float = 0.5
    _max_delay_s: float = 8.0
    _jitter_factor: float = 0.5

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
        """Resolve the base URL from config or fall back to the provider default.

        Strips trailing version path (e.g. /v1, /v2) from config base URLs
        when _completions_path already contains a version prefix, preventing
        doubled paths like /v1/v1/chat/completions.
        """
        if self._config.base_url:
            url = self._config.base_url.rstrip("/")
            # Deduplicate version path: if completions_path starts with /v1
            # and base_url ends with /v1, strip it from base_url
            if self._completions_path.startswith("/v"):
                version_prefix = self._completions_path.split("/")[1]  # e.g. "v1"
                if url.endswith(f"/{version_prefix}"):
                    url = url[: -len(version_prefix) - 1]
            return url
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
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict[str, object]:
        """Construct the JSON body for chat completions."""
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"
        body: dict[str, object] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        return body

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """Determine if an HTTP status code warrants a retry.

        Retries on: 429 (rate limited), 500, 502, 503, 504 (server errors).
        Does NOT retry on 4xx client errors (except 429).
        """
        assert isinstance(status_code, int), "status_code must be an int"
        return status_code == 429 or status_code >= 500

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter for a given attempt.

        Uses full jitter: uniform random in [0, min(base * 2^attempt, max_delay)].
        """
        assert attempt >= 0, "attempt must be non-negative"
        exp_delay = min(self._base_delay_s * (2**attempt), self._max_delay_s)
        jitter = random.uniform(0, self._jitter_factor * exp_delay)
        delay: float = exp_delay + jitter
        assert delay >= 0, "computed delay must be non-negative"
        return delay

    def _handle_http_retry(
        self,
        exc: httpx.HTTPStatusError,
        attempt: int,
    ) -> float | None:
        """Handle an HTTP error during retry loop.

        Returns the backoff delay if the error is retryable and retries remain,
        or None if this was the last attempt. Raises RuntimeError for
        non-retryable status codes.
        """
        if not self._is_retryable_status(exc.response.status_code):
            self._status = BackendStatus.ERROR
            raise RuntimeError(f"{self._provider_name} API error: {exc}") from exc
        if attempt < self._max_retries - 1:
            delay = self._compute_backoff_delay(attempt)
            logger.warning(
                "retrying_after_http_error",
                provider=self._provider_name,
                status_code=exc.response.status_code,
                attempt=attempt + 1,
                max_retries=self._max_retries,
                delay_s=round(delay, 3),
            )
            return delay
        return None

    def _handle_connection_retry(
        self,
        exc: httpx.ConnectError | httpx.TimeoutException,
        attempt: int,
    ) -> float | None:
        """Handle a connection/timeout error during retry loop.

        Returns the backoff delay if retries remain, or None on last attempt.
        """
        if attempt < self._max_retries - 1:
            delay = self._compute_backoff_delay(attempt)
            logger.warning(
                "retrying_after_connection_error",
                provider=self._provider_name,
                error_type=type(exc).__name__,
                attempt=attempt + 1,
                max_retries=self._max_retries,
                delay_s=round(delay, 3),
            )
            return delay
        return None

    def _raise_retries_exhausted(self, last_exc: Exception | None) -> None:
        """Raise a RuntimeError after all retry attempts are exhausted."""
        self._status = BackendStatus.ERROR
        if isinstance(last_exc, httpx.HTTPStatusError):
            raise RuntimeError(f"{self._provider_name} API error: {last_exc}") from last_exc
        raise RuntimeError(f"{self._provider_name} connection failed: {last_exc}") from last_exc

    def _validate_and_prepare_request(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stream: bool,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        """Validate API key and build (url, headers, payload) for a request."""
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
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )
        return url, headers, payload

    # DEVIATION CS-004: generate is 59 lines.
    # Justification: Async generator with yield inside a retry loop cannot be extracted
    # into a sub-function without breaking the generator protocol. Retry handling and
    # request preparation are already extracted into helpers.
    # Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using an OpenAI-compatible chat completions API.

        Implements retry with exponential backoff + jitter for transient
        errors (5xx, 429, connection failures, timeouts) before propagating
        the error to the health tracker / circuit breaker.
        """
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        url, headers, payload = self._validate_and_prepare_request(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with (
                    self._make_client(timeout=30.0) as client,
                    client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                        timeout=60.0,
                    ) as response,
                ):
                    response.raise_for_status()
                    if stream:
                        async for chunk in self._parse_sse_stream(response):
                            yield chunk
                    else:
                        content = self._extract_non_stream_content(response.json())
                        if content:
                            assert isinstance(content, str), "response content must be a string"
                            yield content
                return  # Success — exit retry loop
            except httpx.HTTPStatusError as e:
                last_exc = e
                delay = self._handle_http_retry(e, attempt)
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                delay = self._handle_connection_retry(e, attempt)
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"{self._provider_name} request failed: {e}") from e

        self._raise_retries_exhausted(last_exc)

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict[str, Any]:
        """Generate a non-streaming response with tool-use support.

        Unlike generate(), this returns the full parsed message dict from the
        API response, including tool_calls when present. Uses non-streaming
        mode because tool_call accumulation from SSE deltas adds significant
        complexity for minimal latency benefit in agentic loops.

        Returns a dict with keys:
            content: str | None — text content (may be None for pure tool calls)
            tool_calls: list[dict] | None — tool_call objects from the API
            finish_reason: str | None — "stop", "tool_calls", etc.
        """
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages must not be empty"

        url, headers, payload = self._validate_and_prepare_request(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with self._make_client(timeout=60.0) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    return self._extract_non_stream_full_message(response.json())
            except httpx.HTTPStatusError as e:
                last_exc = e
                delay = self._handle_http_retry(e, attempt)
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                delay = self._handle_connection_retry(e, attempt)
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"{self._provider_name} request failed: {e}") from e

        self._raise_retries_exhausted(last_exc)
        # Unreachable — _raise_retries_exhausted always raises — but keeps mypy happy
        raise RuntimeError("retries exhausted")  # pragma: no cover

    def _extract_non_stream_full_message(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Extract the full message dict from a non-streaming API response.

        Returns a dict with content, tool_calls, and finish_reason suitable
        for tool-use conversations.
        """
        choices = response_data.get("choices")
        if not choices:
            return {"content": "", "tool_calls": None, "finish_reason": None}

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content) if content else None

        tool_calls = message.get("tool_calls")

        return {
            "content": content or "",
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }

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
        content = delta.get("content")
        if content is None:
            return None
        if not isinstance(content, str):
            return str(content) if content else None
        return content

    def _extract_non_stream_content(self, response_data: dict[str, Any]) -> str:
        """Extract content from a non-streaming response."""
        choices = response_data.get("choices")
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            return str(content) if content else ""
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
