"""OpenRouter adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import os
import json
from collections.abc import AsyncIterator
from typing import Dict, Any
import httpx

from dragonlight_router.core.types import (
    GenerativeBackend,
    BackendConfig,
    BackendStatus,
    EngineResponse,
)


class OpenRouterBackend(GenerativeBackend):
    """OpenRouter backend adapter."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._status = BackendStatus.AVAILABLE

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
        """Generate text using OpenRouter's OpenAI-compatible API."""
        if not self._config.env_key:
            raise ValueError("API key not configured for OpenRouter backend")

        api_key = os.getenv(self._config.env_key)
        if not api_key:
            raise ValueError(f"Environment variable {self._config.env_key} not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter uses OpenAI-compatible chat completions endpoint
        url = f"{self._config.base_url.rstrip('/')}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST", url, headers=headers, json=payload, timeout=60.0
                ) as response:
                    response.raise_for_status()
                    if stream:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        delta = chunk["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    # Skip invalid JSON lines
                                    continue
                    else:
                        # Non-streaming response
                        response_data = await response.json()
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            content = response_data["choices"][0].get("message", {}).get("content", "")
                            yield content
            except httpx.HTTPStatusError as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"OpenRouter API error: {e}") from e
            except Exception as e:
                self._status = BackendStatus.ERROR
                raise RuntimeError(f"OpenRouter request failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        if not self._config.env_key:
            self._status = BackendStatus.ERROR
            return False

        api_key = os.getenv(self._config.env_key)
        if not api_key:
            self._status = BackendStatus.ERROR
            return False

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # Use OpenRouter's models endpoint for health check
        url = f"{self._config.base_url.rstrip('/')}/v1/models"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.is_success:
                    self._status = BackendStatus.AVAILABLE
                    return True
                else:
                    self._status = BackendStatus.ERROR
                    return False
            except Exception:
                self._status = BackendStatus.ERROR
                return False

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage for budgeting and health tracking.
        
        In a full implementation, this would update internal metrics
        or communicate with a usage tracking service.
        """
        # For now, we just note the usage - actual tracking is done elsewhere
        pass