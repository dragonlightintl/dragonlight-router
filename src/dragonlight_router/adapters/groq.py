"""Groq adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Dict, Any

from dragonlight_router.core.types import (
    GenerativeBackend,
    BackendConfig,
    BackendStatus,
)


class GroqBackend(GenerativeBackend):
    """Groq backend adapter."""

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
        """Generate text using Groq's API (mocked for development)."""
        # Mock implementation: yield a mock response without making actual API calls
        if not self._config.env_key:
            yield f"[Mock Groq] Error: API key not configured for backend {self._config.name}"
            return

        api_key = os.getenv(self._config.env_key)
        if not api_key:
            yield f"[Mock Groq] Error: Environment variable {self._config.env_key} not set"
            return

        # Simulate a mock response based on the first user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        mock_response = f"[Mock Groq {self._config.model}] Response to: {user_message[:50]}..."
        yield mock_response

    async def health_check(self) -> bool:
        """Mock health check - always returns True for development."""
        return True

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Mock usage recording - does nothing in development."""
        pass