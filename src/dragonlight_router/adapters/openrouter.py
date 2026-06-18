"""OpenRouter adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import os

import httpx

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend
from dragonlight_router.core.types import (
    BackendConfig,
    BackendStatus,
)


class OpenRouterBackend(OpenAICompatibleBackend):
    """OpenRouter backend adapter."""

    _provider_name = "OpenRouter"
    _default_base_url = "https://openrouter.ai/api"

    def __init__(
        self,
        config: BackendConfig,
        *,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        # OpenRouter resolves the API key lazily from env_key at call time,
        # but we still store it during init for the base class contract.
        super().__init__(config, _transport=_transport)
        # Re-resolve: OpenRouter checks env_key presence separately
        if config.env_key:
            self._api_key = os.getenv(config.env_key, "")

    def _validate_api_key(self) -> str:
        """Validate and return the API key, raising on missing config or env var."""
        if not self._config.env_key:
            raise ValueError("API key not configured for OpenRouter backend")
        api_key = os.getenv(self._config.env_key, "")
        if not api_key:
            raise ValueError(f"Environment variable {self._config.env_key} not set")
        return api_key

    def _build_auth_headers(self) -> dict[str, str]:
        """Build auth headers using the validated API key."""
        api_key = self._validate_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        if not self._config.env_key:
            self._status = BackendStatus.ERROR
            return False

        api_key = os.getenv(self._config.env_key, "")
        if not api_key:
            self._status = BackendStatus.ERROR
            return False

        # Delegate to base with the key already validated
        self._api_key = api_key
        return await super().health_check()
