"""Cerebras adapter implementing GenerativeBackend protocol.

Cerebras is OpenAI-compatible -- uses the same /v1/chat/completions
request/response format with a different base URL.
"""

from __future__ import annotations

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend

_DEFAULT_BASE_URL = "https://api.cerebras.ai"


class CerebrasBackend(OpenAICompatibleBackend):
    """Cerebras backend adapter using OpenAI-compatible chat completions."""

    _provider_name = "Cerebras"
    _default_base_url = _DEFAULT_BASE_URL
