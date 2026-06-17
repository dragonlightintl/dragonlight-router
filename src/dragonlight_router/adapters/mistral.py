"""Mistral adapter implementing GenerativeBackend protocol.

Mistral's API is OpenAI-compatible (same /v1/chat/completions format),
so this adapter inherits from OpenAICompatibleBackend with Mistral's
default base URL.
"""

from __future__ import annotations

import httpx  # noqa: F401 — re-exported for test mock patching

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend


class MistralBackend(OpenAICompatibleBackend):
    """Mistral backend adapter using the OpenAI-compatible chat completions API."""

    _provider_name = "Mistral"
    _default_base_url = "https://api.mistral.ai"
