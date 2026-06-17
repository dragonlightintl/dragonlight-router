"""Groq adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import httpx  # noqa: F401 — re-exported for test mock patching

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqBackend(OpenAICompatibleBackend):
    """Groq backend adapter using Groq's OpenAI-compatible chat completions API."""

    _provider_name = "Groq"
    _default_base_url = _GROQ_BASE_URL
    _completions_path = "/chat/completions"
    _models_path = "/models"
