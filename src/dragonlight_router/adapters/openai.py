"""OpenAI adapter implementing GenerativeBackend protocol."""

from __future__ import annotations

import httpx  # noqa: F401 — re-exported for test mock patching

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend


class OpenAIBackend(OpenAICompatibleBackend):
    """OpenAI backend adapter using the OpenAI-compatible chat completions API."""

    _provider_name = "OpenAI"
    _default_base_url = "https://api.openai.com"
