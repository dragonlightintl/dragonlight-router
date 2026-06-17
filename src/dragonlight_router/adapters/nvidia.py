"""NVIDIA NIM adapter implementing GenerativeBackend protocol.

NVIDIA NIM is OpenAI-compatible -- uses the same /v1/chat/completions
request/response format with a different base URL and model naming
convention (namespace/model, e.g. ``nvidia/llama-3.1-nemotron-70b-instruct``).
"""

from __future__ import annotations

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com"


class NvidiaBackend(OpenAICompatibleBackend):
    """NVIDIA NIM backend adapter using OpenAI-compatible chat completions."""

    _provider_name = "NVIDIA NIM"
    _default_base_url = _DEFAULT_BASE_URL
