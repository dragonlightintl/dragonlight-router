"""Adapter package for dragonlight-router.

Exports all GenerativeBackend implementations for supported providers.
"""

from dragonlight_router.core.types import BackendConfig, GenerativeBackend

from .anthropic import AnthropicBackend
from .cerebras import CerebrasBackend
from .cohere import CohereBackend
from .google import GoogleBackend
from .groq import GroqBackend
from .local import LocalBackend
from .mistral import MistralBackend
from .nvidia import NvidiaBackend
from .openai import OpenAIBackend
from .openrouter import OpenRouterBackend
from .together import TogetherBackend

__all__ = [
    "AnthropicBackend",
    "CerebrasBackend",
    "CohereBackend",
    "GoogleBackend",
    "GroqBackend",
    "LocalBackend",
    "MistralBackend",
    "NvidiaBackend",
    "OpenAIBackend",
    "OpenRouterBackend",
    "TogetherBackend",
    "create_adapter",
]

_PROVIDER_MAP: dict[str, type[GenerativeBackend]] = {
    "anthropic": AnthropicBackend,
    "cerebras": CerebrasBackend,
    "cohere": CohereBackend,
    "google": GoogleBackend,
    "groq": GroqBackend,
    "local": LocalBackend,
    "mistral": MistralBackend,
    "nvidia": NvidiaBackend,
    "openai": OpenAIBackend,
    "openrouter": OpenRouterBackend,
    "together": TogetherBackend,
}


def create_adapter(config: BackendConfig) -> GenerativeBackend:
    """Instantiate the correct adapter for a given BackendConfig."""
    assert isinstance(config, BackendConfig), f"config must be BackendConfig, got {type(config)}"
    assert isinstance(config.provider, str) and config.provider, "config.provider must be a non-empty string"

    cls = _PROVIDER_MAP.get(config.provider)
    if cls is None:
        raise ValueError(f"No adapter registered for provider: {config.provider!r}")
    return cls(config)
