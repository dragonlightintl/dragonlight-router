"""Tests for the adapter factory (create_adapter).

Spec traceability: TM-005 (Adapter factory)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dragonlight_router.adapters import create_adapter
from dragonlight_router.adapters.anthropic import AnthropicBackend
from dragonlight_router.adapters.cerebras import CerebrasBackend
from dragonlight_router.adapters.cohere import CohereBackend
from dragonlight_router.adapters.google import GoogleBackend
from dragonlight_router.adapters.groq import GroqBackend
from dragonlight_router.adapters.local import LocalBackend
from dragonlight_router.adapters.mistral import MistralBackend
from dragonlight_router.adapters.nvidia import NvidiaBackend
from dragonlight_router.adapters.openai import OpenAIBackend
from dragonlight_router.adapters.openrouter import OpenRouterBackend
from dragonlight_router.adapters.together import TogetherBackend
from dragonlight_router.core.types import GenerativeBackend


# ---------------------------------------------------------------------------
# Provider → expected class mapping
# ---------------------------------------------------------------------------

PROVIDER_CLASS_MAP: dict[str, type] = {
    "openrouter": OpenRouterBackend,
    "openai": OpenAIBackend,
    "groq": GroqBackend,
    "anthropic": AnthropicBackend,
    "cerebras": CerebrasBackend,
    "google": GoogleBackend,
    "local": LocalBackend,
    "cohere": CohereBackend,
    "mistral": MistralBackend,
    "together": TogetherBackend,
    "nvidia": NvidiaBackend,
}


# ---------------------------------------------------------------------------
# test_create_adapter_all_providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", list(PROVIDER_CLASS_MAP.keys()))
def test_create_adapter_all_providers(make_backend_config, provider):
    """create_adapter returns a GenerativeBackend instance for every supported provider."""
    config = make_backend_config(
        name=f"{provider}-test",
        provider=provider,
        model=f"{provider}-test-model",
        base_url=f"https://api.{provider}.example.com/v1",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        adapter = create_adapter(config)

    assert isinstance(adapter, GenerativeBackend)


# ---------------------------------------------------------------------------
# test_create_adapter_unknown_provider
# ---------------------------------------------------------------------------


def test_create_adapter_unknown_provider(make_backend_config):
    """create_adapter raises ValueError for an unknown provider name."""
    config = make_backend_config(
        name="unknown-test",
        provider="nonexistent-provider",
        model="some-model",
        base_url="https://api.nowhere.example.com/v1",
        env_key=None,
    )
    with pytest.raises(ValueError, match="No adapter registered for provider"):
        create_adapter(config)


# ---------------------------------------------------------------------------
# test_create_adapter_returns_correct_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,expected_class",
    list(PROVIDER_CLASS_MAP.items()),
    ids=list(PROVIDER_CLASS_MAP.keys()),
)
def test_create_adapter_returns_correct_class(
    make_backend_config, provider, expected_class
):
    """create_adapter returns the correct concrete class for each provider."""
    config = make_backend_config(
        name=f"{provider}-test",
        provider=provider,
        model=f"{provider}-test-model",
        base_url=f"https://api.{provider}.example.com/v1",
        env_key=None,
    )
    with patch.dict("os.environ", {}, clear=True):
        adapter = create_adapter(config)

    assert type(adapter) is expected_class
