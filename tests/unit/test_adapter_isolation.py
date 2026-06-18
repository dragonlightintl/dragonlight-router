"""Tests for HAZ-014 mitigation — adapter status isolation.

Validates that each dispatch attempt creates a fresh adapter instance,
preventing concurrent adapter state mutation from affecting routing.

Spec traceability: HAZ-014 (Concurrent Adapter State Mutation)
"""
from __future__ import annotations

import os
from unittest.mock import patch

from dragonlight_router.adapters import create_adapter
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
)


def _make_config(provider: str = "groq") -> BackendConfig:
    """Create a minimal BackendConfig for adapter isolation tests."""
    return BackendConfig(
        name="test-backend",
        provider=provider,
        model="test-model",
        tier=BackendTier.SIMPLE,
        base_url="http://localhost:9999",
        env_key="TEST_KEY",
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(rpm=60, rpd=1000, tpm=100000, daily_token_cap=0),
    )


class TestAdapterIsolation:
    """HAZ-014: Fresh adapter per dispatch prevents shared mutable state."""

    def test_create_adapter_returns_fresh_instance(self):
        """[HAZ-014 AC-1] create_adapter returns a new instance each call."""
        config = _make_config()
        with patch.dict(os.environ, {"TEST_KEY": "test-value"}):
            adapter1 = create_adapter(config)
            adapter2 = create_adapter(config)
        assert adapter1 is not adapter2

    def test_fresh_adapter_starts_available(self):
        """[HAZ-014 AC-2] Fresh adapter starts with AVAILABLE status."""
        config = _make_config()
        with patch.dict(os.environ, {"TEST_KEY": "test-value"}):
            adapter = create_adapter(config)
        assert adapter.status == BackendStatus.AVAILABLE

    def test_adapter_status_mutation_does_not_affect_new_instance(self):
        """[HAZ-014 AC-3] Mutating one adapter's status doesn't affect a new one."""
        config = _make_config()
        with patch.dict(os.environ, {"TEST_KEY": "test-value"}):
            adapter1 = create_adapter(config)
            adapter1._status = BackendStatus.ERROR

            adapter2 = create_adapter(config)
        assert adapter2.status == BackendStatus.AVAILABLE
        assert adapter1.status == BackendStatus.ERROR

    def test_all_providers_return_fresh_instances(self):
        """[HAZ-014 AC-4] All provider adapters return fresh instances."""
        providers_with_env = [
            ("groq", "GROQ_KEY"),
            ("openai", "OPENAI_KEY"),
            ("anthropic", "ANTHROPIC_KEY"),
            ("mistral", "MISTRAL_KEY"),
            ("together", "TOGETHER_KEY"),
            ("cerebras", "CEREBRAS_KEY"),
            ("nvidia", "NVIDIA_KEY"),
            ("openrouter", "OPENROUTER_KEY"),
            ("cohere", "COHERE_KEY"),
            ("google", "GOOGLE_KEY"),
            ("local", None),
        ]
        for provider, env_key in providers_with_env:
            config = BackendConfig(
                name=f"test-{provider}",
                provider=provider,
                model="test-model",
                tier=BackendTier.SIMPLE if provider != "local" else BackendTier.LOCAL,
                base_url="http://localhost:9999",
                env_key=env_key,
                capabilities=BackendCapabilities(
                    max_context_tokens=4096,
                    supports_tool_use=False,
                    supports_streaming=True,
                    supports_json_mode=False,
                    supports_system_prompts=True,
                ),
                cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
                rate_limits=BackendRateLimits(rpm=60, rpd=1000, tpm=100000, daily_token_cap=0),
            )
            env_patch = {env_key: "test-value"} if env_key else {}
            with patch.dict(os.environ, env_patch):
                a1 = create_adapter(config)
                a2 = create_adapter(config)
            assert a1 is not a2, f"Provider {provider} should return fresh instances"
            assert a2.status == BackendStatus.AVAILABLE, (
                f"Provider {provider} fresh adapter should start AVAILABLE"
            )
