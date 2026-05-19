"""Tests for dragonlight_router.core.types — frozen dataclasses and enums."""
from __future__ import annotations

import pytest

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendError,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    CatalogEntry,
    ComplexityEstimate,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    ModelScore,
    ProviderConfig,
)


class TestBackendTier:
    def test_values(self):
        assert BackendTier.LOCAL.value == "local"
        assert BackendTier.HAIKU.value == "haiku"
        assert BackendTier.SONNET.value == "sonnet"
        assert BackendTier.OPUS.value == "opus"

    def test_all_members(self):
        assert len(BackendTier) == 4


class TestBackendStatus:
    def test_values(self):
        assert BackendStatus.AVAILABLE.value == "available"
        assert BackendStatus.RATE_LIMITED.value == "rate_limited"
        assert BackendStatus.DAILY_CAP_HIT.value == "daily_cap_hit"
        assert BackendStatus.ERROR.value == "error"
        assert BackendStatus.CIRCUIT_OPEN.value == "circuit_open"
        assert BackendStatus.OFFLINE.value == "offline"

    def test_all_members(self):
        assert len(BackendStatus) == 6


class TestBackendCapabilities:
    def test_frozen(self):
        caps = BackendCapabilities(
            max_context_tokens=128_000,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        )
        with pytest.raises(Exception):
            caps.max_context_tokens = 256_000  # type: ignore[misc]

    def test_fields(self):
        caps = BackendCapabilities(
            max_context_tokens=32_000,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        )
        assert caps.max_context_tokens == 32_000
        assert caps.supports_tool_use is False


class TestBackendCostProfile:
    def test_defaults(self):
        cost = BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0)
        assert cost.cache_read_per_mtok == 0.0
        assert cost.cache_write_per_mtok == 0.0

    def test_frozen(self):
        cost = BackendCostProfile(input_per_mtok=15.0, output_per_mtok=75.0)
        with pytest.raises(Exception):
            cost.input_per_mtok = 0.0  # type: ignore[misc]


class TestBackendRateLimits:
    def test_fields(self):
        limits = BackendRateLimits(rpm=30, rpd=1000, tpm=6000, daily_token_cap=0)
        assert limits.rpm == 30
        assert limits.rpd == 1000
        assert limits.tpm == 6000
        assert limits.daily_token_cap == 0


class TestBackendConfig:
    def test_frozen(self):
        config = BackendConfig(
            name="test",
            provider="groq",
            model="llama-3.3-70b",
            tier=BackendTier.HAIKU,
            base_url="https://api.groq.com/openai/v1",
            env_key="GROQ_API_KEY",
            capabilities=BackendCapabilities(128_000, True, True, True, True),
            cost=BackendCostProfile(0.0, 0.0),
            rate_limits=BackendRateLimits(30, 1000, 6000, 0),
        )
        with pytest.raises(Exception):
            config.name = "changed"  # type: ignore[misc]

    def test_default_priority(self):
        config = BackendConfig(
            name="test",
            provider="groq",
            model="llama-3.3-70b",
            tier=BackendTier.HAIKU,
            base_url="https://api.groq.com/openai/v1",
            env_key=None,
            capabilities=BackendCapabilities(128_000, True, True, True, True),
            cost=BackendCostProfile(0.0, 0.0),
            rate_limits=BackendRateLimits(30, 1000, 6000, 0),
        )
        assert config.priority == 0


class TestProviderConfig:
    def test_fields(self):
        pc = ProviderConfig(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            catalog_url="https://api.groq.com/openai/v1/models",
            env_key="GROQ_API_KEY",
            model_prefix="groq/",
            rpm_limit=30,
            rpd_limit=1000,
            tpm_limit=6000,
        )
        assert pc.name == "groq"
        assert pc.rpd_limit == 1000

    def test_nullable_limits(self):
        pc = ProviderConfig(
            name="nvidia_nim",
            base_url="https://integrate.api.nvidia.com/v1",
            catalog_url="https://integrate.api.nvidia.com/v1/models",
            env_key="NVIDIA_NIM_API_KEY",
            model_prefix="nvidia_nim/",
            rpm_limit=40,
            rpd_limit=None,
            tpm_limit=None,
        )
        assert pc.rpd_limit is None
        assert pc.tpm_limit is None


class TestDispatchOrder:
    def test_frozen(self):
        order = DispatchOrder(
            intent_category="engineering_build",
            specific_intent="code.generate",
            operator_message="Write a function",
            system_prompt="You are an engineer",
            context_tokens=2000,
        )
        with pytest.raises(Exception):
            order.context_tokens = 5000  # type: ignore[misc]

    def test_defaults(self):
        order = DispatchOrder(
            intent_category="general",
            specific_intent="chat",
            operator_message="hello",
            system_prompt="sys",
            context_tokens=100,
        )
        assert order.requires_tool_use is False
        assert order.persona is None
        assert order.stream_id is None


class TestEngineResponse:
    def test_fields(self):
        resp = EngineResponse(
            content="result",
            backend_used="groq_llama70b",
            backend_tier=BackendTier.HAIKU,
            tokens_in=100,
            tokens_out=500,
            estimated_cost_usd=0.0,
            latency_ms=1234.5,
            was_fallback=False,
            fallback_chain=[],
        )
        assert resp.backend_tier == BackendTier.HAIKU
        assert resp.latency_ms == 1234.5


class TestDispatchFailure:
    def test_fields(self):
        failure = DispatchFailure(
            message="All exhausted",
            attempted_backends=["a", "b"],
            error_details={"a": "rpm", "b": "circuit"},
        )
        assert len(failure.attempted_backends) == 2


class TestModelScore:
    def test_fields(self):
        score = ModelScore(
            model_id="groq/llama-3.3-70b",
            provider="groq",
            rank=75,
            budget_score=80.0,
            health_score=100.0,
            composite=82.5,
        )
        assert score.composite == 82.5


class TestCatalogEntry:
    def test_defaults(self):
        entry = CatalogEntry(model_id="groq/llama-3.3-70b", provider="groq")
        assert entry.created is None

    def test_with_created(self):
        entry = CatalogEntry(model_id="nim/kimi-k2.6", provider="nvidia_nim", created=1747000000)
        assert entry.created == 1747000000


class TestComplexityEstimate:
    def test_fields(self):
        est = ComplexityEstimate(
            tier=BackendTier.SONNET,
            confidence=0.85,
            signals=["large_context(50000)", "requires_tools"],
        )
        assert est.tier == BackendTier.SONNET
        assert len(est.signals) == 2


class TestBackendError:
    def test_defaults(self):
        err = BackendError(
            backend_name="groq_llama70b",
            error_type="timeout",
            message="Request timed out",
        )
        assert err.http_status is None
        assert err.retryable is False
