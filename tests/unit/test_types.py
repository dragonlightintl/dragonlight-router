"""Tests for dragonlight_router.core.types — frozen dataclasses and enums.

Spec traceability: TM-017 (Core type definitions)
"""
from __future__ import annotations

import dataclasses

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
        """[TM-017 AC-1] BackendTier enum has correct string values."""
        assert BackendTier.LOCAL.value == "local"
        assert BackendTier.SIMPLE.value == "simple"
        assert BackendTier.MODERATE.value == "moderate"
        assert BackendTier.COMPLEX.value == "complex"

    def test_all_members(self):
        """[TM-017 AC-1] BackendTier has exactly 4 members."""
        assert len(BackendTier) == 4


class TestBackendStatus:
    def test_values(self):
        """[TM-017 AC-1] BackendStatus enum has correct string values."""
        assert BackendStatus.AVAILABLE.value == "available"
        assert BackendStatus.RATE_LIMITED.value == "rate_limited"
        assert BackendStatus.DAILY_CAP_HIT.value == "daily_cap_hit"
        assert BackendStatus.ERROR.value == "error"
        assert BackendStatus.CIRCUIT_OPEN.value == "circuit_open"
        assert BackendStatus.OFFLINE.value == "offline"

    def test_key_invalid_value(self):
        """[TM-017 AC-1] BackendStatus.KEY_INVALID has correct string value."""
        assert BackendStatus.KEY_INVALID.value == "key_invalid"

    def test_all_members(self):
        """[TM-017 AC-1] BackendStatus has exactly 9 members."""
        assert len(BackendStatus) == 9


class TestBackendCapabilities:
    def test_frozen(self):
        """[TM-017 AC-2] BackendCapabilities is frozen (immutable)."""
        caps = BackendCapabilities(
            max_context_tokens=128_000,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.max_context_tokens = 256_000  # type: ignore[misc]

    def test_fields(self):
        """[TM-017 AC-2] BackendCapabilities fields are accessible."""
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
        """[TM-017 AC-2] BackendCostProfile cache fields default to 0.0."""
        cost = BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0)
        assert cost.cache_read_per_mtok == 0.0
        assert cost.cache_write_per_mtok == 0.0

    def test_frozen(self):
        """[TM-017 AC-2] BackendCostProfile is frozen (immutable)."""
        cost = BackendCostProfile(input_per_mtok=15.0, output_per_mtok=75.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cost.input_per_mtok = 0.0  # type: ignore[misc]


class TestBackendRateLimits:
    def test_fields(self):
        """[TM-017 AC-2] BackendRateLimits fields are accessible."""
        limits = BackendRateLimits(rpm=30, rpd=1000, tpm=6000, daily_token_cap=0)
        assert limits.rpm == 30
        assert limits.rpd == 1000
        assert limits.tpm == 6000
        assert limits.daily_token_cap == 0


class TestBackendConfig:
    def test_frozen(self):
        """[TM-017 AC-2] BackendConfig is frozen (immutable)."""
        config = BackendConfig(
            name="test",
            provider="groq",
            model="llama-3.3-70b",
            tier=BackendTier.SIMPLE,
            base_url="https://api.groq.com/openai/v1",
            env_key="GROQ_API_KEY",
            capabilities=BackendCapabilities(128_000, True, True, True, True),
            cost=BackendCostProfile(0.0, 0.0),
            rate_limits=BackendRateLimits(30, 1000, 6000, 0),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "changed"  # type: ignore[misc]

    def test_default_priority(self):
        """[TM-017 AC-2] BackendConfig default priority is 0."""
        config = BackendConfig(
            name="test",
            provider="groq",
            model="llama-3.3-70b",
            tier=BackendTier.SIMPLE,
            base_url="https://api.groq.com/openai/v1",
            env_key=None,
            capabilities=BackendCapabilities(128_000, True, True, True, True),
            cost=BackendCostProfile(0.0, 0.0),
            rate_limits=BackendRateLimits(30, 1000, 6000, 0),
        )
        assert config.priority == 0


class TestProviderConfig:
    def test_fields(self):
        """[TM-017 AC-2] ProviderConfig fields are accessible."""
        pc = ProviderConfig(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            catalog_url="https://api.groq.com/openai/v1/models",
            env_key="GROQ_API_KEY",
            model_prefix="groq/",
            rpm_limit=30,
            rpd_limit=1000,
            tpm_limit=6000,
            daily_token_cap=None,
        )
        assert pc.name == "groq"
        assert pc.rpd_limit == 1000

    def test_nullable_limits(self):
        """[TM-017 AC-2] ProviderConfig allows None for optional rate limits."""
        pc = ProviderConfig(
            name="nvidia_nim",
            base_url="https://integrate.api.nvidia.com/v1",
            catalog_url="https://integrate.api.nvidia.com/v1/models",
            env_key="NVIDIA_NIM_API_KEY",
            model_prefix="nvidia_nim/",
            rpm_limit=40,
            rpd_limit=None,
            tpm_limit=None,
            daily_token_cap=None,
        )
        assert pc.rpd_limit is None
        assert pc.tpm_limit is None


class TestDispatchOrder:
    def test_frozen(self):
        """[TM-017 AC-2] DispatchOrder is frozen (immutable)."""
        order = DispatchOrder(
            intent_category="engineering_build",
            specific_intent="code.generate",
            operator_message="Write a function",
            system_prompt="You are an engineer",
            context_tokens=2000,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            order.context_tokens = 5000  # type: ignore[misc]

    def test_defaults(self):
        """[TM-017 AC-2] DispatchOrder optional fields default correctly."""
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
        """[TM-017 AC-2] EngineResponse fields are accessible."""
        resp = EngineResponse(
            content="result",
            backend_used="groq_llama70b",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=100,
            tokens_out=500,
            estimated_cost_usd=0.0,
            latency_ms=1234.5,
            was_fallback=False,
            fallback_chain=[],
        )
        assert resp.backend_tier == BackendTier.SIMPLE
        assert resp.latency_ms == 1234.5


class TestDispatchFailure:
    def test_fields(self):
        """[TM-017 AC-2] DispatchFailure fields are accessible."""
        failure = DispatchFailure(
            message="All exhausted",
            attempted_backends=["a", "b"],
            error_details={"a": "rpm", "b": "circuit"},
        )
        assert len(failure.attempted_backends) == 2


class TestModelScore:
    def test_fields(self):
        """[TM-017 AC-2] ModelScore fields are accessible."""
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
        """[TM-017 AC-2] CatalogEntry defaults created to None."""
        entry = CatalogEntry(model_id="groq/llama-3.3-70b", provider="groq")
        assert entry.created is None

    def test_with_created(self):
        """[TM-017 AC-2] CatalogEntry stores created timestamp."""
        entry = CatalogEntry(model_id="nim/kimi-k2.6", provider="nvidia_nim", created=1747000000)
        assert entry.created == 1747000000


class TestComplexityEstimate:
    def test_fields(self):
        """[TM-017 AC-2] ComplexityEstimate fields are accessible."""
        est = ComplexityEstimate(
            tier=BackendTier.MODERATE,
            confidence=0.85,
            signals=["large_context(50000)", "requires_tools"],
        )
        assert est.tier == BackendTier.MODERATE
        assert len(est.signals) == 2


class TestBackendError:
    def test_defaults(self):
        """[TM-017 AC-2] BackendError defaults http_status to None and retryable to False."""
        err = BackendError(
            backend_name="groq_llama70b",
            error_type="timeout",
            message="Request timed out",
        )
        assert err.http_status is None
        assert err.retryable is False


class TestOkErrMethods:
    def test_ok_unwrap_err_raises(self):
        """[TM-017 AC-3] Ok.unwrap_err() raises AssertionError (line 35)."""
        from dragonlight_router.core.types import Ok
        ok = Ok(value=42)
        with pytest.raises(AssertionError, match="unwrap_err on Ok"):
            ok.unwrap_err()

    def test_err_unwrap_raises(self):
        """[TM-017 AC-3] Err.unwrap() raises AssertionError (line 52)."""
        from dragonlight_router.core.types import Err
        err = Err(error="something failed")
        with pytest.raises(AssertionError, match="unwrap on Err"):
            err.unwrap()
