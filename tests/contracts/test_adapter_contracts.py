"""Contract tests for adapter implementations.

Verifies that ALL GenerativeBackend adapter implementations conform to the
base protocol interface, accept the correct parameter types, and handle
error conditions consistently.

Spec traceability: adapter protocol contract (GenerativeBackend)
"""

from __future__ import annotations

import inspect

import pytest

from dragonlight_router.adapters import (
    _PROVIDER_MAP,
    AnthropicBackend,
    CerebrasBackend,
    CohereBackend,
    GoogleBackend,
    GroqBackend,
    LocalBackend,
    MistralBackend,
    NvidiaBackend,
    OpenAIBackend,
    OpenRouterBackend,
    TogetherBackend,
    create_adapter,
)
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    GenerativeBackend,
)

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_ADAPTER_CLASSES: list[type] = [
    AnthropicBackend,
    CerebrasBackend,
    CohereBackend,
    GoogleBackend,
    GroqBackend,
    LocalBackend,
    MistralBackend,
    NvidiaBackend,
    OpenAIBackend,
    OpenRouterBackend,
    TogetherBackend,
]

# Provider name -> adapter class, from the canonical _PROVIDER_MAP
PROVIDER_ADAPTER_PAIRS: list[tuple[str, type]] = list(_PROVIDER_MAP.items())


def _make_config(provider: str) -> BackendConfig:
    """Build a minimal BackendConfig for the given provider."""
    return BackendConfig(
        name=f"{provider}/test-model",
        provider=provider,
        model="test-model",
        tier=BackendTier.SIMPLE,
        base_url="https://test.example.com/v1",
        env_key=None,  # No API key -> triggers missing-key behavior
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
        rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
    )


def _instantiate_adapter(provider: str, cls: type) -> GenerativeBackend:
    """Instantiate an adapter with a no-key config."""
    config = _make_config(provider)
    return cls(config)


# ---------------------------------------------------------------------------
# Contract: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every adapter must satisfy the GenerativeBackend runtime protocol."""

    @pytest.mark.parametrize(
        "provider,cls",
        PROVIDER_ADAPTER_PAIRS,
        ids=[p for p, _ in PROVIDER_ADAPTER_PAIRS],
    )
    def test_adapter_is_runtime_checkable_protocol_instance(
        self,
        provider: str,
        cls: type,
    ) -> None:
        """Each adapter instance must pass isinstance(x, GenerativeBackend)."""
        adapter = _instantiate_adapter(provider, cls)
        assert isinstance(adapter, GenerativeBackend), (
            f"{cls.__name__} does not satisfy GenerativeBackend protocol"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_has_generate_method(self, cls: type) -> None:
        """Every adapter must have a generate() method."""
        assert hasattr(cls, "generate"), f"{cls.__name__} missing generate()"
        assert callable(cls.generate), f"{cls.__name__}.generate is not callable"

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_has_health_check_method(self, cls: type) -> None:
        """Every adapter must have a health_check() method."""
        assert hasattr(cls, "health_check"), f"{cls.__name__} missing health_check()"
        assert callable(cls.health_check), f"{cls.__name__}.health_check is not callable"

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_has_record_usage_method(self, cls: type) -> None:
        """Every adapter must have a record_usage() method."""
        assert hasattr(cls, "record_usage"), f"{cls.__name__} missing record_usage()"
        assert callable(cls.record_usage), f"{cls.__name__}.record_usage is not callable"

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_has_config_property(self, cls: type) -> None:
        """Every adapter must expose a config property."""
        assert isinstance(getattr(cls, "config", None), property), (
            f"{cls.__name__} must have a 'config' property"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_has_status_property(self, cls: type) -> None:
        """Every adapter must expose a status property."""
        assert isinstance(getattr(cls, "status", None), property), (
            f"{cls.__name__} must have a 'status' property"
        )


# ---------------------------------------------------------------------------
# Contract: Method signatures
# ---------------------------------------------------------------------------


class TestMethodSignatures:
    """Verify generate() signatures match the protocol contract."""

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_generate_signature_has_correct_parameters(self, cls: type) -> None:
        """generate() must accept messages, max_tokens, temperature, stream."""
        sig = inspect.signature(cls.generate)
        params = list(sig.parameters.keys())
        assert "messages" in params, f"{cls.__name__}.generate missing 'messages' param"
        assert "max_tokens" in params, f"{cls.__name__}.generate missing 'max_tokens' param"
        assert "temperature" in params, f"{cls.__name__}.generate missing 'temperature' param"
        assert "stream" in params, f"{cls.__name__}.generate missing 'stream' param"

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_generate_is_async_generator(self, cls: type) -> None:
        """generate() must be an async generator function."""
        assert inspect.isasyncgenfunction(cls.generate), (
            f"{cls.__name__}.generate must be an async generator (yields values)"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_health_check_is_async(self, cls: type) -> None:
        """health_check() must be a coroutine function."""
        assert inspect.iscoroutinefunction(cls.health_check), (
            f"{cls.__name__}.health_check must be async"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_record_usage_accepts_tokens(self, cls: type) -> None:
        """record_usage() must accept tokens_in and tokens_out."""
        sig = inspect.signature(cls.record_usage)
        params = list(sig.parameters.keys())
        assert "tokens_in" in params, f"{cls.__name__}.record_usage missing 'tokens_in'"
        assert "tokens_out" in params, f"{cls.__name__}.record_usage missing 'tokens_out'"

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_generate_default_max_tokens(self, cls: type) -> None:
        """generate() max_tokens default must be 4096."""
        sig = inspect.signature(cls.generate)
        default = sig.parameters["max_tokens"].default
        assert default == 4096, (
            f"{cls.__name__}.generate max_tokens default is {default}, expected 4096"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_generate_default_temperature(self, cls: type) -> None:
        """generate() temperature default must be 0.7."""
        sig = inspect.signature(cls.generate)
        default = sig.parameters["temperature"].default
        assert default == 0.7, (
            f"{cls.__name__}.generate temperature default is {default}, expected 0.7"
        )

    @pytest.mark.parametrize(
        "cls",
        ALL_ADAPTER_CLASSES,
        ids=[c.__name__ for c in ALL_ADAPTER_CLASSES],
    )
    def test_generate_default_stream(self, cls: type) -> None:
        """generate() stream default must be True."""
        sig = inspect.signature(cls.generate)
        default = sig.parameters["stream"].default
        assert default is True, (
            f"{cls.__name__}.generate stream default is {default}, expected True"
        )


# ---------------------------------------------------------------------------
# Contract: Instance properties
# ---------------------------------------------------------------------------


class TestInstanceProperties:
    """Adapter instances must expose correct config/status properties."""

    @pytest.mark.parametrize(
        "provider,cls",
        PROVIDER_ADAPTER_PAIRS,
        ids=[p for p, _ in PROVIDER_ADAPTER_PAIRS],
    )
    def test_config_returns_backend_config(self, provider: str, cls: type) -> None:
        """config property must return the BackendConfig passed to __init__."""
        config = _make_config(provider)
        adapter = cls(config)
        assert adapter.config is config, (
            f"{cls.__name__}.config does not return the original BackendConfig"
        )

    @pytest.mark.parametrize(
        "provider,cls",
        PROVIDER_ADAPTER_PAIRS,
        ids=[p for p, _ in PROVIDER_ADAPTER_PAIRS],
    )
    def test_status_is_backend_status(self, provider: str, cls: type) -> None:
        """status property must return a BackendStatus enum value."""
        adapter = _instantiate_adapter(provider, cls)
        assert isinstance(adapter.status, BackendStatus), (
            f"{cls.__name__}.status returned {type(adapter.status)}, expected BackendStatus"
        )

    @pytest.mark.parametrize(
        "provider,cls",
        PROVIDER_ADAPTER_PAIRS,
        ids=[p for p, _ in PROVIDER_ADAPTER_PAIRS],
    )
    def test_initial_status_is_available(self, provider: str, cls: type) -> None:
        """Adapters must start in AVAILABLE status."""
        adapter = _instantiate_adapter(provider, cls)
        assert adapter.status == BackendStatus.AVAILABLE, (
            f"{cls.__name__} initial status is {adapter.status}, expected AVAILABLE"
        )


# ---------------------------------------------------------------------------
# Contract: Missing API key handling
# ---------------------------------------------------------------------------


class TestMissingApiKeyHandling:
    """All adapters must raise ValueError (not crash) when API key is missing.

    LocalBackend is excluded because it does not require an API key.
    """

    _API_KEY_ADAPTERS: list[tuple[str, type]] = [
        (p, c) for p, c in PROVIDER_ADAPTER_PAIRS if p != "local"
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider,cls",
        _API_KEY_ADAPTERS,
        ids=[p for p, _ in _API_KEY_ADAPTERS],
    )
    async def test_generate_raises_on_missing_key(
        self,
        provider: str,
        cls: type,
    ) -> None:
        """generate() must raise ValueError when API key is not configured."""
        adapter = _instantiate_adapter(provider, cls)
        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(ValueError, match="(?i)api key|not configured"):
            async for _ in adapter.generate(messages):
                pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider,cls",
        _API_KEY_ADAPTERS,
        ids=[p for p, _ in _API_KEY_ADAPTERS],
    )
    async def test_status_set_to_error_on_missing_key(
        self,
        provider: str,
        cls: type,
    ) -> None:
        """After a missing-key error, status must be ERROR."""
        adapter = _instantiate_adapter(provider, cls)
        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(ValueError):
            async for _ in adapter.generate(messages):
                pass
        assert adapter.status == BackendStatus.ERROR, (
            f"{cls.__name__} status after missing key is {adapter.status}, expected ERROR"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider,cls",
        _API_KEY_ADAPTERS,
        ids=[p for p, _ in _API_KEY_ADAPTERS],
    )
    async def test_health_check_returns_false_on_missing_key(
        self,
        provider: str,
        cls: type,
    ) -> None:
        """health_check() must return False when API key is not configured."""
        adapter = _instantiate_adapter(provider, cls)
        result = await adapter.health_check()
        assert result is False, (
            f"{cls.__name__}.health_check() returned {result} with no API key, expected False"
        )


# ---------------------------------------------------------------------------
# Contract: Factory function
# ---------------------------------------------------------------------------


class TestAdapterFactory:
    """create_adapter() must correctly instantiate adapters by provider name."""

    @pytest.mark.parametrize(
        "provider,expected_cls",
        PROVIDER_ADAPTER_PAIRS,
        ids=[p for p, _ in PROVIDER_ADAPTER_PAIRS],
    )
    def test_create_adapter_returns_correct_type(
        self,
        provider: str,
        expected_cls: type,
    ) -> None:
        """create_adapter must return the correct adapter class for each provider."""
        config = _make_config(provider)
        adapter = create_adapter(config)
        assert isinstance(adapter, expected_cls), (
            f"create_adapter({provider!r}) returned {type(adapter).__name__}, "
            f"expected {expected_cls.__name__}"
        )

    def test_create_adapter_raises_on_unknown_provider(self) -> None:
        """create_adapter must raise ValueError for unknown provider names."""
        config = _make_config("unknown_provider_xyz")
        with pytest.raises(ValueError, match="No adapter registered"):
            create_adapter(config)

    @pytest.mark.parametrize(
        "provider",
        list(_PROVIDER_MAP.keys()),
        ids=list(_PROVIDER_MAP.keys()),
    )
    def test_provider_map_is_complete(self, provider: str) -> None:
        """Every entry in _PROVIDER_MAP must map to a GenerativeBackend subclass."""
        cls = _PROVIDER_MAP[provider]
        adapter = _instantiate_adapter(provider, cls)
        assert isinstance(adapter, GenerativeBackend), (
            f"_PROVIDER_MAP[{provider!r}] -> {cls.__name__} does not satisfy GenerativeBackend"
        )
