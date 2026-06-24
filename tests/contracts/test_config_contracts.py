"""Contract tests for configuration schema.

Verifies that RouterConfig and related Pydantic models maintain their
documented field contracts, defaults, validation, and serialization.

Spec traceability: config/schema.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dragonlight_router.config.schema import (
    IntentClassificationConfig,
    PinnedDispatchConfig,
    ProviderSchema,
    RateLimitSchema,
    RouterConfig,
)
from dragonlight_router.core.errors import RouterConfigError

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Contract: RouterConfig defaults
# ---------------------------------------------------------------------------


class TestRouterConfigDefaults:
    """RouterConfig must have documented default values for all fields."""

    def test_default_state_dir(self) -> None:
        """state_dir defaults to './router_state'."""
        config = RouterConfig()
        assert config.state_dir == Path("./router_state")

    def test_default_catalog_ttl_hours(self) -> None:
        """catalog_ttl_hours defaults to 24."""
        config = RouterConfig()
        assert config.catalog_ttl_hours == 24

    def test_default_budget_flush_interval(self) -> None:
        """budget_flush_interval_s defaults to 5."""
        config = RouterConfig()
        assert config.budget_flush_interval_s == 5

    def test_default_top_n(self) -> None:
        """default_top_n defaults to 12."""
        config = RouterConfig()
        assert config.default_top_n == 12

    def test_default_max_consecutive_same_provider(self) -> None:
        """max_consecutive_same_provider defaults to 2."""
        config = RouterConfig()
        assert config.max_consecutive_same_provider == 2

    def test_default_providers_empty(self) -> None:
        """providers defaults to empty list."""
        config = RouterConfig()
        assert config.providers == []

    def test_default_admin_api_key_none(self) -> None:
        """admin_api_key defaults to None."""
        config = RouterConfig()
        assert config.admin_api_key is None

    def test_default_intent_classification(self) -> None:
        """intent_classification defaults to IntentClassificationConfig()."""
        config = RouterConfig()
        assert isinstance(config.intent_classification, IntentClassificationConfig)
        assert config.intent_classification.enabled is False

    def test_default_pinned_dispatch(self) -> None:
        """pinned_dispatch defaults to PinnedDispatchConfig()."""
        config = RouterConfig()
        assert isinstance(config.pinned_dispatch, PinnedDispatchConfig)
        assert config.pinned_dispatch.honor_health is True


# ---------------------------------------------------------------------------
# Contract: RouterConfig has all documented fields
# ---------------------------------------------------------------------------


class TestRouterConfigFields:
    """RouterConfig must expose all documented fields."""

    EXPECTED_FIELDS = {
        "state_dir",
        "catalog_ttl_hours",
        "budget_flush_interval_s",
        "default_top_n",
        "max_consecutive_same_provider",
        "providers",
        "admin_api_key",
        "intent_classification",
        "pinned_dispatch",
    }

    def test_all_fields_present(self) -> None:
        """Every documented field must exist on the model."""
        actual_fields = set(RouterConfig.model_fields.keys())
        for field in self.EXPECTED_FIELDS:
            assert field in actual_fields, f"RouterConfig missing field: {field}"

    def test_no_undocumented_fields(self) -> None:
        """RouterConfig should not have fields beyond what is documented."""
        actual_fields = set(RouterConfig.model_fields.keys())
        extra = actual_fields - self.EXPECTED_FIELDS
        assert not extra, f"RouterConfig has undocumented fields: {extra}"


# ---------------------------------------------------------------------------
# Contract: ProviderSchema validation
# ---------------------------------------------------------------------------


class TestProviderSchemaContract:
    """ProviderSchema must validate required fields."""

    def test_valid_provider(self) -> None:
        """A fully specified provider must validate."""
        provider = ProviderSchema(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model_prefix="groq_",
            rate_limits=RateLimitSchema(rpm=30),
        )
        assert provider.name == "groq"
        assert provider.base_url == "https://api.groq.com/openai/v1"

    def test_missing_name_raises(self) -> None:
        """ProviderSchema without name must raise ValidationError."""
        with pytest.raises(ValidationError):
            ProviderSchema(
                base_url="https://api.test.com",
                model_prefix="t_",
                rate_limits=RateLimitSchema(rpm=10),
            )  # type: ignore[call-arg]

    def test_missing_base_url_raises(self) -> None:
        """ProviderSchema without base_url must raise ValidationError."""
        with pytest.raises(ValidationError):
            ProviderSchema(
                name="test",
                model_prefix="t_",
                rate_limits=RateLimitSchema(rpm=10),
            )  # type: ignore[call-arg]

    def test_missing_rate_limits_raises(self) -> None:
        """ProviderSchema without rate_limits must raise ValidationError."""
        with pytest.raises(ValidationError):
            ProviderSchema(
                name="test",
                base_url="https://api.test.com",
                model_prefix="t_",
            )  # type: ignore[call-arg]

    def test_missing_model_prefix_raises(self) -> None:
        """ProviderSchema without model_prefix must raise ValidationError."""
        with pytest.raises(ValidationError):
            ProviderSchema(
                name="test",
                base_url="https://api.test.com",
                rate_limits=RateLimitSchema(rpm=10),
            )  # type: ignore[call-arg]

    def test_optional_fields_default_none(self) -> None:
        """Optional fields (catalog_url, env_key) default to None."""
        provider = ProviderSchema(
            name="test",
            base_url="https://api.test.com",
            model_prefix="t_",
            rate_limits=RateLimitSchema(rpm=10),
        )
        assert provider.catalog_url is None
        assert provider.env_key is None


# ---------------------------------------------------------------------------
# Contract: RateLimitSchema
# ---------------------------------------------------------------------------


class TestRateLimitSchemaContract:
    """RateLimitSchema must validate correctly."""

    def test_rpm_required(self) -> None:
        """rpm is a required field."""
        with pytest.raises(ValidationError):
            RateLimitSchema()  # type: ignore[call-arg]

    def test_optional_fields_default_none(self) -> None:
        """rpd, tpm, daily_token_cap default to None."""
        rl = RateLimitSchema(rpm=60)
        assert rl.rpd is None
        assert rl.tpm is None
        assert rl.daily_token_cap is None

    def test_all_fields_set(self) -> None:
        """All fields can be set explicitly."""
        rl = RateLimitSchema(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000)
        assert rl.rpm == 60
        assert rl.rpd == 14400
        assert rl.tpm == 100000
        assert rl.daily_token_cap == 1000000


# ---------------------------------------------------------------------------
# Contract: IntentClassificationConfig
# ---------------------------------------------------------------------------


class TestIntentClassificationConfigContract:
    """IntentClassificationConfig must have documented defaults."""

    def test_defaults(self) -> None:
        """All fields must match documented defaults."""
        ic = IntentClassificationConfig()
        assert ic.enabled is False
        assert ic.timeout_ms == 100
        assert ic.cache_ttl_s == 300
        assert ic.cache_max_entries == 5000
        assert ic.confidence_threshold == 0.6
        assert ic.profile_confidence_threshold == 0.3
        assert ic.spectrograph_match_weight == 0.15
        assert ic.spectrograph_match_weight_governor == 0.05


# ---------------------------------------------------------------------------
# Contract: PinnedDispatchConfig
# ---------------------------------------------------------------------------


class TestPinnedDispatchConfigContract:
    """PinnedDispatchConfig must have documented defaults."""

    def test_defaults(self) -> None:
        """honor_health defaults to True."""
        pd = PinnedDispatchConfig()
        assert pd.honor_health is True


# ---------------------------------------------------------------------------
# Contract: Invalid configs produce errors, not crashes
# ---------------------------------------------------------------------------


class TestInvalidConfigHandling:
    """Invalid configuration values must produce clear errors."""

    def test_unknown_field_ignored_or_rejected(self) -> None:
        """Extra/unknown fields must not crash."""
        import contextlib

        with contextlib.suppress(ValidationError):
            RouterConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_wrong_type_raises_validation_error(self) -> None:
        """Wrong type for catalog_ttl_hours must raise ValidationError."""
        with pytest.raises(ValidationError):
            RouterConfig(catalog_ttl_hours="not_an_int")  # type: ignore[arg-type]

    def test_wrong_type_for_providers_raises(self) -> None:
        """providers must be a list, not a string."""
        with pytest.raises(ValidationError):
            RouterConfig(providers="not_a_list")  # type: ignore[arg-type]

    def test_router_config_error_is_structured(self) -> None:
        """RouterConfigError must carry message and optional config_path."""
        err = RouterConfigError(message="bad config", config_path="/tmp/test.yaml")
        assert err.message == "bad config"
        assert err.config_path == "/tmp/test.yaml"

    def test_router_config_error_path_optional(self) -> None:
        """RouterConfigError config_path defaults to None."""
        err = RouterConfigError(message="bad config")
        assert err.config_path is None


# ---------------------------------------------------------------------------
# Contract: Serialization / deserialization
# ---------------------------------------------------------------------------


class TestConfigSerialization:
    """Config types must be serializable to/from dicts."""

    def test_router_config_to_dict(self) -> None:
        """RouterConfig must be convertible to a dict."""
        config = RouterConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "state_dir" in data
        assert "providers" in data

    def test_router_config_round_trip(self) -> None:
        """RouterConfig -> dict -> RouterConfig must preserve values."""
        original = RouterConfig(
            catalog_ttl_hours=48,
            default_top_n=20,
            admin_api_key="test-key",
        )
        data = original.model_dump()
        restored = RouterConfig(**data)
        assert restored.catalog_ttl_hours == original.catalog_ttl_hours
        assert restored.default_top_n == original.default_top_n
        assert restored.admin_api_key == original.admin_api_key

    def test_provider_schema_to_dict(self) -> None:
        """ProviderSchema must be convertible to a dict."""
        provider = ProviderSchema(
            name="test",
            base_url="https://api.test.com",
            model_prefix="t_",
            rate_limits=RateLimitSchema(rpm=60),
        )
        data = provider.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert "rate_limits" in data

    def test_provider_schema_round_trip(self) -> None:
        """ProviderSchema -> dict -> ProviderSchema must preserve values."""
        original = ProviderSchema(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            catalog_url="https://api.groq.com/openai/v1/models",
            env_key="GROQ_API_KEY",
            model_prefix="groq_",
            rate_limits=RateLimitSchema(rpm=30, rpd=14400),
        )
        data = original.model_dump()
        restored = ProviderSchema(**data)
        assert restored.name == original.name
        assert restored.catalog_url == original.catalog_url
        assert restored.rate_limits.rpm == original.rate_limits.rpm

    def test_config_frozen(self) -> None:
        """RouterConfig must be frozen (immutable)."""
        config = RouterConfig()
        with pytest.raises(ValidationError):
            config.catalog_ttl_hours = 999  # type: ignore[misc]

    def test_provider_schema_frozen(self) -> None:
        """ProviderSchema must be frozen (immutable)."""
        provider = ProviderSchema(
            name="test",
            base_url="https://api.test.com",
            model_prefix="t_",
            rate_limits=RateLimitSchema(rpm=60),
        )
        with pytest.raises(ValidationError):
            provider.name = "changed"  # type: ignore[misc]
