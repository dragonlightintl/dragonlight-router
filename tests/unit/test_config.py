"""Tests for config/loader.py and config/schema.py.

Spec traceability: TM-016 (Configuration loading and schema)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dragonlight_router.config.loader import load_config
from dragonlight_router.config.schema import RouterConfig, ProviderSchema, RateLimitSchema
from dragonlight_router.result import Ok


class TestRouterConfigDefaults:
    def test_default_config_loads(self):
        """[TM-016 AC-1] Loading with no path returns sensible defaults."""
        config = load_config(config_path=None)
        assert isinstance(config, Ok), f"Expected Ok, got {config}"
        assert config.unwrap().default_top_n == 12
        assert config.unwrap().max_consecutive_same_provider == 2
        assert config.unwrap().catalog_ttl_hours == 24

    def test_default_state_dir(self):
        """[TM-016 AC-1] Default state directory is ./router_state."""
        config = load_config(config_path=None)
        assert config.unwrap().state_dir == Path("./router_state")


class TestConfigFromYAML:
    def test_loads_yaml_file(self, tmp_path: Path):
        """[TM-016 AC-2] Config loads from a YAML file with providers and rate limits."""
        config_data = {
            "state_dir": str(tmp_path / "state"),
            "catalog_ttl_hours": 12,
            "default_top_n": 8,
            "max_consecutive_same_provider": 3,
            "providers": [
                {
                    "name": "groq",
                    "base_url": "https://api.groq.com/openai/v1",
                    "model_prefix": "groq_",
                    "env_key": "GROQ_API_KEY",
                    "rate_limits": {"rpm": 30, "rpd": 14400},
                }
            ],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config_data))
        config = load_config(config_path=config_path)
        assert config.unwrap().catalog_ttl_hours == 12
        assert config.unwrap().default_top_n == 8
        assert len(config.unwrap().providers) == 1
        assert config.unwrap().providers[0].name == "groq"
        assert config.unwrap().providers[0].rate_limits.rpm == 30

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        """[TM-016 AC-2] Missing config file falls back to defaults."""
        config = load_config(config_path=tmp_path / "missing.yaml")
        assert isinstance(config, Ok), f"Expected Ok, got {config}"
        assert config.unwrap().providers == []


class TestProviderSchema:
    def test_minimal_provider(self):
        """[TM-016 AC-3] Minimal provider schema has correct defaults."""
        p = ProviderSchema(
            name="groq",
            base_url="http://localhost",
            model_prefix="groq_",
            rate_limits=RateLimitSchema(rpm=30),
        )
        assert p.name == "groq"
        assert p.catalog_url is None
        assert p.env_key is None
        assert p.rate_limits.rpd is None

    def test_full_provider(self):
        """[TM-016 AC-3] Full provider schema preserves all fields."""
        p = ProviderSchema(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            catalog_url="https://integrate.api.nvidia.com/v1/models",
            env_key="NVIDIA_API_KEY",
            model_prefix="nvidia_",
            rate_limits=RateLimitSchema(rpm=60, rpd=5000, tpm=100000),
        )
        assert p.catalog_url == "https://integrate.api.nvidia.com/v1/models"
        assert p.rate_limits.rpd == 5000


class TestRateLimitSchema:
    def test_defaults(self):
        """[TM-016 AC-3] Rate limit schema defaults rpd and tpm to None."""
        r = RateLimitSchema(rpm=30)
        assert r.rpm == 30
        assert r.rpd is None
        assert r.tpm is None

    def test_all_set(self):
        """[TM-016 AC-3] Rate limit schema preserves all values when set."""
        r = RateLimitSchema(rpm=60, rpd=14400, tpm=200000)
        assert r.rpd == 14400
