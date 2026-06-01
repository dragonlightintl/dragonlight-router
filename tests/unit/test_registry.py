"""Tests for dragonlight_router.core.registry — BackendRegistry."""
from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
)


def _make_config(name: str = "test_backend", provider: str = "groq") -> BackendConfig:
    return BackendConfig(
        name=name,
        provider=provider,
        model="llama-3.3-70b",
        tier=BackendTier.SIMPLE,
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        capabilities=BackendCapabilities(128_000, True, True, True, True),
        cost=BackendCostProfile(0.0, 0.0),
        rate_limits=BackendRateLimits(30, 1000, 6000, 0),
    )


class FakeBackend:
    """Minimal GenerativeBackend implementation for testing."""

    def __init__(self, cfg: BackendConfig) -> None:
        self._config = cfg
        self._status = BackendStatus.AVAILABLE

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def status(self) -> BackendStatus:
        return self._status

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        yield "hello"
        yield ""

    async def health_check(self) -> bool:
        return True

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        pass


class TestBackendRegistry:
    def test_register_and_get(self):
        registry = BackendRegistry()
        backend = FakeBackend(_make_config("test_a"))
        registry.register(backend)

        retrieved, state = registry.get("test_a")
        assert retrieved is backend
        assert state is not None
        assert isinstance(state, BackendState)

    def test_get_nonexistent_returns_none(self):
        registry = BackendRegistry()
        backend, state = registry.get("nonexistent")
        assert backend is None
        assert state is None

    def test_duplicate_registration_asserts(self):
        registry = BackendRegistry()
        backend = FakeBackend(_make_config("dup"))
        registry.register(backend)
        with pytest.raises(AssertionError, match="Duplicate"):
            registry.register(backend)

    def test_all_backends(self):
        registry = BackendRegistry()
        registry.register(FakeBackend(_make_config("a", "groq")))
        registry.register(FakeBackend(_make_config("b", "cerebras")))
        registry.register(FakeBackend(_make_config("c", "gemini")))

        all_backends = registry.all_backends()
        assert len(all_backends) == 3
        names = [name for name, _, _ in all_backends]
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_health_snapshot(self):
        registry = BackendRegistry()
        backend = FakeBackend(_make_config("snap_test", "groq"))
        registry.register(backend)

        _, state = registry.get("snap_test")
        state.record_request()
        state.record_success(50, 100, 200.0)

        snapshot = registry.health_snapshot()
        assert "snap_test" in snapshot
        entry = snapshot["snap_test"]
        assert entry["provider"] == "groq"
        assert entry["tier"] == "haiku"
        assert entry["status"] == "available"
        assert entry["requests_today"] == 1
        assert entry["tokens_today"] == 150
        assert entry["avg_latency_ms"] == 200.0
        assert entry["circuit_open"] is False

    def test_fresh_state_per_backend(self):
        registry = BackendRegistry()
        registry.register(FakeBackend(_make_config("x")))
        registry.register(FakeBackend(_make_config("y")))

        _, state_x = registry.get("x")
        _, state_y = registry.get("y")
        state_x.record_request()

        assert state_x.requests_today == 1
        assert state_y.requests_today == 0
