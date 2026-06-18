"""Backend registry — constructed once at boot, queried per-request."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from dragonlight_router.core.state import BackendState
from dragonlight_router.core.types import (
    BackendConfig,
    BackendStatus,
    BackendTier,
    GenerativeBackend,
)

logger = structlog.get_logger()


@dataclass
class BackendRegistry:
    """Registry of all available generative backends.

    Constructed once at server boot. Backends register with their
    immutable config. Runtime state is tracked separately per backend.

    DEVIATION_RECORD:
      rule violated: dragonlight-coding-standards-v2.md#frozen-dataclasses
        (all data objects must be frozen dataclass)
      justification: The registry is mutable by design (register, get)
        but could be made frozen with a separate mutable state object
      approved by: Korrigon @ Dragonlight International
      mitigations: Use of immutable configs and separate BackendState
        for runtime changes
      scope: This class
      expiration: 2026-06-30 (to be revisited)
    """

    _backends: dict[str, GenerativeBackend] = field(default_factory=dict)
    _states: dict[str, BackendState] = field(default_factory=dict)

    def register(self, backend: GenerativeBackend) -> None:
        """Register a backend. Initializes fresh state."""
        name = backend.config.name
        assert name not in self._backends, f"Duplicate backend name: {name}"
        logger.debug(
            "registering_backend",
            name=name,
            provider=backend.config.provider,
            model=backend.config.model,
        )
        self._backends[name] = backend
        self._states[name] = BackendState()
        logger.info("backend_registered", name=name, total_backends=len(self._backends))

    def get(self, name: str) -> tuple[GenerativeBackend | None, BackendState | None]:
        """Look up a backend by name. Returns (None, None) if not registered."""
        backend = self._backends.get(name)
        state = self._states.get(name)
        return backend, state

    def all_backends(self) -> list[tuple[str, GenerativeBackend, BackendState]]:
        """Return all registered backends with state."""
        return [
            (name, self._backends[name], self._states[name])
            for name in self._backends
        ]

    def get_by_tier(self, tier: BackendTier) -> list[BackendConfig]:
        """Get all backends matching the specified tier.
        
        Args:
            tier: The BackendTier to filter by.
            
        Returns:
            List of BackendConfig objects for backends in the specified tier.
        """
        result = []
        for _name, backend, _state in self.all_backends():
            if backend.config.tier == tier:
                result.append(backend.config)
        return result

    def retire(self, name: str) -> bool:
        """Mark a backend as retired, excluding it from all routing.

        Args:
            name: The backend name to retire.

        Returns:
            True if found and retired, False if not found.
        """
        backend, state = self.get(name)
        if backend is None or state is None:
            return False
        state.status = BackendStatus.RETIRED
        logger.info("backend_retired", name=name)
        return True

    def reinstate(self, name: str) -> bool:
        """Reinstate a previously retired backend, returning it to the active pool.

        Args:
            name: The backend name to reinstate.

        Returns:
            True if found and was retired, False otherwise.
        """
        backend, state = self.get(name)
        if backend is None or state is None:
            return False
        if state.status != BackendStatus.RETIRED:
            return False
        state.status = BackendStatus.AVAILABLE
        logger.info("backend_reinstated", name=name)
        return True

    def health_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a health snapshot for observability."""
        snapshot: dict[str, dict[str, Any]] = {}
        for name, backend, state in self.all_backends():
            snapshot[name] = {
                "provider": backend.config.provider,
                "model": backend.config.model,
                "tier": backend.config.tier.value,
                "status": state.status.value,
                "requests_today": state.requests_today,
                "tokens_today": state.tokens_today,
                "avg_latency_ms": round(state.avg_latency_ms, 1),
                "circuit_open": state.is_circuit_open(),
            }
        return snapshot
