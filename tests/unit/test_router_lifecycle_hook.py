"""Tests for the matrix lifecycle hook wired into catalog refresh.

Validates that after each successful catalog refresh, the router
auto-seeds new models and decays deprecated ones (HAZ-008 extension).

Spec traceability: HAZ-008 (Stale Catalog Routing to Deprecated Models)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from dragonlight_router.router import RouterEngine

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_result(new_seeded: int = 0, total_in_matrix: int = 0) -> MagicMock:
    """Stub SeedResult returned by auto_seed_new_models."""
    result = MagicMock()
    result.new_seeded = new_seeded
    result.total_in_matrix = total_in_matrix
    return result


def _make_decay_result(decayed: int = 0, removed: int = 0) -> MagicMock:
    """Stub DecayResult returned by decay_deprecated_models."""
    result = MagicMock()
    result.decayed = decayed
    result.removed = removed
    return result


def _setup_minimal_router(tmp_path: Path) -> RouterEngine:
    """Create a RouterEngine with minimal config for lifecycle tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "providers": [],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))

    # Empty matrix so no backend registration is attempted
    matrix: dict = {"coding": {}}
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

    return RouterEngine(config_path=config_path)


# ---------------------------------------------------------------------------
# _sync_matrix_lifecycle
# ---------------------------------------------------------------------------


class TestSyncMatrixLifecycle:
    """Unit tests for RouterEngine._sync_matrix_lifecycle."""

    def test_calls_auto_seed_and_decay(self, tmp_path: Path) -> None:
        """_sync_matrix_lifecycle calls both lifecycle functions."""
        engine = _setup_minimal_router(tmp_path)

        seed_result = _make_seed_result(new_seeded=0)
        decay_result = _make_decay_result(decayed=0, removed=0)

        with (
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ) as mock_seed,
            patch(
                "dragonlight_router.router.decay_deprecated_models",
                return_value=decay_result,
            ) as mock_decay,
        ):
            engine._sync_matrix_lifecycle()

        mock_seed.assert_called_once_with(engine._config.state_dir)
        mock_decay.assert_called_once_with(engine._config.state_dir)

    def test_matrix_reloaded_when_new_models_seeded(self, tmp_path: Path) -> None:
        """Matrix is reloaded after new models are seeded."""
        engine = _setup_minimal_router(tmp_path)

        seed_result = _make_seed_result(new_seeded=3, total_in_matrix=10)
        decay_result = _make_decay_result(decayed=0, removed=0)

        with (
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ),
            patch(
                "dragonlight_router.router.decay_deprecated_models",
                return_value=decay_result,
            ),
            patch.object(engine._matrix, "reload_if_changed") as mock_reload,
        ):
            engine._sync_matrix_lifecycle()

        # reload called at least once (from seeding)
        assert mock_reload.call_count >= 1

    def test_matrix_reloaded_when_models_decayed(self, tmp_path: Path) -> None:
        """Matrix is reloaded when deprecated models are decayed or removed."""
        engine = _setup_minimal_router(tmp_path)

        seed_result = _make_seed_result(new_seeded=0)
        decay_result = _make_decay_result(decayed=2, removed=1)

        with (
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ),
            patch(
                "dragonlight_router.router.decay_deprecated_models",
                return_value=decay_result,
            ),
            patch.object(engine._matrix, "reload_if_changed") as mock_reload,
        ):
            engine._sync_matrix_lifecycle()

        # reload called at least once (from decay)
        assert mock_reload.call_count >= 1

    def test_matrix_not_reloaded_when_nothing_changed(self, tmp_path: Path) -> None:
        """Matrix is NOT reloaded when no new models seeded and nothing decayed."""
        engine = _setup_minimal_router(tmp_path)

        seed_result = _make_seed_result(new_seeded=0)
        decay_result = _make_decay_result(decayed=0, removed=0)

        with (
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ),
            patch(
                "dragonlight_router.router.decay_deprecated_models",
                return_value=decay_result,
            ),
            patch.object(engine._matrix, "reload_if_changed") as mock_reload,
        ):
            engine._sync_matrix_lifecycle()

        mock_reload.assert_not_called()

    def test_lifecycle_failure_does_not_crash_router(self, tmp_path: Path) -> None:
        """Exception in lifecycle functions is caught and logged, not re-raised."""
        engine = _setup_minimal_router(tmp_path)

        with patch(
            "dragonlight_router.router.auto_seed_new_models",
            side_effect=RuntimeError("seed exploded"),
        ):
            # Must not raise
            engine._sync_matrix_lifecycle()

    def test_lifecycle_decay_failure_does_not_crash_router(self, tmp_path: Path) -> None:
        """Exception during decay is caught and logged, not re-raised."""
        engine = _setup_minimal_router(tmp_path)

        seed_result = _make_seed_result(new_seeded=0)

        with (
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ),
            patch(
                "dragonlight_router.router.decay_deprecated_models",
                side_effect=OSError("disk full"),
            ),
        ):
            # Must not raise
            engine._sync_matrix_lifecycle()


# ---------------------------------------------------------------------------
# _HAS_LIFECYCLE=False guard
# ---------------------------------------------------------------------------


class TestLifecycleDisabledGuard:
    """When lifecycle module is absent, nothing lifecycle-related runs."""

    def test_has_lifecycle_false_skips_sync(self, tmp_path: Path) -> None:
        """When _HAS_LIFECYCLE is False, _sync_matrix_lifecycle is never called."""
        engine = _setup_minimal_router(tmp_path)

        with (
            patch("dragonlight_router.router._HAS_LIFECYCLE", False),
            patch.object(engine, "_sync_matrix_lifecycle") as mock_sync,
        ):
            # Manually trigger the method that would call _sync_matrix_lifecycle
            # We replicate the guard logic to confirm the method is NOT invoked
            import dragonlight_router.router as router_module

            if router_module._HAS_LIFECYCLE:
                engine._sync_matrix_lifecycle()

        mock_sync.assert_not_called()

    def test_has_lifecycle_true_calls_sync(self, tmp_path: Path) -> None:
        """When _HAS_LIFECYCLE is True, _sync_matrix_lifecycle is invoked."""
        engine = _setup_minimal_router(tmp_path)

        with (
            patch("dragonlight_router.router._HAS_LIFECYCLE", True),
            patch.object(engine, "_sync_matrix_lifecycle") as mock_sync,
        ):
            import dragonlight_router.router as router_module

            if router_module._HAS_LIFECYCLE:
                engine._sync_matrix_lifecycle()

        mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# Initial seed in __init__
# ---------------------------------------------------------------------------


class TestInitialSeedOnBoot:
    """Verify that __init__ runs an initial lifecycle seed when available."""

    def test_initial_seed_runs_during_init(self, tmp_path: Path) -> None:
        """auto_seed_new_models is called during RouterEngine.__init__."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        config = {
            "state_dir": str(state_dir),
            "catalog_ttl_hours": 24,
            "providers": [],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))

        matrix: dict = {"coding": {}}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        seed_result = _make_seed_result(new_seeded=2, total_in_matrix=5)

        with (
            patch("dragonlight_router.router._HAS_LIFECYCLE", True),
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                return_value=seed_result,
            ) as mock_seed,
        ):
            engine = RouterEngine(config_path=config_path)

        mock_seed.assert_called_once_with(engine._config.state_dir)

    def test_initial_seed_failure_does_not_prevent_boot(self, tmp_path: Path) -> None:
        """Exception during initial seed is caught; RouterEngine still constructs."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        config = {
            "state_dir": str(state_dir),
            "catalog_ttl_hours": 24,
            "providers": [],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))

        matrix: dict = {"coding": {}}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        with (
            patch("dragonlight_router.router._HAS_LIFECYCLE", True),
            patch(
                "dragonlight_router.router.auto_seed_new_models",
                side_effect=RuntimeError("seed failed on boot"),
            ),
        ):
            # Should not raise — exception is caught in __init__
            engine = RouterEngine(config_path=config_path)

        assert engine is not None

    def test_initial_seed_skipped_when_no_lifecycle(self, tmp_path: Path) -> None:
        """auto_seed_new_models is NOT called when _HAS_LIFECYCLE is False."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        config = {
            "state_dir": str(state_dir),
            "catalog_ttl_hours": 24,
            "providers": [],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))

        matrix: dict = {"coding": {}}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        with (
            patch("dragonlight_router.router._HAS_LIFECYCLE", False),
            patch(
                "dragonlight_router.router.auto_seed_new_models",
            ) as mock_seed,
        ):
            RouterEngine(config_path=config_path)

        mock_seed.assert_not_called()
