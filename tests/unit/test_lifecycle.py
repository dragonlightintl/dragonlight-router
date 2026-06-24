"""Unit tests for roles/lifecycle.py — matrix lifecycle management.

Covers catalog diff detection, auto-seeding, rank decay, spectrography
tracking, and lifecycle state persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dragonlight_router.roles.lifecycle import (
    CatalogDiff,
    auto_seed_new_models,
    decay_deprecated_models,
    detect_catalog_changes,
    get_models_needing_spectrography,
    load_lifecycle_state,
    mark_spectrography_complete,
    save_lifecycle_state,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROLES = ("coding", "testing", "review", "spec", "reasoning")


def _write_catalog(state_dir: Path, catalog: dict[str, list[dict[str, Any]]]) -> None:
    payload = {"timestamp": 9999999999.0, "catalog": catalog}
    (state_dir / "provider_catalog.json").write_text(json.dumps(payload))


def _write_matrix(state_dir: Path, matrix: dict[str, Any]) -> None:
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))


def _read_matrix(state_dir: Path) -> dict[str, Any]:
    return json.loads((state_dir / "model_role_matrix.json").read_text())


def _read_lifecycle(state_dir: Path) -> dict[str, Any]:
    return json.loads((state_dir / "matrix_lifecycle.json").read_text())


def _empty_matrix() -> dict[str, Any]:
    return {
        "version": 1,
        "default_rank": 20,
        "roles": {role: [] for role in _ROLES},
    }


def _matrix_with_model(model_id: str, rank: int = 50) -> dict[str, Any]:
    return {
        "version": 1,
        "default_rank": 20,
        "roles": {role: [{"model_id": model_id, "rank": rank}] for role in _ROLES},
    }


def _catalog_entry(model_id: str, provider: str = "groq") -> dict[str, Any]:
    return {"model_id": model_id, "provider": provider, "created": 1}


# ---------------------------------------------------------------------------
# load_lifecycle_state / save_lifecycle_state
# ---------------------------------------------------------------------------


class TestLifecycleStatePersistence:
    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        state = load_lifecycle_state(tmp_path)
        assert state == {"models": {}}

    def test_save_creates_file(self, tmp_path: Path) -> None:
        state = {"models": {"foo/bar": {"source": "heuristic"}}}
        save_lifecycle_state(tmp_path, state)
        assert (tmp_path / "matrix_lifecycle.json").exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        state: dict[str, Any] = {
            "models": {
                "groq/llama-3.3-70b-versatile": {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, state)
        loaded = load_lifecycle_state(tmp_path)
        assert loaded == state

    def test_load_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "matrix_lifecycle.json").write_text("{bad json")
        state = load_lifecycle_state(tmp_path)
        assert state == {"models": {}}

    def test_load_missing_models_key_adds_it(self, tmp_path: Path) -> None:
        (tmp_path / "matrix_lifecycle.json").write_text('{"version": 1}')
        state = load_lifecycle_state(tmp_path)
        assert "models" in state
        assert state["models"] == {}

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """Save should not leave tmp files behind."""
        state = {"models": {}}
        save_lifecycle_state(tmp_path, state)
        tmp_files = list(tmp_path.glob(".lifecycle_*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# detect_catalog_changes
# ---------------------------------------------------------------------------


class TestDetectCatalogChanges:
    def test_new_model_detected(self, tmp_path: Path) -> None:
        _write_catalog(
            tmp_path,
            {"groq": [_catalog_entry("groq/llama-3.3-70b-versatile")]},
        )
        _write_matrix(tmp_path, _empty_matrix())

        diff = detect_catalog_changes(tmp_path)

        assert isinstance(diff, CatalogDiff)
        assert "groq/llama-3.3-70b-versatile" in diff.new_models
        assert diff.missing_models == []

    def test_missing_model_detected(self, tmp_path: Path) -> None:
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model("groq/llama-3.3-70b-versatile"))

        diff = detect_catalog_changes(tmp_path)

        assert "groq/llama-3.3-70b-versatile" in diff.missing_models
        assert diff.new_models == []

    def test_unchanged_model_counted(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        diff = detect_catalog_changes(tmp_path)

        assert diff.unchanged == 1
        assert diff.new_models == []
        assert diff.missing_models == []

    def test_mix_of_new_missing_unchanged(self, tmp_path: Path) -> None:
        in_both = "groq/llama-3.3-70b-versatile"
        only_catalog = "groq/deepseek-r1-distill-llama-70b"
        only_matrix = "groq/qwq-32b"

        _write_catalog(
            tmp_path,
            {
                "groq": [
                    _catalog_entry(in_both),
                    _catalog_entry(only_catalog),
                ]
            },
        )
        matrix: dict[str, Any] = {
            "version": 1,
            "default_rank": 20,
            "roles": {
                role: [
                    {"model_id": in_both, "rank": 50},
                    {"model_id": only_matrix, "rank": 40},
                ]
                for role in _ROLES
            },
        }
        _write_matrix(tmp_path, matrix)

        diff = detect_catalog_changes(tmp_path)

        assert only_catalog in diff.new_models
        assert only_matrix in diff.missing_models
        assert diff.unchanged == 1

    def test_excluded_catalog_model_not_reported_as_new(self, tmp_path: Path) -> None:
        """Embedding/TTS/guard models should not appear in new_models."""
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    _catalog_entry("groq/whisper-large-v3"),
                    _catalog_entry("groq/llama-3.3-70b-versatile"),
                ]
            },
        )
        _write_matrix(tmp_path, _empty_matrix())

        diff = detect_catalog_changes(tmp_path)

        assert "groq/whisper-large-v3" not in diff.new_models
        assert "groq/llama-3.3-70b-versatile" in diff.new_models

    def test_empty_catalog_all_matrix_models_missing(self, tmp_path: Path) -> None:
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model("groq/llama-3.3-70b-versatile"))

        diff = detect_catalog_changes(tmp_path)

        assert "groq/llama-3.3-70b-versatile" in diff.missing_models
        assert diff.new_models == []
        assert diff.unchanged == 0

    def test_no_matrix_file_all_catalog_models_new(self, tmp_path: Path) -> None:
        _write_catalog(
            tmp_path,
            {"groq": [_catalog_entry("groq/llama-3.3-70b-versatile")]},
        )
        # No matrix file

        diff = detect_catalog_changes(tmp_path)

        assert "groq/llama-3.3-70b-versatile" in diff.new_models
        assert diff.missing_models == []


# ---------------------------------------------------------------------------
# auto_seed_new_models
# ---------------------------------------------------------------------------


class TestAutoSeedNewModels:
    def test_new_model_added_to_all_roles(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        result = auto_seed_new_models(tmp_path)

        assert result.new_seeded == 1
        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            ids = [e["model_id"] for e in matrix["roles"][role]]
            assert model_id in ids, f"model missing from role {role}"

    def test_seed_result_counts_correct(self, tmp_path: Path) -> None:
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    _catalog_entry("groq/llama-3.3-70b-versatile"),
                    _catalog_entry("groq/deepseek-r1-distill-llama-70b"),
                ]
            },
        )
        _write_matrix(tmp_path, _empty_matrix())

        result = auto_seed_new_models(tmp_path)

        assert result.new_seeded == 2
        assert result.missing_detected == 0

    def test_missing_detected_count(self, tmp_path: Path) -> None:
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model("groq/llama-3.3-70b-versatile"))

        result = auto_seed_new_models(tmp_path)

        assert result.new_seeded == 0
        assert result.missing_detected == 1

    def test_new_model_lifecycle_state_recorded_as_heuristic(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        auto_seed_new_models(tmp_path)

        lifecycle = _read_lifecycle(tmp_path)
        assert model_id in lifecycle["models"]
        assert lifecycle["models"][model_id]["source"] == "heuristic"
        assert lifecycle["models"][model_id]["consecutive_misses"] == 0

    def test_excluded_model_not_seeded(self, tmp_path: Path) -> None:
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    _catalog_entry("groq/whisper-large-v3"),
                    _catalog_entry("groq/llama-3.3-70b-versatile"),
                ]
            },
        )
        _write_matrix(tmp_path, _empty_matrix())

        result = auto_seed_new_models(tmp_path)

        # Only the non-excluded model should be seeded
        assert result.new_seeded == 1
        matrix = _read_matrix(tmp_path)
        ids = [e["model_id"] for e in matrix["roles"]["coding"]]
        assert "groq/whisper-large-v3" not in ids

    def test_total_in_matrix_correct(self, tmp_path: Path) -> None:
        existing = "groq/deepseek-r1-distill-llama-70b"
        new_model = "groq/llama-3.3-70b-versatile"
        _write_catalog(
            tmp_path,
            {"groq": [_catalog_entry(existing), _catalog_entry(new_model)]},
        )
        _write_matrix(tmp_path, _matrix_with_model(existing))

        result = auto_seed_new_models(tmp_path)

        assert result.total_in_matrix == 2

    def test_seeded_ranks_are_positive_integers(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        auto_seed_new_models(tmp_path)

        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            entry = next(e for e in matrix["roles"][role] if e["model_id"] == model_id)
            assert isinstance(entry["rank"], int)
            assert entry["rank"] > 0

    def test_no_duplicates_after_reseed(self, tmp_path: Path) -> None:
        """Calling auto_seed twice on the same catalog should not duplicate entries."""
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        auto_seed_new_models(tmp_path)
        auto_seed_new_models(tmp_path)

        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            ids = [e["model_id"] for e in matrix["roles"][role]]
            assert len(ids) == len(set(ids)), f"duplicates in role {role}"


# ---------------------------------------------------------------------------
# consecutive_misses tracking
# ---------------------------------------------------------------------------


class TestConsecutiveMissesTracking:
    def test_miss_increments_consecutive_misses(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        # Seed once to create lifecycle entry
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        auto_seed_new_models(tmp_path)

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["consecutive_misses"] == 1

    def test_miss_increments_multiple_times(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-18",
                    "consecutive_misses": 2,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        auto_seed_new_models(tmp_path)

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["consecutive_misses"] == 3

    def test_reappearance_resets_consecutive_misses(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        # Model is in catalog again
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-18",
                    "last_catalog_hit": "2026-06-18",
                    "consecutive_misses": 2,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        auto_seed_new_models(tmp_path)

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["consecutive_misses"] == 0


# ---------------------------------------------------------------------------
# decay_deprecated_models
# ---------------------------------------------------------------------------


class TestDecayDeprecatedModels:
    def _make_lifecycle_with_misses(self, model_id: str, misses: int) -> dict[str, Any]:
        return {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-15",
                    "last_catalog_hit": "2026-06-15",
                    "consecutive_misses": misses,
                    "spectrography_run_id": None,
                }
            }
        }

    def test_no_deprecated_models_returns_zero_counts(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=60))
        save_lifecycle_state(tmp_path, self._make_lifecycle_with_misses(model_id, 0))

        result = decay_deprecated_models(tmp_path)

        assert result.decayed == 0
        assert result.removed == 0
        assert result.remaining == 1

    def test_rank_decayed_for_deprecated_model(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=80))
        save_lifecycle_state(tmp_path, self._make_lifecycle_with_misses(model_id, 3))

        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        assert result.decayed == 1
        assert result.removed == 0
        matrix = _read_matrix(tmp_path)
        entry = matrix["roles"]["coding"][0]
        assert entry["model_id"] == model_id
        assert entry["rank"] == 40  # 80 * 0.5

    def test_model_removed_when_rank_below_10(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=15))
        save_lifecycle_state(tmp_path, self._make_lifecycle_with_misses(model_id, 3))

        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        # 15 * 0.5 = 7 < 10 → removed
        assert result.removed == 1
        assert result.decayed == 0
        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            ids = [e["model_id"] for e in matrix["roles"][role]]
            assert model_id not in ids

    def test_model_exactly_at_threshold_not_decayed(self, tmp_path: Path) -> None:
        """Models with consecutive_misses == max_misses - 1 are not decayed."""
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=80))
        save_lifecycle_state(tmp_path, self._make_lifecycle_with_misses(model_id, 2))

        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        assert result.decayed == 0
        assert result.removed == 0
        matrix = _read_matrix(tmp_path)
        entry = matrix["roles"]["coding"][0]
        assert entry["rank"] == 80  # unchanged

    def test_remaining_count_correct(self, tmp_path: Path) -> None:
        survivor = "groq/llama-3.3-70b-versatile"
        doomed = "groq/llama-3.1-8b-instant"
        matrix: dict[str, Any] = {
            "version": 1,
            "default_rank": 20,
            "roles": {
                role: [
                    {"model_id": survivor, "rank": 60},
                    {"model_id": doomed, "rank": 12},
                ]
                for role in _ROLES
            },
        }
        _write_matrix(tmp_path, matrix)
        lifecycle: dict[str, Any] = {
            "models": {
                survivor: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
                doomed: {
                    "source": "heuristic",
                    "first_seen": "2026-06-15",
                    "last_catalog_hit": "2026-06-15",
                    "consecutive_misses": 5,
                    "spectrography_run_id": None,
                },
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        # doomed rank 12 * 0.5 = 6 < 10 → removed; survivor stays
        assert result.removed == 1
        assert result.remaining == 1

    def test_custom_decay_rate(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=100))
        save_lifecycle_state(tmp_path, self._make_lifecycle_with_misses(model_id, 4))

        decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.8)

        matrix = _read_matrix(tmp_path)
        entry = matrix["roles"]["coding"][0]
        assert entry["rank"] == 80  # 100 * 0.8


# ---------------------------------------------------------------------------
# get_models_needing_spectrography
# ---------------------------------------------------------------------------


class TestGetModelsNeedingSpectrography:
    def test_returns_only_heuristic_source_models(self, tmp_path: Path) -> None:
        model_a = "groq/llama-3.3-70b-versatile"
        model_b = "groq/deepseek-r1-distill-llama-70b"
        model_c = "groq/llama-3.1-8b-instant"

        matrix: dict[str, Any] = {
            "version": 1,
            "default_rank": 20,
            "roles": {
                role: [
                    {"model_id": model_a, "rank": 60},
                    {"model_id": model_b, "rank": 55},
                    {"model_id": model_c, "rank": 30},
                ]
                for role in _ROLES
            },
        }
        _write_matrix(tmp_path, matrix)

        lifecycle: dict[str, Any] = {
            "models": {
                model_a: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
                model_b: {
                    "source": "empirical",
                    "first_seen": "2026-06-19",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": "run-2026-06-19-abc",
                },
                model_c: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        result = get_models_needing_spectrography(tmp_path)

        assert model_b not in result
        assert model_a in result
        assert model_c in result

    def test_sorted_by_max_rank_descending(self, tmp_path: Path) -> None:
        high_rank_model = "groq/llama-3.3-70b-versatile"
        low_rank_model = "groq/llama-3.1-8b-instant"

        matrix: dict[str, Any] = {
            "version": 1,
            "default_rank": 20,
            "roles": {
                role: [
                    {"model_id": high_rank_model, "rank": 80},
                    {"model_id": low_rank_model, "rank": 28},
                ]
                for role in _ROLES
            },
        }
        _write_matrix(tmp_path, matrix)

        lifecycle: dict[str, Any] = {
            "models": {
                high_rank_model: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
                low_rank_model: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        result = get_models_needing_spectrography(tmp_path)

        assert result[0] == high_rank_model
        assert result[1] == low_rank_model

    def test_empty_when_no_heuristic_models(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "empirical",
                    "first_seen": "2026-06-19",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": "run-abc",
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        result = get_models_needing_spectrography(tmp_path)

        assert result == []

    def test_empty_lifecycle_returns_empty(self, tmp_path: Path) -> None:
        _write_matrix(tmp_path, _matrix_with_model("groq/llama-3.3-70b-versatile"))
        # No lifecycle file — load_lifecycle_state returns {"models": {}}

        result = get_models_needing_spectrography(tmp_path)

        assert result == []


# ---------------------------------------------------------------------------
# mark_spectrography_complete
# ---------------------------------------------------------------------------


class TestMarkSpectrographyComplete:
    def test_updates_source_to_empirical(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        mark_spectrography_complete(tmp_path, [model_id], "run-2026-06-20-abc123")

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["source"] == "empirical"

    def test_records_run_id(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        run_id = "run-2026-06-20-abc123"
        mark_spectrography_complete(tmp_path, [model_id], run_id)

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["spectrography_run_id"] == run_id

    def test_updates_multiple_models(self, tmp_path: Path) -> None:
        model_a = "groq/llama-3.3-70b-versatile"
        model_b = "groq/deepseek-r1-distill-llama-70b"
        lifecycle: dict[str, Any] = {
            "models": {
                model_a: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
                model_b: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                },
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        mark_spectrography_complete(tmp_path, [model_a, model_b], "run-xyz")

        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_a]["source"] == "empirical"
        assert updated["models"][model_b]["source"] == "empirical"

    def test_creates_entry_if_model_not_in_lifecycle(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        save_lifecycle_state(tmp_path, {"models": {}})

        mark_spectrography_complete(tmp_path, [model_id], "run-new")

        updated = _read_lifecycle(tmp_path)
        assert model_id in updated["models"]
        assert updated["models"][model_id]["source"] == "empirical"
        assert updated["models"][model_id]["spectrography_run_id"] == "run-new"

    def test_heuristic_models_no_longer_need_spectrography_after_mark(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, _matrix_with_model(model_id))

        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        mark_spectrography_complete(tmp_path, [model_id], "run-abc")

        result = get_models_needing_spectrography(tmp_path)
        assert model_id not in result


# ---------------------------------------------------------------------------
# Integration: full lifecycle flow
# ---------------------------------------------------------------------------


class TestLifecycleIntegration:
    def test_seed_then_decay_removes_missing_model(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"

        # Matrix has model at rank 15; catalog is empty (model gone)
        _write_catalog(tmp_path, {})
        _write_matrix(tmp_path, _matrix_with_model(model_id, rank=15))

        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-15",
                    "last_catalog_hit": "2026-06-15",
                    "consecutive_misses": 2,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        # Third miss — now at 3
        auto_seed_new_models(tmp_path)
        # Decay: 15 * 0.5 = 7 < 10 → removed
        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        assert result.removed == 1
        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            ids = [e["model_id"] for e in matrix["roles"][role]]
            assert model_id not in ids

    def test_seed_marks_heuristic_then_spectrography_clears_queue(self, tmp_path: Path) -> None:
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        auto_seed_new_models(tmp_path)

        # Should now need spectrography
        needing = get_models_needing_spectrography(tmp_path)
        assert model_id in needing

        # After marking complete, queue clears
        mark_spectrography_complete(tmp_path, [model_id], "run-2026-06-20-xyz")
        needing_after = get_models_needing_spectrography(tmp_path)
        assert model_id not in needing_after


# ---------------------------------------------------------------------------
# Coverage: dict-format matrix branches and defensive guards
# ---------------------------------------------------------------------------


class TestDictFormatMatrixBranches:
    """Exercises the dict-format matrix branches in internal helpers."""

    def _dict_format_matrix(self, model_id: str, rank: int = 50) -> dict[str, Any]:
        """Write a matrix using the flat dict format (role → {model_id: rank})."""
        return {
            "version": 1,
            "default_rank": 20,
            "roles": {role: {model_id: rank} for role in _ROLES},
        }

    def test_detect_changes_with_dict_format_matrix(self, tmp_path: Path) -> None:
        """_get_all_matrix_model_ids handles dict-format role entries."""
        model_id = "groq/llama-3.3-70b-versatile"
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, self._dict_format_matrix(model_id))

        diff = detect_catalog_changes(tmp_path)

        # Model present in both — should be unchanged, not new/missing
        assert diff.unchanged == 1
        assert model_id not in diff.new_models
        assert model_id not in diff.missing_models

    def test_decay_with_dict_format_matrix_skips_role(self, tmp_path: Path) -> None:
        """decay_deprecated_models skips non-list role entries (dict format)."""
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, self._dict_format_matrix(model_id, rank=80))
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-15",
                    "last_catalog_hit": "2026-06-15",
                    "consecutive_misses": 5,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        # With dict format, the entries are dicts not lists — decay silently skips
        result = decay_deprecated_models(tmp_path, max_misses=3, decay_rate=0.5)

        # No list entries to iterate, so nothing decayed or removed
        assert result.decayed == 0
        assert result.removed == 0

    def test_get_models_needing_spectrography_with_dict_format_matrix(self, tmp_path: Path) -> None:
        """get_models_needing_spectrography handles dict-format role entries."""
        model_id = "groq/llama-3.3-70b-versatile"
        _write_matrix(tmp_path, self._dict_format_matrix(model_id, rank=60))
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-20",
                    "last_catalog_hit": "2026-06-20",
                    "consecutive_misses": 0,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        # Dict-format entries are skipped by the rank scanner, model still returned
        result = get_models_needing_spectrography(tmp_path)
        assert model_id in result

    def test_matrix_load_corrupt_json_falls_back_to_empty(self, tmp_path: Path) -> None:
        """_load_matrix_raw returns empty structure on JSON parse failure."""
        (tmp_path / "model_role_matrix.json").write_text("{broken json")
        _write_catalog(tmp_path, {"groq": [_catalog_entry("groq/llama-3.3-70b-versatile")]})

        # detect_catalog_changes will call _load_matrix_raw → gets fallback empty
        diff = detect_catalog_changes(tmp_path)

        # All catalog models appear as new since matrix parsed as empty
        assert "groq/llama-3.3-70b-versatile" in diff.new_models


class TestAutoSeedKnownModelReAddedToMatrix:
    """Covers the 'model in lifecycle but new to matrix' branch in auto_seed_new_models."""

    def test_known_model_reappears_in_catalog_after_being_absent_from_matrix(
        self, tmp_path: Path
    ) -> None:
        """Model previously in lifecycle but removed from matrix is re-seeded."""
        model_id = "groq/llama-3.3-70b-versatile"
        # Catalog has the model; matrix is empty (it was removed)
        _write_catalog(tmp_path, {"groq": [_catalog_entry(model_id)]})
        _write_matrix(tmp_path, _empty_matrix())

        # Pre-populate lifecycle state for this model (already known)
        lifecycle: dict[str, Any] = {
            "models": {
                model_id: {
                    "source": "heuristic",
                    "first_seen": "2026-06-15",
                    "last_catalog_hit": "2026-06-18",
                    "consecutive_misses": 1,
                    "spectrography_run_id": None,
                }
            }
        }
        save_lifecycle_state(tmp_path, lifecycle)

        result = auto_seed_new_models(tmp_path)

        # Model is new to matrix, so seeded
        assert result.new_seeded == 1

        # Lifecycle entry updated — misses reset
        updated = _read_lifecycle(tmp_path)
        assert updated["models"][model_id]["consecutive_misses"] == 0

        # Model added to all roles
        matrix = _read_matrix(tmp_path)
        for role in _ROLES:
            ids = [e["model_id"] for e in matrix["roles"][role]]
            assert model_id in ids
