"""Tests for roles/spectrography_targets.py — spectrography target filtering.

Spec traceability: ST-001 (Spectrography Targets)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dragonlight_router.roles.spectrography_targets import get_spectrography_targets

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MATRIX = {
    "version": 1,
    "default_rank": 20,
    "roles": {
        "coding": [
            {"model_id": "prov/model-alpha", "rank": 82},
            {"model_id": "prov/model-beta", "rank": 55},
            {"model_id": "prov/model-gamma", "rank": 28},
        ],
        "testing": [
            {"model_id": "prov/model-alpha", "rank": 70},
            {"model_id": "prov/model-beta", "rank": 60},
        ],
        "review": [
            {"model_id": "prov/model-gamma", "rank": 35},
        ],
        "spec": [],
        "reasoning": [
            {"model_id": "prov/model-delta", "rank": 85},
        ],
    },
}

# max ranks: alpha=82, beta=60, gamma=35, delta=85


def _write_lifecycle(path: Path, models: dict) -> None:
    path.write_text(json.dumps({"models": models}))


def _write_matrix(path: Path, data: dict | None = None) -> None:
    path.write_text(json.dumps(data if data is not None else _MATRIX))


# ---------------------------------------------------------------------------
# Returns empty when no lifecycle state exists
# ---------------------------------------------------------------------------


class TestNoLifecycleState:
    def test_returns_empty_list_when_lifecycle_missing(self, tmp_path: Path) -> None:
        """No matrix_lifecycle.json → empty list returned."""
        _write_matrix(tmp_path / "model_role_matrix.json")
        result = get_spectrography_targets(tmp_path)
        assert result == []

    def test_returns_empty_list_when_lifecycle_has_no_models(self, tmp_path: Path) -> None:
        """Empty models dict → empty list."""
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", {})
        _write_matrix(tmp_path / "model_role_matrix.json")
        result = get_spectrography_targets(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# Filters to heuristic-only models
# ---------------------------------------------------------------------------


class TestHeuristicFilter:
    def test_returns_only_heuristic_models(self, tmp_path: Path) -> None:
        """Only models with source='heuristic' are returned."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},
            "prov/model-beta": {"source": "empirical"},
            "prov/model-delta": {"source": "heuristic"},
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert set(result) == {"prov/model-alpha", "prov/model-delta"}

    def test_models_with_source_empirical_are_excluded(self, tmp_path: Path) -> None:
        """source='empirical' models never appear in results."""
        models = {
            "prov/model-alpha": {"source": "empirical"},
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert result == []

    def test_models_with_source_operator_are_excluded(self, tmp_path: Path) -> None:
        """source='operator' models never appear in results."""
        models = {
            "prov/model-alpha": {"source": "operator"},
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert result == []


# ---------------------------------------------------------------------------
# Sorts by highest rank across roles
# ---------------------------------------------------------------------------


class TestRankSorting:
    def test_sorted_by_highest_rank_descending(self, tmp_path: Path) -> None:
        """Results are ordered by max rank descending (delta=85, alpha=82, beta=60)."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},  # max rank 82
            "prov/model-beta": {"source": "heuristic"},  # max rank 60
            "prov/model-delta": {"source": "heuristic"},  # max rank 85
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert result == ["prov/model-delta", "prov/model-alpha", "prov/model-beta"]

    def test_max_rank_is_across_all_roles(self, tmp_path: Path) -> None:
        """A model with low coding rank but high reasoning rank sorts by reasoning."""
        matrix = {
            "version": 1,
            "roles": {
                "coding": [{"model_id": "x/low-coder", "rank": 20}],
                "reasoning": [{"model_id": "x/low-coder", "rank": 90}],
            },
        }
        models = {
            "x/low-coder": {"source": "heuristic"},
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json", matrix)

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert result == ["x/low-coder"]


# ---------------------------------------------------------------------------
# min_rank filter
# ---------------------------------------------------------------------------


class TestMinRankFilter:
    def test_min_rank_excludes_low_ranked_models(self, tmp_path: Path) -> None:
        """Models with max rank < min_rank are excluded."""
        # gamma has max rank 35 (review role), below default min_rank=30
        models = {
            "prov/model-alpha": {"source": "heuristic"},  # max 82 — passes
            "prov/model-gamma": {"source": "heuristic"},  # max 35 — passes 30
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=40)
        assert "prov/model-alpha" in result
        assert "prov/model-gamma" not in result

    def test_min_rank_zero_includes_all_heuristic(self, tmp_path: Path) -> None:
        """min_rank=0 includes even unranked models."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},
            "prov/unknown": {"source": "heuristic"},  # not in matrix → max rank 0
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0)
        assert "prov/model-alpha" in result
        assert "prov/unknown" in result

    def test_default_min_rank_is_30(self, tmp_path: Path) -> None:
        """Default min_rank=30 is applied when not specified."""
        # prov/model-gamma has max rank 35 (above 30) — included
        # prov/model-tiny doesn't appear in matrix (rank 0) — excluded
        models = {
            "prov/model-gamma": {"source": "heuristic"},  # max 35
            "prov/model-tiny": {"source": "heuristic"},  # max 0 → excluded
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path)
        assert "prov/model-gamma" in result
        assert "prov/model-tiny" not in result


# ---------------------------------------------------------------------------
# limit parameter
# ---------------------------------------------------------------------------


class TestLimit:
    def test_limit_caps_results(self, tmp_path: Path) -> None:
        """--limit N returns at most N models."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},  # rank 82
            "prov/model-beta": {"source": "heuristic"},  # rank 60
            "prov/model-delta": {"source": "heuristic"},  # rank 85
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0, limit=2)
        assert len(result) == 2

    def test_limit_returns_highest_ranked_first(self, tmp_path: Path) -> None:
        """With --limit, the top-ranked models are returned."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},  # rank 82
            "prov/model-beta": {"source": "heuristic"},  # rank 60
            "prov/model-delta": {"source": "heuristic"},  # rank 85
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0, limit=1)
        assert result == ["prov/model-delta"]

    def test_limit_larger_than_results_returns_all(self, tmp_path: Path) -> None:
        """Limit larger than available results returns everything."""
        models = {
            "prov/model-alpha": {"source": "heuristic"},
        }
        _write_lifecycle(tmp_path / "matrix_lifecycle.json", models)
        _write_matrix(tmp_path / "model_role_matrix.json")

        result = get_spectrography_targets(tmp_path, min_rank=0, limit=100)
        assert result == ["prov/model-alpha"]
