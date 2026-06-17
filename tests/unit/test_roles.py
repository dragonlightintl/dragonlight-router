"""Tests for roles/matrix.py — role-to-model ranking matrix.

Spec traceability: TM-018 (Role matrix ranking)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dragonlight_router.roles.matrix import RoleMatrix


def _write_matrix(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


class TestRoleMatrix:
    def test_get_ranked_models(self, tmp_path: Path):
        """[TM-018 AC-1] Ranked models returned in descending rank order."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {
            "coding": {
                "mistral_codestral": 95,
                "groq_llama70b": 85,
                "ollama_qwen": 50,
            }
        })
        matrix = RoleMatrix(matrix_path=path)
        ranked = matrix.get_ranked_models("coding")
        assert ranked[0] == ("mistral_codestral", 95)
        assert ranked[1] == ("groq_llama70b", 85)
        assert ranked[2] == ("ollama_qwen", 50)

    def test_sorted_by_rank_descending(self, tmp_path: Path):
        """[TM-018 AC-1] Models sorted by rank descending."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {
            "coding": {
                "a": 50,
                "b": 90,
                "c": 70,
            }
        })
        matrix = RoleMatrix(matrix_path=path)
        ranked = matrix.get_ranked_models("coding")
        ranks = [r[1] for r in ranked]
        assert ranks == sorted(ranks, reverse=True)

    def test_unknown_role_returns_empty(self, tmp_path: Path):
        """[TM-018 AC-2] Unknown role returns empty list."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {"coding": {"a": 90}})
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_ranked_models("nonexistent") == []

    def test_get_rank_returns_value(self, tmp_path: Path):
        """[TM-018 AC-1] get_rank returns the configured rank value."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {"coding": {"mistral_codestral": 95}})
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_rank("mistral_codestral", "coding") == 95

    def test_get_rank_unknown_model_returns_default(self, tmp_path: Path):
        """[TM-018 AC-2] Unknown model returns default rank of 20."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {"coding": {"mistral_codestral": 95}})
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_rank("unknown_model", "coding") == 20

    def test_get_rank_unknown_role_returns_default(self, tmp_path: Path):
        """[TM-018 AC-2] Unknown role returns default rank of 20."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {"coding": {"a": 90}})
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_rank("a", "nonexistent") == 20

    def test_reload_if_changed(self, tmp_path: Path):
        """[TM-018 AC-3] Matrix reloads when source file changes."""
        path = tmp_path / "matrix.json"
        _write_matrix(path, {"coding": {"a": 50}})
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_rank("a", "coding") == 50

        # Modify the file
        import time
        time.sleep(0.01)
        _write_matrix(path, {"coding": {"a": 99}})
        matrix.reload_if_changed()
        assert matrix.get_rank("a", "coding") == 99

    def test_missing_file_empty_matrix(self, tmp_path: Path):
        """[TM-018 AC-2] Missing matrix file results in empty rankings."""
        path = tmp_path / "missing.json"
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_ranked_models("coding") == []
