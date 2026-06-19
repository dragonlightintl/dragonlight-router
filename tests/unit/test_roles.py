"""Tests for roles/matrix.py — role-to-model ranking matrix.

Spec traceability: TM-018 (Role matrix ranking)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dragonlight_router.roles.matrix import RoleMatrix

pytestmark = pytest.mark.unit


def _write_matrix(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


class TestRoleMatrix:
    def test_get_ranked_models(self, tmp_path: Path):
        """[TM-018 AC-1] Ranked models returned in descending rank order."""
        path = tmp_path / "matrix.json"
        _write_matrix(
            path,
            {
                "coding": {
                    "mistral_codestral": 95,
                    "groq_llama70b": 85,
                    "ollama_qwen": 50,
                }
            },
        )
        matrix = RoleMatrix(matrix_path=path)
        ranked = matrix.get_ranked_models("coding")
        assert ranked[0] == ("mistral_codestral", 95)
        assert ranked[1] == ("groq_llama70b", 85)
        assert ranked[2] == ("ollama_qwen", 50)

    def test_sorted_by_rank_descending(self, tmp_path: Path):
        """[TM-018 AC-1] Models sorted by rank descending."""
        path = tmp_path / "matrix.json"
        _write_matrix(
            path,
            {
                "coding": {
                    "a": 50,
                    "b": 90,
                    "c": 70,
                }
            },
        )
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

    def test_reload_if_changed_file_not_exist_returns_early(self, tmp_path: Path):
        """[TM-010 AC-1] reload_if_changed returns silently when the file doesn't exist."""
        path = tmp_path / "existing.json"
        path.write_text(json.dumps({"coding": {"a": 50}}))
        matrix = RoleMatrix(matrix_path=path)
        path.unlink()
        matrix.reload_if_changed()
        assert matrix.get_rank("a", "coding") == 50

    def test_reload_if_changed_oserror_logs_warning(self, tmp_path: Path):
        """[TM-010 AC-1] reload_if_changed handles OSError from os.path.getmtime gracefully."""
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps({"coding": {"a": 70}}))
        matrix = RoleMatrix(matrix_path=path)
        mtime_path = "dragonlight_router.roles.matrix.os.path.getmtime"
        with patch(mtime_path, side_effect=OSError("stat failed")):
            matrix.reload_if_changed()
        assert matrix.get_rank("a", "coding") == 70

    def test_load_read_json_returns_none_sets_empty_matrix(self, tmp_path: Path):
        """[TM-010 AC-2] _load sets empty matrix when _read_json returns None."""
        path = tmp_path / "matrix.json"
        path.write_text("not valid json {{{{")
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_ranked_models("coding") == []

    def test_read_json_json_decode_error_returns_none(self, tmp_path: Path):
        """[TM-010 AC-2] _read_json returns None on JSONDecodeError."""
        path = tmp_path / "matrix.json"
        path.write_text("{bad json")
        matrix = RoleMatrix.__new__(RoleMatrix)
        matrix._path = path
        matrix._mtime = 0.0
        matrix._matrix = {}
        result = matrix._read_json()
        assert result is None

    def test_read_json_oserror_returns_none(self, tmp_path: Path):
        """[TM-010 AC-2] _read_json returns None on OSError."""
        path = tmp_path / "ghost.json"
        path.write_text("{}")
        matrix = RoleMatrix.__new__(RoleMatrix)
        matrix._path = path
        matrix._mtime = 0.0
        matrix._matrix = {}
        with patch("pathlib.Path.read_text", side_effect=OSError("cannot read")):
            result = matrix._read_json()
        assert result is None

    def test_parse_full_schema_entries_as_dict(self, tmp_path: Path):
        """[TM-010 AC-3] _parse_full_schema handles dict entries (not list)."""
        roles_raw = {
            "coding": {"gpt4": 90, "llama": 70},
        }
        result = RoleMatrix._parse_full_schema(roles_raw)
        assert result == {"coding": {"gpt4": 90, "llama": 70}}

    def test_load_full_schema_with_roles_key(self, tmp_path: Path):
        """[TM-010 AC-3] Full schema format with 'roles' key parsed correctly."""
        path = tmp_path / "matrix.json"
        data = {
            "version": 1,
            "roles": {
                "coding": [{"model_id": "gpt4", "rank": 95}],
            },
        }
        path.write_text(json.dumps(data))
        matrix = RoleMatrix(matrix_path=path)
        assert matrix.get_rank("gpt4", "coding") == 95
