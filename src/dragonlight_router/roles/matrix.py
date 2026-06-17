"""Role matrix -- maps roles to ranked model lists.

Loaded from a JSON file. Supports hot-reload via mtime check.
Returns ranked tuples of (model_id, rank) for a given role.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_DEFAULT_RANK = 20


class RoleMatrix:
    """Maps roles to ranked model IDs. File-backed with hot-reload."""

    def __init__(self, matrix_path: Path) -> None:
        assert isinstance(matrix_path, Path), "matrix_path must be a Path instance"
        self._path = matrix_path
        self._mtime: float = 0.0
        self._matrix: dict[str, dict[str, int]] = {}
        self._load()

    def get_ranked_models(self, role: str) -> list[tuple[str, int]]:
        """Return [(model_id, rank), ...] sorted by rank descending.

        Returns empty list for unknown roles.
        """
        assert isinstance(role, str), "role must be a string"
        role_data = self._matrix.get(role, {})
        ranked = [(model_id, rank) for model_id, rank in role_data.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked), "ranked items must be 2-tuples"
        return ranked

    def get_rank(self, model_id: str, role: str) -> int:
        """Return rank for a model in a role, or default_rank (20) for unknowns."""
        assert isinstance(model_id, str), "model_id must be a string"
        assert isinstance(role, str), "role must be a string"
        role_data = self._matrix.get(role, {})
        return role_data.get(model_id, _DEFAULT_RANK)

    def reload_if_changed(self) -> None:
        """Check file mtime and reload if the file has been modified."""
        if not self._path.exists():
            return

        try:
            current_mtime = os.path.getmtime(self._path)
            if current_mtime > self._mtime:
                self._load()
        except OSError as exc:
            logger.warning("matrix_stat_failed", error=str(exc))

    def _load(self) -> None:
        """Load matrix from JSON file.

        Supports two formats:
        1. Full schema: {"version": 1, "default_rank": 20, "roles": {"coding": [{"model_id": "x", "rank": 90}]}}
        2. Flat dict: {"coding": {"model_id": rank, ...}}
        """
        if not self._path.exists():
            self._matrix = {}
            return

        raw = self._read_json()
        if raw is None:
            self._matrix = {}
            return

        if "roles" in raw:
            self._matrix = self._parse_full_schema(raw["roles"])
        else:
            self._matrix = raw

        assert isinstance(self._matrix, dict), "matrix must be a dict after load"

    def _read_json(self) -> dict[str, Any] | None:
        """Read and parse the JSON matrix file. Returns None on failure."""
        try:
            text = self._path.read_text()
            raw: dict[str, Any] = json.loads(text)
            self._mtime = os.path.getmtime(self._path)
            return raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("role_matrix_load_failed", path=str(self._path), error=str(exc))
            return None

    @staticmethod
    def _parse_full_schema(roles_raw: dict[str, Any]) -> dict[str, dict[str, int]]:
        """Parse the full schema format with version/roles structure."""
        matrix: dict[str, dict[str, int]] = {}
        for role, entries in roles_raw.items():
            if isinstance(entries, list):
                matrix[role] = {
                    e["model_id"]: e["rank"] for e in entries
                }
            elif isinstance(entries, dict):
                matrix[role] = entries
        return matrix
