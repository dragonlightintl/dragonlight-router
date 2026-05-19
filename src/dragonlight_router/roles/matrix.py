"""Role matrix — maps roles to ranked model lists.

Loaded from a JSON file. Supports hot-reload via mtime check.
Returns ranked tuples of (model_id, rank) for a given role.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import structlog

logger = structlog.get_logger()

_DEFAULT_RANK = 20


class RoleMatrix:
    """Maps roles to ranked model IDs. File-backed with hot-reload."""

    def __init__(self, matrix_path: Path) -> None:
        self._path = matrix_path
        self._mtime: float = 0.0
        self._matrix: dict[str, dict[str, int]] = {}
        self._load()

    def get_ranked_models(self, role: str) -> list[tuple[str, int]]:
        """Return [(model_id, rank), ...] sorted by rank descending.

        Returns empty list for unknown roles.
        """
        role_data = self._matrix.get(role, {})
        ranked = [(model_id, rank) for model_id, rank in role_data.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_rank(self, model_id: str, role: str) -> int:
        """Return rank for a model in a role, or default_rank (20) for unknowns."""
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
        except OSError:
            pass

    def _load(self) -> None:
        """Load matrix from JSON file.

        Supports two formats:
        1. Full schema: {"version": 1, "default_rank": 20, "roles": {"coding": [{"model_id": "x", "rank": 90}]}}
        2. Flat dict: {"coding": {"model_id": rank, ...}}
        """
        if not self._path.exists():
            self._matrix = {}
            return

        try:
            text = self._path.read_text()
            raw = json.loads(text)
            self._mtime = os.path.getmtime(self._path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("role_matrix_load_failed", path=str(self._path), error=str(exc))
            self._matrix = {}
            return

        if "roles" in raw:
            roles_raw = raw["roles"]
            self._matrix = {}
            for role, entries in roles_raw.items():
                if isinstance(entries, list):
                    self._matrix[role] = {
                        e["model_id"]: e["rank"] for e in entries
                    }
                elif isinstance(entries, dict):
                    self._matrix[role] = entries
        else:
            self._matrix = raw
