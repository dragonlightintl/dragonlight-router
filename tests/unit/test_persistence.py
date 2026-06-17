"""Tests for budget/persistence.py — atomic state file I/O.

Spec traceability: TM-022 (Budget state persistence)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dragonlight_router.budget.persistence import load_budget_state, save_budget_state


class TestSaveBudgetState:
    def test_creates_file(self, tmp_path: Path):
        """[TM-022 AC-1] save_budget_state creates the file on disk."""
        path = tmp_path / "budget.json"
        save_budget_state({"groq": {"rpm_used": 5}}, path)
        assert path.exists()

    def test_valid_json(self, tmp_path: Path):
        """[TM-022 AC-1] Saved state is valid JSON matching the input."""
        path = tmp_path / "budget.json"
        state = {"groq": {"rpm_used": 5, "rpd_used": 100}}
        save_budget_state(state, path)
        loaded = json.loads(path.read_text())
        assert loaded == state

    def test_atomic_write_no_partial(self, tmp_path: Path):
        """[TM-022 AC-2] Atomic write replaces existing file without partial state."""
        path = tmp_path / "budget.json"
        save_budget_state({"old": True}, path)
        save_budget_state({"new": True}, path)
        loaded = json.loads(path.read_text())
        assert loaded == {"new": True}

    def test_creates_parent_dirs(self, tmp_path: Path):
        """[TM-022 AC-1] save_budget_state creates parent directories."""
        path = tmp_path / "sub" / "dir" / "budget.json"
        save_budget_state({"x": 1}, path)
        assert path.exists()


class TestLoadBudgetState:
    def test_loads_existing(self, tmp_path: Path):
        """[TM-022 AC-3] load_budget_state returns existing state as Ok."""
        path = tmp_path / "budget.json"
        path.write_text(json.dumps({"groq": {"rpm_used": 5}}))
        result = load_budget_state(path)
        assert result.is_ok()
        assert result.unwrap() == {"groq": {"rpm_used": 5}}

    def test_missing_file_returns_none(self, tmp_path: Path):
        """[TM-022 AC-3] Missing file returns Ok(None)."""
        path = tmp_path / "nonexistent.json"
        result = load_budget_state(path)
        assert result.is_ok()
        assert result.unwrap() is None

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        """[TM-022 AC-3] Corrupt JSON file returns Ok(None)."""
        path = tmp_path / "budget.json"
        path.write_text("not json {{{")
        result = load_budget_state(path)
        assert result.is_ok()
        assert result.unwrap() is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        """[TM-022 AC-3] Empty file returns Ok(None)."""
        path = tmp_path / "budget.json"
        path.write_text("")
        result = load_budget_state(path)
        assert result.is_ok()
        assert result.unwrap() is None

    def test_oserror_on_load_returns_err(self, tmp_path: Path):
        """[TM-022 AC-3] OSError during read returns Err(StatePersistenceError) (lines 78-80)."""
        from unittest.mock import patch, MagicMock

        path = tmp_path / "budget.json"
        path.write_text('{"x": 1}')
        with patch.object(type(path), "read_text", side_effect=OSError("disk error")):
            result = load_budget_state(path)
        assert result.is_err()


class TestSaveBudgetStateError:
    def test_oserror_during_write_returns_err(self, tmp_path: Path):
        """[TM-022 AC-2] OSError during atomic write returns Err(StatePersistenceError) (lines 46-51)."""
        import os
        from unittest.mock import patch

        path = tmp_path / "budget.json"
        with patch("os.rename", side_effect=OSError("rename failed")):
            result = save_budget_state({"x": 1}, path)
        assert result.is_err()
