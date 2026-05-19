"""Tests for budget/persistence.py — atomic state file I/O."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dragonlight_router.budget.persistence import load_budget_state, save_budget_state


class TestSaveBudgetState:
    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / "budget.json"
        save_budget_state({"groq": {"rpm_used": 5}}, path)
        assert path.exists()

    def test_valid_json(self, tmp_path: Path):
        path = tmp_path / "budget.json"
        state = {"groq": {"rpm_used": 5, "rpd_used": 100}}
        save_budget_state(state, path)
        loaded = json.loads(path.read_text())
        assert loaded == state

    def test_atomic_write_no_partial(self, tmp_path: Path):
        """If file already exists, new write replaces atomically."""
        path = tmp_path / "budget.json"
        save_budget_state({"old": True}, path)
        save_budget_state({"new": True}, path)
        loaded = json.loads(path.read_text())
        assert loaded == {"new": True}

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "budget.json"
        save_budget_state({"x": 1}, path)
        assert path.exists()


class TestLoadBudgetState:
    def test_loads_existing(self, tmp_path: Path):
        path = tmp_path / "budget.json"
        path.write_text(json.dumps({"groq": {"rpm_used": 5}}))
        result = load_budget_state(path)
        assert result == {"groq": {"rpm_used": 5}}

    def test_missing_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        result = load_budget_state(path)
        assert result is None

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "budget.json"
        path.write_text("not json {{{")
        result = load_budget_state(path)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "budget.json"
        path.write_text("")
        result = load_budget_state(path)
        assert result is None
