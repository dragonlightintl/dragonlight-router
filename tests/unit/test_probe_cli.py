"""Tests for cli/probe_cli.py — on-demand model probing CLI.

Tests the argument parsing, show command, stale detection, and history display.
Probe execution tests are kept lightweight (dry-run only) since they require
live API adapters tested in test_spectrography_runner_full.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from dragonlight_router.cli.probe_cli import (
    _build_parser,
    _cmd_history,
    _cmd_show,
    _cmd_stale,
    _parse_delays,
    _parse_scores_from_json,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# Argument parser tests
# ===========================================================================


class TestBuildParser:
    """Tests for _build_parser."""

    def test_creates_parser(self):
        parser = _build_parser()
        assert parser is not None

    def test_probe_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["probe", "--model", "test/model", "--dry-run"])
        assert args.command == "probe"
        assert args.model == ["test/model"]
        assert args.dry_run is True

    def test_show_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["show", "--model", "test/model"])
        assert args.command == "show"
        assert args.model == "test/model"

    def test_history_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["history"])
        assert args.command == "history"

    def test_stale_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["stale"])
        assert args.command == "stale"

    def test_probe_multiple_models(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "m2", "m3",
        ])
        assert args.model == ["m1", "m2", "m3"]

    def test_probe_with_axes(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--probes", "style", "edge_case",
        ])
        assert args.probes == ["style", "edge_case"]

    def test_default_judge_model(self):
        parser = _build_parser()
        args = parser.parse_args(["probe", "--model", "m1", "--dry-run"])
        assert args.judge_model == "gemini/gemini-2.5-pro"

    def test_custom_judge_model(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--judge-model", "custom/judge",
        ])
        assert args.judge_model == "custom/judge"

    def test_resume_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--resume",
        ])
        assert args.resume is True

    def test_resume_from(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--resume-from", "run-123",
        ])
        assert args.resume_from == "run-123"

    def test_merge_checkpoints(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--merge-checkpoints",
        ])
        assert args.merge_checkpoints is True

    def test_write_profiles_flag(self):
        parser = _build_parser()
        args = parser.parse_args([
            "probe", "--model", "m1", "--write-profiles",
        ])
        assert args.write_profiles is True

    def test_stale_max_age(self):
        parser = _build_parser()
        args = parser.parse_args(["stale", "--max-age", "14"])
        assert args.max_age == 14

    def test_history_limit(self):
        parser = _build_parser()
        args = parser.parse_args(["history", "--limit", "5"])
        assert args.limit == 5

    def test_custom_db_path(self):
        parser = _build_parser()
        args = parser.parse_args(["--db-path", "/tmp/test.db", "history"])
        assert args.db_path == "/tmp/test.db"


# ===========================================================================
# Parse helpers tests
# ===========================================================================


class TestParseDelays:
    """Tests for _parse_delays."""

    def test_parses_key_value_pairs(self):
        result = _parse_delays(["gemini=2.0", "groq=3.5"])
        assert result == {"gemini": 2.0, "groq": 3.5}

    def test_none_returns_none(self):
        assert _parse_delays(None) is None

    def test_empty_returns_none(self):
        assert _parse_delays([]) is None

    def test_invalid_format_raises(self):
        with pytest.raises(SystemExit, match="Invalid --provider-delay"):
            _parse_delays(["bad-no-equals"])


class TestParseScoresFromJson:
    """Tests for _parse_scores_from_json."""

    def test_parses_dict_format(self):
        raw = {
            "generation": {"score": 0.8, "confidence": 0.9, "sample_count": 10},
            "analysis": {"score": 0.6, "confidence": 0.7, "sample_count": 5},
        }
        result = _parse_scores_from_json(raw)
        assert result["generation"].score == pytest.approx(0.8)
        assert result["generation"].confidence == pytest.approx(0.9)
        assert result["generation"].sample_count == 10
        assert result["analysis"].score == pytest.approx(0.6)

    def test_parses_flat_format(self):
        raw = {"generation": 0.85, "analysis": 0.7}
        result = _parse_scores_from_json(raw)
        assert result["generation"].score == pytest.approx(0.85)
        assert result["generation"].confidence == pytest.approx(1.0)
        assert result["generation"].sample_count == 0

    def test_empty_returns_empty(self):
        result = _parse_scores_from_json({})
        assert result == {}


# ===========================================================================
# Show command tests
# ===========================================================================


class TestCmdShow:
    """Tests for _cmd_show via YAML fallback."""

    def test_show_unknown_model_prints_not_found(self, capsys, monkeypatch, tmp_path):
        """Show for unknown model should print 'No profile found'."""
        import argparse

        from dragonlight_router.spectrography.lifecycle import load_existing_fingerprints

        # Monkeypatch _CONFIG_DIR to a temp dir with no profiles
        monkeypatch.setattr(
            "dragonlight_router.cli.probe_cli._CONFIG_DIR",
            tmp_path,
        )
        args = argparse.Namespace(
            model="nonexistent/model",
            db_path="/tmp/nonexistent.db",
        )
        _cmd_show(args)
        captured = capsys.readouterr()
        assert "No profile found" in captured.out


# ===========================================================================
# History command tests
# ===========================================================================


class TestCmdHistory:
    """Tests for _cmd_history."""

    def test_history_no_db_prints_message(self, capsys):
        import argparse

        args = argparse.Namespace(
            db_path="/tmp/nonexistent_history_test.db",
            limit=20,
        )
        _cmd_history(args)
        captured = capsys.readouterr()
        assert "No spectrography database found" in captured.out

    def test_history_with_empty_db(self, tmp_path, capsys):
        import argparse

        from dragonlight_router.spectrography.storage import SpectrographyStore

        db_path = tmp_path / "hist.db"
        store = SpectrographyStore(db_path)
        store.open()
        store.close()

        args = argparse.Namespace(
            db_path=str(db_path),
            limit=20,
        )
        _cmd_history(args)
        captured = capsys.readouterr()
        assert "No spectrography runs recorded" in captured.out

    def test_history_with_runs(self, tmp_path, capsys):
        import argparse

        from dragonlight_router.spectrography.storage import SpectrographyStore

        db_path = tmp_path / "hist2.db"
        store = SpectrographyStore(db_path)
        store.open()
        store.record_run_start("run-001", "judge/model", 5, 80)
        store.record_run_complete("run-001", error_count=2)
        store.close()

        args = argparse.Namespace(
            db_path=str(db_path),
            limit=20,
        )
        _cmd_history(args)
        captured = capsys.readouterr()
        assert "run-001" in captured.out
        assert "complete" in captured.out


# ===========================================================================
# Stale command tests
# ===========================================================================


class TestCmdStale:
    """Tests for _cmd_stale."""

    def test_stale_with_no_models(self, capsys, monkeypatch, tmp_path):
        import argparse

        # Create empty matrix and profiles
        matrix = {"roles": {}}
        matrix_path = tmp_path / "model_role_matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        profiles_path = tmp_path / "model_spectrograph_profiles.yaml"
        profiles_path.write_text("profiles: {}")

        monkeypatch.setattr(
            "dragonlight_router.cli.probe_cli._CONFIG_DIR",
            tmp_path,
        )
        args = argparse.Namespace(max_age=30)
        _cmd_stale(args)
        captured = capsys.readouterr()
        assert "All models have fresh profiles" in captured.out

    def test_stale_with_missing_profiles(self, capsys, monkeypatch, tmp_path):
        import argparse

        matrix = {
            "roles": {
                "coding": [
                    {"model_id": "test/model-a", "rank": 90},
                    {"model_id": "test/model-b", "rank": 80},
                ],
            },
        }
        matrix_path = tmp_path / "model_role_matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        profiles_path = tmp_path / "model_spectrograph_profiles.yaml"
        profiles_path.write_text("profiles: {}")

        monkeypatch.setattr(
            "dragonlight_router.cli.probe_cli._CONFIG_DIR",
            tmp_path,
        )
        args = argparse.Namespace(max_age=30)
        _cmd_stale(args)
        captured = capsys.readouterr()
        assert "Models needing spectrography" in captured.out
        assert "test/model-a" in captured.out
        assert "missing profile" in captured.out
