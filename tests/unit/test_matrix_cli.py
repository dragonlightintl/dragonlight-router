"""Tests for cli/matrix_cli.py -- role matrix management CLI.

Spec traceability: TM-019 (Matrix CLI)
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dragonlight_router.cli.matrix_cli import (
    _build_parser,
    _cmd_profile_pending,
    _cmd_show,
    _cmd_stats,
    _extract_roles,
    _load_matrix,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_MATRIX_FULL_SCHEMA = {
    "version": 1,
    "default_rank": 20,
    "roles": {
        "coding": [
            {"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 95},
            {"model_id": "nvidia_nim/deepseek-ai/deepseek-v4-pro", "rank": 90},
            {"model_id": "groq/llama-3.3-70b-versatile", "rank": 75},
        ],
        "testing": [
            {"model_id": "nvidia_nim/qwen/qwen3.5-397b-a17b", "rank": 92},
            {"model_id": "groq/llama-3.3-70b-versatile", "rank": 75},
        ],
        "review": [
            {"model_id": "nvidia_nim/qwen/qwen3.5-397b-a17b", "rank": 95},
            {"model_id": "groq/llama-3.3-70b-versatile", "rank": 72},
        ],
        "spec": [
            {"model_id": "nvidia_nim/deepseek-ai/deepseek-v4-pro", "rank": 95},
            {"model_id": "groq/llama-3.3-70b-versatile", "rank": 70},
        ],
        "reasoning": [
            {"model_id": "nvidia_nim/qwen/qwen3.5-397b-a17b", "rank": 95},
            {"model_id": "groq/deepseek-r1-distill-llama-70b", "rank": 82},
            {"model_id": "groq/llama-3.3-70b-versatile", "rank": 70},
            {"model_id": "openrouter/qwen/qwen3-coder:free", "rank": 55},
        ],
    },
}

_SAMPLE_MATRIX_FLAT = {
    "coding": {
        "nvidia_nim/moonshotai/kimi-k2.6": 95,
        "groq/llama-3.3-70b-versatile": 75,
    },
    "testing": {
        "nvidia_nim/qwen/qwen3.5-397b-a17b": 92,
    },
}


def _write_matrix(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _make_show_args(state_dir: str, role: str | None = None):
    """Build a minimal argparse.Namespace for the show command."""
    import argparse

    ns = argparse.Namespace()
    ns.state_dir = state_dir
    ns.role = role
    return ns


def _make_stats_args(state_dir: str):
    """Build a minimal argparse.Namespace for the stats command."""
    import argparse

    ns = argparse.Namespace()
    ns.state_dir = state_dir
    return ns


# ---------------------------------------------------------------------------
# _load_matrix
# ---------------------------------------------------------------------------


class TestLoadMatrix:
    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """Missing matrix file returns None."""
        result = _load_matrix(tmp_path)
        assert result is None

    def test_returns_dict_on_valid_file(self, tmp_path: Path):
        """Valid JSON file returns dict."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        result = _load_matrix(tmp_path)
        assert result is not None
        assert "roles" in result

    def test_returns_none_on_invalid_json(self, tmp_path: Path):
        """Malformed JSON returns None."""
        (tmp_path / "model_role_matrix.json").write_text("{not valid json")
        result = _load_matrix(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_roles
# ---------------------------------------------------------------------------


class TestExtractRoles:
    def test_full_schema_format(self):
        """Full schema with 'roles' key is parsed correctly."""
        roles = _extract_roles(_SAMPLE_MATRIX_FULL_SCHEMA)
        assert "coding" in roles
        assert len(roles["coding"]) == 3
        assert roles["coding"][0]["model_id"] == "nvidia_nim/moonshotai/kimi-k2.6"
        assert roles["coding"][0]["rank"] == 95

    def test_flat_dict_format(self):
        """Flat dict format is converted to list-of-dicts correctly."""
        roles = _extract_roles(_SAMPLE_MATRIX_FLAT)
        assert "coding" in roles
        coding_ids = {e["model_id"] for e in roles["coding"]}
        assert "nvidia_nim/moonshotai/kimi-k2.6" in coding_ids

    def test_full_schema_with_dict_entries(self):
        """Full schema that uses dict entries (not list) is handled."""
        data = {
            "roles": {
                "coding": {
                    "nvidia_nim/moonshotai/kimi-k2.6": 95,
                    "groq/llama-3.3-70b-versatile": 75,
                },
            }
        }
        roles = _extract_roles(data)
        assert len(roles["coding"]) == 2


# ---------------------------------------------------------------------------
# show command
# ---------------------------------------------------------------------------


class TestCmdShow:
    def test_show_all_roles(self, tmp_path: Path, capsys):
        """show prints all roles when no --role filter is given."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_show_args(str(tmp_path))
        rc = _cmd_show(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Role: coding" in captured.out
        assert "Role: testing" in captured.out
        assert "nvidia_nim/moonshotai/kimi-k2.6" in captured.out

    def test_show_filtered_role(self, tmp_path: Path, capsys):
        """show with --role filters to only that role."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_show_args(str(tmp_path), role="coding")
        rc = _cmd_show(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Role: coding" in captured.out
        assert "Role: testing" not in captured.out

    def test_show_sorted_by_rank_descending(self, tmp_path: Path, capsys):
        """show displays models sorted by rank descending."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_show_args(str(tmp_path), role="coding")
        _cmd_show(args)
        captured = capsys.readouterr()
        rank_prefixes = ("9", "8", "7", "6", "5")
        lines = [ln for ln in captured.out.splitlines() if ln.strip().startswith(rank_prefixes)]
        ranks = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                with contextlib.suppress(ValueError):
                    ranks.append(int(parts[0]))
        assert ranks == sorted(ranks, reverse=True)

    def test_show_missing_matrix(self, tmp_path: Path, capsys):
        """show returns exit code 1 when matrix file is missing."""
        args = _make_show_args(str(tmp_path))
        rc = _cmd_show(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_show_unknown_role_filter(self, tmp_path: Path, capsys):
        """show with an unknown --role prints a message to stderr."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_show_args(str(tmp_path), role="nonexistent_role")
        rc = _cmd_show(args)
        assert rc == 0  # exits 0, just warns
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_show_model_count_in_header(self, tmp_path: Path, capsys):
        """show includes model count in the role header line."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_show_args(str(tmp_path), role="coding")
        _cmd_show(args)
        captured = capsys.readouterr()
        assert "3 models" in captured.out


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


class TestCmdStats:
    def test_stats_total_unique_models(self, tmp_path: Path, capsys):
        """stats counts unique model IDs across all roles."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_stats_args(str(tmp_path))
        rc = _cmd_stats(args)
        assert rc == 0
        captured = capsys.readouterr()
        # groq/llama-3.3-70b-versatile appears in 5 roles but counts once
        assert "Total unique models:" in captured.out
        # 6 unique models in _SAMPLE_MATRIX_FULL_SCHEMA
        assert "6" in captured.out

    def test_stats_roles_listed(self, tmp_path: Path, capsys):
        """stats lists all roles with their model counts."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_stats_args(str(tmp_path))
        _cmd_stats(args)
        captured = capsys.readouterr()
        assert "coding" in captured.out
        assert "testing" in captured.out
        assert "reasoning" in captured.out

    def test_stats_providers_listed(self, tmp_path: Path, capsys):
        """stats lists providers with their entry counts."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_stats_args(str(tmp_path))
        _cmd_stats(args)
        captured = capsys.readouterr()
        assert "nvidia_nim" in captured.out
        assert "groq" in captured.out

    def test_stats_rank_distribution(self, tmp_path: Path, capsys):
        """stats shows min, max, and mean rank."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FULL_SCHEMA)
        args = _make_stats_args(str(tmp_path))
        _cmd_stats(args)
        captured = capsys.readouterr()
        assert "min=" in captured.out
        assert "max=" in captured.out
        assert "mean=" in captured.out
        # min rank is 55, max is 95
        assert "min=55" in captured.out
        assert "max=95" in captured.out

    def test_stats_missing_matrix(self, tmp_path: Path, capsys):
        """stats returns exit code 1 when matrix file is missing."""
        args = _make_stats_args(str(tmp_path))
        rc = _cmd_stats(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_stats_flat_format(self, tmp_path: Path, capsys):
        """stats works with flat-dict format matrix."""
        _write_matrix(tmp_path / "model_role_matrix.json", _SAMPLE_MATRIX_FLAT)
        args = _make_stats_args(str(tmp_path))
        rc = _cmd_stats(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Total unique models:" in captured.out


# ---------------------------------------------------------------------------
# Argparse parsing
# ---------------------------------------------------------------------------


class TestArgparseParsing:
    def test_seed_defaults(self):
        """seed subcommand: defaults are correct."""
        parser = _build_parser()
        args = parser.parse_args(["seed"])
        assert args.command == "seed"
        assert args.state_dir is None
        assert args.config is None
        assert args.merge is True  # --merge is default

    def test_seed_no_merge(self):
        """seed --no-merge sets merge=False."""
        parser = _build_parser()
        args = parser.parse_args(["seed", "--no-merge"])
        assert args.merge is False

    def test_seed_with_state_dir(self):
        """seed --state-dir sets the state_dir."""
        parser = _build_parser()
        args = parser.parse_args(["seed", "--state-dir", "/tmp/state"])
        assert args.state_dir == "/tmp/state"

    def test_seed_with_config(self):
        """seed --config sets the config path."""
        parser = _build_parser()
        args = parser.parse_args(["seed", "--config", "/tmp/router.yaml"])
        assert args.config == "/tmp/router.yaml"

    def test_update_defaults(self):
        """update subcommand: defaults are correct."""
        parser = _build_parser()
        args = parser.parse_args(["update"])
        assert args.command == "update"
        assert args.state_dir is None
        assert args.spectrography_dir is None
        assert args.blend == 0.7

    def test_update_with_blend(self):
        """update --blend sets the blend weight."""
        parser = _build_parser()
        args = parser.parse_args(["update", "--blend", "0.5"])
        assert args.blend == pytest.approx(0.5)

    def test_update_with_spectrography_dir(self):
        """update --spectrography-dir sets the spectrography_dir."""
        parser = _build_parser()
        args = parser.parse_args(["update", "--spectrography-dir", "/tmp/spec"])
        assert args.spectrography_dir == "/tmp/spec"

    def test_show_defaults(self):
        """show subcommand: defaults are correct."""
        parser = _build_parser()
        args = parser.parse_args(["show"])
        assert args.command == "show"
        assert args.state_dir is None
        assert args.role is None

    def test_show_with_role(self):
        """show --role sets the role filter."""
        parser = _build_parser()
        args = parser.parse_args(["show", "--role", "coding"])
        assert args.role == "coding"

    def test_stats_defaults(self):
        """stats subcommand: defaults are correct."""
        parser = _build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"
        assert args.state_dir is None

    def test_stats_with_state_dir(self):
        """stats --state-dir sets the state_dir."""
        parser = _build_parser()
        args = parser.parse_args(["stats", "--state-dir", "/tmp/state"])
        assert args.state_dir == "/tmp/state"

    def test_missing_subcommand_exits(self):
        """No subcommand causes SystemExit."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_subcommand_exits(self):
        """Unknown subcommand causes SystemExit."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["notacommand"])


# ---------------------------------------------------------------------------
# ImportError handling (seed / update)
# ---------------------------------------------------------------------------


class TestImportErrorHandling:
    def test_seed_import_error(self, tmp_path: Path, capsys):
        """seed prints a clear message when auto_populate is missing."""
        import argparse

        ns = argparse.Namespace()
        ns.state_dir = str(tmp_path)
        ns.config = None
        ns.merge = True

        with patch.dict("sys.modules", {"dragonlight_router.roles.auto_populate": None}):
            from dragonlight_router.cli.matrix_cli import _cmd_seed

            rc = _cmd_seed(ns)

        assert rc == 1
        captured = capsys.readouterr()
        assert "auto_populate" in captured.err

    def test_update_import_error(self, tmp_path: Path, capsys):
        """update prints a clear message when matrix_updater is missing."""
        import argparse

        ns = argparse.Namespace()
        ns.state_dir = str(tmp_path)
        ns.spectrography_dir = None
        ns.blend = 0.7

        with patch.dict("sys.modules", {"dragonlight_router.roles.matrix_updater": None}):
            from dragonlight_router.cli.matrix_cli import _cmd_update

            rc = _cmd_update(ns)

        assert rc == 1
        captured = capsys.readouterr()
        assert "matrix_updater" in captured.err


# ---------------------------------------------------------------------------
# profile-pending command
# ---------------------------------------------------------------------------


class TestCmdProfilePending:
    def _make_args(self, state_dir: str, limit: int | None = None):
        """Build a minimal argparse.Namespace for the profile-pending command."""
        import argparse

        ns = argparse.Namespace()
        ns.state_dir = state_dir
        ns.limit = limit
        return ns

    def test_output_lists_models(self, tmp_path: Path, capsys):
        """profile-pending prints one model per line."""
        from unittest.mock import patch

        model_ids = ["prov/model-alpha", "prov/model-beta"]
        with patch(
            "dragonlight_router.roles.lifecycle.get_models_needing_spectrography",
            return_value=model_ids,
        ):
            args = self._make_args(str(tmp_path))
            rc = _cmd_profile_pending(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "prov/model-alpha" in captured.out
        assert "prov/model-beta" in captured.out

    def test_output_includes_count_header(self, tmp_path: Path, capsys):
        """profile-pending prints a header with the total count."""
        from unittest.mock import patch

        model_ids = ["prov/model-alpha", "prov/model-beta"]
        with patch(
            "dragonlight_router.roles.lifecycle.get_models_needing_spectrography",
            return_value=model_ids,
        ):
            args = self._make_args(str(tmp_path))
            rc = _cmd_profile_pending(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "2" in captured.out

    def test_with_limit_flag(self, tmp_path: Path, capsys):
        """profile-pending --limit N returns at most N models."""
        from unittest.mock import patch

        model_ids = ["prov/model-a", "prov/model-b", "prov/model-c"]
        with patch(
            "dragonlight_router.roles.lifecycle.get_models_needing_spectrography",
            return_value=model_ids,
        ):
            args = self._make_args(str(tmp_path), limit=2)
            rc = _cmd_profile_pending(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "prov/model-a" in captured.out
        assert "prov/model-b" in captured.out
        assert "prov/model-c" not in captured.out

    def test_empty_result_prints_informative_message(self, tmp_path: Path, capsys):
        """profile-pending prints a message when no models need profiling."""
        from unittest.mock import patch

        with patch(
            "dragonlight_router.roles.lifecycle.get_models_needing_spectrography",
            return_value=[],
        ):
            args = self._make_args(str(tmp_path))
            rc = _cmd_profile_pending(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "No models" in captured.out

    def test_import_error_when_lifecycle_unavailable(self, tmp_path: Path, capsys):
        """profile-pending prints a clear error when lifecycle module is missing."""
        with patch.dict("sys.modules", {"dragonlight_router.roles.lifecycle": None}):
            from dragonlight_router.cli.matrix_cli import _cmd_profile_pending as _cmd

            args = self._make_args(str(tmp_path))
            rc = _cmd(args)

        assert rc == 1
        captured = capsys.readouterr()
        assert "lifecycle" in captured.err

    def test_argparse_profile_pending_defaults(self):
        """profile-pending subcommand: defaults are correct."""
        parser = _build_parser()
        args = parser.parse_args(["profile-pending"])
        assert args.command == "profile-pending"
        assert args.state_dir is None
        assert args.limit is None

    def test_argparse_profile_pending_with_limit(self):
        """profile-pending --limit sets the limit."""
        parser = _build_parser()
        args = parser.parse_args(["profile-pending", "--limit", "10"])
        assert args.limit == 10

    def test_argparse_profile_pending_with_state_dir(self):
        """profile-pending --state-dir sets the state_dir."""
        parser = _build_parser()
        args = parser.parse_args(["profile-pending", "--state-dir", "/tmp/state"])
        assert args.state_dir == "/tmp/state"
