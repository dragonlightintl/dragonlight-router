"""CLI entry point for role matrix management.

Commands:
  seed    -- Auto-populate the role matrix from the provider catalog.
  update  -- Update the role matrix from spectrography results.
  show    -- Display the current role matrix in a readable table format.
  stats   -- Show matrix statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.config.loader import load_config

logger = structlog.get_logger()

_DEFAULT_BLEND = 0.7


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------


def _resolve_state_dir(args_state_dir: str | None) -> Path:
    """Resolve state_dir from CLI arg or router config."""
    if args_state_dir:
        return Path(args_state_dir)
    result = load_config()
    if result.is_ok():
        return result.unwrap().state_dir
    return Path("./router_state")


def _load_matrix(state_dir: Path) -> dict[str, Any] | None:
    """Load model_role_matrix.json from state_dir. Returns None if missing/invalid."""
    matrix_path = state_dir / "model_role_matrix.json"
    if not matrix_path.exists():
        logger.warning("matrix_file_missing", path=str(matrix_path))
        return None
    try:
        raw: dict[str, Any] = json.loads(matrix_path.read_text())
        return raw
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("matrix_load_failed", path=str(matrix_path), error=str(exc))
        return None


def _extract_roles(matrix_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract the roles dict from either full-schema or flat-dict format.

    Full schema: {"version": 1, "roles": {"coding": [{"model_id": ..., "rank": ...}]}}
    Flat dict:   {"coding": {"model_id": rank, ...}}

    Always returns: {"role": [{"model_id": str, "rank": int}, ...]}
    """
    if "roles" in matrix_data:
        roles_raw = matrix_data["roles"]
        roles: dict[str, list[dict[str, Any]]] = {}
        for role, entries in roles_raw.items():
            if isinstance(entries, list):
                roles[role] = entries
            elif isinstance(entries, dict):
                roles[role] = [{"model_id": mid, "rank": rank} for mid, rank in entries.items()]
            else:
                roles[role] = []
        return roles
    # Flat dict format
    flat_roles: dict[str, list[dict[str, Any]]] = {}
    for role, entries in matrix_data.items():
        if isinstance(entries, dict):
            flat_roles[role] = [{"model_id": mid, "rank": rank} for mid, rank in entries.items()]
        elif isinstance(entries, list):
            flat_roles[role] = entries
    return flat_roles


# ---------------------------------------------------------------------------
# seed command
# ---------------------------------------------------------------------------


def _cmd_seed(args: argparse.Namespace) -> int:
    """Auto-populate the role matrix from the provider catalog."""
    state_dir = _resolve_state_dir(args.state_dir)
    merge = args.merge
    config_path = Path(args.config) if args.config else None

    try:
        from dragonlight_router.roles.auto_populate import auto_populate_matrix
    except ImportError as exc:
        print(
            f"ERROR: auto_populate module not available: {exc}\n"
            "roles/auto_populate.py has not been built yet. "
            "Run this command once that module is in place.",
            file=sys.stderr,
        )
        return 1

    logger.info("matrix_seed_starting", state_dir=str(state_dir), merge=merge)
    try:
        auto_populate_matrix(state_dir, merge_existing=merge, config_path=config_path)
        print(f"Matrix seeded successfully -> {state_dir / 'model_role_matrix.json'}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("matrix_seed_failed", error=str(exc))
        print(f"ERROR: seed failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# update command
# ---------------------------------------------------------------------------


def _cmd_update(args: argparse.Namespace) -> int:
    """Update the role matrix from spectrography results."""
    state_dir = _resolve_state_dir(args.state_dir)
    spectrography_dir = Path(args.spectrography_dir) if args.spectrography_dir else None
    blend = args.blend

    try:
        from dragonlight_router.roles.matrix_updater import update_matrix_from_spectrography
    except ImportError as exc:
        print(
            f"ERROR: matrix_updater module not available: {exc}\n"
            "roles/matrix_updater.py has not been built yet. "
            "Run this command once that module is in place.",
            file=sys.stderr,
        )
        return 1

    logger.info(
        "matrix_update_starting",
        state_dir=str(state_dir),
        spectrography_dir=str(spectrography_dir),
        blend=blend,
    )
    try:
        update_matrix_from_spectrography(state_dir, spectrography_dir, blend_weight=blend)
        print(f"Matrix updated successfully -> {state_dir / 'model_role_matrix.json'}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("matrix_update_failed", error=str(exc))
        print(f"ERROR: update failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# show command
# ---------------------------------------------------------------------------


def _cmd_show(args: argparse.Namespace) -> int:
    """Display the current role matrix in a readable table format."""
    state_dir = _resolve_state_dir(args.state_dir)
    filter_role: str | None = args.role

    matrix_data = _load_matrix(state_dir)
    if matrix_data is None:
        print(
            f"ERROR: No matrix file found at {state_dir / 'model_role_matrix.json'}",
            file=sys.stderr,
        )
        return 1

    roles = _extract_roles(matrix_data)
    if not roles:
        print("Matrix is empty — no roles defined.")
        return 0

    role_names = [filter_role] if filter_role else sorted(roles.keys())

    for role in role_names:
        if role not in roles:
            print(f"Role '{role}' not found in matrix.", file=sys.stderr)
            continue
        entries = sorted(roles[role], key=lambda e: e["rank"], reverse=True)
        print(f"\nRole: {role} ({len(entries)} models)")
        for entry in entries:
            rank = entry["rank"]
            model_id = entry["model_id"]
            print(f"  {rank:<4} {model_id}")

    return 0


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


def _cmd_stats(args: argparse.Namespace) -> int:
    """Show matrix statistics."""
    state_dir = _resolve_state_dir(args.state_dir)

    matrix_data = _load_matrix(state_dir)
    if matrix_data is None:
        print(
            f"ERROR: No matrix file found at {state_dir / 'model_role_matrix.json'}",
            file=sys.stderr,
        )
        return 1

    roles = _extract_roles(matrix_data)
    if not roles:
        print("Matrix is empty — no roles defined.")
        return 0

    # Collect all model entries (deduplicated by model_id)
    all_model_ids: set[str] = set()
    provider_counts: dict[str, int] = {}
    all_ranks: list[int] = []
    role_model_counts: dict[str, int] = {}

    for role, entries in roles.items():
        role_model_counts[role] = len(entries)
        for entry in entries:
            mid = entry["model_id"]
            rank = entry["rank"]
            all_model_ids.add(mid)
            all_ranks.append(rank)
            provider = mid.split("/")[0] if "/" in mid else mid
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

    total_models = len(all_model_ids)
    roles_summary = ", ".join(
        f"{role} ({count})" for role, count in sorted(role_model_counts.items())
    )
    providers_summary = ", ".join(
        f"{prov} ({count})"
        for prov, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)
    )

    rank_min = min(all_ranks) if all_ranks else 0
    rank_max = max(all_ranks) if all_ranks else 0
    rank_mean = sum(all_ranks) / len(all_ranks) if all_ranks else 0.0

    print("Role Matrix Statistics")
    print(f"  Total unique models: {total_models}")
    print(f"  Roles: {roles_summary}")
    print(f"  Providers: {providers_summary}")
    print(f"  Rank distribution: min={rank_min}, max={rank_max}, mean={rank_mean:.1f}")

    return 0


# ---------------------------------------------------------------------------
# profile-pending command
# ---------------------------------------------------------------------------


def _cmd_profile_pending(args: argparse.Namespace) -> int:
    """List models that need spectrography profiling (heuristic ranks only)."""
    state_dir = _resolve_state_dir(args.state_dir)
    limit: int | None = args.limit

    try:
        from dragonlight_router.roles.lifecycle import get_models_needing_spectrography
    except ImportError as exc:
        print(
            f"ERROR: lifecycle module not available: {exc}\n"
            "roles/lifecycle.py has not been built yet. "
            "Run this command once that module is in place.",
            file=sys.stderr,
        )
        return 1

    model_ids = get_models_needing_spectrography(state_dir)

    if limit is not None:
        model_ids = model_ids[:limit]

    if not model_ids:
        print("No models need spectrography profiling.")
        return 0

    print(f"Models needing spectrography profiling ({len(model_ids)}):")
    for model_id in model_ids:
        print(f"  {model_id}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Role matrix management for dragonlight-router.",
        prog="dragonlight-matrix",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- seed --
    seed_parser = subparsers.add_parser(
        "seed",
        help="Auto-populate the role matrix from the provider catalog.",
    )
    seed_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )
    seed_parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to router config YAML (default: auto-resolved).",
    )
    seed_parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve existing operator-curated ranks (default: --merge).",
    )

    # -- update --
    update_parser = subparsers.add_parser(
        "update",
        help="Update the role matrix from spectrography results.",
    )
    update_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )
    update_parser.add_argument(
        "--spectrography-dir",
        metavar="PATH",
        default=None,
        help="Path to spectrography output directory.",
    )
    update_parser.add_argument(
        "--blend",
        type=float,
        default=_DEFAULT_BLEND,
        metavar="FLOAT",
        help=f"Blend weight for empirical scores (default: {_DEFAULT_BLEND}).",
    )

    # -- show --
    show_parser = subparsers.add_parser(
        "show",
        help="Display the current role matrix in readable table format.",
    )
    show_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )
    show_parser.add_argument(
        "--role",
        metavar="ROLE",
        default=None,
        help="Filter output to a single role.",
    )

    # -- stats --
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show matrix statistics.",
    )
    stats_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )

    # -- profile-pending --
    profile_pending_parser = subparsers.add_parser(
        "profile-pending",
        help="List models that need spectrography profiling (heuristic ranks only).",
    )
    profile_pending_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )
    profile_pending_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N models (sorted by rank descending).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS = {
    "seed": _cmd_seed,
    "update": _cmd_update,
    "show": _cmd_show,
    "stats": _cmd_stats,
    "profile-pending": _cmd_profile_pending,
}


def main() -> None:
    """CLI entry point: dragonlight-matrix"""
    parser = _build_parser()
    args = parser.parse_args()

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    exit_code = handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
