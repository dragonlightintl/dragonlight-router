"""CLI entry point for on-demand model probing.

Runs spectrography probes against one or more models and outputs flavor profiles.
Designed for quick, targeted probing without running the full spectrography suite.

Commands:
  probe     -- Run probes against specified models.
  show      -- Display stored profile for a model.
  history   -- Show recent spectrography runs.
  stale     -- List models needing re-probing.

Usage:
  dragonlight-probe probe --model nvidia_nim/kimi-k2.6
  dragonlight-probe probe --model groq/llama-3.3-70b-versatile --probes style edge_case
  dragonlight-probe show --model gemini/gemini-2.5-pro
  dragonlight-probe history
  dragonlight-probe stale
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_DEFAULT_JUDGE = "gemini/gemini-2.5-pro"
_DEFAULT_OUTPUT_DIR = "spectrography_results"
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "config" / "spectrography.db"


# ---------------------------------------------------------------------------
# Subcommand: probe
# ---------------------------------------------------------------------------


def _cmd_probe(args: argparse.Namespace) -> None:
    """Run probes against specified models."""
    from dragonlight_router.benchmark.calibration_audit import _get_all_model_ids
    from dragonlight_router.spectrography.runner import run_spectrography

    models = args.model if args.model else _get_all_model_ids()
    if not models:
        print("No model targets resolved. Check config/model_role_matrix.json.", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        from dragonlight_router.spectrography.probes import get_all_probes, get_probes_by_axis

        if args.probes:
            probe_count = sum(len(get_probes_by_axis(a)) for a in args.probes)
        else:
            probe_count = len(get_all_probes())
        total_pairs = len(models) * probe_count
        print(f"Dry run: {len(models)} models x {probe_count} probes = {total_pairs} pairs")
        print(f"Models: {', '.join(models)}")
        print(f"Judge: {args.judge_model}")
        return

    output_dir = Path(args.output_dir)

    asyncio.run(
        run_spectrography(
            models=models,
            judge_model=args.judge_model,
            output_dir=output_dir,
            provider_delays=_parse_delays(args.provider_delay),
            write_profiles=args.write_profiles,
            resume=args.resume,
            resume_from=args.resume_from,
            merge_checkpoints=args.merge_checkpoints,
        )
    )

    # Store results in SQLite if db_path is available
    _store_results_to_sqlite(output_dir, Path(args.db_path))


def _store_results_to_sqlite(output_dir: Path, db_path: Path) -> None:
    """Post-run: store results from the latest run into SQLite."""
    from dragonlight_router.spectrography.storage import SpectrographyStore

    # Find the most recent run directory
    if not output_dir.exists():
        return

    run_dirs = sorted(output_dir.iterdir(), reverse=True)
    if not run_dirs:
        return

    latest_run = run_dirs[0]
    report_path = latest_run / "report.json"
    if not report_path.exists():
        return

    import json

    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("sqlite_store_report_load_failed", error=str(exc))
        return

    store = SpectrographyStore(db_path)
    store.open()

    try:
        run_id = report.get("run_id", latest_run.name)

        store.record_run_start(
            run_id=run_id,
            judge_model=report.get("judge_model", "unknown"),
            model_count=len(report.get("models_evaluated", [])),
            probe_count=report.get("total_probes", 0),
        )

        store.record_run_complete(
            run_id=run_id,
            error_count=report.get("total_errors", 0),
            status="complete",
        )

        # Store profiles
        profiles_raw = report.get("profiles", {})
        if profiles_raw:
            from dragonlight_router.core.types import ModelSpectrographProfile, SpectrographScore

            profiles = {}
            for mid, pdata in profiles_raw.items():
                task_scores = _parse_scores_from_json(pdata.get("task_scores", {}))
                domain_scores = _parse_scores_from_json(pdata.get("domain_scores", {}))
                qs_scores = _parse_scores_from_json(pdata.get("qs_scores", {}))
                profiles[mid] = ModelSpectrographProfile(
                    model_id=mid,
                    version=1,
                    updated_at=report.get("completed_at", ""),
                    task_scores=task_scores,
                    domain_scores=domain_scores,
                    qs_scores=qs_scores,
                )
            store.store_profiles_batch(profiles, run_id)

        logger.info("sqlite_store_complete", run_id=run_id, db_path=str(db_path))
    finally:
        store.close()


def _parse_scores_from_json(
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Parse score entries from JSON report format into SpectrographScore instances."""
    from dragonlight_router.core.types import SpectrographScore

    scores = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            scores[key] = SpectrographScore(
                score=float(val.get("score", 0.5)),
                confidence=float(val.get("confidence", 0.0)),
                sample_count=int(val.get("sample_count", 0)),
            )
        elif isinstance(val, (int, float)):
            scores[key] = SpectrographScore(
                score=float(val),
                confidence=1.0,
                sample_count=0,
            )
    return scores


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------


def _cmd_show(args: argparse.Namespace) -> None:
    """Display stored profile for a model."""
    db_path = Path(args.db_path)

    if db_path.exists():
        _show_from_sqlite(args.model, db_path)
    else:
        _show_from_yaml(args.model)


def _show_from_sqlite(model_id: str, db_path: Path) -> None:
    """Display profile from SQLite store."""
    from dragonlight_router.spectrography.storage import SpectrographyStore

    store = SpectrographyStore(db_path)
    store.open()
    try:
        profile = store.load_profile(model_id)
        if profile is None:
            print(f"No profile found for {model_id} in SQLite store.")
            print("Falling back to YAML profiles...")
            _show_from_yaml(model_id)
            return
        _print_profile(profile)
    finally:
        store.close()


def _show_from_yaml(model_id: str) -> None:
    """Display profile from YAML config file."""
    from dragonlight_router.spectrography.lifecycle import load_existing_fingerprints

    profiles_path = _CONFIG_DIR / "model_spectrograph_profiles.yaml"
    profiles = load_existing_fingerprints(profiles_path)

    if model_id not in profiles:
        print(f"No profile found for {model_id}.")
        if profiles:
            print(f"\nAvailable models ({len(profiles)}):")
            for mid in sorted(profiles.keys()):
                print(f"  {mid}")
        return

    _print_profile(profiles[model_id])


def _print_profile(profile: Any) -> None:
    """Pretty-print a model spectrograph profile."""
    print(f"\nModel: {profile.model_id}")
    print(f"Updated: {profile.updated_at}")
    print(f"Version: {profile.version}")

    print("\nTask Scores:")
    print(f"  {'Task Type':<20} {'Score':>8} {'Confidence':>12} {'Samples':>9}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 9}")
    for key in sorted(profile.task_scores.keys()):
        fs = profile.task_scores[key]
        print(f"  {key:<20} {fs.score:>8.4f} {fs.confidence:>12.4f} {fs.sample_count:>9}")

    print("\nDomain Scores:")
    print(f"  {'Domain':<20} {'Score':>8} {'Confidence':>12} {'Samples':>9}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 9}")
    for key in sorted(profile.domain_scores.keys()):
        fs = profile.domain_scores[key]
        print(f"  {key:<20} {fs.score:>8.4f} {fs.confidence:>12.4f} {fs.sample_count:>9}")

    print("\nQuality/Speed Scores:")
    print(f"  {'QS Mode':<20} {'Score':>8} {'Confidence':>12} {'Samples':>9}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 9}")
    for key in sorted(profile.qs_scores.keys()):
        fs = profile.qs_scores[key]
        print(f"  {key:<20} {fs.score:>8.4f} {fs.confidence:>12.4f} {fs.sample_count:>9}")


# ---------------------------------------------------------------------------
# Subcommand: history
# ---------------------------------------------------------------------------


def _cmd_history(args: argparse.Namespace) -> None:
    """Show recent spectrography runs."""
    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"No spectrography database found at {db_path}.")
        print("Run `dragonlight-probe probe --model <model>` to create one.")
        return

    from dragonlight_router.spectrography.storage import SpectrographyStore

    store = SpectrographyStore(db_path)
    store.open()
    try:
        runs = store.get_run_history(limit=args.limit)
        if not runs:
            print("No spectrography runs recorded.")
            return

        print(f"\nRecent Spectrography Runs ({len(runs)}):")
        print(
            f"  {'Run ID':<36} {'Status':<10} {'Models':>7} "
            f"{'Probes':>7} {'Errors':>7} {'Started':<20}"
        )
        print(f"  {'-' * 36} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 20}")
        for run in runs:
            started = run.get("started_at", "")[:19]
            print(
                f"  {run['run_id']:<36} {run['status']:<10} "
                f"{run['model_count']:>7} {run['probe_count']:>7} "
                f"{run['error_count']:>7} {started:<20}"
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Subcommand: stale
# ---------------------------------------------------------------------------


def _cmd_stale(args: argparse.Namespace) -> None:
    """List models needing re-probing."""
    from dragonlight_router.spectrography.lifecycle import (
        get_models_needing_spectrography,
        load_existing_fingerprints,
    )

    profiles_path = _CONFIG_DIR / "model_spectrograph_profiles.yaml"
    matrix_path = _CONFIG_DIR / "model_role_matrix.json"

    existing = load_existing_fingerprints(profiles_path)
    needs_probing = get_models_needing_spectrography(
        matrix_path,
        existing,
        staleness_days=args.max_age,
    )

    if not needs_probing:
        print("All models have fresh profiles. Nothing to probe.")
        return

    print(f"\nModels needing spectrography ({len(needs_probing)}):")
    for mid in needs_probing:
        reason = "missing profile" if mid not in existing else "stale profile"
        print(f"  {mid:<60} ({reason})")

    print(f"\nRun: dragonlight-probe probe --model {' '.join(needs_probing[:3])}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_delays(raw: list[str] | None) -> dict[str, float] | None:
    """Parse --provider-delay key=value pairs into a dict."""
    if not raw:
        return None
    delays: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid --provider-delay: {item!r} (expected key=value)")
        k, v = item.split("=", 1)
        delays[k.strip()] = float(v.strip())
    return delays


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Dragonlight Spectrograph Probe -- on-demand model profiling.",
        prog="dragonlight-probe",
    )
    parser.add_argument(
        "--db-path",
        default=str(_DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {_DEFAULT_DB_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- probe ---
    probe_parser = subparsers.add_parser(
        "probe",
        help="Run probes against specified models",
    )
    probe_parser.add_argument(
        "--model",
        nargs="+",
        help="Model IDs to probe (default: all from role matrix)",
    )
    probe_parser.add_argument(
        "--probes",
        nargs="*",
        choices=["style", "edge_case", "reasoning_depth", "domain_cross",
                 "instruction_following", "speed_quality"],
        help="Probe axes to run (default: all)",
    )
    probe_parser.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE,
        help=f"Model to use as judge (default: {_DEFAULT_JUDGE})",
    )
    probe_parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {_DEFAULT_OUTPUT_DIR})",
    )
    probe_parser.add_argument(
        "--write-profiles",
        action="store_true",
        help="Write discovered profiles to config/ after completion",
    )
    probe_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    probe_parser.add_argument(
        "--resume-from",
        metavar="RUN_ID",
        help="Resume using checkpoint from a specific prior run",
    )
    probe_parser.add_argument(
        "--merge-checkpoints",
        action="store_true",
        help="Merge all prior checkpoints and skip completed pairs",
    )
    probe_parser.add_argument(
        "--provider-delay",
        nargs="*",
        metavar="K=V",
        help="Per-provider delay overrides (e.g. groq=2.0)",
    )
    probe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be probed without running",
    )
    probe_parser.set_defaults(func=_cmd_probe)

    # --- show ---
    show_parser = subparsers.add_parser(
        "show",
        help="Display stored profile for a model",
    )
    show_parser.add_argument(
        "--model",
        required=True,
        help="Model ID to display",
    )
    show_parser.set_defaults(func=_cmd_show)

    # --- history ---
    history_parser = subparsers.add_parser(
        "history",
        help="Show recent spectrography runs",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of runs to show (default: 20)",
    )
    history_parser.set_defaults(func=_cmd_history)

    # --- stale ---
    stale_parser = subparsers.add_parser(
        "stale",
        help="List models needing re-probing",
    )
    stale_parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max profile age in days before considered stale (default: 30)",
    )
    stale_parser.set_defaults(func=_cmd_stale)

    return parser


def main() -> None:
    """CLI entry point: dragonlight-probe."""
    parser = _build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)

    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
