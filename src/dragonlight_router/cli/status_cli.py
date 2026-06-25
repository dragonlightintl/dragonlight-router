"""CLI entry point for router status inspection.

Reads local state files and reports router health, budget, and
provider status without starting the HTTP server.

Commands:
  status   -- Show overall health summary (providers, circuits, retired models).
  budget   -- Show per-provider rate limit usage (RPM, RPD remaining).
  retired  -- List all retired models with retirement reason and timestamp.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from dragonlight_router.config.loader import load_config

# ---------------------------------------------------------------------------
# State directory resolution
# ---------------------------------------------------------------------------


def _resolve_state_dir(args_state_dir: str | None) -> Path:
    """Resolve state_dir from CLI arg or router config."""
    if args_state_dir:
        return Path(args_state_dir)
    result = load_config()
    if result.is_ok():
        return result.unwrap().state_dir
    return Path("./router_state")


def _resolve_config() -> Any:
    """Load router config, returning RouterConfig or None."""
    result = load_config()
    if result.is_ok():
        return result.unwrap()
    return None


# ---------------------------------------------------------------------------
# State file readers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON state file. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        text = path.read_text()
        if not text.strip():
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


def _query_budget_db(
    db_path: Path,
    provider_name: str,
) -> dict[str, int]:
    """Query budget.db for a provider's current RPM and RPD usage.

    Returns {"rpm_used": int, "rpd_used": int, "tpm_used": int, "daily_tokens": int}.
    """
    result = {"rpm_used": 0, "rpd_used": 0, "tpm_used": 0, "daily_tokens": 0}
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        now = time.time()
        cutoff_minute = now - 60.0
        start_of_day = dt.datetime.now(dt.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        # RPM: requests in last 60 seconds
        row = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp > ?",
            (provider_name, cutoff_minute),
        ).fetchone()
        result["rpm_used"] = row[0] if row else 0

        # RPD: requests since start of UTC day
        row = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp >= ?",
            (provider_name, start_of_day),
        ).fetchone()
        result["rpd_used"] = row[0] if row else 0

        # TPM: tokens in last 60 seconds
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens_used), 0)"
            " FROM request_log WHERE provider = ? AND timestamp > ?",
            (provider_name, cutoff_minute),
        ).fetchone()
        result["tpm_used"] = row[0] if row else 0

        # Daily tokens
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens_used), 0)"
            " FROM request_log WHERE provider = ? AND timestamp >= ?",
            (provider_name, start_of_day),
        ).fetchone()
        result["daily_tokens"] = row[0] if row else 0

        conn.close()
    except (sqlite3.Error, OSError):
        pass
    return result


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    """Show overall router health summary."""
    state_dir = _resolve_state_dir(args.state_dir)
    health_state = _load_json(state_dir / "health_state.json")
    budget_state = _load_json(state_dir / "budget_state.json")

    print(f"State directory: {state_dir.resolve()}")
    print()

    # --- Provider health from circuit breakers ---
    if health_state is None:
        print("Health state: not found (no health_state.json)")
    else:
        breaker_states = health_state.get("breaker_states", {})
        error_counts = health_state.get("error_counts", {})
        retired = health_state.get("retired", {})

        # Compute overall availability
        all_models = set(breaker_states.keys()) | set(retired.keys())
        available_count = 0
        for model_id in all_models:
            if model_id in retired:
                continue
            breaker = breaker_states.get(model_id, {})
            state = breaker.get("state", "closed")
            if state == "closed" or state == "half_open":
                available_count += 1
            elif state == "open":
                opened_at = breaker.get("opened_at", 0.0)
                # Default cooldown ~60s; conservative check
                if time.time() >= opened_at + 120.0:
                    available_count += 1

        total = len(all_models)
        if total == 0:
            availability = "healthy (no models tracked)"
        elif available_count == 0:
            availability = "UNAVAILABLE"
        elif available_count < total:
            availability = f"DEGRADED ({available_count}/{total} available)"
        else:
            availability = f"healthy ({available_count}/{total} available)"

        print(f"Overall availability: {availability}")
        print()

        # Circuit breaker summary
        print("Circuit breakers:")
        if not breaker_states:
            print("  (none tracked)")
        else:
            for model_id in sorted(breaker_states.keys()):
                breaker = breaker_states[model_id]
                state = breaker.get("state", "closed")
                errors = error_counts.get(model_id, 0)
                error_ts = breaker.get("error_timestamps", [])
                opened_at = breaker.get("opened_at", 0.0)

                status_str = state.upper()
                detail_parts = []
                if errors > 0:
                    detail_parts.append(f"{errors} errors")
                if state == "open" and opened_at > 0:
                    opened_dt = dt.datetime.fromtimestamp(opened_at, tz=dt.UTC)
                    detail_parts.append(f"opened {opened_dt.strftime('%H:%M:%S UTC')}")
                if error_ts:
                    detail_parts.append(f"{len(error_ts)} recent")

                detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
                print(f"  {model_id:55s} {status_str}{detail}")

        # Retired models summary (brief)
        if retired:
            print()
            print(f"Retired models: {len(retired)}")
            for model_id in sorted(retired.keys()):
                ts = retired[model_id]
                retired_dt = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
                print(f"  {model_id:55s} retired {retired_dt.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            print()
            print("Retired models: 0")

    # --- Budget summary ---
    print()
    if budget_state is None:
        print("Budget state: not found (no budget_state.json)")
    else:
        rpd_counts = budget_state.get("rpd_counts", {})
        daily_token_counts = budget_state.get("daily_token_counts", {})
        day_reset_at = budget_state.get("day_reset_at", 0.0)

        if day_reset_at > 0:
            reset_dt = dt.datetime.fromtimestamp(day_reset_at, tz=dt.UTC)
            print(f"Budget day resets at: {reset_dt.strftime('%Y-%m-%d %H:%M UTC')}")

        if rpd_counts:
            active_providers = {k: v for k, v in rpd_counts.items() if v > 0}
            if active_providers:
                print(f"Providers with daily requests: {len(active_providers)}")
                sorted_provs = sorted(
                    active_providers.items(), key=lambda x: x[1], reverse=True,
                )
                for prov, count in sorted_provs:
                    tokens = daily_token_counts.get(prov, 0)
                    print(f"  {prov:20s} {count:>6} requests, {tokens:>8} tokens")
            else:
                print("No provider requests recorded today.")
        else:
            print("No provider requests recorded today.")

    return 0


# ---------------------------------------------------------------------------
# budget command
# ---------------------------------------------------------------------------


def _cmd_budget(args: argparse.Namespace) -> int:
    """Show per-provider rate limit usage."""
    state_dir = _resolve_state_dir(args.state_dir)
    config = _resolve_config()

    if config is None or not config.providers:
        print("ERROR: No provider configuration found.", file=sys.stderr)
        print(
            "Set DRAGONLIGHT_ROUTER_CONFIG or ensure config/router.yaml exists.",
            file=sys.stderr,
        )
        return 1

    db_path = state_dir / "budget.db"
    has_db = db_path.exists()

    print(f"Budget status (state: {state_dir.resolve()})")
    if not has_db:
        print("  budget.db not found — no request history available.")
        print()

    for provider in config.providers:
        name = provider.name
        rpm_limit = provider.rate_limits.rpm
        rpd_limit = provider.rate_limits.rpd
        tpm_limit = provider.rate_limits.tpm
        daily_token_cap = provider.rate_limits.daily_token_cap

        if has_db:
            usage = _query_budget_db(db_path, name)
        else:
            usage = {"rpm_used": 0, "rpd_used": 0, "tpm_used": 0, "daily_tokens": 0}

        print()
        print(f"  {name}")

        # RPM
        rpm_remaining = max(0, rpm_limit - usage["rpm_used"])
        rpm_used = usage["rpm_used"]
        print(
            f"    RPM:  {rpm_used:>6} used / {rpm_limit:>6} limit"
            f"  ({rpm_remaining} remaining)"
        )

        # RPD
        rpd_used = usage["rpd_used"]
        if rpd_limit is not None:
            rpd_remaining = max(0, rpd_limit - rpd_used)
            print(
                f"    RPD:  {rpd_used:>6} used / {rpd_limit:>6} limit"
                f"  ({rpd_remaining} remaining)"
            )
        else:
            print(f"    RPD:  {rpd_used:>6} used / unlimited")

        # TPM
        tpm_used = usage["tpm_used"]
        if tpm_limit is not None:
            tpm_remaining = max(0, tpm_limit - tpm_used)
            print(
                f"    TPM:  {tpm_used:>6} used / {tpm_limit:>6} limit"
                f"  ({tpm_remaining} remaining)"
            )
        else:
            print(f"    TPM:  {tpm_used:>6} used / unlimited")

        # Daily token cap
        if daily_token_cap is not None:
            daily_used = usage["daily_tokens"]
            daily_remaining = max(0, daily_token_cap - daily_used)
            print(
                f"    Daily tokens: {daily_used:>8} used"
                f" / {daily_token_cap:>8} cap"
                f"  ({daily_remaining} remaining)"
            )

    return 0


# ---------------------------------------------------------------------------
# retired command
# ---------------------------------------------------------------------------


def _cmd_retired(args: argparse.Namespace) -> int:
    """List all retired models with retirement timestamp."""
    state_dir = _resolve_state_dir(args.state_dir)
    health_state = _load_json(state_dir / "health_state.json")

    if health_state is None:
        print("Health state: not found (no health_state.json)")
        print("No retired models to report.")
        return 0

    retired = health_state.get("retired", {})
    if not retired:
        print("No retired models.")
        return 0

    print(f"Retired models ({len(retired)}):")
    print()
    for model_id in sorted(retired.keys()):
        ts = retired[model_id]
        retired_dt = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
        age = time.time() - ts
        age_str = _format_age(age)
        print(f"  {model_id}")
        print(f"    Retired: {retired_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({age_str} ago)")

    return 0


def _format_age(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    return f"{days}d {hours}h"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dragonlight-router health, budget, and"
            " provider status without starting the server."
        ),
        prog="dragonlight-status",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- status --
    status_parser = subparsers.add_parser(
        "status",
        help="Show overall health summary (providers, circuits, retired models).",
    )
    status_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )

    # -- budget --
    budget_parser = subparsers.add_parser(
        "budget",
        help="Show per-provider rate limit usage (RPM, RPD, TPM remaining).",
    )
    budget_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )

    # -- retired --
    retired_parser = subparsers.add_parser(
        "retired",
        help="List all retired models with retirement timestamp.",
    )
    retired_parser.add_argument(
        "--state-dir",
        metavar="PATH",
        default=None,
        help="Path to router state directory (default: from config).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS = {
    "status": _cmd_status,
    "budget": _cmd_budget,
    "retired": _cmd_retired,
}


def main() -> None:
    """CLI entry point: dragonlight-status"""
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
