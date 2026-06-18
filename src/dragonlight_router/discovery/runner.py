"""Model Flavor Discovery -- direct-adapter orchestrator for profiling model strengths.

Evaluates models DIRECTLY through GenerativeBackend adapters (not the router
HTTP API) to isolate intrinsic model behavior from router machinery.  Produces
flavor fingerprints, calibration deltas, and ranked summaries.

Spec reference: model-flavor-discovery-v0.1.0-spec.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.benchmark.dogfood import _get_all_model_ids
from dragonlight_router.benchmark.judge import (
    _JUDGE_SYSTEM_PROMPT,
    _JUDGE_USER_TEMPLATE,
    _normalize_scores,
    _parse_judge_scores,
)
from dragonlight_router.core.types import (
    GenerativeBackend,
    ModelFlavorProfile,
)
from dragonlight_router.discovery.analyzer import (
    ProbeResult,
    aggregate_scores,
    build_fingerprints_yaml,
    build_model_rankings,
    compute_calibration_deltas,
    rank_normalize,
)
from dragonlight_router.discovery.lifecycle import (
    load_existing_fingerprints,
    merge_incremental,
    write_fingerprints_yaml,
)
from dragonlight_router.discovery.probes import DiscoveryProbe, get_all_probes

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE = "gemini/gemini-2.5-pro"
_DEFAULT_OUTPUT_DIR = "discovery_results"
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_BACKOFF_SCHEDULE = [5.0, 10.0, 20.0, 40.0]
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Mutable state for a discovery run."""

    run_id: str
    started_at: str
    results: list[ProbeResult] = field(default_factory=list)
    completed_pairs: set[tuple[str, str]] = field(default_factory=set)
    total_errors: int = 0
    shutdown_requested: bool = False


# ---------------------------------------------------------------------------
# Provider pacing
# ---------------------------------------------------------------------------

class ProviderPacer:
    """Track per-provider last-request timestamps and enforce delays."""

    _DEFAULT_DELAYS: dict[str, float] = {
        "gemini": 1.0, "groq": 1.5, "nvidia_nim": 1.0, "openrouter": 2.0,
    }

    def __init__(self, overrides: dict[str, float] | None = None) -> None:
        self._delays = dict(self._DEFAULT_DELAYS)
        if overrides:
            self._delays.update(overrides)
        self._last: dict[str, float] = {}

    async def wait(self, provider: str) -> None:
        """Sleep until minimum inter-request delay has elapsed."""
        delay = self._delays.get(provider, 1.0)
        elapsed = time.monotonic() - self._last.get(provider, 0.0)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last[provider] = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_provider(model_id: str) -> str:
    """Extract provider prefix from model_id (e.g. 'gemini/...' -> 'gemini')."""
    return model_id.split("/")[0] if "/" in model_id else "unknown"


# ---------------------------------------------------------------------------
# Adapter creation
# ---------------------------------------------------------------------------

def _create_adapter(model_id: str) -> GenerativeBackend | None:
    """Create a fresh adapter for the given model. Returns None on failure.

    # TODO: wire to actual adapter factory once available
    Adapter instantiation requires building a full BackendConfig with
    provider-specific base URLs, env keys, capabilities, cost profiles,
    and rate limits.  This placeholder logs a warning and returns None;
    the runner handles None adapters gracefully by skipping the model.
    """
    logger.warning("adapter_creation_placeholder", model_id=model_id,
                   msg="adapter factory not yet wired -- skipping model")
    return None


# ---------------------------------------------------------------------------
# Streaming response collector
# ---------------------------------------------------------------------------

async def _collect_streaming_response(
    adapter: GenerativeBackend,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str | None:
    """Collect full text from a streaming adapter. Returns None on error."""
    assert isinstance(messages, list), "messages must be a list"
    assert len(messages) > 0, "messages must not be empty"

    try:
        chunks: list[str] = []
        async for chunk in adapter.generate(
            messages, max_tokens=max_tokens, temperature=temperature, stream=True,
        ):
            chunks.append(chunk)
        return "".join(chunks) if chunks else None
    except Exception as exc:
        logger.warning("adapter_generate_error", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def _evaluate_probe(
    model_id: str,
    probe: DiscoveryProbe,
    model_adapter: GenerativeBackend,
    judge_adapter: GenerativeBackend,
    judge_model_id: str,
) -> ProbeResult:
    """Evaluate a single (model, probe) pair and judge the response.

    Steps:
        1. Collect response from model adapter (streaming, max_tokens=512, temp=0.7).
        2. If empty response, return ProbeResult with score 0.0.
        3. Build judge messages using _JUDGE_USER_TEMPLATE.
        4. Collect judge response (max_tokens=256, temperature=0.0).
        5. Parse and normalize judge scores.
        6. Return ProbeResult.
    """
    assert isinstance(model_id, str), "model_id must be a string"
    assert isinstance(probe, DiscoveryProbe), "probe must be a DiscoveryProbe"

    is_self_eval = (model_id == judge_model_id)

    # Step 1: collect model response
    model_messages = [{"role": "user", "content": probe.prompt}]
    model_response = await _collect_streaming_response(
        model_adapter, model_messages, max_tokens=512, temperature=0.7,
    )

    # Step 2: handle empty response
    if not model_response or not model_response.strip():
        logger.warning("empty_model_response", model_id=model_id, probe_id=probe.id)
        return ProbeResult(
            model_id=model_id,
            probe_id=probe.id,
            task_type=probe.task_type,
            domain=probe.domain,
            quality_speed=probe.quality_speed,
            judge_scores=None,
            normalized_score=0.0,
            is_self_eval=is_self_eval,
            error="empty_model_response",
        )

    # Step 3: build judge messages
    judge_user_text = _JUDGE_USER_TEMPLATE.format(
        original_prompt=probe.prompt,
        judge_criteria=probe.judge_criteria,
        quality_speed=probe.quality_speed,
        model_response=model_response,
    )
    judge_messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_text},
    ]

    # Step 4: collect judge response
    judge_raw = await _collect_streaming_response(
        judge_adapter, judge_messages, max_tokens=256, temperature=0.0,
    )

    if judge_raw is None:
        logger.warning("judge_call_failed", model_id=model_id, probe_id=probe.id)
        return ProbeResult(
            model_id=model_id,
            probe_id=probe.id,
            task_type=probe.task_type,
            domain=probe.domain,
            quality_speed=probe.quality_speed,
            judge_scores=None,
            normalized_score=0.5,
            is_self_eval=is_self_eval,
            error="judge_call_failed",
        )

    # Step 5: parse and normalize judge scores
    scores = _parse_judge_scores(judge_raw)
    if scores is None:
        logger.warning("judge_parse_failed", model_id=model_id, probe_id=probe.id)
        return ProbeResult(
            model_id=model_id,
            probe_id=probe.id,
            task_type=probe.task_type,
            domain=probe.domain,
            quality_speed=probe.quality_speed,
            judge_scores=None,
            normalized_score=0.5,
            is_self_eval=is_self_eval,
            error="judge_parse_failed",
        )

    normalized = _normalize_scores(scores)

    # Step 6: return ProbeResult
    if is_self_eval:
        logger.info("self_evaluation_detected", model_id=model_id, probe_id=probe.id)

    return ProbeResult(
        model_id=model_id,
        probe_id=probe.id,
        task_type=probe.task_type,
        domain=probe.domain,
        quality_speed=probe.quality_speed,
        judge_scores=scores,
        normalized_score=normalized,
        is_self_eval=is_self_eval,
        error=None,
    )


# ---------------------------------------------------------------------------
# Interleaved scheduling
# ---------------------------------------------------------------------------

def _interleaved_schedule(
    models: list[str], probes: list[DiscoveryProbe],
) -> list[tuple[str, DiscoveryProbe]]:
    """Build provider-interleaved (model, probe) schedule.

    Iterates probes as the outer loop and models as the inner loop so that
    consecutive calls alternate across providers rather than hammering a
    single provider for all probes before moving on.
    """
    return [(m, p) for p in probes for m in models]


# ---------------------------------------------------------------------------
# Checkpoint (JSONL resume support)
# ---------------------------------------------------------------------------

def _cp_path(output_dir: Path, run_id: str) -> Path:
    """Checkpoint file path for a run."""
    return output_dir / run_id / "checkpoint.jsonl"


def _load_checkpoint(path: Path) -> set[tuple[str, str]]:
    """Load completed (model, probe_id) pairs from checkpoint JSONL."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            completed.add((d["model_id"], d["probe_id"]))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("checkpoint_line_skip", error=str(exc))
            continue
    logger.info("checkpoint_loaded", completed_pairs=len(completed))
    return completed


def _append_checkpoint(path: Path, result: ProbeResult) -> None:
    """Append one result to checkpoint JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "model_id": result.model_id,
        "probe_id": result.probe_id,
        "task_type": result.task_type,
        "domain": result.domain,
        "quality_speed": result.quality_speed,
        "judge_scores": result.judge_scores,
        "normalized_score": result.normalized_score,
        "is_self_eval": result.is_self_eval,
        "error": result.error,
    })
    with open(path, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_json_report(
    run_id: str,
    started_at: str,
    completed_at: str,
    judge_model: str,
    results: list[ProbeResult],
    profiles: dict[str, ModelFlavorProfile],
    deltas: dict[str, dict[str, Any]],
    rankings: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Write the full JSON report."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Serialize profiles to JSON-safe dicts
    profiles_json: dict[str, dict[str, Any]] = {}
    for mid, profile in profiles.items():
        profiles_json[mid] = {
            "model_id": profile.model_id,
            "version": profile.version,
            "updated_at": profile.updated_at,
            "task_scores": {
                k: {"score": v.score, "confidence": v.confidence,
                     "sample_count": v.sample_count}
                for k, v in profile.task_scores.items()
            },
            "domain_scores": {
                k: {"score": v.score, "confidence": v.confidence,
                     "sample_count": v.sample_count}
                for k, v in profile.domain_scores.items()
            },
            "qs_scores": {
                k: {"score": v.score, "confidence": v.confidence,
                     "sample_count": v.sample_count}
                for k, v in profile.qs_scores.items()
            },
        }

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "judge_model": judge_model,
        "models_evaluated": sorted(profiles_json.keys()),
        "total_probes": len(results),
        "total_errors": sum(1 for r in results if r.error),
        "self_eval_count": sum(1 for r in results if r.is_self_eval),
        "profiles": profiles_json,
        "calibration_deltas": deltas,
        "rankings": rankings,
        "per_probe_results": [
            {
                "model_id": r.model_id,
                "probe_id": r.probe_id,
                "task_type": r.task_type,
                "domain": r.domain,
                "quality_speed": r.quality_speed,
                "judge_scores": r.judge_scores,
                "normalized_score": r.normalized_score,
                "is_self_eval": r.is_self_eval,
                "error": r.error,
            }
            for r in results
        ],
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("json_report_written", path=str(report_path))


def _write_markdown_summary(
    run_id: str,
    started_at: str,
    completed_at: str,
    judge_model: str,
    results: list[ProbeResult],
    profiles: dict[str, ModelFlavorProfile],
    deltas: dict[str, dict[str, Any]],
    rankings: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Write human-readable markdown summary."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    self_eval_count = sum(1 for r in results if r.is_self_eval)
    error_count = sum(1 for r in results if r.error)

    lines = [
        "# Model Flavor Discovery Report", "",
        f"- **Run ID:** {run_id}",
        f"- **Started:** {started_at}",
        f"- **Completed:** {completed_at}",
        f"- **Judge model:** {judge_model}",
        f"- **Models evaluated:** {len(profiles)}",
        f"- **Total probe evaluations:** {len(results)}",
        f"- **Errors:** {error_count}",
        f"- **Self-evaluations:** {self_eval_count}",
        "",
    ]

    # Per-model average scores, sorted descending
    lines += ["## Model Rankings (Overall Average)", "",
              "| Rank | Model | Avg Score |", "|------|-------|-----------|"]
    model_avgs: list[tuple[str, float]] = []
    for mid, profile in profiles.items():
        all_scores = (
            list(profile.task_scores.values())
            + list(profile.domain_scores.values())
            + list(profile.qs_scores.values())
        )
        avg = (
            sum(s.score for s in all_scores) / len(all_scores)
            if all_scores else 0.5
        )
        model_avgs.append((mid, avg))
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    for rank, (mid, avg) in enumerate(model_avgs, 1):
        lines.append(f"| {rank} | {mid} | {avg:.4f} |")
    lines.append("")

    # Per-dimension rankings
    if rankings:
        lines += ["## Dimension Rankings", ""]
        for dim_key, ranked_models in sorted(rankings.items()):
            lines.append(f"**{dim_key}:** {', '.join(ranked_models[:5])}")
        lines.append("")

    # Proficiencies and deficiencies
    lines += ["## Proficiencies & Deficiencies", ""]
    for mid in sorted(profiles.keys()):
        profile = profiles[mid]
        dims: list[tuple[str, float]] = []
        for k, v in profile.task_scores.items():
            dims.append((f"task/{k}", v.score))
        for k, v in profile.domain_scores.items():
            dims.append((f"domain/{k}", v.score))
        for k, v in profile.qs_scores.items():
            dims.append((f"qs/{k}", v.score))
        dims.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"**{mid}**")
        if dims:
            lines.append(
                f"- Top: {', '.join(f'{k} ({v:.2f})' for k, v in dims[:3])}")
            lines.append(
                f"- Low: {', '.join(f'{k} ({v:.2f})' for k, v in dims[-3:])}")
        lines.append("")

    # Calibration deltas
    if deltas:
        lines += [
            "## Calibration Deltas", "",
            "| Model | Dimension | Delta |",
            "|-------|-----------|-------|",
        ]
        for mid, dim_deltas in sorted(deltas.items()):
            for dk, dv in sorted(dim_deltas.items()):
                if isinstance(dv, dict):
                    delta_val = dv.get("delta", 0.0)
                    lines.append(f"| {mid} | {dk} | {delta_val:+.4f} |")
                else:
                    lines.append(f"| {mid} | {dk} | {dv:+.4f} |")
        lines.append("")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("markdown_summary_written", path=str(summary_path))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_discovery(
    models: list[str],
    judge_model: str,
    output_dir: Path,
    provider_delays: dict[str, float] | None,
    write_profiles: bool,
    resume: bool,
) -> None:
    """Main entry point for flavor discovery.

    Coordinates model evaluation, judge scoring, fingerprint computation,
    and report generation.  Calls adapters directly (not through the
    router HTTP API) to isolate intrinsic model behavior.
    """
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    started_at = datetime.now(UTC).isoformat()
    probes = get_all_probes()
    assert len(probes) > 0, "No discovery probes found"

    logger.info(
        "discovery_starting",
        run_id=run_id,
        model_count=len(models),
        probe_count=len(probes),
        judge_model=judge_model,
    )

    # --- Create adapters ---
    model_adapters: dict[str, GenerativeBackend] = {}
    for mid in models:
        adapter = _create_adapter(mid)
        if adapter is not None:
            model_adapters[mid] = adapter
        else:
            logger.warning("model_skipped_no_adapter", model_id=mid)

    judge_adapter = _create_adapter(judge_model)
    if judge_adapter is None:
        logger.error("judge_adapter_creation_failed", judge_model=judge_model)
        raise SystemExit(
            f"Cannot create adapter for judge model {judge_model} -- aborting"
        )

    reachable_models = sorted(model_adapters.keys())
    if not reachable_models:
        logger.error("no_models_available")
        raise SystemExit("No model adapters could be created -- aborting")

    logger.info(
        "adapters_created",
        reachable=len(reachable_models),
        skipped=len(models) - len(reachable_models),
    )

    # --- State and checkpoint ---
    state = RunState(run_id=run_id, started_at=started_at)
    cp = _cp_path(output_dir, run_id)
    if resume:
        state.completed_pairs = _load_checkpoint(cp)

    # --- Signal handling for graceful shutdown ---
    def _on_signal(signum: int, frame: Any) -> None:
        logger.info("signal_received", signal=signum)
        state.shutdown_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Build interleaved schedule ---
    pacer = ProviderPacer(overrides=provider_delays)
    schedule = _interleaved_schedule(reachable_models, probes)

    logger.info("schedule_built", total_pairs=len(schedule))

    # --- Main evaluation loop ---
    for idx, (model_id, probe) in enumerate(schedule):
        if state.shutdown_requested:
            logger.info("shutdown_requested", completed=len(state.results))
            break

        if (model_id, probe.id) in state.completed_pairs:
            continue

        logger.info(
            "evaluating",
            model_id=model_id,
            probe_id=probe.id,
            progress=f"{idx + 1}/{len(schedule)}",
        )

        # Pace per provider
        provider = _extract_provider(model_id)
        await pacer.wait(provider)

        # Evaluate
        result = await _evaluate_probe(
            model_id, probe, model_adapters[model_id], judge_adapter,
            judge_model_id=judge_model,
        )

        if result.error:
            state.total_errors += 1

        state.results.append(result)
        state.completed_pairs.add((model_id, probe.id))
        _append_checkpoint(cp, result)

    # --- Aggregation pipeline ---
    completed_at = datetime.now(UTC).isoformat()

    logger.info(
        "aggregation_starting",
        total_results=len(state.results),
        total_errors=state.total_errors,
    )

    # Step 6: aggregate scores -> raw fingerprints
    raw_scores = aggregate_scores(state.results)

    # Step 7: rank normalize -> ModelFlavorProfiles
    profiles = rank_normalize(raw_scores)

    # Step 8: compute calibration deltas against operator-declared profiles
    declared_path = _CONFIG_DIR / "model_flavor_profiles.yaml"
    deltas = compute_calibration_deltas(profiles, declared_path)

    # Serialize deltas for JSON report
    deltas_json: dict[str, dict[str, Any]] = {}
    for mid, dim_deltas in deltas.items():
        deltas_json[mid] = {
            dk: {"declared": dv.declared, "empirical": dv.empirical,
                 "delta": dv.delta, "recommendation": dv.recommendation}
            for dk, dv in dim_deltas.items()
        }

    # Step 9: build rankings
    rankings = build_model_rankings(profiles)

    # --- Write reports ---
    _write_json_report(
        run_id, started_at, completed_at, judge_model,
        state.results, profiles, deltas_json, rankings, output_dir,
    )
    _write_markdown_summary(
        run_id, started_at, completed_at, judge_model,
        state.results, profiles, deltas_json, rankings, output_dir,
    )

    # Write YAML fingerprints
    fingerprints_yaml = build_fingerprints_yaml(profiles, run_id)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fingerprints.yaml").write_text(fingerprints_yaml)
    logger.info("fingerprints_yaml_written", path=str(run_dir / "fingerprints.yaml"))

    # Optionally write profiles to config/
    if write_profiles:
        existing_profiles = load_existing_fingerprints(
            _CONFIG_DIR / "model_flavor_profiles.yaml",
        )
        merged = merge_incremental(existing_profiles, profiles)
        merged_yaml = build_fingerprints_yaml(merged, run_id)
        write_fingerprints_yaml(
            merged_yaml, _CONFIG_DIR / "model_flavor_profiles.yaml",
        )
        logger.info(
            "profiles_written_to_config",
            path=str(_CONFIG_DIR / "model_flavor_profiles.yaml"),
        )

    logger.info(
        "discovery_complete",
        run_id=run_id,
        models_evaluated=len(reachable_models),
        total_probes=len(state.results),
        total_errors=state.total_errors,
        partial=state.shutdown_requested,
    )


# ---------------------------------------------------------------------------
# CLI
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


def main() -> None:
    """CLI entry point: python -m dragonlight_router.discovery.runner"""
    parser = argparse.ArgumentParser(
        description="Model Flavor Discovery -- direct-adapter profiling.",
        prog="dragonlight-router-discovery",
    )
    parser.add_argument(
        "--judge-model", default=_DEFAULT_JUDGE,
        help=f"Model to use as judge (default: {_DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models", nargs="*",
        help="Subset of model IDs to evaluate (default: all from role matrix)",
    )
    parser.add_argument(
        "--provider-delay", nargs="*", metavar="K=V",
        help="Per-provider delay overrides (e.g. groq=2.0)",
    )
    parser.add_argument(
        "--write-profiles", action="store_true",
        help="Write discovered profiles to config/ after completion",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip already-completed pairs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve targets and probes, then exit without evaluating",
    )

    args = parser.parse_args()

    # Resolve model targets
    model_targets = args.models if args.models else _get_all_model_ids()
    if not model_targets:
        raise SystemExit("No model targets resolved -- check config/model_role_matrix.json")

    provider_delays = _parse_delays(args.provider_delay)

    if args.dry_run:
        probes = get_all_probes()
        logger.info(
            "dry_run_complete",
            model_count=len(model_targets),
            probe_count=len(probes),
            judge_model=args.judge_model,
            total_pairs=len(model_targets) * len(probes),
        )
        return

    asyncio.run(run_discovery(
        models=model_targets,
        judge_model=args.judge_model,
        output_dir=Path(args.output_dir),
        provider_delays=provider_delays,
        write_profiles=args.write_profiles,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
