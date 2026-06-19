"""Model Spectrography -- direct-adapter orchestrator for profiling model strengths.

Evaluates models DIRECTLY through GenerativeBackend adapters (not the router
HTTP API) to isolate intrinsic model behavior from router machinery.  Produces
flavor fingerprints, calibration deltas, and ranked summaries.

Spec reference: model-spectrography-v0.1.0-spec.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from dragonlight_router.adapters import create_adapter
from dragonlight_router.benchmark.calibration_audit import _get_all_model_ids
from dragonlight_router.benchmark.judge import (
    _JUDGE_SYSTEM_PROMPT,
    _JUDGE_USER_TEMPLATE,
    _normalize_scores,
    _parse_judge_scores,
)
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    GenerativeBackend,
    ModelFlavorProfile,
)
from dragonlight_router.spectrography.analyzer import (
    ProbeResult,
    aggregate_scores,
    build_fingerprints_yaml,
    build_model_rankings,
    compute_calibration_deltas,
    rank_normalize,
)
from dragonlight_router.spectrography.lifecycle import (
    load_existing_fingerprints,
    merge_incremental,
    write_fingerprints_yaml,
)
from dragonlight_router.spectrography.probes import SpectrographyProbe, get_all_probes

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE = "gemini/gemini-2.5-pro"
_DEFAULT_OUTPUT_DIR = "spectrography_results"
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_BACKOFF_SCHEDULE = [5.0, 10.0, 20.0, 40.0]
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    """Mutable state for a spectrography run."""

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
        "gemini": 1.0,
        "groq": 1.5,
        "nvidia_nim": 1.0,
        "openrouter": 2.0,
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

_PROVIDER_ADAPTER_KEY: dict[str, str] = {
    "nvidia_nim": "nvidia",
    "groq": "groq",
    "openrouter": "openrouter",
    "gemini": "google",
    "cerebras": "cerebras",
    "mistral": "mistral",
    "anthropic": "anthropic",
    "openai": "openai",
}


def _load_provider_configs() -> dict[str, dict[str, Any]]:
    """Load provider configs from router.yaml for adapter construction."""
    yaml_path = _CONFIG_DIR / "router.yaml"
    if not yaml_path.exists():
        logger.warning("router_yaml_missing", path=str(yaml_path))
        return {}
    try:
        raw = yaml.safe_load(yaml_path.read_text()) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("router_yaml_load_failed", error=str(exc))
        return {}
    providers = raw.get("providers", [])
    return {p["name"]: p for p in providers if isinstance(p, dict) and "name" in p}


# DEVIATION CS-MUTABLE-002: intentionally mutable — runtime cache/singleton.
_CACHED_PROVIDERS: dict[str, dict[str, Any]] | None = None


def _get_providers() -> dict[str, dict[str, Any]]:
    """Lazy-load and cache provider configs."""
    global _CACHED_PROVIDERS  # noqa: PLW0603
    if _CACHED_PROVIDERS is None:
        _CACHED_PROVIDERS = _load_provider_configs()
    return _CACHED_PROVIDERS


def _build_backend_config(
    model_id: str,
    provider_cfg: dict[str, Any],
    adapter_key: str,
) -> BackendConfig:
    """Build a BackendConfig dataclass from provider config and model ID."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
    assert isinstance(adapter_key, str) and adapter_key, "adapter_key must be non-empty"

    provider_prefix = _extract_provider(model_id)
    prefix = provider_cfg.get("model_prefix", f"{provider_prefix}/")
    bare_model = model_id[len(prefix) :] if model_id.startswith(prefix) else model_id
    env_key = provider_cfg.get("env_key")
    rl = provider_cfg.get("rate_limits", {})

    caps = BackendCapabilities(
        max_context_tokens=131072,
        supports_tool_use=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_system_prompts=True,
    )
    limits = BackendRateLimits(
        rpm=rl.get("rpm", 30),
        rpd=rl.get("rpd") or 999999,
        tpm=rl.get("tpm") or 9999999,
        daily_token_cap=rl.get("daily_token_cap") or 9999999,
    )
    config = BackendConfig(
        name=model_id,
        provider=adapter_key,
        model=bare_model,
        tier=BackendTier.MODERATE,
        base_url=provider_cfg.get("base_url", ""),
        env_key=env_key,
        capabilities=caps,
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=limits,
    )

    assert config.name == model_id, "config name must match model_id"
    return config


def _create_adapter(model_id: str) -> GenerativeBackend | None:
    """Create a fresh adapter for the given model. Returns None on failure."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"

    provider_prefix = _extract_provider(model_id)
    providers = _get_providers()
    provider_cfg = providers.get(provider_prefix)
    if provider_cfg is None:
        logger.warning("adapter_no_provider_config", model_id=model_id, provider=provider_prefix)
        return None

    adapter_key = _PROVIDER_ADAPTER_KEY.get(provider_prefix)
    if adapter_key is None:
        logger.warning("adapter_no_adapter_key", model_id=model_id, provider=provider_prefix)
        return None

    env_key = provider_cfg.get("env_key")
    if env_key and not os.environ.get(env_key):
        logger.warning("adapter_missing_env_key", model_id=model_id, env_key=env_key)
        return None

    config = _build_backend_config(model_id, provider_cfg, adapter_key)
    try:
        adapter = create_adapter(config)
        logger.info("adapter_created", model_id=model_id, adapter_key=adapter_key)
        return adapter
    except (ValueError, TypeError) as exc:
        logger.warning("adapter_creation_failed", model_id=model_id, error=str(exc))
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
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            chunks.append(chunk)
        return "".join(chunks) if chunks else None
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("adapter_generate_error", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


async def _call_model_adapter(
    model_adapter: GenerativeBackend,
    probe: SpectrographyProbe,
) -> str | None:
    """Send the probe prompt to the model and collect the response.

    Returns the full response text, or None if the model returned nothing.
    """
    assert isinstance(probe, SpectrographyProbe), "probe must be a SpectrographyProbe"
    assert hasattr(model_adapter, "generate"), "model_adapter must support generate"

    model_messages = [{"role": "user", "content": probe.prompt}]
    response = await _collect_streaming_response(
        model_adapter,
        model_messages,
        max_tokens=512,
        temperature=0.7,
    )

    assert response is None or isinstance(response, str), "response must be str or None"
    return response


async def _call_judge_adapter(
    judge_adapter: GenerativeBackend,
    probe: SpectrographyProbe,
    model_response: str,
) -> tuple[dict[str, int] | None, str | None]:
    """Send the judge prompt and parse scores from the response.

    Returns (scores_dict, error_string). On success error_string is None.
    """
    assert isinstance(model_response, str) and model_response.strip(), (
        "model_response must be non-empty"
    )
    assert isinstance(probe, SpectrographyProbe), "probe must be a SpectrographyProbe"

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

    judge_raw = await _collect_streaming_response(
        judge_adapter,
        judge_messages,
        max_tokens=256,
        temperature=0.0,
    )

    if judge_raw is None:
        return None, "judge_call_failed"

    scores = _parse_judge_scores(judge_raw)
    if scores is None:
        return None, "judge_parse_failed"

    assert isinstance(scores, dict), "parsed scores must be a dict"
    return scores, None


def _make_error_result(
    model_id: str,
    probe: SpectrographyProbe,
    error_info: tuple[bool, float, str],
) -> ProbeResult:
    """Build a ProbeResult for an error case.

    error_info is (is_self_eval, fallback_score, error_string).
    """
    assert isinstance(error_info, tuple) and len(error_info) == 3, "need 3-tuple"
    is_self_eval, score, error = error_info
    assert isinstance(error, str) and error, "error must be non-empty"
    return ProbeResult(
        model_id=model_id,
        probe_id=probe.id,
        task_type=probe.task_type,
        domain=probe.domain,
        quality_speed=probe.quality_speed,
        judge_scores=None,
        normalized_score=score,
        is_self_eval=is_self_eval,
        error=error,
    )


# DEVIATION CS-PARAM-001: _evaluate_probe takes 5 params
# -- dataclass grouping would break API.
async def _evaluate_probe(
    model_id: str,
    probe: SpectrographyProbe,
    model_adapter: GenerativeBackend,
    judge_adapter: GenerativeBackend,
    judge_model_id: str,
) -> ProbeResult:
    """Evaluate a single (model, probe) pair: call model, judge, build result."""
    assert isinstance(model_id, str), "model_id must be a string"
    assert isinstance(probe, SpectrographyProbe), "probe must be a SpectrographyProbe"

    is_self_eval = model_id == judge_model_id
    model_response = await _call_model_adapter(model_adapter, probe)

    if not model_response or not model_response.strip():
        logger.warning("empty_model_response", model_id=model_id, probe_id=probe.id)
        return _make_error_result(model_id, probe, (is_self_eval, 0.0, "empty_model_response"))

    scores, error = await _call_judge_adapter(judge_adapter, probe, model_response)
    if error is not None:
        logger.warning(error, model_id=model_id, probe_id=probe.id)
        return _make_error_result(model_id, probe, (is_self_eval, 0.5, error))

    normalized = _normalize_scores(scores)
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
    models: list[str],
    probes: list[SpectrographyProbe],
) -> list[tuple[str, SpectrographyProbe]]:
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
    line = json.dumps(
        {
            "model_id": result.model_id,
            "probe_id": result.probe_id,
            "task_type": result.task_type,
            "domain": result.domain,
            "quality_speed": result.quality_speed,
            "judge_scores": result.judge_scores,
            "normalized_score": result.normalized_score,
            "is_self_eval": result.is_self_eval,
            "error": result.error,
        }
    )
    with open(path, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@dataclass
class ReportContext:
    """Bundles all data needed for report generation (keeps param count <= 4)."""

    run_id: str
    started_at: str
    completed_at: str
    judge_model: str
    results: list[ProbeResult]
    profiles: dict[str, ModelFlavorProfile]
    deltas: dict[str, dict[str, Any]]
    rankings: dict[str, list[str]]
    output_dir: Path


def _serialize_profiles(
    profiles: dict[str, ModelFlavorProfile],
) -> dict[str, dict[str, Any]]:
    """Convert ModelFlavorProfile map to JSON-safe dicts."""
    assert isinstance(profiles, dict), "profiles must be a dict"

    profiles_json: dict[str, dict[str, Any]] = {}
    for mid, profile in profiles.items():
        profiles_json[mid] = {
            "model_id": profile.model_id,
            "version": profile.version,
            "updated_at": profile.updated_at,
            "task_scores": {
                k: {"score": v.score, "confidence": v.confidence, "sample_count": v.sample_count}
                for k, v in profile.task_scores.items()
            },
            "domain_scores": {
                k: {"score": v.score, "confidence": v.confidence, "sample_count": v.sample_count}
                for k, v in profile.domain_scores.items()
            },
            "qs_scores": {
                k: {"score": v.score, "confidence": v.confidence, "sample_count": v.sample_count}
                for k, v in profile.qs_scores.items()
            },
        }

    assert len(profiles_json) == len(profiles), "all profiles must be serialized"
    return profiles_json


def _build_json_report_data(ctx: ReportContext) -> dict[str, Any]:
    """Build the full JSON report dict from a ReportContext."""
    assert isinstance(ctx, ReportContext), "ctx must be a ReportContext"

    profiles_json = _serialize_profiles(ctx.profiles)
    report: dict[str, Any] = {
        "run_id": ctx.run_id,
        "started_at": ctx.started_at,
        "completed_at": ctx.completed_at,
        "judge_model": ctx.judge_model,
        "models_evaluated": sorted(profiles_json.keys()),
        "total_probes": len(ctx.results),
        "total_errors": sum(1 for r in ctx.results if r.error),
        "self_eval_count": sum(1 for r in ctx.results if r.is_self_eval),
        "profiles": profiles_json,
        "calibration_deltas": ctx.deltas,
        "rankings": ctx.rankings,
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
            for r in ctx.results
        ],
    }

    assert "run_id" in report, "report must contain run_id"
    return report


def _write_json_report(ctx: ReportContext) -> None:
    """Serialize and write the full JSON report to disk."""
    assert isinstance(ctx, ReportContext), "ctx must be a ReportContext"

    run_dir = ctx.output_dir / ctx.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = _build_json_report_data(ctx)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    assert report_path.exists(), "report file must exist after write"
    logger.info("json_report_written", path=str(report_path))


def _md_header_section(ctx: ReportContext) -> list[str]:
    """Build the markdown header/metadata section."""
    assert isinstance(ctx, ReportContext), "ctx must be a ReportContext"

    self_eval_count = sum(1 for r in ctx.results if r.is_self_eval)
    error_count = sum(1 for r in ctx.results if r.error)

    lines = [
        "# Model Spectrography Report",
        "",
        f"- **Run ID:** {ctx.run_id}",
        f"- **Started:** {ctx.started_at}",
        f"- **Completed:** {ctx.completed_at}",
        f"- **Judge model:** {ctx.judge_model}",
        f"- **Models evaluated:** {len(ctx.profiles)}",
        f"- **Total probe evaluations:** {len(ctx.results)}",
        f"- **Errors:** {error_count}",
        f"- **Self-evaluations:** {self_eval_count}",
        "",
    ]

    assert len(lines) > 0, "header section must produce lines"
    return lines


def _md_rankings_section(
    profiles: dict[str, ModelFlavorProfile],
    rankings: dict[str, list[str]],
) -> list[str]:
    """Build the overall rankings and dimension rankings markdown sections."""
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert isinstance(rankings, dict), "rankings must be a dict"

    lines = [
        "## Model Rankings (Overall Average)",
        "",
        "| Rank | Model | Avg Score |",
        "|------|-------|-----------|",
    ]
    model_avgs: list[tuple[str, float]] = []
    for mid, profile in profiles.items():
        all_scores = (
            list(profile.task_scores.values())
            + list(profile.domain_scores.values())
            + list(profile.qs_scores.values())
        )
        avg = sum(s.score for s in all_scores) / len(all_scores) if all_scores else 0.5
        model_avgs.append((mid, avg))
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    for rank, (mid, avg) in enumerate(model_avgs, 1):
        lines.append(f"| {rank} | {mid} | {avg:.4f} |")
    lines.append("")

    if rankings:
        lines += ["## Dimension Rankings", ""]
        for dim_key, ranked_models in sorted(rankings.items()):
            lines.append(f"**{dim_key}:** {', '.join(ranked_models[:5])}")
        lines.append("")

    return lines


def _md_proficiencies_section(
    profiles: dict[str, ModelFlavorProfile],
) -> list[str]:
    """Build the proficiencies and deficiencies markdown section."""
    assert isinstance(profiles, dict), "profiles must be a dict"

    lines: list[str] = ["## Proficiencies & Deficiencies", ""]
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
            lines.append(f"- Top: {', '.join(f'{k} ({v:.2f})' for k, v in dims[:3])}")
            lines.append(f"- Low: {', '.join(f'{k} ({v:.2f})' for k, v in dims[-3:])}")
        lines.append("")

    assert len(lines) >= 2, "proficiencies section must have content"
    return lines


def _md_calibration_section(
    deltas: dict[str, dict[str, Any]],
) -> list[str]:
    """Build the calibration deltas markdown section."""
    assert isinstance(deltas, dict), "deltas must be a dict"

    if not deltas:
        return []

    lines = [
        "## Calibration Deltas",
        "",
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

    assert len(lines) > 4, "calibration section must have table rows"
    return lines


def _write_markdown_summary(ctx: ReportContext) -> None:
    """Assemble and write the human-readable markdown summary."""
    assert isinstance(ctx, ReportContext), "ctx must be a ReportContext"

    run_dir = ctx.output_dir / ctx.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = _md_header_section(ctx)
    lines += _md_rankings_section(ctx.profiles, ctx.rankings)
    lines += _md_proficiencies_section(ctx.profiles)
    lines += _md_calibration_section(ctx.deltas)

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines))

    assert summary_path.exists(), "summary file must exist after write"
    logger.info("markdown_summary_written", path=str(summary_path))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _SpectrographySetup:
    """Return bundle from _setup_spectrography (keeps param count <= 4)."""

    state: RunState
    model_adapters: dict[str, GenerativeBackend]
    judge_adapter: GenerativeBackend
    schedule: list[tuple[str, SpectrographyProbe]]
    pacer: ProviderPacer
    checkpoint_path: Path


@dataclass
class _SpectrographyConfig:
    """Input bundle for run_spectrography (keeps param count <= 4)."""

    models: list[str]
    judge_model: str
    output_dir: Path
    provider_delays: dict[str, float] | None
    write_profiles: bool
    resume: bool


def _create_all_adapters(
    models: list[str],
    judge_model: str,
) -> tuple[dict[str, GenerativeBackend], GenerativeBackend]:
    """Create model + judge adapters. Raises SystemExit on fatal failures."""
    assert isinstance(models, list) and models, "models must be non-empty"
    assert isinstance(judge_model, str) and judge_model, "judge_model must be non-empty"

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
        raise SystemExit(f"Cannot create adapter for judge model {judge_model} -- aborting")

    if not model_adapters:
        logger.error("no_models_available")
        raise SystemExit("No model adapters could be created -- aborting")

    logger.info(
        "adapters_created",
        reachable=len(model_adapters),
        skipped=len(models) - len(model_adapters),
    )
    return model_adapters, judge_adapter


def _setup_spectrography(cfg: _SpectrographyConfig) -> _SpectrographySetup:
    """Create adapters, state, schedule, and signal handlers for a run."""
    assert isinstance(cfg, _SpectrographyConfig), "cfg must be a _SpectrographyConfig"

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    started_at = datetime.now(UTC).isoformat()
    probes = get_all_probes()
    assert len(probes) > 0, "No spectrography probes found"

    logger.info(
        "spectrography_starting",
        run_id=run_id,
        model_count=len(cfg.models),
        probe_count=len(probes),
        judge_model=cfg.judge_model,
    )

    model_adapters, judge_adapter = _create_all_adapters(cfg.models, cfg.judge_model)
    reachable_models = sorted(model_adapters.keys())

    state = RunState(run_id=run_id, started_at=started_at)
    cp = _cp_path(cfg.output_dir, run_id)
    if cfg.resume:
        state.completed_pairs = _load_checkpoint(cp)

    def _on_signal(signum: int, frame: Any) -> None:
        logger.info("signal_received", signal=signum)
        state.shutdown_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pacer = ProviderPacer(overrides=cfg.provider_delays)
    schedule = _interleaved_schedule(reachable_models, probes)
    logger.info("schedule_built", total_pairs=len(schedule))

    assert len(schedule) > 0, "schedule must not be empty"
    return _SpectrographySetup(
        state=state,
        model_adapters=model_adapters,
        judge_adapter=judge_adapter,
        schedule=schedule,
        pacer=pacer,
        checkpoint_path=cp,
    )


async def _run_probe_loop(
    setup: _SpectrographySetup,
    judge_model: str,
) -> None:
    """Execute the main evaluation loop over the interleaved schedule."""
    assert isinstance(setup, _SpectrographySetup), "setup must be a _SpectrographySetup"
    assert isinstance(judge_model, str), "judge_model must be a string"

    state = setup.state
    for idx, (model_id, probe) in enumerate(setup.schedule):
        if state.shutdown_requested:
            logger.info("shutdown_requested", completed=len(state.results))
            break

        if (model_id, probe.id) in state.completed_pairs:
            continue

        logger.info(
            "evaluating",
            model_id=model_id,
            probe_id=probe.id,
            progress=f"{idx + 1}/{len(setup.schedule)}",
        )

        provider = _extract_provider(model_id)
        await setup.pacer.wait(provider)

        result = await _evaluate_probe(
            model_id,
            probe,
            setup.model_adapters[model_id],
            setup.judge_adapter,
            judge_model_id=judge_model,
        )

        if result.error:
            state.total_errors += 1

        state.results.append(result)
        state.completed_pairs.add((model_id, probe.id))
        _append_checkpoint(setup.checkpoint_path, result)


def _serialize_deltas(
    deltas: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Serialize CalibrationDelta objects to JSON-safe dicts."""
    assert isinstance(deltas, dict), "deltas must be a dict"

    result: dict[str, dict[str, Any]] = {}
    for mid, dim_deltas in deltas.items():
        result[mid] = {
            dk: {
                "declared": dv.declared,
                "empirical": dv.empirical,
                "delta": dv.delta,
                "recommendation": dv.recommendation,
            }
            for dk, dv in dim_deltas.items()
        }

    assert len(result) == len(deltas), "all deltas must be serialized"
    return result


def _write_config_profiles(
    profiles: dict[str, ModelFlavorProfile],
    run_id: str,
) -> None:
    """Merge and write discovered profiles to the config directory."""
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert isinstance(run_id, str) and run_id, "run_id must be non-empty"

    existing_profiles = load_existing_fingerprints(
        _CONFIG_DIR / "model_flavor_profiles.yaml",
    )
    merged = merge_incremental(existing_profiles, profiles)
    merged_yaml = build_fingerprints_yaml(merged, run_id)
    write_fingerprints_yaml(merged_yaml, _CONFIG_DIR / "model_flavor_profiles.yaml")
    logger.info(
        "profiles_written_to_config",
        path=str(_CONFIG_DIR / "model_flavor_profiles.yaml"),
    )


def _generate_spectrography_reports(
    setup: _SpectrographySetup,
    cfg: _SpectrographyConfig,
) -> None:
    """Aggregate results, compute deltas, and write all reports."""
    assert isinstance(setup, _SpectrographySetup), "setup must be a _SpectrographySetup"
    assert isinstance(cfg, _SpectrographyConfig), "cfg must be a _SpectrographyConfig"

    state = setup.state
    completed_at = datetime.now(UTC).isoformat()
    logger.info(
        "aggregation_starting", total_results=len(state.results), total_errors=state.total_errors
    )

    raw_scores = aggregate_scores(state.results)
    profiles = rank_normalize(raw_scores)
    deltas = compute_calibration_deltas(profiles, _CONFIG_DIR / "model_flavor_profiles.yaml")
    deltas_json = _serialize_deltas(deltas)
    rankings = build_model_rankings(profiles)

    ctx = ReportContext(
        run_id=state.run_id,
        started_at=state.started_at,
        completed_at=completed_at,
        judge_model=cfg.judge_model,
        results=state.results,
        profiles=profiles,
        deltas=deltas_json,
        rankings=rankings,
        output_dir=cfg.output_dir,
    )
    _write_json_report(ctx)
    _write_markdown_summary(ctx)

    run_dir = cfg.output_dir / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fp_yaml = build_fingerprints_yaml(profiles, state.run_id)
    (run_dir / "fingerprints.yaml").write_text(fp_yaml)
    logger.info("fingerprints_yaml_written", path=str(run_dir / "fingerprints.yaml"))

    if cfg.write_profiles:
        _write_config_profiles(profiles, state.run_id)

    logger.info(
        "spectrography_complete",
        run_id=state.run_id,
        models_evaluated=len(setup.model_adapters),
        total_probes=len(state.results),
        total_errors=state.total_errors,
        partial=state.shutdown_requested,
    )


# DEVIATION CS-PARAM-001: run_spectrography takes 6 params —
# grouping into dataclass would break public API.
async def run_spectrography(
    models: list[str],
    judge_model: str,
    output_dir: Path,
    provider_delays: dict[str, float] | None,
    write_profiles: bool,
    resume: bool,
) -> None:
    """Main entry point for model spectrography.

    Orchestrates: setup -> probe loop -> report generation.
    """
    cfg = _SpectrographyConfig(
        models=models,
        judge_model=judge_model,
        output_dir=output_dir,
        provider_delays=provider_delays,
        write_profiles=write_profiles,
        resume=resume,
    )
    setup = _setup_spectrography(cfg)
    await _run_probe_loop(setup, judge_model)
    _generate_spectrography_reports(setup, cfg)


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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    assert argparse is not None, "argparse module must be available"

    parser = argparse.ArgumentParser(
        description="Model Spectrography -- direct-adapter profiling.",
        prog="dragonlight-router-spectrography",
    )
    parser.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE,
        help=f"Model to use as judge (default: {_DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of model IDs to evaluate (default: all from role matrix)",
    )
    parser.add_argument(
        "--provider-delay",
        nargs="*",
        metavar="K=V",
        help="Per-provider delay overrides (e.g. groq=2.0)",
    )
    parser.add_argument(
        "--write-profiles",
        action="store_true",
        help="Write discovered profiles to config/ after completion",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-completed pairs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve targets and probes, then exit without evaluating",
    )

    assert isinstance(parser, argparse.ArgumentParser), "must return ArgumentParser"
    return parser


def main() -> None:
    """CLI entry point: python -m dragonlight_router.spectrography.runner"""
    args = _build_arg_parser().parse_args()

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

    asyncio.run(
        run_spectrography(
            models=model_targets,
            judge_model=args.judge_model,
            output_dir=Path(args.output_dir),
            provider_delays=provider_delays,
            write_profiles=args.write_profiles,
            resume=args.resume,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
