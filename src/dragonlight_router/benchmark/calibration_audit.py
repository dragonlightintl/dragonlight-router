"""Calibration audit -- HTTP client that calibrates models through the router's own API.

Sends pinned dispatch requests to POST /v1/dispatch for each (model, prompt) pair,
judges responses via the same router, and produces calibration reports comparing
empirical scores against operator-declared flavor profiles.

Spec reference: calibration-audit-v0.1.0-spec.md
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import signal
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

from dragonlight_router.benchmark.judge import (
    _JUDGE_SYSTEM_PROMPT,
    _JUDGE_USER_TEMPLATE,
    _normalize_scores,
    _parse_judge_scores,
)
from dragonlight_router.benchmark.prompts import EvalPrompt, get_all_prompts

logger = structlog.get_logger()

# Constants
_DEFAULT_ROUTER_URL = "http://localhost:8000"
_DEFAULT_JUDGE = "gemini/gemini-2.5-pro"
_FALLBACK_JUDGE = "nvidia_nim/qwen/qwen3.5-397b-a17b"
_DEFAULT_OUTPUT_DIR = "benchmark_results"
_CTX_TOKENS = 100
_DEFAULT_PROVIDER_DELAYS: types.MappingProxyType[str, float] = types.MappingProxyType(
    {
        "gemini": 1.0,
        "groq": 1.5,
        "nvidia_nim": 1.0,
        "openrouter": 2.0,
    }
)
_BACKOFF_SCHEDULE = [5.0, 10.0, 20.0, 40.0]
_MAX_RETRIES = 3
_DELTA_THRESHOLD = 0.15
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


# --- Result types ---


@dataclass(frozen=True)
class PromptResult:
    """Result of a single (model, prompt) evaluation."""

    model: str
    prompt_id: str
    http_status: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    content: str
    judge_scores: dict[str, int] | None
    normalized_score: float
    error: str | None


@dataclass
class RunState:
    """Mutable state for a benchmark run."""

    run_id: str
    started_at: str
    results: list[PromptResult] = field(default_factory=list)
    completed_pairs: set[tuple[str, str]] = field(default_factory=set)
    rate_limit_hits: int = 0
    budget_exhaustions: int = 0
    circuit_breaker_trips: int = 0
    total_errors: int = 0
    shutdown_requested: bool = False


def _extract_provider(model_id: str) -> str:
    """Extract provider prefix from model_id (e.g. 'gemini/...' -> 'gemini')."""
    return model_id.split("/")[0] if "/" in model_id else "unknown"


# --- HTTP helpers ---


# DEVIATION CS-PARAM-001: 6 params — dataclass would break API.
async def _dispatch_pinned(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt_text: str,
    run_id: str,
    prompt_id: str,
) -> dict[str, Any]:
    """POST /v1/dispatch with model pinned."""
    body: dict[str, Any] = {
        "model": model,
        "operator_message": prompt_text,
        "context_tokens": _CTX_TOKENS,
    }
    if prompt_id:
        body["metadata"] = {"benchmark_run_id": run_id, "prompt_id": prompt_id}
    resp = await client.post(f"{url}/v1/dispatch", json=body, timeout=120.0)
    return {"status": resp.status_code, "body": resp.json()}


async def _check_health(client: httpx.AsyncClient, url: str) -> bool:
    """GET /v1/health -- verify router is running."""
    try:
        return (await client.get(f"{url}/v1/health", timeout=10.0)).status_code == 200
    except httpx.HTTPError:
        return False


async def _check_model_reachable(
    client: httpx.AsyncClient,
    url: str,
    model: str,
) -> bool:
    """Send a trivial pinned dispatch to verify model reachability."""
    try:
        r = await _dispatch_pinned(client, url, model, "Respond with OK", "", "")
        if r["status"] == 200:
            return True
        logger.warning("model_unreachable", model=model, status=r["status"])
        return False
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning("model_unreachable", model=model, error=str(exc))
        return False


# --- Pre-flight (spec 3.1) ---


async def run_preflight(
    client: httpx.AsyncClient,
    url: str,
    requested: list[str],
    judge_model: str,
) -> tuple[list[str], str]:
    """Run pre-flight checks. Returns (reachable_models, resolved_judge)."""
    if not await _check_health(client, url):
        raise SystemExit(f"Router unreachable at {url}")
    logger.info("preflight_health_ok", router_url=url)

    reachable = [m for m in requested if await _check_model_reachable(client, url, m)]
    for m in requested:
        if m not in reachable:
            logger.warning("preflight_model_excluded", model=m)
    if not reachable:
        raise SystemExit("No models reachable -- cannot run benchmark")

    judge = judge_model
    if judge not in reachable and not await _check_model_reachable(client, url, judge):
        logger.warning("judge_unreachable", judge=judge)
        judge = _FALLBACK_JUDGE
        if not await _check_model_reachable(client, url, judge):
            raise SystemExit("No judge model reachable")

    logger.info("preflight_complete", reachable=len(reachable), judge=judge)
    return reachable, judge


# --- Pacing (spec section 4) ---


class ProviderPacer:
    """Track per-provider last-request timestamps and enforce delays."""

    def __init__(self, delays: dict[str, float]) -> None:
        self._delays = delays
        self._last: dict[str, float] = {}

    async def wait(self, provider: str) -> None:
        """Sleep until minimum inter-request delay has elapsed."""
        delay = self._delays.get(provider, 1.0)
        elapsed = time.monotonic() - self._last.get(provider, 0.0)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last[provider] = time.monotonic()


def _interleaved_schedule(
    models: list[str],
    prompts: list[EvalPrompt],
) -> list[tuple[str, EvalPrompt]]:
    """Build provider-interleaved (model, prompt) schedule."""
    return [(m, p) for p in prompts for m in models]


# --- Retry with backoff (spec 4.3) ---


# DEVIATION CS-PARAM-001: 8 params — dataclass would break API.
async def _dispatch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    text: str,
    run_id: str,
    prompt_id: str,
    pacer: ProviderPacer,
    state: RunState,
) -> dict[str, Any]:
    """Dispatch with 429 backoff retry."""
    provider = _extract_provider(model)
    last: dict[str, Any] = {"status": 0, "body": {"error": "no_attempts"}}
    for attempt in range(_MAX_RETRIES + 1):
        await pacer.wait(provider)
        last = await _dispatch_pinned(client, url, model, text, run_id, prompt_id)
        if last["status"] != 429:
            return last
        state.rate_limit_hits += 1
        if attempt >= _MAX_RETRIES:
            logger.warning("retry_exhausted", model=model, prompt_id=prompt_id)
            return last
        backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
        wait = float(last["body"].get("retry_after", backoff))
        await asyncio.sleep(min(wait, 60.0))
    return last  # pragma: no cover — loop always returns via early exit or retry exhaustion


# --- Model evaluation (spec 3.2) ---


# DEVIATION CS-PARAM-001: 7 params — dataclass would break API.
async def evaluate_prompt(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: EvalPrompt,
    run_id: str,
    pacer: ProviderPacer,
    state: RunState,
) -> PromptResult | None:
    """Evaluate a single (model, prompt) pair."""
    try:
        resp = await _dispatch_with_retry(
            client,
            url,
            model,
            prompt.prompt,
            run_id,
            prompt.id,
            pacer,
            state,
        )
    except httpx.HTTPError as exc:
        state.total_errors += 1
        return PromptResult(
            model=model,
            prompt_id=prompt.id,
            http_status=0,
            latency_ms=0.0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            content="",
            judge_scores=None,
            normalized_score=0.0,
            error=str(exc),
        )
    status = resp["status"]
    if status == 429:
        state.budget_exhaustions += 1
    if status in (502, 503):
        state.circuit_breaker_trips += 1
    if status >= 400:
        state.total_errors += 1
        return PromptResult(
            model=model,
            prompt_id=prompt.id,
            http_status=status,
            latency_ms=0.0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            content="",
            judge_scores=None,
            normalized_score=0.0,
            error=resp["body"].get("error", "dispatch_failed"),
        )
    body = resp["body"]
    return PromptResult(
        model=model,
        prompt_id=prompt.id,
        http_status=status,
        latency_ms=body.get("latency_ms", 0.0),
        tokens_in=body.get("tokens_in", 0),
        tokens_out=body.get("tokens_out", 0),
        cost_usd=body.get("estimated_cost_usd", 0.0),
        content=body.get("content", ""),
        judge_scores=None,
        normalized_score=0.0,
        error=None,
    )


# --- Judge evaluation (spec 3.3) ---


# DEVIATION CS-PARAM-001: 8 params — dataclass would break API.
async def judge_single(
    client: httpx.AsyncClient,
    url: str,
    judge_model: str,
    prompt: EvalPrompt,
    model_response: str,
    run_id: str,
    pacer: ProviderPacer,
    state: RunState,
) -> tuple[dict[str, int] | None, float]:
    """Judge a single response. Returns (raw_scores, normalized_score)."""
    judge_text = f"[SYSTEM] {_JUDGE_SYSTEM_PROMPT}\n\n" + _JUDGE_USER_TEMPLATE.format(
        original_prompt=prompt.prompt,
        judge_criteria=prompt.judge_criteria,
        quality_speed=prompt.quality_speed,
        model_response=model_response,
    )
    try:
        resp = await _dispatch_with_retry(
            client,
            url,
            judge_model,
            judge_text,
            run_id,
            f"judge-{prompt.id}",
            pacer,
            state,
        )
    except httpx.HTTPError as exc:
        logger.warning("judge_dispatch_failed", prompt_id=prompt.id, error=str(exc))
        return None, 0.5
    if resp["status"] != 200:
        logger.warning("judge_dispatch_error", prompt_id=prompt.id, status=resp["status"])
        return None, 0.5
    scores = _parse_judge_scores(resp["body"].get("content", ""))
    if scores is None:
        logger.warning("judge_parse_failed", prompt_id=prompt.id)
        return None, 0.5
    return scores, _normalize_scores(scores)


# --- Checkpoint (spec 6.1) ---


def _cp_path(output_dir: Path, run_id: str) -> Path:
    """Checkpoint file path for a run."""
    return output_dir / run_id / "checkpoint.jsonl"


def _append_checkpoint(path: Path, r: PromptResult) -> None:
    """Append a single result to the checkpoint JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "model": r.model,
            "prompt_id": r.prompt_id,
            "http_status": r.http_status,
            "latency_ms": r.latency_ms,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "cost_usd": r.cost_usd,
            "judge_scores": r.judge_scores,
            "normalized_score": r.normalized_score,
            "error": r.error,
        }
    )
    with open(path, "a") as f:
        f.write(line + "\n")


def _load_checkpoint(path: Path) -> tuple[list[PromptResult], set[tuple[str, str]]]:
    """Load checkpoint file. Returns (results, completed_pairs)."""
    if not path.exists():
        return [], set()
    results: list[PromptResult] = []
    completed: set[tuple[str, str]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        results.append(
            PromptResult(
                model=d["model"],
                prompt_id=d["prompt_id"],
                http_status=d["http_status"],
                latency_ms=d["latency_ms"],
                tokens_in=d["tokens_in"],
                tokens_out=d["tokens_out"],
                cost_usd=d["cost_usd"],
                content="",
                judge_scores=d.get("judge_scores"),
                normalized_score=d["normalized_score"],
                error=d.get("error"),
            )
        )
        completed.add((d["model"], d["prompt_id"]))
    logger.info("checkpoint_loaded", count=len(results))
    return results, completed


# --- Aggregation (spec 3.4) ---


def _aggregate_model_scores(
    results: list[PromptResult],
    prompts_by_id: dict[str, EvalPrompt],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-model scores into SpectrographScore-format profiles."""
    from dragonlight_router.core.types import IBR_DOMAINS, IBR_QUALITY_SPEED, IBR_TASK_TYPES

    by_model: dict[str, list[PromptResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    profiles: dict[str, dict[str, Any]] = {}
    for mid, mrs in by_model.items():
        ta: dict[str, list[float]] = {t: [] for t in IBR_TASK_TYPES}
        da: dict[str, list[float]] = {d: [] for d in IBR_DOMAINS}
        qa: dict[str, list[float]] = {q: [] for q in IBR_QUALITY_SPEED}
        for r in mrs:
            p = prompts_by_id.get(r.prompt_id)
            if p is None or (r.normalized_score == 0.0 and r.error):
                continue
            ta[p.task_type].append(r.normalized_score)
            da[p.domain].append(r.normalized_score)
            qa[p.quality_speed].append(r.normalized_score)
        profiles[mid] = {
            "task_scores": _flavor_dict(ta),
            "domain_scores": _flavor_dict(da),
            "qs_scores": _flavor_dict(qa),
        }
    return profiles


def _flavor_dict(accum: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """Build SpectrographScore-format dict from accumulated score lists."""
    out: dict[str, dict[str, Any]] = {}
    for k, vs in accum.items():
        if vs:
            out[k] = {
                "score": round(sum(vs) / len(vs), 4),
                "confidence": round(min(1.0, len(vs) / 50.0), 4),
                "sample_count": len(vs),
            }
        else:
            out[k] = {"score": 0.5, "confidence": 0.0, "sample_count": 0}
    return out


# --- Calibration delta (spec 3.4, 5.1) ---


def _load_declared_profiles(cfg: Path) -> dict[str, dict[str, Any]]:
    """Load operator-declared profiles from model_spectrograph_profiles.yaml."""
    path = cfg / "model_spectrograph_profiles.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "profiles" not in data:
        return {}
    raw_profiles: dict[str, dict[str, Any]] = data["profiles"]
    return raw_profiles


def _score_val(v: Any) -> float:
    """Extract float score from dict or scalar."""
    if isinstance(v, dict):
        return float(v.get("score", 0.5))
    return float(v) if isinstance(v, (int, float)) else 0.5


def _calibration_deltas(
    empirical: dict[str, dict[str, Any]],
    declared: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare empirical vs declared profiles, flag deltas > threshold."""
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for mid, emp in empirical.items():
        decl = declared.get(mid, {})
        if not decl:
            continue
        md: dict[str, dict[str, float]] = {}
        for dim in ("task_scores", "domain_scores", "qs_scores"):
            ed, dd = emp.get(dim, {}), decl.get(dim, {})
            for k in set(ed) | set(dd):
                ev, dv = _score_val(ed.get(k)), _score_val(dd.get(k))
                d = ev - dv
                if abs(d) > _DELTA_THRESHOLD:
                    md[f"{dim}/{k}"] = {
                        "declared": round(dv, 4),
                        "measured": round(ev, 4),
                        "delta": round(d, 4),
                    }
        if md:
            deltas[mid] = md
    return deltas


# --- Report generation (spec section 5) ---


# DEVIATION CS-PARAM-001: 8 params — dataclass would break API.
def _json_report(
    state: RunState,
    models: list[str],
    judge: str,
    prompts: list[EvalPrompt],
    profiles: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, dict[str, float]]],
    done_at: str,
    partial: bool = False,
) -> dict[str, Any]:
    """Build the JSON report (spec 5.1)."""
    return {
        "run_id": state.run_id,
        "started_at": state.started_at,
        "completed_at": done_at,
        "partial": partial,
        "judge_model": judge,
        "models_benchmarked": models,
        "prompts_per_model": len(prompts),
        "total_dispatch_calls": len(state.results),
        "total_cost_usd": round(sum(r.cost_usd for r in state.results), 6),
        "profiles": profiles,
        "calibration_deltas": deltas,
        "per_prompt_results": [
            {
                "model": r.model,
                "prompt_id": r.prompt_id,
                "latency_ms": r.latency_ms,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "judge_scores": r.judge_scores,
                "normalized_score": r.normalized_score,
                "http_status": r.http_status,
                "error": r.error,
            }
            for r in state.results
        ],
        "router_stats": {
            "rate_limit_hits": state.rate_limit_hits,
            "budget_exhaustions": state.budget_exhaustions,
            "circuit_breaker_trips": state.circuit_breaker_trips,
            "total_errors": state.total_errors,
        },
    }


def _md_results_table(profiles: dict[str, dict[str, Any]]) -> list[str]:
    """Build per-model score table and proficiency/deficiency lines.

    Returns markdown lines for the results and proficiency sections.
    """
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert all(isinstance(v, dict) for v in profiles.values()), "profile values must be dicts"
    lines: list[str] = [
        "## Per-Model Scores",
        "",
        "| Model | Avg Score |",
        "|-------|-----------|",
    ]
    rows: list[tuple[str, float]] = []
    for mid, prof in profiles.items():
        scores = [
            (e.get("score", 0.5) if isinstance(e, dict) else 0.5)
            for dim in ("task_scores", "domain_scores", "qs_scores")
            for e in prof.get(dim, {}).values()
        ]
        rows.append((mid, sum(scores) / len(scores) if scores else 0.5))
    for mid, avg in sorted(rows, key=lambda x: x[1], reverse=True):
        lines.append(f"| {mid} | {avg:.4f} |")
    lines += ["", "## Proficiencies & Deficiencies", ""]
    for mid, prof in sorted(profiles.items()):
        dims = [
            (f"{d}/{k}", (e.get("score", 0.5) if isinstance(e, dict) else 0.5))
            for d in ("task_scores", "domain_scores", "qs_scores")
            for k, e in prof.get(d, {}).items()
        ]
        dims.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"**{mid}**")
        lines.append(f"- Top: {', '.join(f'{k} ({v:.2f})' for k, v in dims[:3])}")
        lines.append(f"- Low: {', '.join(f'{k} ({v:.2f})' for k, v in dims[-3:])}")
        lines.append("")
    return lines


def _md_calibration_section(
    deltas: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    """Build the calibration-delta markdown table.

    Returns markdown lines for the calibration section, or empty list.
    """
    assert isinstance(deltas, dict), "deltas must be a dict"
    assert all(isinstance(v, dict) for v in deltas.values()), "delta values must be dicts"
    if not deltas:
        return []
    lines: list[str] = [
        "## Calibration Deltas (|delta| > 0.15)",
        "",
        "| Model | Dimension | Declared | Measured | Delta |",
        "|-------|-----------|----------|----------|-------|",
    ]
    for mid, dims in sorted(deltas.items()):
        for dk, v in sorted(dims.items()):
            lines.append(
                f"| {mid} | {dk} | {v['declared']:.4f} | {v['measured']:.4f} | {v['delta']:+.4f} |"
            )
    lines.append("")
    return lines


def _md_summary(report: dict[str, Any]) -> str:
    """Build the markdown summary (spec 5.2)."""
    tag = " (PARTIAL)" if report.get("partial") else ""
    lines = [
        f"# Calibration Audit Report{tag}",
        "",
        f"- **Run ID:** {report['run_id']}",
        f"- **Started:** {report['started_at']}",
        f"- **Completed:** {report['completed_at']}",
        f"- **Judge model:** {report['judge_model']}",
        f"- **Total dispatch calls:** {report['total_dispatch_calls']}",
        f"- **Total cost:** ${report['total_cost_usd']:.4f}",
        "",
    ]
    lines += _md_results_table(report["profiles"])
    lines += _md_calibration_section(report["calibration_deltas"])
    s = report["router_stats"]
    lines += [
        "## Router Operational Stats",
        "",
        f"- Rate limit hits (429): {s['rate_limit_hits']}",
        f"- Budget exhaustions: {s['budget_exhaustions']}",
        f"- Circuit breaker trips: {s['circuit_breaker_trips']}",
        f"- Total errors: {s['total_errors']}",
        "",
    ]
    return "\n".join(lines)


# DEVIATION CS-PARAM-001: 7 params — dataclass would break API.
def _write_reports(
    out: Path,
    run_id: str,
    state: RunState,
    models: list[str],
    judge: str,
    prompts: list[EvalPrompt],
    partial: bool = False,
) -> None:
    """Aggregate results, compute deltas, write JSON + markdown reports."""
    pbi = {p.id: p for p in prompts}
    emp = _aggregate_model_scores(state.results, pbi)
    decl = _load_declared_profiles(_CONFIG_DIR)
    deltas = _calibration_deltas(emp, decl)
    done = datetime.now(UTC).isoformat()
    report = _json_report(state, models, judge, prompts, emp, deltas, done, partial)
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "summary.md").write_text(_md_summary(report))
    logger.info("reports_written", path=str(run_dir))


# --- Main benchmark loop ---


@dataclass
class _CalibrationSetup:
    """Intermediate state produced by _setup_calibration."""

    run_id: str
    prompts: list[EvalPrompt]
    state: RunState
    cp: Path
    pacer: ProviderPacer


def _setup_calibration(
    output_dir: Path,
    resume: bool,
    model_filter: list[str] | None,
    provider_delays: dict[str, float],
) -> _CalibrationSetup:
    """Build initial run state, checkpoint path, pacer, and signal handlers.

    Returns a _CalibrationSetup with everything the main loop needs.
    """
    assert isinstance(output_dir, Path), "output_dir must be a Path"
    assert isinstance(provider_delays, dict), "provider_delays must be a dict"
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    prompts = get_all_prompts()
    state = RunState(run_id=run_id, started_at=datetime.now(UTC).isoformat())
    cp = _cp_path(output_dir, run_id)
    if resume:
        state.results, state.completed_pairs = _load_checkpoint(cp)

    def _on_signal(signum: int, frame: Any) -> None:
        logger.info("signal_received", signal=signum)
        state.shutdown_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pacer = ProviderPacer(provider_delays)
    return _CalibrationSetup(
        run_id=run_id,
        prompts=prompts,
        state=state,
        cp=cp,
        pacer=pacer,
    )


# DEVIATION CS-PARAM-001: 7 params — dataclass would break API.
async def run_calibration_audit(
    router_url: str,
    judge_model: str,
    output_dir: Path,
    resume: bool,
    provider_delays: dict[str, float],
    model_filter: list[str] | None,
    dry_run: bool,
) -> None:
    """Run the full calibration audit pipeline."""
    requested = model_filter if model_filter else _get_all_model_ids()
    setup = _setup_calibration(output_dir, resume, model_filter, provider_delays)
    state, prompts = setup.state, setup.prompts
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        reachable, judge = await run_preflight(client, router_url, requested, judge_model)
        if dry_run:
            logger.info(
                "dry_run_complete", reachable=reachable, judge=judge, prompt_count=len(prompts)
            )
            return
        pairs = _interleaved_schedule(reachable, prompts)
        for idx, (model, prompt) in enumerate(pairs):
            if state.shutdown_requested:
                break
            if (model, prompt.id) in state.completed_pairs:
                continue
            logger.info(
                "evaluating", model=model, prompt_id=prompt.id, progress=f"{idx + 1}/{len(pairs)}"
            )
            result = await evaluate_prompt(
                client, router_url, model, prompt, setup.run_id, setup.pacer, state
            )
            if result is None:
                continue
            if result.error is None and result.content:
                scores, norm = await judge_single(
                    client,
                    router_url,
                    judge,
                    prompt,
                    result.content,
                    setup.run_id,
                    setup.pacer,
                    state,
                )
                result = dataclasses.replace(
                    result, judge_scores=scores, normalized_score=norm, error=None
                )
            state.results.append(result)
            state.completed_pairs.add((model, prompt.id))
            _append_checkpoint(setup.cp, result)
    _write_reports(
        output_dir, setup.run_id, state, reachable, judge, prompts, partial=state.shutdown_requested
    )
    logger.info(
        "calibration_audit_complete",
        run_id=setup.run_id,
        results=len(state.results),
        errors=state.total_errors,
    )


# --- Model discovery ---

# Provider priority: prefer native/direct > high-RPM inference > meta-proxy.
_PROVIDER_PRIORITY: types.MappingProxyType[str, int] = types.MappingProxyType(
    {
        "gemini": 1,
        "groq": 2,
        "nvidia_nim": 3,
        "openrouter": 4,
    }
)


def _base_model_name(model_id: str) -> str:
    """Extract the base model identity, stripping provider prefix and org namespace."""
    parts = model_id.split("/", 1)
    bare = parts[1] if len(parts) > 1 else parts[0]
    if bare.startswith("models/"):
        bare = bare[7:]
    org_parts = bare.split("/")
    if len(org_parts) == 2:
        bare = org_parts[1]
    return bare.lower().replace(":free", "")


def _get_all_model_ids() -> list[str]:
    """Build a deduplicated benchmark target list from the role matrix.

    The role matrix may list the same base model under multiple providers
    (e.g. llama-3.3-70b on both groq and nvidia_nim). For benchmarking,
    a model's "flavor" is intrinsic — it performs the same regardless of
    which provider serves it. So we pick one provider per base model,
    preferring native/direct providers over meta-proxies.

    The catalog intentionally retains duplicates for production load
    balancing — this dedup is benchmark-only.
    """
    path = _CONFIG_DIR / "model_role_matrix.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())

    all_ids: list[str] = []
    for entries in data.get("roles", {}).values():
        for e in entries:
            all_ids.append(e["model_id"])

    base_to_best: dict[str, tuple[int, str]] = {}
    for mid in all_ids:
        provider = mid.split("/")[0] if "/" in mid else "unknown"
        prio = _PROVIDER_PRIORITY.get(provider, 99)
        is_free = ":free" in mid
        rank = prio + (10 if is_free else 0)
        bm = _base_model_name(mid)
        if bm not in base_to_best or rank < base_to_best[bm][0]:
            base_to_best[bm] = (rank, mid)

    targets = sorted(mid for _rank, mid in base_to_best.values())

    logger.info(
        "benchmark_targets_resolved",
        role_matrix_models=len(all_ids),
        unique_base_models=len(targets),
    )
    return targets


# --- CLI (spec section 9) ---


def _parse_delays(raw: list[str] | None) -> dict[str, float]:
    """Parse --provider-delay key=value pairs."""
    delays = dict(_DEFAULT_PROVIDER_DELAYS)
    if not raw:
        return delays
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid --provider-delay: {item!r} (expected key=value)")
        k, v = item.split("=", 1)
        delays[k.strip()] = float(v.strip())
    return delays


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Calibration audit -- LLM-as-judge calibration through the router API.",
        prog="dragonlight-router-calibration-audit",
    )
    p.add_argument(
        "--router-url",
        default=_DEFAULT_ROUTER_URL,
        help=f"Router base URL (default: {_DEFAULT_ROUTER_URL})",
    )
    p.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE,
        help=f"Model for judging (default: {_DEFAULT_JUDGE})",
    )
    p.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument(
        "--provider-delay",
        nargs="*",
        metavar="K=V",
        help="Per-provider delay overrides (e.g. groq=2.0)",
    )
    p.add_argument("--models", nargs="*", help="Subset of models to benchmark")
    p.add_argument("--dry-run", action="store_true", help="Pre-flight only")
    return p


def main() -> None:
    """CLI entry point for python -m dragonlight_router.benchmark.calibration_audit."""
    args = _build_cli_parser().parse_args()
    asyncio.run(
        run_calibration_audit(
            router_url=args.router_url,
            judge_model=args.judge_model,
            output_dir=Path(args.output_dir),
            resume=args.resume,
            provider_delays=_parse_delays(args.provider_delay),
            model_filter=args.models,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
