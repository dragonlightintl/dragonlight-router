"""Benchmark runner for IBR automated flavor profiling.

Orchestrates the benchmark pipeline: sends eval prompts to each model,
collects responses, scores them via LLM-as-judge, aggregates results
into ModelSpectrographProfile instances, and persists to JSON.

Spec reference: intent-based-router-v0.1.0-spec.md section 3.2, Method 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.benchmark.judge import judge_response
from dragonlight_router.benchmark.prompts import EvalPrompt, get_all_prompts
from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    GenerativeBackend,
    ModelSpectrographProfile,
    SpectrographScore,
)

logger = structlog.get_logger()

# IBR-FLV-06: Decay constants for stale benchmark profiles.
_DECAY_THRESHOLD_DAYS: int = 30
_DECAY_RATE_PER_DAY: float = 0.01
_DECAY_TARGET: float = 0.5
_PROFILE_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Decay logic (IBR-FLV-06)
# ---------------------------------------------------------------------------


def apply_decay(
    profile: ModelSpectrographProfile,
    now: datetime | None = None,
) -> ModelSpectrographProfile:
    """Apply time-based decay to a benchmark profile.

    Profiles older than 30 days decay toward 0.5 at 0.01/day.
    Returns a new profile with adjusted scores and updated timestamp.

    IBR-FLV-06: Benchmark profiles older than 30 days MUST decay toward 0.5.
    """
    assert isinstance(profile, ModelSpectrographProfile), "profile must be ModelSpectrographProfile"

    if now is None:
        now = datetime.now(UTC)

    updated_at = datetime.fromisoformat(profile.updated_at)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    age_days = (now - updated_at).total_seconds() / 86400.0

    if age_days <= _DECAY_THRESHOLD_DAYS:
        return profile

    decay_days = age_days - _DECAY_THRESHOLD_DAYS
    assert decay_days > 0, "decay_days must be positive when past threshold"

    task_scores = _decay_dimension(profile.task_scores, decay_days)
    domain_scores = _decay_dimension(profile.domain_scores, decay_days)
    qs_scores = _decay_dimension(profile.qs_scores, decay_days)

    return ModelSpectrographProfile(
        model_id=profile.model_id,
        version=profile.version,
        updated_at=profile.updated_at,  # Preserve original timestamp
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _decay_dimension(
    scores: dict[str, SpectrographScore],
    decay_days: float,
) -> dict[str, SpectrographScore]:
    """Apply decay to all scores in a dimension dict."""
    assert isinstance(scores, dict), "scores must be a dict"
    assert decay_days > 0, "decay_days must be positive"

    result: dict[str, SpectrographScore] = {}
    for key, fs in scores.items():
        decayed_score = _decay_single_score(fs.score, decay_days)
        decayed_confidence = max(0.0, fs.confidence - decay_days * _DECAY_RATE_PER_DAY)
        result[key] = SpectrographScore(
            score=decayed_score,
            confidence=decayed_confidence,
            sample_count=fs.sample_count,
        )
    return result


def _decay_single_score(score: float, decay_days: float) -> float:
    """Decay a single score toward 0.5 by decay_days * rate."""
    assert 0.0 <= score <= 1.0, f"score must be in [0.0, 1.0], got {score}"
    assert decay_days > 0, "decay_days must be positive"

    distance = score - _DECAY_TARGET
    decay_amount = min(abs(distance), decay_days * _DECAY_RATE_PER_DAY)

    if distance > 0:
        result = score - decay_amount
    elif distance < 0:
        result = score + decay_amount
    else:
        result = score

    result = max(0.0, min(1.0, result))
    assert 0.0 <= result <= 1.0, f"decayed score out of range: {result}"
    return result


# ---------------------------------------------------------------------------
# Response collection helper
# ---------------------------------------------------------------------------


async def _collect_model_response(
    adapter: GenerativeBackend,
    prompt: EvalPrompt,
) -> str:
    """Send an eval prompt to a model and collect the full response.

    Returns empty string on adapter failure.
    """
    assert isinstance(prompt, EvalPrompt), "prompt must be an EvalPrompt"

    messages = [{"role": "user", "content": prompt.prompt}]
    try:
        chunks: list[str] = []
        async for chunk in adapter.generate(
            messages,
            max_tokens=512,
            temperature=0.7,
            stream=True,
        ):
            chunks.append(chunk)
        return "".join(chunks)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "benchmark_model_error",
            prompt_id=prompt.id,
            error=str(exc),
        )
        return ""


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


def _aggregate_scores(
    scored_prompts: list[tuple[EvalPrompt, float]],
) -> ModelSpectrographProfile:
    """Aggregate per-prompt scores into a ModelSpectrographProfile.

    Groups scores by task_type, domain, and quality_speed, computing
    the mean score for each dimension. Confidence is based on sample count.
    """
    assert isinstance(scored_prompts, list), "scored_prompts must be a list"

    task_accum: dict[str, list[float]] = {t: [] for t in IBR_TASK_TYPES}
    domain_accum: dict[str, list[float]] = {d: [] for d in IBR_DOMAINS}
    qs_accum: dict[str, list[float]] = {q: [] for q in IBR_QUALITY_SPEED}

    for prompt, score in scored_prompts:
        task_accum[prompt.task_type].append(score)
        domain_accum[prompt.domain].append(score)
        qs_accum[prompt.quality_speed].append(score)

    task_scores = _build_flavor_scores(task_accum)
    domain_scores = _build_flavor_scores(domain_accum)
    qs_scores = _build_flavor_scores(qs_accum)

    # model_id will be filled in by the caller
    return ModelSpectrographProfile(
        model_id="",
        version=_PROFILE_SCHEMA_VERSION,
        updated_at=datetime.now(UTC).isoformat(),
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _build_flavor_scores(
    accum: dict[str, list[float]],
) -> dict[str, SpectrographScore]:
    """Build SpectrographScore dict from accumulated score lists."""
    assert isinstance(accum, dict), "accum must be a dict"

    result: dict[str, SpectrographScore] = {}
    for key, values in accum.items():
        if values:
            avg_score = sum(values) / len(values)
            avg_score = max(0.0, min(1.0, avg_score))
            confidence = min(1.0, len(values) / 50.0)
            result[key] = SpectrographScore(
                score=avg_score,
                confidence=confidence,
                sample_count=len(values),
            )
        else:
            result[key] = IBR_NEUTRAL_SPECTROGRAPH
    return result


def _finalize_profile(
    model_id: str,
    profile: ModelSpectrographProfile,
) -> ModelSpectrographProfile:
    """Replace the placeholder model_id in an aggregated profile."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
    assert isinstance(profile, ModelSpectrographProfile), "profile must be ModelSpectrographProfile"

    return ModelSpectrographProfile(
        model_id=model_id,
        version=profile.version,
        updated_at=profile.updated_at,
        task_scores=profile.task_scores,
        domain_scores=profile.domain_scores,
        qs_scores=profile.qs_scores,
    )


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrate benchmark evaluation across multiple models.

    Sends eval prompts to each model, scores responses via LLM-as-judge,
    aggregates into ModelSpectrographProfile instances, and saves to JSON.
    """

    def __init__(
        self,
        eval_prompts: list[EvalPrompt],
        judge_adapter: GenerativeBackend,
        output_path: Path,
    ) -> None:
        assert isinstance(eval_prompts, list), "eval_prompts must be a list"
        assert len(eval_prompts) > 0, "eval_prompts must not be empty"
        assert isinstance(output_path, Path), "output_path must be a Path"

        self._prompts = eval_prompts
        self._judge = judge_adapter
        self._output_path = output_path

    async def _score_all_prompts(
        self,
        model_id: str,
        adapter: GenerativeBackend,
    ) -> list[tuple[EvalPrompt, float]]:
        """Send all eval prompts to a model and score each response."""
        scored: list[tuple[EvalPrompt, float]] = []
        total = len(self._prompts)

        for i, prompt in enumerate(self._prompts):
            logger.info(
                "benchmark_prompt",
                model_id=model_id,
                prompt_id=prompt.id,
                progress=f"{i + 1}/{total}",
            )
            response = await _collect_model_response(adapter, prompt)
            score = await judge_response(prompt, response, self._judge)
            scored.append((prompt, score))

        assert len(scored) == total, f"Expected {total} scores, got {len(scored)}"
        return scored

    async def benchmark_model(
        self,
        model_id: str,
        adapter: GenerativeBackend,
    ) -> ModelSpectrographProfile:
        """Run all eval prompts against a single model and return its profile."""
        assert isinstance(model_id, str) and model_id, "model_id must be a non-empty string"

        scored = await self._score_all_prompts(model_id, adapter)
        profile = _finalize_profile(model_id, _aggregate_scores(scored))

        logger.info(
            "benchmark_model_complete",
            model_id=model_id,
            prompts_scored=len(scored),
        )
        return profile

    async def benchmark_all(
        self,
        models: dict[str, GenerativeBackend],
    ) -> dict[str, ModelSpectrographProfile]:
        """Run benchmarks for all provided models.

        Returns a dict mapping model_id to ModelSpectrographProfile.
        Saves results to output_path/benchmark_profiles.json.
        """
        assert isinstance(models, dict), "models must be a dict"
        assert len(models) > 0, "models must not be empty"

        profiles: dict[str, ModelSpectrographProfile] = {}

        for model_id, adapter in models.items():
            logger.info("benchmark_starting", model_id=model_id)
            profile = await self.benchmark_model(model_id, adapter)
            profiles[model_id] = profile

        self._save_profiles(profiles)
        return profiles

    def _save_profiles(
        self,
        profiles: dict[str, ModelSpectrographProfile],
    ) -> None:
        """Serialize profiles to JSON at the configured output path."""
        assert isinstance(profiles, dict), "profiles must be a dict"

        self._output_path.mkdir(parents=True, exist_ok=True)
        output_file = self._output_path / "benchmark_profiles.json"

        data = _serialize_profiles(profiles)
        output_file.write_text(json.dumps(data, indent=2))

        logger.info(
            "benchmark_profiles_saved",
            path=str(output_file),
            model_count=len(profiles),
        )


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------


def _serialize_profiles(
    profiles: dict[str, ModelSpectrographProfile],
) -> dict[str, Any]:
    """Serialize profiles dict to JSON-compatible structure."""
    assert isinstance(profiles, dict), "profiles must be a dict"

    result: dict[str, Any] = {
        "version": _PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "profiles": {},
    }

    for model_id, profile in profiles.items():
        result["profiles"][model_id] = _serialize_single_profile(profile)

    return result


def _serialize_single_profile(profile: ModelSpectrographProfile) -> dict[str, Any]:
    """Serialize a single ModelSpectrographProfile to dict."""
    assert isinstance(profile, ModelSpectrographProfile), "profile must be ModelSpectrographProfile"

    return {
        "model_id": profile.model_id,
        "version": profile.version,
        "updated_at": profile.updated_at,
        "task_scores": _serialize_scores(profile.task_scores),
        "domain_scores": _serialize_scores(profile.domain_scores),
        "qs_scores": _serialize_scores(profile.qs_scores),
    }


def _serialize_scores(scores: dict[str, SpectrographScore]) -> dict[str, Any]:
    """Serialize SpectrographScore dict to JSON-compatible dict."""
    assert isinstance(scores, dict), "scores must be a dict"
    return {
        key: {
            "score": round(fs.score, 4),
            "confidence": round(fs.confidence, 4),
            "sample_count": fs.sample_count,
        }
        for key, fs in scores.items()
    }


def load_benchmark_profiles(path: Path) -> dict[str, ModelSpectrographProfile]:
    """Load benchmark profiles from JSON file.

    Returns empty dict if file does not exist or is malformed.
    """
    assert isinstance(path, Path), "path must be a Path"

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("benchmark_profiles_load_failed", error=str(exc))
        return {}

    if not isinstance(data, dict) or "profiles" not in data:
        logger.warning("benchmark_profiles_invalid_format")
        return {}

    return _deserialize_profiles(data["profiles"])


def _deserialize_profiles(
    raw: dict[str, Any],
) -> dict[str, ModelSpectrographProfile]:
    """Deserialize profiles from JSON dict."""
    assert isinstance(raw, dict), "raw must be a dict"

    profiles: dict[str, ModelSpectrographProfile] = {}
    for model_id, profile_data in raw.items():
        profile = _deserialize_single_profile(model_id, profile_data)
        if profile is not None:
            profiles[model_id] = profile
    return profiles


def _deserialize_single_profile(
    model_id: str,
    data: dict[str, Any],
) -> ModelSpectrographProfile | None:
    """Deserialize a single profile. Returns None on bad data."""
    assert isinstance(model_id, str), "model_id must be a string"

    if not isinstance(data, dict):
        return None

    try:
        return ModelSpectrographProfile(
            model_id=data.get("model_id", model_id),
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at", "")),
            task_scores=_deserialize_scores(
                data.get("task_scores", {}),
                IBR_TASK_TYPES,
            ),
            domain_scores=_deserialize_scores(
                data.get("domain_scores", {}),
                IBR_DOMAINS,
            ),
            qs_scores=_deserialize_scores(
                data.get("qs_scores", {}),
                IBR_QUALITY_SPEED,
            ),
        )
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "benchmark_profile_deserialize_failed",
            model_id=model_id,
            error=str(exc),
        )
        return None


def _deserialize_scores(
    raw: dict[str, Any],
    allowed_keys: frozenset[str],
) -> dict[str, SpectrographScore]:
    """Deserialize SpectrographScore dict, filling missing keys with neutral."""
    assert isinstance(allowed_keys, frozenset), "allowed_keys must be frozenset"

    scores: dict[str, SpectrographScore] = {}
    parsed = raw if isinstance(raw, dict) else {}

    for key in allowed_keys:
        if key in parsed and isinstance(parsed[key], dict):
            entry = parsed[key]
            score = max(0.0, min(1.0, float(entry.get("score", 0.5))))
            confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.0))))
            sample_count = max(0, int(entry.get("sample_count", 0)))
            scores[key] = SpectrographScore(
                score=score,
                confidence=confidence,
                sample_count=sample_count,
            )
        else:
            scores[key] = IBR_NEUTRAL_SPECTROGRAPH

    return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _resolve_available_models(
    engine: Any,
    model_filter: list[str] | None,
) -> dict[str, GenerativeBackend]:
    """Collect available adapters from the engine registry."""
    available: dict[str, GenerativeBackend] = {}
    for name, backend, _state in engine._registry.all_backends():
        if model_filter is not None and name not in model_filter:
            continue
        available[name] = backend
    return available


async def run_benchmark_cli(
    config_path: str,
    output_path: str,
    models: list[str] | None = None,
    judge_model: str | None = None,
) -> None:
    """Run benchmarks from the command line.

    Loads config, resolves adapters, runs benchmarks, saves results.
    """
    assert isinstance(config_path, str), "config_path must be a string"
    assert isinstance(output_path, str), "output_path must be a string"

    from dragonlight_router.router import RouterEngine

    logger.info("benchmark_cli_starting", config_path=config_path, output_path=output_path)

    engine = RouterEngine(config_path=Path(config_path))
    available = _resolve_available_models(engine, models)

    if not available:
        logger.error("benchmark_no_models_available")
        return

    judge_adapter = _resolve_judge(engine, judge_model)
    if judge_adapter is None:
        logger.error("benchmark_no_judge_available")
        return

    runner = BenchmarkRunner(
        eval_prompts=get_all_prompts(),
        judge_adapter=judge_adapter,
        output_path=Path(output_path),
    )
    await runner.benchmark_all(available)
    logger.info("benchmark_cli_complete", models_benchmarked=len(available))


def _resolve_judge(
    engine: Any,
    judge_model: str | None,
) -> GenerativeBackend | None:
    """Resolve the judge adapter from the engine registry.

    Priority: operator-specified > first classification role > first available.
    """
    if judge_model is not None:
        backend, _state = engine._registry.get(judge_model)
        if backend is not None:
            result: GenerativeBackend = backend
            return result
        logger.warning("benchmark_judge_not_found", model=judge_model)

    # Try classification role (most capable model)
    if engine._classification_adapter is not None:
        adapter: GenerativeBackend = engine._classification_adapter
        return adapter

    # Fall back to first available backend
    for _name, backend, _state in engine._registry.all_backends():
        fallback: GenerativeBackend = backend
        return fallback

    return None


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run IBR flavor benchmarks against registered models.",
        prog="dragonlight-router benchmark-flavors",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to router.yaml config file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for benchmark_profiles.json.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model IDs to benchmark (default: all).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model ID to use as the judge (default: auto-resolve).",
    )
    return parser


def main() -> None:
    """CLI main entry point for python -m dragonlight_router.benchmark.runner."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    asyncio.run(
        run_benchmark_cli(
            config_path=args.config,
            output_path=args.output,
            models=args.models,
            judge_model=args.judge_model,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
