"""Comprehensive tests for benchmark/runner.py — covers all uncovered code paths.

Spec: calibration-audit-v0.1.0-spec
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dragonlight_router.benchmark.prompts import EvalPrompt
from dragonlight_router.benchmark.runner import (
    BenchmarkRunner,
    _build_cli_parser,
    _decay_dimension,
    _deserialize_profiles,
    _deserialize_scores,
    _deserialize_single_profile,
    _resolve_available_models,
    _resolve_judge,
    _serialize_profiles,
    _serialize_scores,
    _serialize_single_profile,
    apply_decay,
    load_benchmark_profiles,
    main,
    run_benchmark_cli,
)
from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    FlavorScore,
    GenerativeBackend,
    ModelFlavorProfile,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_prompt(**overrides: Any) -> EvalPrompt:
    """Build an EvalPrompt with sensible defaults."""
    defaults: dict[str, str] = {
        "id": "test-prompt-001",
        "task_type": "generation",
        "domain": "code",
        "quality_speed": "quality",
        "prompt": "Write a test function.",
        "judge_criteria": "Correctness and completeness.",
    }
    defaults.update(overrides)
    return EvalPrompt(**defaults)


def _make_profile(
    model_id: str = "test-model",
    updated_at: str | None = None,
    score: float = 0.8,
    confidence: float = 0.6,
    sample_count: int = 10,
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with uniform scores."""
    if updated_at is None:
        updated_at = datetime.now(UTC).isoformat()
    fs = FlavorScore(score=score, confidence=confidence, sample_count=sample_count)
    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=updated_at,
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


def _make_mock_adapter(response_text: str) -> MagicMock:
    """Build a mock GenerativeBackend yielding a single chunk."""
    adapter = MagicMock(spec=GenerativeBackend)

    async def _generate(*args: Any, **kwargs: Any) -> Any:
        yield response_text

    adapter.generate = _generate
    return adapter


def _make_raising_adapter(exc: Exception) -> MagicMock:
    """Build a mock adapter whose generate() raises."""
    adapter = MagicMock(spec=GenerativeBackend)

    async def _generate(*args: Any, **kwargs: Any) -> Any:
        raise exc
        yield  # noqa: RET503

    adapter.generate = _generate
    return adapter


# ---------------------------------------------------------------------------
# apply_decay — timezone-naive updated_at
# ---------------------------------------------------------------------------


class TestApplyDecayTimezone:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_naive_timestamp_treated_as_utc(self) -> None:
        now = datetime.now(UTC)
        # Create a naive timestamp (no tzinfo)
        naive_ts = (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S")
        profile = _make_profile(updated_at=naive_ts)
        result = apply_decay(profile, now=now)
        assert result is not profile  # Decay should have been applied
        for fs in result.task_scores.values():
            assert fs.score < 0.8


# ---------------------------------------------------------------------------
# _decay_dimension
# ---------------------------------------------------------------------------


class TestDecayDimension:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_all_scores_decayed(self) -> None:
        scores = {
            "generation": FlavorScore(score=0.9, confidence=0.5, sample_count=10),
            "analysis": FlavorScore(score=0.2, confidence=0.3, sample_count=5),
        }
        result = _decay_dimension(scores, decay_days=10)
        assert result["generation"].score < 0.9
        assert result["analysis"].score > 0.2
        # Confidence decays
        assert result["generation"].confidence < 0.5
        assert result["analysis"].confidence < 0.3

    def test_sample_count_preserved(self) -> None:
        scores = {"gen": FlavorScore(score=0.8, confidence=0.5, sample_count=42)}
        result = _decay_dimension(scores, decay_days=5)
        assert result["gen"].sample_count == 42


# ---------------------------------------------------------------------------
# BenchmarkRunner._save_profiles and benchmark_all
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerSaveProfiles:
    """Spec: calibration-audit-v0.1.0-spec"""

    async def test_benchmark_all(self, tmp_path: Path) -> None:
        model_adapter = _make_mock_adapter("good response")
        judge_scores = '{"accuracy": 4, "completeness": 4, "clarity": 4, "relevance": 4}'
        judge_adapter = _make_mock_adapter(judge_scores)
        prompts = [_make_eval_prompt(id="p1")]

        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=tmp_path,
        )
        profiles = await runner.benchmark_all({"m1": model_adapter})
        assert "m1" in profiles
        assert isinstance(profiles["m1"], ModelFlavorProfile)
        # Should have written the file
        output_file = tmp_path / "benchmark_profiles.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "profiles" in data
        assert "m1" in data["profiles"]

    async def test_benchmark_all_multiple_models(self, tmp_path: Path) -> None:
        adapter1 = _make_mock_adapter("response 1")
        adapter2 = _make_mock_adapter("response 2")
        judge_scores = '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        judge_adapter = _make_mock_adapter(judge_scores)
        prompts = [_make_eval_prompt(id="p1")]

        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=tmp_path,
        )
        profiles = await runner.benchmark_all({"m1": adapter1, "m2": adapter2})
        assert len(profiles) == 2

    async def test_benchmark_all_empty_models_raises(self, tmp_path: Path) -> None:
        judge_adapter = _make_mock_adapter("scores")
        prompts = [_make_eval_prompt()]
        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=tmp_path,
        )
        with pytest.raises(AssertionError, match="must not be empty"):
            await runner.benchmark_all({})

    def test_save_profiles_creates_dir(self, tmp_path: Path) -> None:
        judge_adapter = _make_mock_adapter("scores")
        prompts = [_make_eval_prompt()]
        nested = tmp_path / "a" / "b" / "c"
        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=nested,
        )
        profile = _make_profile(model_id="m1")
        runner._save_profiles({"m1": profile})
        assert (nested / "benchmark_profiles.json").exists()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerializeProfiles:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_serialize_profiles(self) -> None:
        profile = _make_profile(model_id="m1")
        result = _serialize_profiles({"m1": profile})
        assert result["version"] == 1
        assert "generated_at" in result
        assert "m1" in result["profiles"]

    def test_serialize_single_profile(self) -> None:
        profile = _make_profile(model_id="m1")
        result = _serialize_single_profile(profile)
        assert result["model_id"] == "m1"
        assert "task_scores" in result
        assert "domain_scores" in result
        assert "qs_scores" in result

    def test_serialize_scores(self) -> None:
        scores = {
            "gen": FlavorScore(score=0.8, confidence=0.5, sample_count=10),
        }
        result = _serialize_scores(scores)
        assert result["gen"]["score"] == 0.8
        assert result["gen"]["confidence"] == 0.5
        assert result["gen"]["sample_count"] == 10


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


class TestDeserializeProfiles:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_round_trip(self) -> None:
        original = _make_profile(model_id="m1")
        serialized = _serialize_profiles({"m1": original})
        deserialized = _deserialize_profiles(serialized["profiles"])
        assert "m1" in deserialized
        p = deserialized["m1"]
        assert p.model_id == "m1"

    def test_bad_profile_data_skipped(self) -> None:
        raw = {"m1": "not a dict"}
        result = _deserialize_profiles(raw)
        assert "m1" not in result

    def test_missing_fields_defaults(self) -> None:
        raw = {"m1": {"model_id": "m1"}}
        result = _deserialize_profiles(raw)
        assert "m1" in result
        p = result["m1"]
        assert p.version == 1

    def test_deserialize_error_returns_none(self) -> None:
        # Trigger a TypeError/ValueError in deserialization
        raw = {"m1": {"version": "not_an_int"}}
        result = _deserialize_profiles(raw)
        # Should either succeed with coercion or skip
        # The int() call on "not_an_int" should raise ValueError
        assert "m1" not in result


class TestDeserializeSingleProfile:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_valid_data(self) -> None:
        data: dict[str, Any] = {
            "model_id": "m1",
            "version": 1,
            "updated_at": "2026-01-01T00:00:00",
            "task_scores": {},
            "domain_scores": {},
            "qs_scores": {},
        }
        result = _deserialize_single_profile("m1", data)
        assert result is not None
        assert result.model_id == "m1"

    def test_non_dict_returns_none(self) -> None:
        result = _deserialize_single_profile("m1", "bad")  # type: ignore[arg-type]
        assert result is None

    def test_type_error_returns_none(self) -> None:
        # Pass data that will cause a TypeError during construction
        data: dict[str, Any] = {
            "version": "bad",
            "task_scores": None,
        }
        result = _deserialize_single_profile("m1", data)
        assert result is None


class TestDeserializeScores:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_valid_scores(self) -> None:
        raw: dict[str, Any] = {
            "generation": {"score": 0.8, "confidence": 0.5, "sample_count": 10},
        }
        result = _deserialize_scores(raw, IBR_TASK_TYPES)
        assert result["generation"].score == pytest.approx(0.8)
        assert result["generation"].confidence == pytest.approx(0.5)
        assert result["generation"].sample_count == 10

    def test_missing_keys_get_neutral(self) -> None:
        result = _deserialize_scores({}, IBR_TASK_TYPES)
        for key in IBR_TASK_TYPES:
            assert result[key] == IBR_NEUTRAL_FLAVOR

    def test_non_dict_entry_gets_neutral(self) -> None:
        raw: dict[str, Any] = {"generation": "not a dict"}
        result = _deserialize_scores(raw, IBR_TASK_TYPES)
        assert result["generation"] == IBR_NEUTRAL_FLAVOR

    def test_scores_clamped(self) -> None:
        raw: dict[str, Any] = {
            "generation": {"score": 2.0, "confidence": -1.0, "sample_count": -5},
        }
        result = _deserialize_scores(raw, IBR_TASK_TYPES)
        assert result["generation"].score == 1.0
        assert result["generation"].confidence == 0.0
        assert result["generation"].sample_count == 0

    def test_non_dict_raw_treated_as_empty(self) -> None:
        result = _deserialize_scores("bad", IBR_TASK_TYPES)  # type: ignore[arg-type]
        for key in IBR_TASK_TYPES:
            assert result[key] == IBR_NEUTRAL_FLAVOR


# ---------------------------------------------------------------------------
# load_benchmark_profiles
# ---------------------------------------------------------------------------


class TestLoadBenchmarkProfiles:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = load_benchmark_profiles(tmp_path / "missing.json")
        assert result == {}

    def test_valid_file(self, tmp_path: Path) -> None:
        profile = _make_profile(model_id="m1")
        data = _serialize_profiles({"m1": profile})
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        result = load_benchmark_profiles(path)
        assert "m1" in result

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        path.write_text("{bad json")
        result = load_benchmark_profiles(path)
        assert result == {}

    def test_no_profiles_key(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"other": "data"}))
        result = load_benchmark_profiles(path)
        assert result == {}

    def test_non_dict_data(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps([1, 2, 3]))
        result = load_benchmark_profiles(path)
        assert result == {}


# ---------------------------------------------------------------------------
# _resolve_available_models
# ---------------------------------------------------------------------------


class TestResolveAvailableModels:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_all_models(self) -> None:
        adapter1 = MagicMock(spec=GenerativeBackend)
        adapter2 = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._registry.all_backends.return_value = [
            ("m1", adapter1, "healthy"),
            ("m2", adapter2, "healthy"),
        ]
        result = _resolve_available_models(engine, model_filter=None)
        assert "m1" in result
        assert "m2" in result

    def test_with_filter(self) -> None:
        adapter1 = MagicMock(spec=GenerativeBackend)
        adapter2 = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._registry.all_backends.return_value = [
            ("m1", adapter1, "healthy"),
            ("m2", adapter2, "healthy"),
        ]
        result = _resolve_available_models(engine, model_filter=["m1"])
        assert "m1" in result
        assert "m2" not in result


# ---------------------------------------------------------------------------
# _resolve_judge
# ---------------------------------------------------------------------------


class TestResolveJudge:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_specified_judge_found(self) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._registry.get.return_value = (adapter, "healthy")
        result = _resolve_judge(engine, "judge-model")
        assert result is adapter

    def test_specified_judge_not_found_fallback_classification(self) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._registry.get.return_value = (None, None)
        engine._classification_adapter = adapter
        result = _resolve_judge(engine, "missing-judge")
        assert result is adapter

    def test_fallback_to_first_backend(self) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._registry.get.return_value = (None, None)
        engine._classification_adapter = None
        engine._registry.all_backends.return_value = [
            ("m1", adapter, "healthy"),
        ]
        result = _resolve_judge(engine, "missing-judge")
        assert result is adapter

    def test_no_judge_available(self) -> None:
        engine = MagicMock()
        engine._registry.get.return_value = (None, None)
        engine._classification_adapter = None
        engine._registry.all_backends.return_value = []
        result = _resolve_judge(engine, "missing")
        assert result is None

    def test_no_judge_model_specified_uses_classification(self) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._classification_adapter = adapter
        result = _resolve_judge(engine, None)
        assert result is adapter

    def test_no_judge_model_no_classification_fallback(self) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        engine = MagicMock()
        engine._classification_adapter = None
        engine._registry.all_backends.return_value = [("m1", adapter, "ok")]
        result = _resolve_judge(engine, None)
        assert result is adapter

    def test_no_judge_model_nothing_available(self) -> None:
        engine = MagicMock()
        engine._classification_adapter = None
        engine._registry.all_backends.return_value = []
        result = _resolve_judge(engine, None)
        assert result is None


# ---------------------------------------------------------------------------
# run_benchmark_cli
# ---------------------------------------------------------------------------


class TestRunBenchmarkCli:
    """Spec: calibration-audit-v0.1.0-spec"""

    async def test_no_models_available(self, tmp_path: Path) -> None:
        with patch("dragonlight_router.router.RouterEngine") as mock_engine_cls:
            engine = MagicMock()
            engine._registry.all_backends.return_value = []
            mock_engine_cls.return_value = engine
            await run_benchmark_cli(
                config_path=str(tmp_path / "config.yaml"),
                output_path=str(tmp_path / "output"),
            )
        # Should return without writing anything
        assert not (tmp_path / "output" / "benchmark_profiles.json").exists()

    async def test_no_judge_available(self, tmp_path: Path) -> None:
        adapter = MagicMock(spec=GenerativeBackend)
        with (
            patch("dragonlight_router.router.RouterEngine") as mock_engine_cls,
            patch(
                "dragonlight_router.benchmark.runner._resolve_judge",
                return_value=None,
            ),
        ):
            engine = MagicMock()
            engine._registry.all_backends.return_value = [("m1", adapter, "ok")]
            mock_engine_cls.return_value = engine
            await run_benchmark_cli(
                config_path=str(tmp_path / "config.yaml"),
                output_path=str(tmp_path / "output"),
            )
        assert not (tmp_path / "output" / "benchmark_profiles.json").exists()

    async def test_successful_run(self, tmp_path: Path) -> None:
        model_adapter = _make_mock_adapter("model response")
        judge_scores = '{"accuracy": 4, "completeness": 4, "clarity": 4, "relevance": 4}'
        judge_adapter = _make_mock_adapter(judge_scores)

        with (
            patch("dragonlight_router.router.RouterEngine") as mock_engine_cls,
            patch(
                "dragonlight_router.benchmark.runner.get_all_prompts",
                return_value=[_make_eval_prompt()],
            ),
        ):
            engine = MagicMock()
            engine._registry.all_backends.return_value = [
                ("m1", model_adapter, "healthy"),
            ]
            engine._registry.get.return_value = (judge_adapter, "healthy")
            mock_engine_cls.return_value = engine
            await run_benchmark_cli(
                config_path=str(tmp_path / "config.yaml"),
                output_path=str(tmp_path / "output"),
                models=None,
                judge_model="judge-m",
            )
        assert (tmp_path / "output" / "benchmark_profiles.json").exists()

    async def test_with_model_filter(self, tmp_path: Path) -> None:
        adapter1 = _make_mock_adapter("response 1")
        adapter2 = _make_mock_adapter("response 2")
        judge_scores = '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        judge_adapter = _make_mock_adapter(judge_scores)

        with (
            patch("dragonlight_router.router.RouterEngine") as mock_engine_cls,
            patch(
                "dragonlight_router.benchmark.runner.get_all_prompts",
                return_value=[_make_eval_prompt()],
            ),
        ):
            engine = MagicMock()
            engine._registry.all_backends.return_value = [
                ("m1", adapter1, "ok"),
                ("m2", adapter2, "ok"),
            ]
            engine._registry.get.return_value = (judge_adapter, "ok")
            mock_engine_cls.return_value = engine
            await run_benchmark_cli(
                config_path=str(tmp_path / "config.yaml"),
                output_path=str(tmp_path / "output"),
                models=["m1"],
                judge_model="judge",
            )
        data = json.loads((tmp_path / "output" / "benchmark_profiles.json").read_text())
        assert "m1" in data["profiles"]
        assert "m2" not in data["profiles"]


# ---------------------------------------------------------------------------
# CLI parser and main
# ---------------------------------------------------------------------------


class TestBuildRunnerCliParser:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_required_args(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(
            [
                "--config",
                "/path/to/config.yaml",
                "--output",
                "/path/to/output",
            ]
        )
        assert args.config == "/path/to/config.yaml"
        assert args.output == "/path/to/output"
        assert args.models is None
        assert args.judge_model is None

    def test_optional_args(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(
            [
                "--config",
                "c.yaml",
                "--output",
                "out",
                "--models",
                "m1",
                "m2",
                "--judge-model",
                "judge",
            ]
        )
        assert args.models == ["m1", "m2"]
        assert args.judge_model == "judge"

    def test_missing_required_exits(self) -> None:
        parser = _build_cli_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestRunnerMain:
    """Spec: calibration-audit-v0.1.0-spec"""

    def test_main_invokes_asyncio_run(self) -> None:
        with (
            patch(
                "dragonlight_router.benchmark.runner._build_cli_parser",
            ) as mock_parser,
            patch("dragonlight_router.benchmark.runner.asyncio.run") as mock_run,
        ):
            args = MagicMock()
            args.config = "/path/config.yaml"
            args.output = "/path/output"
            args.models = None
            args.judge_model = None
            mock_parser.return_value.parse_args.return_value = args
            main()
            assert mock_run.called


# ---------------------------------------------------------------------------
# _score_all_prompts (via benchmark_model)
# ---------------------------------------------------------------------------


class TestScoreAllPrompts:
    """Spec: calibration-audit-v0.1.0-spec"""

    async def test_scores_all_prompts(self, tmp_path: Path) -> None:
        model_adapter = _make_mock_adapter("good text")
        judge_scores = '{"accuracy": 5, "completeness": 5, "clarity": 5, "relevance": 5}'
        judge_adapter = _make_mock_adapter(judge_scores)
        prompts = [
            _make_eval_prompt(
                id="p1", task_type="generation", domain="code", quality_speed="quality"
            ),
            _make_eval_prompt(
                id="p2", task_type="analysis", domain="technical", quality_speed="balanced"
            ),
        ]
        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=tmp_path,
        )
        profile = await runner.benchmark_model("m1", model_adapter)
        assert profile.task_scores["generation"].sample_count == 1
        assert profile.task_scores["analysis"].sample_count == 1

    async def test_model_adapter_failure(self, tmp_path: Path) -> None:
        model_adapter = _make_raising_adapter(RuntimeError("fail"))
        judge_scores = '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        judge_adapter = _make_mock_adapter(judge_scores)
        prompts = [_make_eval_prompt(id="p1")]
        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=tmp_path,
        )
        profile = await runner.benchmark_model("m1", model_adapter)
        # Empty response from failure -> 0.0 score
        assert profile.task_scores["generation"].score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coverage gap: apply_decay without now parameter (line 59)
# ---------------------------------------------------------------------------


class TestApplyDecayDefaultNow:
    """Cover apply_decay with now=None (line 59: now = datetime.now(UTC))."""

    def test_apply_decay_without_now_uses_current_time(self) -> None:
        """apply_decay() without now parameter uses datetime.now(UTC) as default."""
        from datetime import timedelta

        # Create a profile old enough to trigger decay (>30 days)
        old_ts = (datetime.now(UTC) - timedelta(days=45)).isoformat()
        profile = _make_profile(model_id="decay-test", updated_at=old_ts, score=0.8)

        # Call without now parameter — covers line 59
        result = apply_decay(profile)

        # Profile should have decayed (it's 45 days old, 15 days past threshold)
        assert result is not profile  # New object created
        for fs in result.task_scores.values():
            assert fs.score < 0.8

    def test_apply_decay_without_now_fresh_profile_unchanged(self) -> None:
        """apply_decay() without now parameter does not decay fresh profiles."""
        fresh_ts = datetime.now(UTC).isoformat()
        profile = _make_profile(model_id="fresh-test", updated_at=fresh_ts, score=0.8)

        result = apply_decay(profile)
        assert result is profile  # Same object returned (no decay)
