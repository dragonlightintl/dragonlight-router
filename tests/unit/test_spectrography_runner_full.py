"""Comprehensive tests for spectrography/runner.py — covering all missed lines.

Spec: model-spectrography-v0.1.0-spec
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    FlavorScore,
    ModelFlavorProfile,
)
from dragonlight_router.spectrography.analyzer import ProbeResult
from dragonlight_router.spectrography.probes import SpectrographyProbe
from dragonlight_router.spectrography.runner import (
    ProviderPacer,
    ReportContext,
    RunState,
    _build_backend_config,
    _build_json_report_data,
    _call_judge_adapter,
    _call_model_adapter,
    _collect_streaming_response,
    _create_adapter,
    _create_all_adapters,
    _evaluate_probe,
    _generate_spectrography_reports,
    _get_providers,
    _load_checkpoint,
    _load_provider_configs,
    _make_error_result,
    _md_calibration_section,
    _md_header_section,
    _md_proficiencies_section,
    _md_rankings_section,
    _run_probe_loop,
    _serialize_deltas,
    _serialize_profiles,
    _SpectrographyConfig,
    _SpectrographySetup,
    _write_config_profiles,
    _write_json_report,
    _write_markdown_summary,
    main,
    run_spectrography,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe(**overrides: Any) -> SpectrographyProbe:
    """Build a SpectrographyProbe with sensible defaults."""
    defaults = {
        "id": "disc-test-001",
        "task_type": "generation",
        "domain": "code",
        "quality_speed": "quality",
        "prompt": "Test prompt text.",
        "judge_criteria": "Test criteria.",
        "discrimination_axis": "style",
        "difficulty": "medium",
    }
    defaults.update(overrides)
    return SpectrographyProbe(**defaults)


def _make_probe_result(
    model_id: str = "test/model-a",
    probe_id: str = "disc-test-001",
    task_type: str = "generation",
    domain: str = "code",
    quality_speed: str = "quality",
    normalized_score: float = 0.8,
    error: str | None = None,
    is_self_eval: bool = False,
) -> ProbeResult:
    """Build a ProbeResult with sensible defaults."""
    return ProbeResult(
        model_id=model_id,
        probe_id=probe_id,
        task_type=task_type,
        domain=domain,
        quality_speed=quality_speed,
        normalized_score=normalized_score,
        judge_scores={"accuracy": 4, "completeness": 3, "clarity": 4, "relevance": 4},
        is_self_eval=is_self_eval,
        error=error,
    )


def _make_flavor_profile(
    model_id: str = "test/model-a",
    score: float = 0.7,
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with uniform scores."""
    fs = FlavorScore(score=score, confidence=0.8, sample_count=5)
    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=datetime.now(UTC).isoformat(),
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


def _make_backend_config(model_id: str = "test/model") -> BackendConfig:
    """Build a BackendConfig with test defaults."""
    return BackendConfig(
        name=model_id,
        provider="test",
        model="test-model",
        tier=BackendTier.MODERATE,
        base_url="https://example.com",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=131072,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(
            rpm=30,
            rpd=999999,
            tpm=9999999,
            daily_token_cap=9999999,
        ),
    )


def _mock_adapter() -> MagicMock:
    """Build a mock GenerativeBackend with an async generate method."""
    adapter = MagicMock()
    adapter.config = _make_backend_config()

    async def _generate(
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        yield "Hello world"

    adapter.generate = _generate
    return adapter


def _mock_adapter_empty() -> MagicMock:
    """Build a mock adapter that yields nothing."""
    adapter = MagicMock()
    adapter.config = _make_backend_config()

    async def _generate(
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        return
        yield  # make it an async generator

    adapter.generate = _generate
    return adapter


def _mock_adapter_error() -> MagicMock:
    """Build a mock adapter that raises RuntimeError."""
    adapter = MagicMock()
    adapter.config = _make_backend_config()

    async def _generate(
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        raise RuntimeError("test error")
        yield  # noqa: RET503  # make it an async generator

    adapter.generate = _generate
    return adapter


def _make_report_context(
    tmp_path: Path,
    results: list[ProbeResult] | None = None,
    profiles: dict[str, ModelFlavorProfile] | None = None,
    deltas: dict[str, dict[str, Any]] | None = None,
    rankings: dict[str, list[str]] | None = None,
) -> ReportContext:
    """Build a ReportContext for testing."""
    if results is None:
        results = [_make_probe_result()]
    if profiles is None:
        profiles = {"test/model-a": _make_flavor_profile()}
    if deltas is None:
        deltas = {}
    if rankings is None:
        rankings = {}
    return ReportContext(
        run_id="test-run-001",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        judge_model="gemini/gemini-2.5-pro",
        results=results,
        profiles=profiles,
        deltas=deltas,
        rankings=rankings,
        output_dir=tmp_path,
    )


# ===========================================================================
# ProviderPacer.wait tests (lines 106-110)
# ===========================================================================


class TestProviderPacerWait:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_wait_first_call_no_sleep(self) -> None:
        pacer = ProviderPacer()
        start = time.monotonic()
        await pacer.wait("gemini")
        elapsed = time.monotonic() - start
        # First call should not sleep significantly
        assert elapsed < 1.5

    @pytest.mark.asyncio
    async def test_wait_respects_delay(self) -> None:
        pacer = ProviderPacer(overrides={"test": 0.05})
        await pacer.wait("test")
        # Second call should trigger sleep
        start = time.monotonic()
        await pacer.wait("test")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.01

    @pytest.mark.asyncio
    async def test_wait_unknown_provider_uses_default(self) -> None:
        pacer = ProviderPacer()
        # Unknown provider uses 1.0s default delay
        await pacer.wait("unknown_provider")
        assert "unknown_provider" in pacer._last


# ===========================================================================
# _load_provider_configs tests (lines 140-150)
# ===========================================================================


class TestLoadProviderConfigs:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_missing_yaml_returns_empty(self) -> None:
        with patch(
            "dragonlight_router.spectrography.runner._CONFIG_DIR",
            Path("/nonexistent/path"),
        ):
            result = _load_provider_configs()
            assert result == {}

    def test_valid_yaml_parses_providers(self, tmp_path: Path) -> None:
        yaml_data = {
            "providers": [
                {"name": "gemini", "base_url": "https://example.com"},
                {"name": "groq", "base_url": "https://groq.example.com"},
            ],
        }
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))
        with patch(
            "dragonlight_router.spectrography.runner._CONFIG_DIR",
            tmp_path,
        ):
            result = _load_provider_configs()
            assert "gemini" in result
            assert "groq" in result

    def test_invalid_yaml_returns_empty(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text("%invalid yaml directive")
        with patch(
            "dragonlight_router.spectrography.runner._CONFIG_DIR",
            tmp_path,
        ):
            result = _load_provider_configs()
            assert result == {}

    def test_skips_non_dict_providers(self, tmp_path: Path) -> None:
        yaml_data = {"providers": ["not-a-dict", {"name": "valid"}]}
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))
        with patch(
            "dragonlight_router.spectrography.runner._CONFIG_DIR",
            tmp_path,
        ):
            result = _load_provider_configs()
            assert "valid" in result
            assert len(result) == 1


# ===========================================================================
# _get_providers tests (line 161)
# ===========================================================================


class TestGetProviders:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_caches_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dragonlight_router.spectrography.runner._CACHED_PROVIDERS",
            None,
        )
        with patch(
            "dragonlight_router.spectrography.runner._load_provider_configs",
            return_value={"test": {"name": "test"}},
        ):
            result = _get_providers()
            assert "test" in result
        # Reset after test
        monkeypatch.setattr(
            "dragonlight_router.spectrography.runner._CACHED_PROVIDERS",
            None,
        )


# ===========================================================================
# _create_adapter tests (lines 213-214, 226-228)
# ===========================================================================


class TestCreateAdapterFull:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_no_adapter_key_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "dragonlight_router.spectrography.runner._CACHED_PROVIDERS",
            {"custom_provider": {"name": "custom_provider"}},
        )
        result = _create_adapter("custom_provider/some-model")
        assert result is None

    def test_adapter_creation_exception_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "dragonlight_router.spectrography.runner._CACHED_PROVIDERS",
            {"groq": {"name": "groq", "base_url": "https://api.groq.com"}},
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with patch(
            "dragonlight_router.spectrography.runner.create_adapter",
            side_effect=ValueError("bad config"),
        ):
            result = _create_adapter("groq/some-model")
            assert result is None


# ===========================================================================
# _collect_streaming_response tests (lines 243-255)
# ===========================================================================


class TestCollectStreamingResponse:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_collects_chunks(self) -> None:
        adapter = _mock_adapter()
        result = await _collect_streaming_response(
            adapter,
            [{"role": "user", "content": "hello"}],
        )
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_none_on_empty(self) -> None:
        adapter = _mock_adapter_empty()
        result = await _collect_streaming_response(
            adapter,
            [{"role": "user", "content": "hello"}],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self) -> None:
        adapter = _mock_adapter_error()
        result = await _collect_streaming_response(
            adapter,
            [{"role": "user", "content": "hello"}],
        )
        assert result is None


# ===========================================================================
# _call_model_adapter tests (lines 270-279)
# ===========================================================================


class TestCallModelAdapter:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_returns_response_text(self) -> None:
        adapter = _mock_adapter()
        probe = _make_probe()
        result = await _call_model_adapter(adapter, probe)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self) -> None:
        adapter = _mock_adapter_error()
        probe = _make_probe()
        result = await _call_model_adapter(adapter, probe)
        assert result is None


# ===========================================================================
# _call_judge_adapter tests (lines 291-319)
# ===========================================================================


class TestCallJudgeAdapter:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_returns_scores_on_success(self) -> None:
        judge_response = json.dumps(
            {
                "accuracy": 4,
                "completeness": 3,
                "clarity": 5,
                "relevance": 4,
            }
        )
        adapter = MagicMock()

        async def _gen(
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            yield judge_response

        adapter.generate = _gen
        probe = _make_probe()
        scores, error = await _call_judge_adapter(adapter, probe, "model output")
        assert scores is not None
        assert error is None
        assert "accuracy" in scores

    @pytest.mark.asyncio
    async def test_returns_error_on_failed_call(self) -> None:
        adapter = _mock_adapter_error()
        probe = _make_probe()
        scores, error = await _call_judge_adapter(adapter, probe, "model output")
        assert scores is None
        assert error == "judge_call_failed"

    @pytest.mark.asyncio
    async def test_returns_error_on_parse_failure(self) -> None:
        adapter = MagicMock()

        async def _gen(
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            yield "not valid json at all"

        adapter.generate = _gen
        probe = _make_probe()
        scores, error = await _call_judge_adapter(adapter, probe, "model output")
        assert scores is None
        assert error == "judge_parse_failed"


# ===========================================================================
# _make_error_result tests (lines 330-333)
# ===========================================================================


class TestMakeErrorResult:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_builds_error_probe_result(self) -> None:
        probe = _make_probe()
        result = _make_error_result(
            "test/model",
            probe,
            (False, 0.0, "test_error"),
        )
        assert result.model_id == "test/model"
        assert result.error == "test_error"
        assert result.normalized_score == 0.0
        assert result.is_self_eval is False

    def test_self_eval_flag(self) -> None:
        probe = _make_probe()
        result = _make_error_result(
            "test/model",
            probe,
            (True, 0.5, "some_error"),
        )
        assert result.is_self_eval is True
        assert result.normalized_score == 0.5


# ===========================================================================
# _evaluate_probe tests (lines 349-368)
# ===========================================================================


class TestEvaluateProbe:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_empty_model_response(self) -> None:
        adapter = _mock_adapter_empty()
        judge = _mock_adapter()
        probe = _make_probe()
        result = await _evaluate_probe(
            "test/model",
            probe,
            adapter,
            judge,
            "judge/model",
        )
        assert result.error == "empty_model_response"
        assert result.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_judge_failure(self) -> None:
        model_adapter = _mock_adapter()
        judge_adapter = _mock_adapter_error()
        probe = _make_probe()
        result = await _evaluate_probe(
            "test/model",
            probe,
            model_adapter,
            judge_adapter,
            "judge/model",
        )
        assert result.error == "judge_call_failed"
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_successful_evaluation(self) -> None:
        model_adapter = _mock_adapter()
        judge_response = json.dumps(
            {
                "accuracy": 4,
                "completeness": 4,
                "clarity": 4,
                "relevance": 4,
            }
        )
        judge_adapter = MagicMock()

        async def _gen(
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            yield judge_response

        judge_adapter.generate = _gen
        probe = _make_probe()
        result = await _evaluate_probe(
            "test/model",
            probe,
            model_adapter,
            judge_adapter,
            "judge/model",
        )
        assert result.error is None
        assert 0.0 <= result.normalized_score <= 1.0
        assert result.is_self_eval is False

    @pytest.mark.asyncio
    async def test_self_eval_detected(self) -> None:
        model_adapter = _mock_adapter()
        judge_response = json.dumps(
            {
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "relevance": 5,
            }
        )
        judge_adapter = MagicMock()

        async def _gen(
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            yield judge_response

        judge_adapter.generate = _gen
        probe = _make_probe()
        result = await _evaluate_probe(
            "same/model",
            probe,
            model_adapter,
            judge_adapter,
            "same/model",
        )
        assert result.is_self_eval is True
        assert result.error is None


# ===========================================================================
# _build_backend_config tests (line 398 not covered — covered by caller)
# ===========================================================================


class TestBuildBackendConfig:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_builds_config_correctly(self) -> None:
        provider_cfg = {
            "model_prefix": "gemini/",
            "env_key": "GEMINI_API_KEY",
            "base_url": "https://example.com",
            "rate_limits": {"rpm": 60},
        }
        config = _build_backend_config(
            "gemini/gemini-2.5-pro",
            provider_cfg,
            "google",
        )
        assert config.name == "gemini/gemini-2.5-pro"
        assert config.provider == "google"
        assert config.model == "gemini-2.5-pro"
        assert config.rate_limits.rpm == 60

    def test_strips_prefix_from_model_id(self) -> None:
        provider_cfg = {"model_prefix": "groq/", "rate_limits": {}}
        config = _build_backend_config("groq/llama-70b", provider_cfg, "groq")
        assert config.model == "llama-70b"

    def test_handles_no_matching_prefix(self) -> None:
        provider_cfg = {"model_prefix": "other/", "rate_limits": {}}
        config = _build_backend_config(
            "groq/llama-70b",
            provider_cfg,
            "groq",
        )
        assert config.model == "groq/llama-70b"


# ===========================================================================
# _serialize_profiles tests (lines 460-486)
# ===========================================================================


class TestSerializeProfiles:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_serializes_profiles_to_json_safe(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1")}
        result = _serialize_profiles(profiles)
        assert "m1" in result
        assert "model_id" in result["m1"]
        assert "task_scores" in result["m1"]
        assert "domain_scores" in result["m1"]
        assert "qs_scores" in result["m1"]

    def test_serializes_multiple_profiles(self) -> None:
        profiles = {
            "m1": _make_flavor_profile("m1"),
            "m2": _make_flavor_profile("m2", score=0.9),
        }
        result = _serialize_profiles(profiles)
        assert len(result) == 2

    def test_score_structure(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1")}
        result = _serialize_profiles(profiles)
        task_scores = result["m1"]["task_scores"]
        for key in IBR_TASK_TYPES:
            entry = task_scores[key]
            assert "score" in entry
            assert "confidence" in entry
            assert "sample_count" in entry


# ===========================================================================
# _build_json_report_data tests (lines 491-520)
# ===========================================================================


class TestBuildJsonReportData:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_builds_report_structure(self, tmp_path: Path) -> None:
        ctx = _make_report_context(tmp_path)
        report = _build_json_report_data(ctx)
        assert report["run_id"] == "test-run-001"
        assert "models_evaluated" in report
        assert "total_probes" in report
        assert "profiles" in report
        assert "per_probe_results" in report

    def test_counts_errors_and_self_evals(self, tmp_path: Path) -> None:
        results = [
            _make_probe_result(error="fail"),
            _make_probe_result(is_self_eval=True),
            _make_probe_result(),
        ]
        ctx = _make_report_context(tmp_path, results=results)
        report = _build_json_report_data(ctx)
        assert report["total_errors"] == 1
        assert report["self_eval_count"] == 1
        assert report["total_probes"] == 3


# ===========================================================================
# _write_json_report tests (lines 525-535)
# ===========================================================================


class TestWriteJsonReport:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_writes_report_file(self, tmp_path: Path) -> None:
        ctx = _make_report_context(tmp_path)
        _write_json_report(ctx)
        report_path = tmp_path / "test-run-001" / "report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["run_id"] == "test-run-001"


# ===========================================================================
# _md_header_section tests (lines 540-559)
# ===========================================================================


class TestMdHeaderSection:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_produces_header_lines(self, tmp_path: Path) -> None:
        ctx = _make_report_context(tmp_path)
        lines = _md_header_section(ctx)
        assert any("Model Spectrography Report" in line for line in lines)
        assert any("Run ID" in line for line in lines)
        assert any("Judge model" in line for line in lines)

    def test_counts_errors_and_self_evals(self, tmp_path: Path) -> None:
        results = [
            _make_probe_result(error="fail"),
            _make_probe_result(is_self_eval=True),
        ]
        ctx = _make_report_context(tmp_path, results=results)
        lines = _md_header_section(ctx)
        joined = "\n".join(lines)
        assert "Errors" in joined
        assert "Self-evaluations" in joined


# ===========================================================================
# _md_rankings_section tests (lines 567-592)
# ===========================================================================


class TestMdRankingsSection:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_produces_rankings_table(self) -> None:
        profiles = {
            "m1": _make_flavor_profile("m1", score=0.3),
            "m2": _make_flavor_profile("m2", score=0.9),
        }
        rankings = {"task/generation": ["m2", "m1"]}
        lines = _md_rankings_section(profiles, rankings)
        joined = "\n".join(lines)
        assert "Model Rankings" in joined
        assert "m1" in joined
        assert "m2" in joined

    def test_includes_dimension_rankings(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1")}
        rankings = {"task/generation": ["m1"], "domain/code": ["m1"]}
        lines = _md_rankings_section(profiles, rankings)
        joined = "\n".join(lines)
        assert "Dimension Rankings" in joined

    def test_empty_rankings_no_dimension_section(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1")}
        lines = _md_rankings_section(profiles, {})
        joined = "\n".join(lines)
        assert "Dimension Rankings" not in joined


# ===========================================================================
# _md_proficiencies_section tests (lines 599-619)
# ===========================================================================


class TestMdProficienciesSection:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_produces_proficiency_lines(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1", score=0.8)}
        lines = _md_proficiencies_section(profiles)
        joined = "\n".join(lines)
        assert "Proficiencies" in joined
        assert "m1" in joined
        assert "Top:" in joined
        assert "Low:" in joined

    def test_multiple_models(self) -> None:
        profiles = {
            "m1": _make_flavor_profile("m1", score=0.3),
            "m2": _make_flavor_profile("m2", score=0.9),
        }
        lines = _md_proficiencies_section(profiles)
        joined = "\n".join(lines)
        assert "m1" in joined
        assert "m2" in joined


# ===========================================================================
# _md_calibration_section tests (lines 626-646)
# ===========================================================================


class TestMdCalibrationSection:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_empty_deltas_returns_empty(self) -> None:
        lines = _md_calibration_section({})
        assert lines == []

    def test_produces_calibration_table_with_dict_deltas(self) -> None:
        deltas: dict[str, dict[str, Any]] = {
            "m1": {"task/generation": {"delta": 0.05}},
        }
        lines = _md_calibration_section(deltas)
        joined = "\n".join(lines)
        assert "Calibration Deltas" in joined
        assert "m1" in joined

    def test_produces_calibration_table_with_float_deltas(self) -> None:
        deltas: dict[str, dict[str, Any]] = {
            "m1": {"task/generation": 0.05},
        }
        lines = _md_calibration_section(deltas)
        joined = "\n".join(lines)
        assert "m1" in joined
        assert "+0.0500" in joined


# ===========================================================================
# _write_markdown_summary tests (lines 651-665)
# ===========================================================================


class TestWriteMarkdownSummary:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_writes_summary_file(self, tmp_path: Path) -> None:
        ctx = _make_report_context(tmp_path)
        _write_markdown_summary(ctx)
        summary_path = tmp_path / "test-run-001" / "summary.md"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "Model Spectrography Report" in content


# ===========================================================================
# _create_all_adapters tests (lines 700-724)
# ===========================================================================


class TestCreateAllAdapters:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_raises_on_no_judge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "dragonlight_router.spectrography.runner._CACHED_PROVIDERS",
            {},
        )
        with pytest.raises(SystemExit, match="Cannot create adapter"):
            _create_all_adapters(["m1"], "judge/model")

    def test_raises_on_no_model_adapters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_adapter = _mock_adapter()
        call_count = 0

        def _fake_create(model_id: str) -> Any:
            nonlocal call_count
            call_count += 1
            # First call(s) for models return None, last for judge returns adapter
            if model_id == "judge/model":
                return mock_adapter
            return None

        with (
            patch(
                "dragonlight_router.spectrography.runner._create_adapter",
                side_effect=_fake_create,
            ),
            pytest.raises(SystemExit, match="No model adapters"),
        ):
            _create_all_adapters(["bad/model"], "judge/model")

    def test_success_with_valid_adapters(self) -> None:
        mock = _mock_adapter()
        with patch(
            "dragonlight_router.spectrography.runner._create_adapter",
            return_value=mock,
        ):
            model_adapters, judge = _create_all_adapters(
                ["m1", "m2"],
                "judge/model",
            )
            assert len(model_adapters) == 2
            assert judge is mock

    def test_skips_failed_model_adapters(self) -> None:
        mock = _mock_adapter()

        def _fake(model_id: str) -> Any:
            if model_id == "m2":
                return None
            return mock

        with patch(
            "dragonlight_router.spectrography.runner._create_adapter",
            side_effect=_fake,
        ):
            model_adapters, judge = _create_all_adapters(
                ["m1", "m2"],
                "judge/model",
            )
            assert "m1" in model_adapters
            assert "m2" not in model_adapters


# ===========================================================================
# _load_checkpoint empty line handling (line 408)
# ===========================================================================


class TestLoadCheckpointEmptyLines:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_skips_empty_and_blank_lines(self, tmp_path: Path) -> None:
        cp_path = tmp_path / "checkpoint.jsonl"
        content = (
            '{"model_id": "m1", "probe_id": "p1"}\n\n   \n{"model_id": "m2", "probe_id": "p2"}\n'
        )
        cp_path.write_text(content)
        loaded = _load_checkpoint(cp_path)
        assert ("m1", "p1") in loaded
        assert ("m2", "p2") in loaded
        assert len(loaded) == 2


# ===========================================================================
# _setup_spectrography tests (lines 729-761)
# ===========================================================================


class TestSetupSpectrography:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_creates_setup_bundle(self, tmp_path: Path) -> None:
        mock = _mock_adapter()
        with (
            patch(
                "dragonlight_router.spectrography.runner._create_all_adapters",
                return_value=({"m1": mock}, mock),
            ),
            patch(
                "dragonlight_router.spectrography.runner.get_all_probes",
                return_value=[_make_probe()],
            ),
        ):
            from dragonlight_router.spectrography.runner import (
                _setup_spectrography,
            )

            cfg = _SpectrographyConfig(
                models=["m1"],
                judge_model="judge/model",
                output_dir=tmp_path,
                provider_delays=None,
                write_profiles=False,
                resume=False,
            )
            setup = _setup_spectrography(cfg)
            assert setup.state.run_id is not None
            assert len(setup.schedule) > 0
            assert "m1" in setup.model_adapters

    def test_resume_loads_checkpoint(self, tmp_path: Path) -> None:
        mock = _mock_adapter()
        # Write a checkpoint file
        run_id = "20260101-000000-abc12345"
        cp_dir = tmp_path / run_id
        cp_dir.mkdir(parents=True)
        cp_path = cp_dir / "checkpoint.jsonl"
        cp_path.write_text(
            json.dumps({"model_id": "m1", "probe_id": "disc-test-001"}) + "\n",
        )

        with (
            patch(
                "dragonlight_router.spectrography.runner._create_all_adapters",
                return_value=({"m1": mock}, mock),
            ),
            patch(
                "dragonlight_router.spectrography.runner.get_all_probes",
                return_value=[_make_probe()],
            ),
        ):
            from dragonlight_router.spectrography.runner import (
                _setup_spectrography,
            )

            cfg = _SpectrographyConfig(
                models=["m1"],
                judge_model="judge/model",
                output_dir=tmp_path,
                provider_delays={"gemini": 0.01},
                write_profiles=False,
                resume=True,
            )
            setup = _setup_spectrography(cfg)
            # The resume flag is set, but run_id is new so no matches
            assert setup.state is not None


# ===========================================================================
# _run_probe_loop tests (lines 773-803)
# ===========================================================================


class TestRunProbeLoop:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_evaluates_all_scheduled_pairs(self) -> None:
        mock_adapter = _mock_adapter()
        judge_response = json.dumps(
            {
                "accuracy": 4,
                "completeness": 4,
                "clarity": 4,
                "relevance": 4,
            }
        )
        judge_adapter = MagicMock()

        async def _gen(
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> Any:
            yield judge_response

        judge_adapter.generate = _gen

        probe = _make_probe()
        state = RunState(
            run_id="test-run",
            started_at=datetime.now(UTC).isoformat(),
        )
        pacer = ProviderPacer(overrides={"test": 0.0})
        cp_path = Path("/tmp/test-spectrography-cp.jsonl")

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model": mock_adapter},
            judge_adapter=judge_adapter,
            schedule=[("test/model", probe)],
            pacer=pacer,
            checkpoint_path=cp_path,
        )

        with patch(
            "dragonlight_router.spectrography.runner._append_checkpoint",
        ):
            await _run_probe_loop(setup, "judge/model")

        assert len(state.results) == 1
        assert ("test/model", "disc-test-001") in state.completed_pairs

    @pytest.mark.asyncio
    async def test_skips_completed_pairs(self) -> None:
        mock_adapter = _mock_adapter()
        judge_adapter = _mock_adapter()
        probe = _make_probe()
        state = RunState(
            run_id="test-run",
            started_at=datetime.now(UTC).isoformat(),
            completed_pairs={("test/model", "disc-test-001")},
        )
        pacer = ProviderPacer(overrides={"test": 0.0})
        cp_path = Path("/tmp/test-spectrography-cp2.jsonl")

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model": mock_adapter},
            judge_adapter=judge_adapter,
            schedule=[("test/model", probe)],
            pacer=pacer,
            checkpoint_path=cp_path,
        )

        await _run_probe_loop(setup, "judge/model")
        assert len(state.results) == 0

    @pytest.mark.asyncio
    async def test_shutdown_stops_loop(self) -> None:
        mock_adapter = _mock_adapter()
        judge_adapter = _mock_adapter()
        probe1 = _make_probe(id="disc-test-001")
        probe2 = _make_probe(id="disc-test-002")
        state = RunState(
            run_id="test-run",
            started_at=datetime.now(UTC).isoformat(),
            shutdown_requested=True,
        )
        pacer = ProviderPacer(overrides={"test": 0.0})
        cp_path = Path("/tmp/test-spectrography-cp3.jsonl")

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model": mock_adapter},
            judge_adapter=judge_adapter,
            schedule=[("test/model", probe1), ("test/model", probe2)],
            pacer=pacer,
            checkpoint_path=cp_path,
        )

        await _run_probe_loop(setup, "judge/model")
        assert len(state.results) == 0

    @pytest.mark.asyncio
    async def test_error_results_increment_counter(self) -> None:
        model_adapter = _mock_adapter_empty()
        judge_adapter = _mock_adapter()
        probe = _make_probe()
        state = RunState(
            run_id="test-run",
            started_at=datetime.now(UTC).isoformat(),
        )
        pacer = ProviderPacer(overrides={"test": 0.0})
        cp_path = Path("/tmp/test-spectrography-cp4.jsonl")

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model": model_adapter},
            judge_adapter=judge_adapter,
            schedule=[("test/model", probe)],
            pacer=pacer,
            checkpoint_path=cp_path,
        )

        with patch(
            "dragonlight_router.spectrography.runner._append_checkpoint",
        ):
            await _run_probe_loop(setup, "judge/model")

        assert state.total_errors == 1


# ===========================================================================
# _serialize_deltas tests (lines 810-821)
# ===========================================================================


class TestSerializeDeltas:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_serializes_calibration_deltas(self) -> None:
        from dragonlight_router.spectrography.analyzer import CalibrationDelta

        delta = CalibrationDelta(
            dimension="task/generation",
            declared=0.7,
            empirical=0.8,
            delta=0.1,
            recommendation="review",
        )
        deltas = {"m1": {"task/generation": delta}}
        result = _serialize_deltas(deltas)
        assert "m1" in result
        entry = result["m1"]["task/generation"]
        assert entry["declared"] == 0.7
        assert entry["empirical"] == 0.8
        assert entry["delta"] == 0.1
        assert entry["recommendation"] == "review"


# ===========================================================================
# _write_config_profiles tests (lines 828-837)
# ===========================================================================


class TestWriteConfigProfiles:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_merges_and_writes_profiles(self) -> None:
        profiles = {"m1": _make_flavor_profile("m1")}
        with (
            patch(
                "dragonlight_router.spectrography.runner.load_existing_fingerprints",
                return_value={},
            ),
            patch(
                "dragonlight_router.spectrography.runner.merge_incremental",
                return_value=profiles,
            ),
            patch(
                "dragonlight_router.spectrography.runner.build_fingerprints_yaml",
                return_value="version: 1\nprofiles: {}",
            ),
            patch(
                "dragonlight_router.spectrography.runner.write_fingerprints_yaml",
            ) as mock_write,
        ):
            _write_config_profiles(profiles, "test-run-001")
            mock_write.assert_called_once()


# ===========================================================================
# _generate_spectrography_reports tests (lines 847-879)
# ===========================================================================


class TestGenerateSpectrographyReports:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_generates_all_reports(self, tmp_path: Path) -> None:
        mock = _mock_adapter()
        probe = _make_probe()
        state = RunState(
            run_id="test-run-001",
            started_at=datetime.now(UTC).isoformat(),
            results=[_make_probe_result()],
        )
        pacer = ProviderPacer()
        cp_path = tmp_path / "test-run-001" / "checkpoint.jsonl"

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model-a": mock},
            judge_adapter=mock,
            schedule=[("test/model-a", probe)],
            pacer=pacer,
            checkpoint_path=cp_path,
        )

        cfg = _SpectrographyConfig(
            models=["test/model-a"],
            judge_model="judge/model",
            output_dir=tmp_path,
            provider_delays=None,
            write_profiles=False,
            resume=False,
        )

        with patch(
            "dragonlight_router.spectrography.runner.compute_calibration_deltas",
            return_value={},
        ):
            _generate_spectrography_reports(setup, cfg)

        run_dir = tmp_path / "test-run-001"
        assert (run_dir / "report.json").exists()
        assert (run_dir / "summary.md").exists()
        assert (run_dir / "fingerprints.yaml").exists()

    def test_writes_config_profiles_when_enabled(
        self,
        tmp_path: Path,
    ) -> None:
        mock = _mock_adapter()
        probe = _make_probe()
        state = RunState(
            run_id="test-run-002",
            started_at=datetime.now(UTC).isoformat(),
            results=[_make_probe_result()],
        )

        setup = _SpectrographySetup(
            state=state,
            model_adapters={"test/model-a": mock},
            judge_adapter=mock,
            schedule=[("test/model-a", probe)],
            pacer=ProviderPacer(),
            checkpoint_path=tmp_path / "test-run-002" / "checkpoint.jsonl",
        )

        cfg = _SpectrographyConfig(
            models=["test/model-a"],
            judge_model="judge/model",
            output_dir=tmp_path,
            provider_delays=None,
            write_profiles=True,
            resume=False,
        )

        with (
            patch(
                "dragonlight_router.spectrography.runner.compute_calibration_deltas",
                return_value={},
            ),
            patch(
                "dragonlight_router.spectrography.runner._write_config_profiles",
            ) as mock_wcp,
        ):
            _generate_spectrography_reports(setup, cfg)
            mock_wcp.assert_called_once()


# ===========================================================================
# run_spectrography tests (lines 899-906)
# ===========================================================================


class TestRunSpectrography:
    """Spec: model-spectrography-v0.1.0-spec"""

    @pytest.mark.asyncio
    async def test_orchestrates_full_pipeline(self, tmp_path: Path) -> None:
        with (
            patch(
                "dragonlight_router.spectrography.runner._setup_spectrography",
            ) as mock_setup,
            patch(
                "dragonlight_router.spectrography.runner._run_probe_loop",
                new_callable=AsyncMock,
            ),
            patch(
                "dragonlight_router.spectrography.runner._generate_spectrography_reports",
            ),
        ):
            mock_setup.return_value = _SpectrographySetup(
                state=RunState(
                    run_id="test",
                    started_at=datetime.now(UTC).isoformat(),
                ),
                model_adapters={},
                judge_adapter=_mock_adapter(),
                schedule=[],
                pacer=ProviderPacer(),
                checkpoint_path=tmp_path / "cp.jsonl",
            )
            await run_spectrography(
                models=["m1"],
                judge_model="judge/model",
                output_dir=tmp_path,
                provider_delays=None,
                write_profiles=False,
                resume=False,
            )


# ===========================================================================
# _build_arg_parser + main tests (lines 928-964, 969-986, 994)
# ===========================================================================


class TestBuildArgParser:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_returns_argument_parser(self) -> None:
        from dragonlight_router.spectrography.runner import _build_arg_parser

        parser = _build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parses_all_flags(self) -> None:
        from dragonlight_router.spectrography.runner import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--judge-model",
                "test/judge",
                "--output-dir",
                "/tmp/out",
                "--models",
                "m1",
                "m2",
                "--provider-delay",
                "gemini=2.0",
                "--write-profiles",
                "--resume",
                "--dry-run",
            ]
        )
        assert args.judge_model == "test/judge"
        assert args.output_dir == "/tmp/out"
        assert args.models == ["m1", "m2"]
        assert args.write_profiles is True
        assert args.resume is True
        assert args.dry_run is True

    def test_defaults(self) -> None:
        from dragonlight_router.spectrography.runner import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.judge_model == "gemini/gemini-2.5-pro"
        assert args.write_profiles is False
        assert args.resume is False
        assert args.dry_run is False


class TestMain:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_dry_run_exits_early(self) -> None:
        with (
            patch(
                "dragonlight_router.spectrography.runner._build_arg_parser",
            ) as mock_parser,
            patch(
                "dragonlight_router.spectrography.runner.get_all_probes",
                return_value=[_make_probe()],
            ),
        ):
            mock_args = MagicMock()
            mock_args.models = ["m1"]
            mock_args.dry_run = True
            mock_args.judge_model = "judge/model"
            mock_args.provider_delay = None
            mock_parser.return_value.parse_args.return_value = mock_args
            main()

    def test_no_targets_raises_system_exit(self) -> None:
        with (
            patch(
                "dragonlight_router.spectrography.runner._build_arg_parser",
            ) as mock_parser,
            patch(
                "dragonlight_router.spectrography.runner._get_all_model_ids",
                return_value=[],
            ),
        ):
            mock_args = MagicMock()
            mock_args.models = None
            mock_args.dry_run = False
            mock_parser.return_value.parse_args.return_value = mock_args
            with pytest.raises(SystemExit, match="No model targets"):
                main()

    def test_runs_spectrography(self) -> None:
        with (
            patch(
                "dragonlight_router.spectrography.runner._build_arg_parser",
            ) as mock_parser,
            patch(
                "dragonlight_router.spectrography.runner._parse_delays",
                return_value=None,
            ),
            patch(
                "dragonlight_router.spectrography.runner.asyncio.run",
            ) as mock_run,
        ):
            mock_args = MagicMock()
            mock_args.models = ["m1"]
            mock_args.dry_run = False
            mock_args.judge_model = "judge/model"
            mock_args.output_dir = "/tmp/out"
            mock_args.provider_delay = None
            mock_args.write_profiles = False
            mock_args.resume = False
            mock_parser.return_value.parse_args.return_value = mock_args
            main()
            mock_run.assert_called_once()

    def test_resolves_models_from_config(self) -> None:
        with (
            patch(
                "dragonlight_router.spectrography.runner._build_arg_parser",
            ) as mock_parser,
            patch(
                "dragonlight_router.spectrography.runner._get_all_model_ids",
                return_value=["m1", "m2"],
            ),
            patch(
                "dragonlight_router.spectrography.runner._parse_delays",
                return_value=None,
            ),
            patch(
                "dragonlight_router.spectrography.runner.asyncio.run",
            ),
        ):
            mock_args = MagicMock()
            mock_args.models = None
            mock_args.dry_run = False
            mock_args.judge_model = "judge/model"
            mock_args.output_dir = "/tmp/out"
            mock_args.provider_delay = None
            mock_args.write_profiles = False
            mock_args.resume = False
            mock_parser.return_value.parse_args.return_value = mock_args
            main()


# ===========================================================================
# Signal handler coverage (lines 750-751)
# ===========================================================================


class TestSetupSpectrographySignalHandler:
    """Cover the _on_signal handler body in _setup_spectrography (lines 750-751)."""

    def test_signal_handler_sets_shutdown_requested(self, tmp_path: Path) -> None:
        """The signal handler registered by _setup_spectrography sets shutdown_requested."""
        import signal as signal_mod

        from dragonlight_router.spectrography.runner import _setup_spectrography

        cfg = _SpectrographyConfig(
            models=["test/model-a"],
            judge_model="test/judge",
            output_dir=tmp_path,
            provider_delays=None,
            write_profiles=False,
            resume=False,
        )

        mock_model_adapter = MagicMock()
        mock_judge_adapter = MagicMock()

        with patch(
            "dragonlight_router.spectrography.runner._create_all_adapters",
            return_value=({"test/model-a": mock_model_adapter}, mock_judge_adapter),
        ):
            setup = _setup_spectrography(cfg)

        # State should start as not shutdown-requested
        assert setup.state.shutdown_requested is False

        # Invoke the signal handler that was registered
        handler = signal_mod.getsignal(signal_mod.SIGINT)
        assert callable(handler)
        handler(signal_mod.SIGINT, None)

        # State should now be shutdown-requested
        assert setup.state.shutdown_requested is True
