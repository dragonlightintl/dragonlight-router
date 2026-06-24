"""Tests for benchmark/ — runner, judge, and prompts.

Spec traceability: IBR spec v0.3.0 section 3.2, Method 3.
AC numbers: IBR-FLV-01 through IBR-FLV-06.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dragonlight_router.benchmark.judge import (
    _build_judge_messages,
    _normalize_scores,
    _parse_judge_scores,
    _score_from_raw,
    judge_response,
)
from dragonlight_router.benchmark.prompts import (
    EvalPrompt,
    _validate_prompt,
    get_all_prompts,
)
from dragonlight_router.benchmark.runner import (
    BenchmarkRunner,
    _aggregate_scores,
    _build_flavor_scores,
    _collect_model_response,
    _decay_single_score,
    _finalize_profile,
    apply_decay,
)
from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    GenerativeBackend,
    ModelSpectrographProfile,
    SpectrographScore,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — mock adapter factories (matches test_classifier.py style)
# ---------------------------------------------------------------------------


def _make_mock_adapter(response_text: str) -> MagicMock:
    """Build a mock GenerativeBackend yielding a single chunk."""
    adapter = MagicMock(spec=GenerativeBackend)

    async def _generate(*args, **kwargs):
        yield response_text

    adapter.generate = _generate
    return adapter


def _make_raising_adapter(exc: Exception) -> MagicMock:
    """Build a mock adapter whose generate() raises *exc*."""
    adapter = MagicMock(spec=GenerativeBackend)

    async def _generate(*args, **kwargs):
        raise exc
        yield  # noqa: RET503

    adapter.generate = _generate
    return adapter


def _make_eval_prompt(**overrides) -> EvalPrompt:
    """Build an EvalPrompt with sensible defaults."""
    defaults = {
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
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with uniform scores across all dimensions."""
    if updated_at is None:
        updated_at = datetime.now(UTC).isoformat()
    fs = SpectrographScore(score=score, confidence=confidence, sample_count=sample_count)
    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at=updated_at,
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


# ===========================================================================
# Prompts
# ===========================================================================


class TestGetAllPrompts:
    """[IBR-FLV-01] Eval prompt bank validation."""

    def test_returns_at_least_48_prompts(self):
        """[IBR-FLV-01] get_all_prompts returns ~50 prompts."""
        prompts = get_all_prompts()
        assert len(prompts) >= 48
        assert len(prompts) <= 60

    def test_all_prompts_have_valid_task_type(self):
        """[IBR-FLV-01] Every prompt has a valid IBR task_type."""
        for p in get_all_prompts():
            assert p.task_type in IBR_TASK_TYPES, (
                f"Prompt {p.id} has invalid task_type: {p.task_type}"
            )

    def test_all_prompts_have_valid_domain(self):
        """[IBR-FLV-01] Every prompt has a valid IBR domain."""
        for p in get_all_prompts():
            assert p.domain in IBR_DOMAINS, f"Prompt {p.id} has invalid domain: {p.domain}"

    def test_all_prompts_have_valid_quality_speed(self):
        """[IBR-FLV-01] Every prompt has a valid quality_speed value."""
        for p in get_all_prompts():
            assert p.quality_speed in {"quality", "balanced", "speed"}, (
                f"Prompt {p.id} has invalid quality_speed: {p.quality_speed}"
            )

    def test_all_prompt_ids_are_unique(self):
        """[IBR-FLV-01] All prompt IDs are unique across the bank."""
        prompts = get_all_prompts()
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids)), "Duplicate prompt IDs found"

    def test_all_prompts_have_populated_fields(self):
        """[IBR-FLV-01] Every EvalPrompt has all fields populated."""
        for p in get_all_prompts():
            assert p.id, "Prompt missing id"
            assert p.task_type, f"Prompt {p.id} missing task_type"
            assert p.domain, f"Prompt {p.id} missing domain"
            assert p.quality_speed, f"Prompt {p.id} missing quality_speed"
            assert p.prompt, f"Prompt {p.id} missing prompt text"
            assert p.judge_criteria, f"Prompt {p.id} missing judge_criteria"

    def test_validate_prompt_valid(self):
        """[IBR-FLV-01] _validate_prompt passes for a valid prompt."""
        p = _make_eval_prompt()
        _validate_prompt(p)  # Should not raise

    def test_validate_prompt_invalid_task_type(self):
        """[IBR-FLV-01] _validate_prompt raises on invalid task_type."""
        p = _make_eval_prompt(task_type="invalid_type")
        with pytest.raises(AssertionError, match="Invalid task_type"):
            _validate_prompt(p)

    def test_validate_prompt_invalid_domain(self):
        """[IBR-FLV-01] _validate_prompt raises on invalid domain."""
        p = _make_eval_prompt(domain="cooking")
        with pytest.raises(AssertionError, match="Invalid domain"):
            _validate_prompt(p)

    def test_validate_prompt_invalid_quality_speed(self):
        """[IBR-FLV-01] _validate_prompt raises on invalid quality_speed."""
        p = _make_eval_prompt(quality_speed="turbo")
        with pytest.raises(AssertionError, match="Invalid quality_speed"):
            _validate_prompt(p)

    def test_validate_prompt_empty_prompt_text(self):
        """[IBR-FLV-01] _validate_prompt raises on empty prompt text."""
        p = _make_eval_prompt(prompt="")
        with pytest.raises(AssertionError, match="Empty prompt text"):
            _validate_prompt(p)

    def test_validate_prompt_empty_judge_criteria(self):
        """[IBR-FLV-01] _validate_prompt raises on empty judge_criteria."""
        p = _make_eval_prompt(judge_criteria="")
        with pytest.raises(AssertionError, match="Empty judge_criteria"):
            _validate_prompt(p)


# ===========================================================================
# Judge — _parse_judge_scores
# ===========================================================================


class TestParseJudgeScores:
    """[IBR-FLV-02] Judge score parsing from LLM output."""

    def test_valid_json(self):
        """[IBR-FLV-02] Valid JSON with all 4 keys parses correctly."""
        raw = '{"accuracy": 4, "completeness": 3, "clarity": 5, "relevance": 4}'
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 4
        assert result["completeness"] == 3
        assert result["clarity"] == 5
        assert result["relevance"] == 4

    def test_invalid_json(self):
        """[IBR-FLV-02] Invalid JSON returns None."""
        assert _parse_judge_scores("{bad json}") is None
        assert _parse_judge_scores("not json at all") is None

    def test_empty_string(self):
        """[IBR-FLV-02] Empty string returns None."""
        assert _parse_judge_scores("") is None

    def test_markdown_fenced_json(self):
        """[IBR-FLV-02] JSON in markdown code fences is extracted."""
        inner = '{"accuracy": 5, "completeness": 4, "clarity": 3, "relevance": 5}'
        raw = f"```json\n{inner}\n```"
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 5
        assert result["clarity"] == 3

    def test_markdown_fenced_no_lang(self):
        """[IBR-FLV-02] Code fences without lang tag are extracted."""
        inner = '{"accuracy": 2, "completeness": 2, "clarity": 2, "relevance": 2}'
        raw = f"```\n{inner}\n```"
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 2

    def test_json_with_preamble_text(self):
        """[IBR-FLV-02] JSON embedded after explanatory text is found."""
        inner = '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        raw = f"Here are my scores: {inner}"
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 3

    def test_missing_required_key(self):
        """[IBR-FLV-02] Missing a required key returns None."""
        raw = '{"accuracy": 4, "completeness": 3, "clarity": 5}'
        assert _parse_judge_scores(raw) is None

    def test_non_integer_values_coerced(self):
        """[IBR-FLV-02] Float values are coerced to int."""
        raw = '{"accuracy": 4.7, "completeness": 3.2, "clarity": 5.0, "relevance": 4.9}'
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 4
        assert result["completeness"] == 3

    def test_values_clamped_to_1_5(self):
        """[IBR-FLV-02] Values outside [1,5] are clamped."""
        raw = '{"accuracy": 0, "completeness": 10, "clarity": -1, "relevance": 6}'
        result = _parse_judge_scores(raw)
        assert result is not None
        assert result["accuracy"] == 1
        assert result["completeness"] == 5
        assert result["clarity"] == 1
        assert result["relevance"] == 5

    def test_non_numeric_value_returns_none(self):
        """[IBR-FLV-02] Non-numeric score values return None."""
        raw = '{"accuracy": "high", "completeness": 3, "clarity": 5, "relevance": 4}'
        assert _parse_judge_scores(raw) is None

    def test_json_array_returns_none(self):
        """[IBR-FLV-02] JSON array (not object) returns None."""
        raw = "[1, 2, 3, 4]"
        assert _parse_judge_scores(raw) is None


# ===========================================================================
# Judge — _normalize_scores
# ===========================================================================


class TestNormalizeScores:
    """[IBR-FLV-02] Normalization of 1-5 integer scores to 0.0-1.0."""

    def test_all_ones(self):
        """[IBR-FLV-02] All 1s normalizes to 0.0."""
        scores = {"accuracy": 1, "completeness": 1, "clarity": 1, "relevance": 1}
        assert _normalize_scores(scores) == pytest.approx(0.0)

    def test_all_fives(self):
        """[IBR-FLV-02] All 5s normalizes to 1.0."""
        scores = {"accuracy": 5, "completeness": 5, "clarity": 5, "relevance": 5}
        assert _normalize_scores(scores) == pytest.approx(1.0)

    def test_all_threes(self):
        """[IBR-FLV-02] All 3s normalizes to 0.5."""
        scores = {"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}
        assert _normalize_scores(scores) == pytest.approx(0.5)

    def test_mixed_scores(self):
        """[IBR-FLV-02] Mixed scores produce expected average."""
        scores = {"accuracy": 4, "completeness": 2, "clarity": 5, "relevance": 3}
        # avg = (4+2+5+3)/4 = 3.5; normalized = (3.5 - 1) / 4 = 0.625
        assert _normalize_scores(scores) == pytest.approx(0.625)

    def test_boundary_value_1(self):
        """[IBR-FLV-02] Single 1 among 5s still in range."""
        scores = {"accuracy": 1, "completeness": 5, "clarity": 5, "relevance": 5}
        result = _normalize_scores(scores)
        assert 0.0 <= result <= 1.0
        # avg = (1+5+5+5)/4 = 4.0; normalized = (4.0 - 1) / 4 = 0.75
        assert result == pytest.approx(0.75)

    def test_boundary_value_5(self):
        """[IBR-FLV-02] Single 5 among 1s still in range."""
        scores = {"accuracy": 5, "completeness": 1, "clarity": 1, "relevance": 1}
        result = _normalize_scores(scores)
        assert 0.0 <= result <= 1.0
        # avg = (5+1+1+1)/4 = 2.0; normalized = (2.0 - 1) / 4 = 0.25
        assert result == pytest.approx(0.25)


# ===========================================================================
# Judge — _score_from_raw
# ===========================================================================


class TestScoreFromRaw:
    """[IBR-FLV-02] End-to-end raw output to normalized score."""

    def test_valid_raw_returns_normalized(self):
        """[IBR-FLV-02] Valid raw output returns proper normalized score."""
        raw = '{"accuracy": 5, "completeness": 5, "clarity": 5, "relevance": 5}'
        assert _score_from_raw(raw, "test-001") == pytest.approx(1.0)

    def test_unparseable_raw_returns_fallback(self):
        """[IBR-FLV-02] Unparseable raw returns 0.5 fallback."""
        assert _score_from_raw("garbage", "test-002") == pytest.approx(0.5)

    def test_empty_raw_returns_fallback(self):
        """[IBR-FLV-02] Empty raw string returns 0.5 fallback."""
        assert _score_from_raw("", "test-003") == pytest.approx(0.5)


# ===========================================================================
# Judge — judge_response (async)
# ===========================================================================


class TestJudgeResponse:
    """[IBR-FLV-02] Full judge pipeline with mocked adapter."""

    async def test_judge_sends_correct_messages(self):
        """[IBR-FLV-02] judge_response sends system+user messages to adapter."""
        captured: list[list[dict]] = []
        raw_score = '{"accuracy": 4, "completeness": 4, "clarity": 4, "relevance": 4}'
        adapter = MagicMock(spec=GenerativeBackend)

        async def _generate(messages, **kwargs):
            captured.append(messages)
            yield raw_score

        adapter.generate = _generate

        prompt = _make_eval_prompt()
        result = await judge_response(prompt, "some model response", adapter)

        assert len(captured) == 1
        msgs = captured[0]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert prompt.prompt in msgs[1]["content"]
        assert prompt.judge_criteria in msgs[1]["content"]
        assert 0.0 <= result <= 1.0

    async def test_judge_returns_normalized_score(self):
        """[IBR-FLV-02] judge_response returns properly normalized score."""
        raw_score = '{"accuracy": 5, "completeness": 5, "clarity": 5, "relevance": 5}'
        adapter = _make_mock_adapter(raw_score)
        prompt = _make_eval_prompt()

        result = await judge_response(prompt, "great response", adapter)
        assert result == pytest.approx(1.0)

    async def test_judge_adapter_failure_returns_fallback(self):
        """[IBR-FLV-02] Adapter failure returns 0.5 neutral fallback."""
        adapter = _make_raising_adapter(RuntimeError("connection lost"))
        prompt = _make_eval_prompt()

        result = await judge_response(prompt, "some response", adapter)
        assert result == pytest.approx(0.5)

    async def test_judge_empty_model_response_returns_zero(self):
        """[IBR-FLV-02] Empty model response scores 0.0."""
        adapter = _make_mock_adapter("anything")
        prompt = _make_eval_prompt()

        result = await judge_response(prompt, "", adapter)
        assert result == pytest.approx(0.0)

    async def test_judge_whitespace_only_response_returns_zero(self):
        """[IBR-FLV-02] Whitespace-only model response scores 0.0."""
        adapter = _make_mock_adapter("anything")
        prompt = _make_eval_prompt()

        result = await judge_response(prompt, "   \n\t  ", adapter)
        assert result == pytest.approx(0.0)

    async def test_judge_bad_parse_returns_fallback(self):
        """[IBR-FLV-02] Bad JSON from judge returns 0.5 fallback."""
        adapter = _make_mock_adapter("I cannot score this properly")
        prompt = _make_eval_prompt()

        result = await judge_response(prompt, "model response text", adapter)
        assert result == pytest.approx(0.5)


# ===========================================================================
# Judge — _build_judge_messages
# ===========================================================================


class TestBuildJudgeMessages:
    """[IBR-FLV-02] Judge message construction."""

    def test_messages_structure(self):
        """[IBR-FLV-02] Returns system + user message pair."""
        prompt = _make_eval_prompt()
        messages = _build_judge_messages(prompt, "test response")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_prompt_fields(self):
        """[IBR-FLV-02] User message includes prompt text, criteria, quality_speed."""
        prompt = _make_eval_prompt(
            prompt="Write a fizzbuzz.",
            judge_criteria="Correctness of loop logic.",
            quality_speed="speed",
        )
        messages = _build_judge_messages(prompt, "def fizzbuzz(): pass")
        user_content = messages[1]["content"]

        assert "Write a fizzbuzz." in user_content
        assert "Correctness of loop logic." in user_content
        assert "speed" in user_content
        assert "def fizzbuzz(): pass" in user_content


# ===========================================================================
# Runner — _decay_single_score
# ===========================================================================


class TestDecaySingleScore:
    """[IBR-FLV-06] Single-score decay toward 0.5."""

    def test_score_above_half_decays_down(self):
        """[IBR-FLV-06] Score > 0.5 decays downward toward 0.5."""
        result = _decay_single_score(0.9, decay_days=10)
        assert result < 0.9
        assert result >= 0.5

    def test_score_below_half_decays_up(self):
        """[IBR-FLV-06] Score < 0.5 decays upward toward 0.5."""
        result = _decay_single_score(0.1, decay_days=10)
        assert result > 0.1
        assert result <= 0.5

    def test_score_at_half_stays(self):
        """[IBR-FLV-06] Score at exactly 0.5 does not move."""
        result = _decay_single_score(0.5, decay_days=10)
        assert result == pytest.approx(0.5)

    def test_small_decay(self):
        """[IBR-FLV-06] 1 day of decay applies 0.01 shift."""
        result = _decay_single_score(0.8, decay_days=1)
        # decay_amount = min(|0.8 - 0.5|, 1 * 0.01) = 0.01
        assert result == pytest.approx(0.79)

    def test_large_decay_capped_at_distance(self):
        """[IBR-FLV-06] Decay amount capped at distance from 0.5."""
        # Score is 0.52, distance to 0.5 is 0.02.
        # 100 days * 0.01/day = 1.0 >> 0.02, so decay_amount = 0.02.
        result = _decay_single_score(0.52, decay_days=100)
        assert result == pytest.approx(0.5)

    def test_result_clamped_to_unit_interval(self):
        """[IBR-FLV-06] Result always in [0.0, 1.0]."""
        assert 0.0 <= _decay_single_score(0.0, decay_days=1) <= 1.0
        assert 0.0 <= _decay_single_score(1.0, decay_days=1) <= 1.0

    def test_invalid_score_raises(self):
        """[IBR-FLV-06] Score outside [0,1] raises AssertionError."""
        with pytest.raises(AssertionError):
            _decay_single_score(1.5, decay_days=1)
        with pytest.raises(AssertionError):
            _decay_single_score(-0.1, decay_days=1)


# ===========================================================================
# Runner — apply_decay
# ===========================================================================


class TestApplyDecay:
    """[IBR-FLV-06] Profile-level time-based decay."""

    def test_profile_within_30_days_no_decay(self):
        """[IBR-FLV-06] Profile younger than 30 days is unchanged."""
        now = datetime.now(UTC)
        profile = _make_profile(updated_at=(now - timedelta(days=15)).isoformat())

        result = apply_decay(profile, now=now)
        # Should be the exact same object (no mutation)
        assert result is profile

    def test_profile_at_exactly_30_days_no_decay(self):
        """[IBR-FLV-06] Profile at exactly 30 days boundary is unchanged."""
        now = datetime.now(UTC)
        profile = _make_profile(updated_at=(now - timedelta(days=30)).isoformat())

        result = apply_decay(profile, now=now)
        assert result is profile

    def test_profile_at_31_days_small_decay(self):
        """[IBR-FLV-06] Profile at 31 days gets small decay (1 day past threshold)."""
        now = datetime.now(UTC)
        profile = _make_profile(
            updated_at=(now - timedelta(days=31)).isoformat(),
            score=0.8,
        )

        result = apply_decay(profile, now=now)
        assert result is not profile  # New object returned

        # Scores should have decayed slightly toward 0.5
        for fs in result.task_scores.values():
            assert fs.score < 0.8
            assert fs.score >= 0.5

    def test_profile_at_60_days_larger_decay(self):
        """[IBR-FLV-06] Profile at 60 days gets larger decay (30 days past threshold)."""
        now = datetime.now(UTC)
        profile = _make_profile(
            updated_at=(now - timedelta(days=60)).isoformat(),
            score=0.8,
            confidence=0.6,
        )

        result = apply_decay(profile, now=now)

        # 30 days of decay at 0.01/day = 0.30 total.
        # score 0.8 is 0.3 from 0.5, so it should decay to 0.5.
        for fs in result.task_scores.values():
            assert fs.score == pytest.approx(0.5)

    def test_decay_confidence_also_decays(self):
        """[IBR-FLV-06] Confidence decays alongside score."""
        now = datetime.now(UTC)
        profile = _make_profile(
            updated_at=(now - timedelta(days=40)).isoformat(),
            confidence=0.5,
        )

        result = apply_decay(profile, now=now)
        # 10 days past threshold, confidence -= 10 * 0.01 = 0.1
        for fs in result.task_scores.values():
            assert fs.confidence == pytest.approx(0.4)

    def test_decay_confidence_floors_at_zero(self):
        """[IBR-FLV-06] Confidence does not go below 0.0."""
        now = datetime.now(UTC)
        profile = _make_profile(
            updated_at=(now - timedelta(days=200)).isoformat(),
            confidence=0.1,
        )

        result = apply_decay(profile, now=now)
        for fs in result.task_scores.values():
            assert fs.confidence >= 0.0

    def test_decay_preserves_model_id_and_timestamp(self):
        """[IBR-FLV-06] Decayed profile keeps original model_id and updated_at."""
        now = datetime.now(UTC)
        ts = (now - timedelta(days=40)).isoformat()
        profile = _make_profile(model_id="my-model", updated_at=ts)

        result = apply_decay(profile, now=now)
        assert result.model_id == "my-model"
        assert result.updated_at == ts

    def test_decay_preserves_sample_count(self):
        """[IBR-FLV-06] Decay does not change sample_count."""
        now = datetime.now(UTC)
        profile = _make_profile(
            updated_at=(now - timedelta(days=40)).isoformat(),
            sample_count=42,
        )

        result = apply_decay(profile, now=now)
        for fs in result.task_scores.values():
            assert fs.sample_count == 42


# ===========================================================================
# Runner — _collect_model_response
# ===========================================================================


class TestCollectModelResponse:
    """[IBR-FLV-03] Response collection from model adapters."""

    async def test_successful_response(self):
        """[IBR-FLV-03] Adapter returning chunks joins them."""
        adapter = _make_mock_adapter("Hello world")
        prompt = _make_eval_prompt()

        result = await _collect_model_response(adapter, prompt)
        assert result == "Hello world"

    async def test_multi_chunk_response(self):
        """[IBR-FLV-03] Multiple chunks are concatenated."""
        adapter = MagicMock(spec=GenerativeBackend)

        async def _generate(*args, **kwargs):
            yield "chunk1"
            yield " "
            yield "chunk2"

        adapter.generate = _generate
        prompt = _make_eval_prompt()

        result = await _collect_model_response(adapter, prompt)
        assert result == "chunk1 chunk2"

    async def test_adapter_failure_returns_empty_string(self):
        """[IBR-FLV-03] Adapter exception returns empty string."""
        adapter = _make_raising_adapter(RuntimeError("network error"))
        prompt = _make_eval_prompt()

        result = await _collect_model_response(adapter, prompt)
        assert result == ""


# ===========================================================================
# Runner — _aggregate_scores
# ===========================================================================


class TestAggregateScores:
    """[IBR-FLV-04] Score aggregation from per-prompt scores."""

    def test_empty_list(self):
        """[IBR-FLV-04] Empty scored_prompts produces all-neutral profile."""
        profile = _aggregate_scores([])

        for key in IBR_TASK_TYPES:
            assert key in profile.task_scores
            assert profile.task_scores[key].score == IBR_NEUTRAL_SPECTROGRAPH.score
        for key in IBR_DOMAINS:
            assert key in profile.domain_scores
        for key in IBR_QUALITY_SPEED:
            assert key in profile.qs_scores

    def test_single_prompt(self):
        """[IBR-FLV-04] Single prompt populates its dimensions correctly."""
        prompt = _make_eval_prompt(
            task_type="generation",
            domain="code",
            quality_speed="quality",
        )
        scored = [(prompt, 0.9)]
        profile = _aggregate_scores(scored)

        assert profile.task_scores["generation"].score == pytest.approx(0.9)
        assert profile.task_scores["generation"].sample_count == 1
        assert profile.domain_scores["code"].score == pytest.approx(0.9)
        assert profile.qs_scores["quality"].score == pytest.approx(0.9)

        # Other dimensions should be neutral
        assert profile.task_scores["analysis"].score == IBR_NEUTRAL_SPECTROGRAPH.score

    def test_multiple_prompts_same_dimension(self):
        """[IBR-FLV-04] Multiple prompts in same dimension are averaged."""
        p1 = _make_eval_prompt(
            id="p1",
            task_type="analysis",
            domain="code",
            quality_speed="quality",
        )
        p2 = _make_eval_prompt(
            id="p2",
            task_type="analysis",
            domain="code",
            quality_speed="quality",
        )

        scored = [(p1, 0.8), (p2, 0.6)]
        profile = _aggregate_scores(scored)

        assert profile.task_scores["analysis"].score == pytest.approx(0.7)
        assert profile.task_scores["analysis"].sample_count == 2
        assert profile.domain_scores["code"].score == pytest.approx(0.7)
        assert profile.domain_scores["code"].sample_count == 2

    def test_prompts_across_dimensions(self):
        """[IBR-FLV-04] Prompts across different dimensions populate independently."""
        p1 = _make_eval_prompt(
            id="p1",
            task_type="generation",
            domain="code",
            quality_speed="quality",
        )
        p2 = _make_eval_prompt(
            id="p2",
            task_type="analysis",
            domain="business",
            quality_speed="speed",
        )

        scored = [(p1, 0.9), (p2, 0.3)]
        profile = _aggregate_scores(scored)

        assert profile.task_scores["generation"].score == pytest.approx(0.9)
        assert profile.task_scores["analysis"].score == pytest.approx(0.3)
        assert profile.domain_scores["code"].score == pytest.approx(0.9)
        assert profile.domain_scores["business"].score == pytest.approx(0.3)
        assert profile.qs_scores["quality"].score == pytest.approx(0.9)
        assert profile.qs_scores["speed"].score == pytest.approx(0.3)

    def test_model_id_is_placeholder(self):
        """[IBR-FLV-04] Aggregated profile has empty model_id placeholder."""
        profile = _aggregate_scores([])
        assert profile.model_id == ""

    def test_confidence_scales_with_sample_count(self):
        """[IBR-FLV-04] Confidence increases with more samples (capped at 1.0)."""
        prompts_and_scores = []
        for i in range(25):
            p = _make_eval_prompt(
                id=f"p{i}",
                task_type="generation",
                domain="code",
                quality_speed="quality",
            )
            prompts_and_scores.append((p, 0.7))

        profile = _aggregate_scores(prompts_and_scores)
        # 25 samples / 50 = 0.5 confidence
        assert profile.task_scores["generation"].confidence == pytest.approx(0.5)


# ===========================================================================
# Runner — _build_flavor_scores
# ===========================================================================


class TestBuildSpectrographScores:
    """[IBR-FLV-04] SpectrographScore construction from accumulated value lists."""

    def test_empty_list_produces_neutral(self):
        """[IBR-FLV-04] Empty list for a key produces IBR_NEUTRAL_SPECTROGRAPH."""
        accum = {"generation": []}
        result = _build_flavor_scores(accum)
        assert result["generation"].score == IBR_NEUTRAL_SPECTROGRAPH.score
        assert result["generation"].confidence == IBR_NEUTRAL_SPECTROGRAPH.confidence
        assert result["generation"].sample_count == IBR_NEUTRAL_SPECTROGRAPH.sample_count

    def test_single_value(self):
        """[IBR-FLV-04] Single value computes correct score and confidence."""
        accum = {"code": [0.8]}
        result = _build_flavor_scores(accum)
        assert result["code"].score == pytest.approx(0.8)
        assert result["code"].sample_count == 1
        # confidence = min(1.0, 1/50) = 0.02
        assert result["code"].confidence == pytest.approx(0.02)

    def test_multiple_values_averaged(self):
        """[IBR-FLV-04] Multiple values are averaged."""
        accum = {"quality": [0.6, 0.8, 1.0]}
        result = _build_flavor_scores(accum)
        assert result["quality"].score == pytest.approx(0.8)
        assert result["quality"].sample_count == 3
        # confidence = min(1.0, 3/50) = 0.06
        assert result["quality"].confidence == pytest.approx(0.06)

    def test_fifty_samples_full_confidence(self):
        """[IBR-FLV-04] 50+ samples produce confidence = 1.0."""
        accum = {"analysis": [0.5] * 50}
        result = _build_flavor_scores(accum)
        assert result["analysis"].confidence == pytest.approx(1.0)

    def test_score_clamped_to_unit_interval(self):
        """[IBR-FLV-04] Averaged scores are clamped to [0.0, 1.0]."""
        accum = {"key": [0.0, 0.0]}
        result = _build_flavor_scores(accum)
        assert 0.0 <= result["key"].score <= 1.0


# ===========================================================================
# Runner — _finalize_profile
# ===========================================================================


class TestFinalizeProfile:
    """[IBR-FLV-04] Profile finalization with model_id replacement."""

    def test_replaces_model_id(self):
        """[IBR-FLV-04] Finalize replaces empty model_id with actual value."""
        profile = _aggregate_scores([])
        assert profile.model_id == ""

        finalized = _finalize_profile("claude-sonnet-4-20250514", profile)
        assert finalized.model_id == "claude-sonnet-4-20250514"

    def test_preserves_scores(self):
        """[IBR-FLV-04] Finalize preserves all score dimensions."""
        p = _make_eval_prompt(task_type="generation", domain="code", quality_speed="quality")
        profile = _aggregate_scores([(p, 0.9)])

        finalized = _finalize_profile("test-model", profile)
        assert finalized.task_scores["generation"].score == pytest.approx(0.9)
        assert finalized.domain_scores["code"].score == pytest.approx(0.9)
        assert finalized.qs_scores["quality"].score == pytest.approx(0.9)

    def test_preserves_version_and_timestamp(self):
        """[IBR-FLV-04] Finalize preserves version and updated_at."""
        profile = _aggregate_scores([])
        finalized = _finalize_profile("test-model", profile)

        assert finalized.version == profile.version
        assert finalized.updated_at == profile.updated_at

    def test_empty_model_id_raises(self):
        """[IBR-FLV-04] Empty model_id raises AssertionError."""
        profile = _aggregate_scores([])
        with pytest.raises(AssertionError):
            _finalize_profile("", profile)


# ===========================================================================
# Runner — BenchmarkRunner initialization and mock pipeline
# ===========================================================================


class TestBenchmarkRunner:
    """[IBR-FLV-03] BenchmarkRunner orchestration."""

    def test_initialization(self):
        """[IBR-FLV-03] Runner initializes with prompts, judge, and output path."""
        prompts = [_make_eval_prompt()]
        judge = _make_mock_adapter("judge response")
        output = Path("/tmp/test_benchmark_output")

        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge,
            output_path=output,
        )
        assert runner._prompts == prompts
        assert runner._output_path == output

    def test_initialization_empty_prompts_raises(self):
        """[IBR-FLV-03] Runner rejects empty prompt list."""
        judge = _make_mock_adapter("judge response")
        with pytest.raises(AssertionError, match="must not be empty"):
            BenchmarkRunner(
                eval_prompts=[],
                judge_adapter=judge,
                output_path=Path("/tmp/test"),
            )

    async def test_benchmark_model_with_mocked_pipeline(self):
        """[IBR-FLV-03] benchmark_model returns valid profile with mocked adapter+judge."""
        model_response = "Here is a well-written function..."
        judge_score = '{"accuracy": 4, "completeness": 4, "clarity": 4, "relevance": 4}'

        model_adapter = _make_mock_adapter(model_response)
        judge_adapter = _make_mock_adapter(judge_score)

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
            output_path=Path("/tmp/test_benchmark"),
        )

        profile = await runner.benchmark_model("test-model", model_adapter)

        assert isinstance(profile, ModelSpectrographProfile)
        assert profile.model_id == "test-model"
        assert profile.version == 1
        assert profile.updated_at  # Non-empty timestamp

        # Both task_types should have scores
        assert profile.task_scores["generation"].score > 0.0
        assert profile.task_scores["analysis"].score > 0.0

    async def test_benchmark_model_all_dimensions_populated(self):
        """[IBR-FLV-03] Returned profile has all IBR dimension keys."""
        model_adapter = _make_mock_adapter("response text")
        judge_score = '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        judge_adapter = _make_mock_adapter(judge_score)

        runner = BenchmarkRunner(
            eval_prompts=[_make_eval_prompt()],
            judge_adapter=judge_adapter,
            output_path=Path("/tmp/test_benchmark"),
        )

        profile = await runner.benchmark_model("my-model", model_adapter)

        for tt in IBR_TASK_TYPES:
            assert tt in profile.task_scores, f"Missing task_type: {tt}"
        for d in IBR_DOMAINS:
            assert d in profile.domain_scores, f"Missing domain: {d}"
        for qs in IBR_QUALITY_SPEED:
            assert qs in profile.qs_scores, f"Missing quality_speed: {qs}"


# ===========================================================================
# Integration sanity — full pipeline mock
# ===========================================================================


class TestIntegrationPipeline:
    """[IBR-FLV-05] Full pipeline mock: adapter -> judge -> profile."""

    async def test_full_pipeline_produces_valid_profile(self):
        """[IBR-FLV-05] Mocked pipeline produces valid ModelSpectrographProfile with correct dims."""
        # Model adapter returns consistent text
        model_adapter = _make_mock_adapter("This is a comprehensive response to the prompt.")

        # Judge adapter returns valid scores
        judge_scores = '{"accuracy": 4, "completeness": 3, "clarity": 5, "relevance": 4}'
        judge_adapter = _make_mock_adapter(judge_scores)

        # Use a representative subset of prompts
        prompts = [
            _make_eval_prompt(
                id="int-gen-code",
                task_type="generation",
                domain="code",
                quality_speed="quality",
            ),
            _make_eval_prompt(
                id="int-analysis-tech",
                task_type="analysis",
                domain="technical",
                quality_speed="balanced",
            ),
            _make_eval_prompt(
                id="int-creative-writing",
                task_type="creative",
                domain="creative_writing",
                quality_speed="speed",
            ),
        ]

        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=Path("/tmp/test_integration_benchmark"),
        )

        profile = await runner.benchmark_model("integration-test-model", model_adapter)

        # Structural assertions
        assert isinstance(profile, ModelSpectrographProfile)
        assert profile.model_id == "integration-test-model"
        assert profile.version == 1
        assert profile.updated_at  # Non-empty

        # All dimension keys present
        assert set(profile.task_scores.keys()) == IBR_TASK_TYPES
        assert set(profile.domain_scores.keys()) == IBR_DOMAINS
        assert set(profile.qs_scores.keys()) == IBR_QUALITY_SPEED

        # Scored dimensions have valid SpectrographScore values
        for dim_scores in (profile.task_scores, profile.domain_scores, profile.qs_scores):
            for _key, fs in dim_scores.items():
                assert isinstance(fs, SpectrographScore)
                assert 0.0 <= fs.score <= 1.0
                assert 0.0 <= fs.confidence <= 1.0
                assert fs.sample_count >= 0

        # Dimensions that received data should have non-neutral scores
        # Judge returns avg = (4+3+5+4)/4=4.0, normalized = 3.0/4.0 = 0.75
        expected_score = pytest.approx(0.75)
        assert profile.task_scores["generation"].score == expected_score
        assert profile.task_scores["analysis"].score == expected_score
        assert profile.task_scores["creative"].score == expected_score

        # Dimensions that got no data should be neutral
        assert profile.task_scores["refactoring"].score == IBR_NEUTRAL_SPECTROGRAPH.score

    async def test_pipeline_with_adapter_failures(self):
        """[IBR-FLV-05] Pipeline gracefully handles adapter failures."""
        # Model adapter fails — returns empty string via the raising path
        model_adapter = _make_raising_adapter(RuntimeError("API down"))

        # Judge still returns valid scores (but won't be called meaningfully
        # because model response is empty)
        judge_adapter = _make_mock_adapter(
            '{"accuracy": 3, "completeness": 3, "clarity": 3, "relevance": 3}'
        )

        prompts = [
            _make_eval_prompt(
                id="fail-test", task_type="generation", domain="code", quality_speed="quality"
            ),
        ]

        runner = BenchmarkRunner(
            eval_prompts=prompts,
            judge_adapter=judge_adapter,
            output_path=Path("/tmp/test_failure_benchmark"),
        )

        profile = await runner.benchmark_model("failing-model", model_adapter)

        # Should still get a valid profile structure
        assert isinstance(profile, ModelSpectrographProfile)
        assert profile.model_id == "failing-model"

        # The generation score should be 0.0 (empty response from adapter failure)
        assert profile.task_scores["generation"].score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coverage for get_prompts_by_task_type and get_prompts_by_domain (lines 983-990)
# ---------------------------------------------------------------------------


class TestGetPromptsByTaskType:
    """[IBR-FLV-05] get_prompts_by_task_type filters prompts correctly."""

    def test_valid_task_type_returns_filtered_list(self):
        """get_prompts_by_task_type returns only prompts matching the task_type."""
        from dragonlight_router.benchmark.prompts import get_prompts_by_task_type
        from dragonlight_router.core.types import IBR_TASK_TYPES

        task_type = sorted(IBR_TASK_TYPES)[0]  # Pick one deterministically
        result = get_prompts_by_task_type(task_type)
        assert isinstance(result, list)
        assert len(result) > 0
        for p in result:
            assert p.task_type == task_type

    def test_invalid_task_type_raises_assertion(self):
        """get_prompts_by_task_type raises AssertionError on invalid task_type."""
        from dragonlight_router.benchmark.prompts import get_prompts_by_task_type

        with pytest.raises(AssertionError, match="Invalid task_type"):
            get_prompts_by_task_type("nonexistent_task_type")


class TestGetPromptsByDomain:
    """[IBR-FLV-05] get_prompts_by_domain filters prompts correctly."""

    def test_valid_domain_returns_filtered_list(self):
        """get_prompts_by_domain returns only prompts matching the domain."""
        from dragonlight_router.benchmark.prompts import get_prompts_by_domain
        from dragonlight_router.core.types import IBR_DOMAINS

        domain = sorted(IBR_DOMAINS)[0]  # Pick one deterministically
        result = get_prompts_by_domain(domain)
        assert isinstance(result, list)
        assert len(result) > 0
        for p in result:
            assert p.domain == domain

    def test_invalid_domain_raises_assertion(self):
        """get_prompts_by_domain raises AssertionError on invalid domain."""
        from dragonlight_router.benchmark.prompts import get_prompts_by_domain

        with pytest.raises(AssertionError, match="Invalid domain"):
            get_prompts_by_domain("nonexistent_domain")
