"""LLM-as-judge scorer for IBR automated benchmarking.

Uses a reference model (the judge) to evaluate another model's response
against structured criteria. Returns normalized scores (0.0-1.0).

Spec reference: intent-based-router-v0.1.0-spec.md section 3.2, Method 3.
"""

from __future__ import annotations

import json
import re

import structlog

from dragonlight_router.benchmark.prompts import EvalPrompt
from dragonlight_router.core.types import GenerativeBackend

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Judge prompt template — structured for consistent scoring
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT: str = (
    "You are an expert evaluator. Score the AI response below on a 1-5 scale "
    "for each criterion. Be strict and consistent. Return ONLY valid JSON."
)

_JUDGE_USER_TEMPLATE: str = """\
## Original Prompt
{original_prompt}

## Evaluation Criteria
{judge_criteria}

## Quality Context
This response should be optimized for: {quality_speed}

## AI Response to Evaluate
{model_response}

## Scoring Rubric
Rate each dimension 1-5:
- **accuracy** (1=wrong/misleading, 3=mostly correct with gaps, 5=fully correct and precise)
- **completeness** (1=missing key points, 3=covers basics, 5=thorough coverage)
- **clarity** (1=confusing/poorly structured, 3=understandable, 5=clear and well-organized)
- **relevance** (1=off-topic, 3=addresses the question, 5=precisely targeted)

Return ONLY this JSON (no other text):
{{"accuracy": <int>, "completeness": <int>, "clarity": <int>, "relevance": <int>}}"""


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def _parse_judge_scores(raw: str) -> dict[str, int] | None:
    """Extract scoring dict from judge response. Returns None on parse failure.

    Handles common LLM response patterns: bare JSON, JSON in code fences,
    or JSON embedded in explanatory text.
    """
    assert isinstance(raw, str), "raw must be a string"

    # Try direct JSON parse first
    text = raw.strip()
    parsed = _try_json_parse(text)
    if parsed is not None:
        return parsed

    # Try extracting from code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        parsed = _try_json_parse(fence_match.group(1))
        if parsed is not None:
            return parsed

    # Try finding first JSON object in text
    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        parsed = _try_json_parse(brace_match.group(0))
        if parsed is not None:
            return parsed

    logger.warning("judge_parse_failed", raw_length=len(raw))
    return None


def _try_json_parse(text: str) -> dict[str, int] | None:
    """Attempt to parse text as JSON scoring dict."""
    assert isinstance(text, str), "text must be a string"
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    required_keys = {"accuracy", "completeness", "clarity", "relevance"}
    if not required_keys.issubset(data.keys()):
        return None

    result: dict[str, int] = {}
    for key in required_keys:
        value = data[key]
        if not isinstance(value, (int, float)):
            return None
        clamped = max(1, min(5, int(value)))
        result[key] = clamped

    assert len(result) == 4, f"Expected 4 scores, got {len(result)}"
    return result


def _normalize_scores(scores: dict[str, int]) -> float:
    """Convert 1-5 integer scores to a single 0.0-1.0 normalized score.

    Averages all dimensions, then maps from [1,5] to [0.0,1.0].
    """
    assert len(scores) == 4, f"Expected 4 scores, got {len(scores)}"
    assert all(1 <= v <= 5 for v in scores.values()), "All scores must be 1-5"

    avg = sum(scores.values()) / len(scores)
    normalized = (avg - 1.0) / 4.0  # Map [1,5] -> [0.0,1.0]

    assert 0.0 <= normalized <= 1.0, f"Normalized score out of range: {normalized}"
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_judge_messages(
    prompt: EvalPrompt,
    model_response: str,
) -> list[dict[str, str]]:
    """Build the system+user message pair for the judge call."""
    assert isinstance(prompt, EvalPrompt), "prompt must be an EvalPrompt"
    assert isinstance(model_response, str), "model_response must be a string"

    user_message = _JUDGE_USER_TEMPLATE.format(
        original_prompt=prompt.prompt,
        judge_criteria=prompt.judge_criteria,
        quality_speed=prompt.quality_speed,
        model_response=model_response,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _score_from_raw(raw: str, prompt_id: str) -> float:
    """Parse and normalize raw judge output. Returns 0.5 on parse failure."""
    assert isinstance(raw, str), "raw must be a string"

    scores = _parse_judge_scores(raw)
    if scores is None:
        logger.warning("judge_score_parse_failed", prompt_id=prompt_id, raw_length=len(raw))
        return 0.5

    normalized = _normalize_scores(scores)
    logger.debug(
        "judge_scored",
        prompt_id=prompt_id,
        raw_scores=scores,
        normalized=round(normalized, 4),
    )
    return normalized


async def judge_response(
    prompt: EvalPrompt,
    model_response: str,
    judge_adapter: GenerativeBackend,
) -> float:
    """Use a reference model to evaluate another model's response.

    Returns a normalized score in [0.0, 1.0].
    Returns 0.5 (neutral) on judge failure to prevent benchmark corruption.
    """
    assert isinstance(prompt, EvalPrompt), "prompt must be an EvalPrompt"
    assert isinstance(model_response, str), "model_response must be a string"

    if not model_response.strip():
        logger.warning("judge_empty_response", prompt_id=prompt.id)
        return 0.0

    messages = _build_judge_messages(prompt, model_response)
    raw_response = await _collect_response(judge_adapter, messages)

    if raw_response is None:
        logger.warning("judge_call_failed", prompt_id=prompt.id)
        return 0.5

    return _score_from_raw(raw_response, prompt.id)


async def _collect_response(
    adapter: GenerativeBackend,
    messages: list[dict[str, str]],
) -> str | None:
    """Collect full text from a streaming adapter. Returns None on error."""
    assert isinstance(messages, list), "messages must be a list"
    assert len(messages) > 0, "messages must not be empty"

    try:
        chunks: list[str] = []
        async for chunk in adapter.generate(
            messages,
            max_tokens=256,
            temperature=0.0,
            stream=True,
        ):
            chunks.append(chunk)
        return "".join(chunks) if chunks else None
    except (RuntimeError, ValueError) as exc:
        logger.warning("judge_adapter_error", error=str(exc))
        return None
