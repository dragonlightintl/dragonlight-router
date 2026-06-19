"""Tests for selection/classifier.py — intent classification engine.

Spec traceability: IBR spec v0.1.0 sections 2, 10.
AC numbers: IBR-CLS-01 through IBR-CLS-07, IBR-SYS-03.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from dragonlight_router.core.types import (
    ClassifiedIntent,
    GenerativeBackend,
)
from dragonlight_router.selection.classifier import (
    DOMAINS,
    QUALITY_SPEED,
    TASK_TYPES,
    _ClassificationCache,
    _compute_cache_key,
    _parse_response,
    _validate_classification,
    classify_intent,
    configure_cache,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — mock adapter factory
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
        yield  # make it an async generator  # noqa: RET503

    adapter.generate = _generate
    return adapter


def _make_slow_adapter(
    delay_s: float,
    response_text: str,
) -> MagicMock:
    """Build a mock adapter that sleeps before yielding."""
    adapter = MagicMock(spec=GenerativeBackend)

    async def _generate(*args, **kwargs):
        await asyncio.sleep(delay_s)
        yield response_text

    adapter.generate = _generate
    return adapter


def _valid_json(**overrides) -> str:
    """Build a valid classification JSON string."""
    payload = {
        "task_type": "analysis",
        "domain": "code",
        "quality_speed": "balanced",
        "confidence": 0.85,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _raw(
    task_type: str = "analysis",
    domain: str = "code",
    quality_speed: str = "balanced",
    confidence: float | str | None = 0.9,
    **extra: object,
) -> dict:
    """Build a raw classification dict for _validate_classification."""
    d: dict = {}
    if task_type is not None:
        d["task_type"] = task_type
    if domain is not None:
        d["domain"] = domain
    if quality_speed is not None:
        d["quality_speed"] = quality_speed
    if confidence is not None:
        d["confidence"] = confidence
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Taxonomy validation
# ---------------------------------------------------------------------------


class TestTaxonomyValidation:
    """[IBR-CLS-05] Taxonomy values validated against allowed sets."""

    def test_all_8_task_types_accepted(self):
        """[IBR-CLS-05] All 8 task_type values pass validation."""
        expected = {
            "generation",
            "analysis",
            "refactoring",
            "summarization",
            "creative",
            "reasoning",
            "lookup",
            "translation",
        }
        assert expected == TASK_TYPES
        for tt in expected:
            raw = _raw(task_type=tt)
            assert _validate_classification(raw) is True

    def test_all_6_domains_accepted(self):
        """[IBR-CLS-05] All 6 domain values pass validation."""
        expected = {
            "code",
            "technical",
            "legal",
            "business",
            "creative_writing",
            "general",
        }
        assert expected == DOMAINS
        for d in expected:
            raw = _raw(domain=d)
            assert _validate_classification(raw) is True

    def test_all_3_quality_speed_accepted(self):
        """[IBR-CLS-05] All 3 quality_speed values pass validation."""
        expected = {"quality", "balanced", "speed"}
        assert expected == QUALITY_SPEED
        for qs in expected:
            raw = _raw(quality_speed=qs)
            assert _validate_classification(raw) is True

    def test_invalid_task_type_rejected(self):
        """[IBR-CLS-05] Invalid task_type returns False."""
        raw = _raw(task_type="invalid_type")
        assert _validate_classification(raw) is False

    def test_invalid_domain_rejected(self):
        """[IBR-CLS-05] Invalid domain returns False."""
        raw = _raw(domain="cooking")
        assert _validate_classification(raw) is False

    def test_invalid_quality_speed_rejected(self):
        """[IBR-CLS-05] Invalid quality_speed returns False."""
        raw = _raw(quality_speed="turbo")
        assert _validate_classification(raw) is False

    def test_empty_string_task_type_rejected(self):
        """[IBR-CLS-05] Empty string task_type returns False."""
        raw = _raw(task_type="")
        assert _validate_classification(raw) is False

    def test_none_task_type_rejected(self):
        """[IBR-CLS-05] None task_type returns False."""
        raw = {
            "task_type": None,
            "domain": "code",
            "quality_speed": "balanced",
            "confidence": 0.9,
        }
        assert _validate_classification(raw) is False

    def test_missing_task_type_rejected(self):
        """[IBR-CLS-05] Missing task_type key returns False."""
        raw = {
            "domain": "code",
            "quality_speed": "balanced",
            "confidence": 0.9,
        }
        assert _validate_classification(raw) is False

    def test_confidence_below_zero_rejected(self):
        """[IBR-CLS-05] Confidence < 0 returns False."""
        raw = _raw(confidence=-0.1)
        assert _validate_classification(raw) is False

    def test_confidence_above_one_rejected(self):
        """[IBR-CLS-05] Confidence > 1 returns False."""
        raw = _raw(confidence=1.1)
        assert _validate_classification(raw) is False

    def test_confidence_zero_accepted(self):
        """[IBR-CLS-05] Confidence = 0.0 is valid."""
        raw = _raw(confidence=0.0)
        assert _validate_classification(raw) is True

    def test_confidence_one_accepted(self):
        """[IBR-CLS-05] Confidence = 1.0 is valid."""
        raw = _raw(confidence=1.0)
        assert _validate_classification(raw) is True

    def test_non_numeric_confidence_rejected(self):
        """[IBR-CLS-05] Non-numeric confidence returns False."""
        raw = _raw(confidence="high")
        assert _validate_classification(raw) is False

    def test_extra_fields_ignored(self):
        """[IBR-CLS-05] Extra fields in dict do not cause rejection."""
        raw = _raw(extra_field="some_value", another=42)
        assert _validate_classification(raw) is True

    def test_taxonomy_constants_cardinality(self):
        """[IBR-CLS-05] Taxonomy constants have expected cardinality."""
        assert len(TASK_TYPES) == 8
        assert len(DOMAINS) == 6
        assert len(QUALITY_SPEED) == 3


# ---------------------------------------------------------------------------
# Classification caching
# ---------------------------------------------------------------------------


class TestClassificationCache:
    """[IBR-CLS-03] SHA-256 caching with TTL and LRU eviction."""

    def _make_intent(self, **kwargs) -> ClassifiedIntent:
        defaults = {
            "task_type": "analysis",
            "domain": "code",
            "quality_speed": "balanced",
            "confidence": 0.9,
            "latency_ms": 10.0,
            "from_cache": False,
        }
        defaults.update(kwargs)
        return ClassifiedIntent(**defaults)

    def test_cache_key_is_sha256(self):
        """[IBR-CLS-03] Cache key is SHA-256 hex digest."""
        msg = "Write a function to sort a list"
        key = _compute_cache_key(msg)
        expected = hashlib.sha256(msg.encode("utf-8")).hexdigest()
        assert key == expected
        assert len(key) == 64

    def test_cache_hit_returns_intent(self):
        """[IBR-CLS-03] Cache get after put returns stored intent."""
        cache = _ClassificationCache(max_entries=10, ttl_s=300.0)
        intent = self._make_intent()
        key = _compute_cache_key("test message")
        cache.put(key, intent)
        result = cache.get(key)
        assert result is not None
        assert result.task_type == intent.task_type

    def test_cache_miss_returns_none(self):
        """[IBR-CLS-03] Cache get for absent key returns None."""
        cache = _ClassificationCache(max_entries=10, ttl_s=300.0)
        key = _compute_cache_key("nonexistent")
        assert cache.get(key) is None

    def test_cache_ttl_expiry(self):
        """[IBR-CLS-03] Expired entries return None on get."""
        cache = _ClassificationCache(max_entries=10, ttl_s=0.01)
        intent = self._make_intent()
        key = _compute_cache_key("ttl test")
        cache.put(key, intent)
        time.sleep(0.02)
        assert cache.get(key) is None

    def test_cache_lru_eviction(self):
        """[IBR-CLS-03] LRU eviction when max_entries exceeded."""
        cache = _ClassificationCache(max_entries=2, ttl_s=300.0)
        intent = self._make_intent()

        key1 = _compute_cache_key("message1")
        key2 = _compute_cache_key("message2")
        key3 = _compute_cache_key("message3")

        cache.put(key1, intent)
        cache.put(key2, intent)
        cache.put(key3, intent)

        # key1 should have been evicted (oldest)
        assert cache.get(key1) is None
        assert cache.get(key2) is not None
        assert cache.get(key3) is not None

    def test_cache_lru_access_refreshes(self):
        """[IBR-CLS-03] Accessing a key refreshes it, preventing eviction."""
        cache = _ClassificationCache(max_entries=2, ttl_s=300.0)
        intent = self._make_intent()

        key1 = _compute_cache_key("msg1")
        key2 = _compute_cache_key("msg2")
        key3 = _compute_cache_key("msg3")

        cache.put(key1, intent)
        cache.put(key2, intent)

        # Access key1 to refresh it
        cache.get(key1)

        # Now add key3 — key2 should be evicted (LRU)
        cache.put(key3, intent)
        assert cache.get(key1) is not None
        assert cache.get(key2) is None
        assert cache.get(key3) is not None

    def test_cache_len(self):
        """[IBR-CLS-03] Cache __len__ reports correct count."""
        cache = _ClassificationCache(max_entries=10, ttl_s=300.0)
        assert len(cache) == 0
        intent = self._make_intent()
        cache.put(_compute_cache_key("a"), intent)
        assert len(cache) == 1
        cache.put(_compute_cache_key("b"), intent)
        assert len(cache) == 2

    def test_cache_clear(self):
        """[IBR-CLS-03] Cache clear removes all entries."""
        cache = _ClassificationCache(max_entries=10, ttl_s=300.0)
        intent = self._make_intent()
        cache.put(_compute_cache_key("c"), intent)
        cache.clear()
        assert len(cache) == 0

    def test_configure_cache_replaces_singleton(self):
        """[IBR-CLS-03] configure_cache replaces the module cache."""
        configure_cache(max_entries=100, ttl_s=60.0)
        # No exception means success; restore default
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_cache_hit_returns_from_cache_true(self):
        """[IBR-CLS-03] Cache hit returns from_cache=True."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        msg = "analyze this code for performance issues"
        r1 = await classify_intent(msg, adapter, timeout_s=5.0)
        assert r1 is not None
        assert r1.from_cache is False

        r2 = await classify_intent(msg, adapter, timeout_s=5.0)
        assert r2 is not None
        assert r2.from_cache is True
        assert r2.task_type == r1.task_type

        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_cache_miss_calls_adapter(self):
        """[IBR-CLS-03] Cache miss invokes the adapter."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        msg = "unique msg " + str(time.monotonic())
        result = await classify_intent(
            msg,
            adapter,
            timeout_s=5.0,
        )
        assert result is not None
        assert result.from_cache is False

        configure_cache(max_entries=5000, ttl_s=300.0)


# ---------------------------------------------------------------------------
# Timeout behavior
# ---------------------------------------------------------------------------


class TestTimeoutBehavior:
    """[IBR-CLS-01] [IBR-CLS-04] Hard timeout and degradation."""

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        """[IBR-CLS-01] Returns None when adapter exceeds timeout."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_slow_adapter(5.0, _valid_json())

        result = await classify_intent(
            "timeout test " + str(time.monotonic()),
            adapter,
            timeout_s=0.01,
        )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise(self):
        """[IBR-CLS-04] Timeout produces None, never an exception."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_slow_adapter(5.0, _valid_json())

        result = await classify_intent(
            "no raise test " + str(time.monotonic()),
            adapter,
            timeout_s=0.01,
        )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self):
        """[IBR-CLS-01] [IBR-OBS-04] Timeout emits warning log."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_slow_adapter(5.0, _valid_json())

        log_path = "dragonlight_router.selection.classifier.logger"
        with patch(log_path) as mock_logger:
            await classify_intent(
                "log test " + str(time.monotonic()),
                adapter,
                timeout_s=0.01,
            )
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "ibr_classification_timeout" in str(call_args)

        configure_cache(max_entries=5000, ttl_s=300.0)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    """[IBR-CLS-04] Response text parsing with various formats."""

    def test_valid_json_parsed(self):
        """[IBR-CLS-05] Valid JSON string parses to dict."""
        text = _valid_json(confidence=0.9)
        result = _parse_response(text)
        assert result is not None
        assert result["task_type"] == "analysis"

    def test_markdown_fenced_json(self):
        """[IBR-CLS-05] Markdown-fenced JSON is extracted."""
        inner = _valid_json(confidence=0.9)
        text = f"```json\n{inner}\n```"
        result = _parse_response(text)
        assert result is not None
        assert result["task_type"] == "analysis"

    def test_markdown_fenced_no_lang(self):
        """[IBR-CLS-05] Plain code fences (no lang) extracted."""
        inner = json.dumps(
            {
                "task_type": "reasoning",
                "domain": "general",
                "quality_speed": "quality",
                "confidence": 0.7,
            }
        )
        text = f"```\n{inner}\n```"
        result = _parse_response(text)
        assert result is not None
        assert result["task_type"] == "reasoning"

    def test_malformed_json_returns_none(self):
        """[IBR-CLS-04] Malformed JSON returns None."""
        assert _parse_response("{bad json}") is None
        assert _parse_response("not json at all") is None
        assert _parse_response("") is None

    def test_missing_required_field_fails_validation(self):
        """[IBR-CLS-05] Missing required fields fails validation."""
        text = '{"task_type": "analysis", "domain": "code"}'
        parsed = _parse_response(text)
        assert parsed is not None
        assert _validate_classification(parsed) is False

    def test_extra_fields_ignored_in_parse(self):
        """[IBR-CLS-05] Extra fields in parsed JSON are fine."""
        text = json.dumps(
            {
                "task_type": "analysis",
                "domain": "code",
                "quality_speed": "balanced",
                "confidence": 0.9,
                "reasoning": "this is extra",
            }
        )
        parsed = _parse_response(text)
        assert parsed is not None
        assert _validate_classification(parsed) is True

    def test_json_with_preamble_text(self):
        """[IBR-CLS-05] JSON embedded in preamble text extracted."""
        inner = json.dumps(
            {
                "task_type": "creative",
                "domain": "creative_writing",
                "quality_speed": "quality",
                "confidence": 0.8,
            }
        )
        text = f"Here is the classification: {inner}"
        result = _parse_response(text)
        assert result is not None
        assert result["task_type"] == "creative"

    def test_empty_json_object(self):
        """[IBR-CLS-05] Empty JSON object fails validation."""
        parsed = _parse_response("{}")
        assert parsed is not None
        assert _validate_classification(parsed) is False

    def test_no_braces_returns_none(self):
        """[IBR-CLS-04] Text with no JSON braces returns None."""
        assert _parse_response("just plain text") is None

    def test_nested_json_extracts_outer(self):
        """[IBR-CLS-05] Nested braces: outer object extracted."""
        text = json.dumps(
            {
                "task_type": "lookup",
                "domain": "general",
                "quality_speed": "speed",
                "confidence": 0.5,
                "meta": {"nested": True},
            }
        )
        result = _parse_response(text)
        assert result is not None
        assert result["task_type"] == "lookup"


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


class TestAdapterIntegration:
    """[IBR-CLS-04] Adapter call, error handling, result construction."""

    @pytest.mark.asyncio
    async def test_valid_adapter_response(self):
        """[IBR-CLS-06] Valid response produces ClassifiedIntent."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        result = await classify_intent(
            "valid response " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is not None
        assert result.task_type == "analysis"
        assert result.domain == "code"
        assert result.quality_speed == "balanced"
        assert result.confidence == 0.85
        assert result.latency_ms >= 0.0
        assert result.from_cache is False

        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_adapter_invalid_json_returns_none(self):
        """[IBR-CLS-04] Invalid JSON from adapter returns None."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter("not json")

        result = await classify_intent(
            "invalid json " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_adapter_invalid_values_returns_none(self):
        """[IBR-CLS-04] Invalid taxonomy values returns None."""
        configure_cache(max_entries=100, ttl_s=300.0)
        bad_json = json.dumps(
            {
                "task_type": "invalid",
                "domain": "code",
                "quality_speed": "balanced",
                "confidence": 0.9,
            }
        )
        adapter = _make_mock_adapter(bad_json)

        result = await classify_intent(
            "invalid values " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_adapter_exception_returns_none(self):
        """[IBR-CLS-04] [IBR-SYS-03] Exception returns None."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_raising_adapter(
            RuntimeError("connection failed"),
        )

        result = await classify_intent(
            "exception test " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_adapter_receives_only_operator_message(self):
        """[IBR-CLS-02] Adapter gets only operator_message."""
        configure_cache(max_entries=100, ttl_s=300.0)
        captured_messages: list[dict] = []

        adapter = MagicMock(spec=GenerativeBackend)

        async def _generate(messages, **kwargs):
            captured_messages.extend(messages)
            yield _valid_json()

        adapter.generate = _generate

        msg = "classify this specific message"
        await classify_intent(
            msg + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )

        # system (prompt) + user (operator_message)
        assert len(captured_messages) == 2
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[1]["role"] == "user"
        assert msg in captured_messages[1]["content"]

        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_all_task_types(self):
        """[IBR-CLS-05] All 8 task types handled from adapter."""
        configure_cache(max_entries=100, ttl_s=300.0)
        for tt in TASK_TYPES:
            resp = json.dumps(
                {
                    "task_type": tt,
                    "domain": "code",
                    "quality_speed": "balanced",
                    "confidence": 0.8,
                }
            )
            adapter = _make_mock_adapter(resp)
            result = await classify_intent(
                f"test {tt} " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
            assert result is not None
            assert result.task_type == tt
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_all_domains(self):
        """[IBR-CLS-05] All 6 domains handled from adapter."""
        configure_cache(max_entries=100, ttl_s=300.0)
        for d in DOMAINS:
            resp = json.dumps(
                {
                    "task_type": "analysis",
                    "domain": d,
                    "quality_speed": "balanced",
                    "confidence": 0.8,
                }
            )
            adapter = _make_mock_adapter(resp)
            result = await classify_intent(
                f"test {d} " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
            assert result is not None
            assert result.domain == d
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_all_quality_speed(self):
        """[IBR-CLS-05] All 3 quality_speed values from adapter."""
        configure_cache(max_entries=100, ttl_s=300.0)
        for qs in QUALITY_SPEED:
            resp = json.dumps(
                {
                    "task_type": "analysis",
                    "domain": "code",
                    "quality_speed": qs,
                    "confidence": 0.8,
                }
            )
            adapter = _make_mock_adapter(resp)
            result = await classify_intent(
                f"test {qs} " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
            assert result is not None
            assert result.quality_speed == qs
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_successful_result_is_cached(self):
        """[IBR-CLS-03] Successful result stored in cache."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        msg = "cache store test " + str(time.monotonic())
        r1 = await classify_intent(msg, adapter, timeout_s=5.0)
        assert r1 is not None

        r2 = await classify_intent(msg, adapter, timeout_s=5.0)
        assert r2 is not None
        assert r2.from_cache is True

        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_failed_result_is_not_cached(self):
        """[IBR-CLS-04] Failed classification (None) not cached."""
        configure_cache(max_entries=100, ttl_s=300.0)
        bad_adapter = _make_mock_adapter("not json")

        msg = "no cache on fail " + str(time.monotonic())
        r1 = await classify_intent(
            msg,
            bad_adapter,
            timeout_s=5.0,
        )
        assert r1 is None

        # Good adapter should not get a cache hit
        good_adapter = _make_mock_adapter(_valid_json())
        r2 = await classify_intent(
            msg,
            good_adapter,
            timeout_s=5.0,
        )
        assert r2 is not None
        assert r2.from_cache is False

        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_integer_confidence_accepted(self):
        """[IBR-CLS-05] Integer confidence coerced to float."""
        configure_cache(max_entries=100, ttl_s=300.0)
        resp = json.dumps(
            {
                "task_type": "analysis",
                "domain": "code",
                "quality_speed": "balanced",
                "confidence": 1,
            }
        )
        adapter = _make_mock_adapter(resp)
        result = await classify_intent(
            "int confidence " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is not None
        assert result.confidence == 1.0
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_result_is_frozen(self):
        """[IBR-DATA-01] ClassifiedIntent from classify is frozen."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())
        result = await classify_intent(
            "frozen test " + str(time.monotonic()),
            adapter,
            timeout_s=5.0,
        )
        assert result is not None
        with pytest.raises(FrozenInstanceError):
            result.task_type = "creative"  # type: ignore[misc]
        configure_cache(max_entries=5000, ttl_s=300.0)


# ---------------------------------------------------------------------------
# Coverage for fallback ClassifiedIntent import (lines 62-72)
# ---------------------------------------------------------------------------


class TestClassifiedIntentFallback:
    """[IBR-CLS-01] ClassifiedIntent is always available (lines 62-72)."""

    def test_classified_intent_is_importable(self):
        """ClassifiedIntent can be imported from core.types."""
        from dragonlight_router.core.types import ClassifiedIntent

        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=10.0,
            from_cache=False,
        )
        assert intent.task_type == "analysis"
        assert intent.domain == "code"
        assert intent.quality_speed == "balanced"
        assert intent.confidence == 0.9
        assert intent.latency_ms == 10.0
        assert intent.from_cache is False

    def test_classified_intent_is_frozen(self):
        """ClassifiedIntent is immutable."""
        from dragonlight_router.core.types import ClassifiedIntent

        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=10.0,
            from_cache=False,
        )
        with pytest.raises(FrozenInstanceError):
            intent.task_type = "creative"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coverage for cache put with existing key (line 131)
# ---------------------------------------------------------------------------


class TestCachePutExistingKey:
    """[IBR-CLS-03] Cache.put replaces existing entry on duplicate key (line 131)."""

    def test_put_replaces_existing_entry(self):
        """Putting the same key twice replaces the old value."""
        cache = _ClassificationCache(max_entries=100, ttl_s=300.0)
        key = hashlib.sha256(b"test message").hexdigest()

        intent1 = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.8,
            latency_ms=5.0,
            from_cache=False,
        )
        intent2 = ClassifiedIntent(
            task_type="creative",
            domain="general",
            quality_speed="quality",
            confidence=0.95,
            latency_ms=3.0,
            from_cache=False,
        )

        cache.put(key, intent1)
        cache.put(key, intent2)

        result = cache.get(key)
        assert result is not None
        assert result.task_type == "creative"
        assert result.domain == "general"


# ---------------------------------------------------------------------------
# Coverage for classify_intent unexpected error (lines 358-362)
# ---------------------------------------------------------------------------


class TestClassifyIntentUnexpectedError:
    """[IBR-SYS-03] classify_intent catches unexpected errors (lines 375-381)."""

    @pytest.mark.asyncio
    async def test_runtime_error_in_call_classifier_returns_none(self):
        """[IBR-SYS-03] RuntimeError from _call_classifier → returns None (line 375)."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        async def _raise_runtime(*args, **kwargs):
            raise RuntimeError("unexpected crash in classifier")

        with patch(
            "dragonlight_router.selection.classifier._call_classifier",
            side_effect=_raise_runtime,
        ):
            result = await classify_intent(
                "runtime error outer " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_oserror_in_call_classifier_returns_none(self):
        """[IBR-SYS-03] OSError from _call_classifier → returns None (line 375)."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        async def _raise_oserror(*args, **kwargs):
            raise OSError("network down in classifier")

        with patch(
            "dragonlight_router.selection.classifier._call_classifier",
            side_effect=_raise_oserror,
        ):
            result = await classify_intent(
                "oserror outer " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)

    @pytest.mark.asyncio
    async def test_value_error_in_call_classifier_returns_none(self):
        """[IBR-SYS-03] ValueError from _call_classifier → returns None (line 375)."""
        configure_cache(max_entries=100, ttl_s=300.0)
        adapter = _make_mock_adapter(_valid_json())

        async def _raise_valueerror(*args, **kwargs):
            raise ValueError("bad data in classifier")

        with patch(
            "dragonlight_router.selection.classifier._call_classifier",
            side_effect=_raise_valueerror,
        ):
            result = await classify_intent(
                "valueerror outer " + str(time.monotonic()),
                adapter,
                timeout_s=5.0,
            )
        assert result is None
        configure_cache(max_entries=5000, ttl_s=300.0)
