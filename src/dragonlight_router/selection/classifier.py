"""Intent classification engine for the Intent Based Router (IBR).

Classifies operator messages into task_type, domain, and quality_speed
dimensions via a lightweight LLM call. Implements SHA-256 caching,
hard timeout enforcement, and graceful degradation on failure.

Spec reference: intent-based-router-v0.1.0-spec.md section 2.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import structlog

from dragonlight_router.core.types import GenerativeBackend

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Taxonomy constants (IBR spec section 2.1)
# ---------------------------------------------------------------------------

TASK_TYPES: frozenset[str] = frozenset({
    "generation",
    "analysis",
    "refactoring",
    "summarization",
    "creative",
    "reasoning",
    "lookup",
    "translation",
})

DOMAINS: frozenset[str] = frozenset({
    "code",
    "technical",
    "legal",
    "business",
    "creative_writing",
    "general",
})

QUALITY_SPEED: frozenset[str] = frozenset({
    "quality",
    "balanced",
    "speed",
})

# ---------------------------------------------------------------------------
# ClassifiedIntent — local definition (another agent adds to core/types.py)
# ---------------------------------------------------------------------------

try:
    from dragonlight_router.core.types import ClassifiedIntent
except ImportError:

    @dataclass(frozen=True)
    class ClassifiedIntent:  # type: ignore[no-redef]
        """Classification output for a single operator message."""
        task_type: str
        domain: str
        quality_speed: str
        confidence: float
        latency_ms: float
        from_cache: bool

# ---------------------------------------------------------------------------
# Classification prompt template
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT: str = (
    "Classify the user message into exactly one value per dimension. "
    "Return ONLY valid JSON, no other text.\n\n"
    "task_type: generation | analysis | refactoring | summarization "
    "| creative | reasoning | lookup | translation\n"
    "domain: code | technical | legal | business | creative_writing | general\n"
    "quality_speed: quality | balanced | speed\n"
    "confidence: float 0.0-1.0\n\n"
    'Output: {"task_type":"...","domain":"...","quality_speed":"...","confidence":0.0}'
)

# ---------------------------------------------------------------------------
# Classification cache — thread-safe LRU with TTL
# ---------------------------------------------------------------------------


class _ClassificationCache:
    """Thread-safe LRU cache with TTL for ClassifiedIntent results.

    Keys are SHA-256 hex digests of operator_message strings.
    Entries are evicted LRU-first when max_entries is exceeded,
    and on access when TTL has elapsed.
    """

    def __init__(self, max_entries: int = 5000, ttl_s: float = 300.0) -> None:
        assert max_entries > 0, f"max_entries must be positive, got {max_entries}"
        assert ttl_s >= 0, f"ttl_s must be non-negative, got {ttl_s}"
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        # OrderedDict provides LRU semantics via move_to_end
        self._entries: OrderedDict[str, tuple[ClassifiedIntent, float]] = OrderedDict()

    def get(self, key: str) -> ClassifiedIntent | None:
        """Retrieve a cached ClassifiedIntent. Returns None if absent or expired."""
        assert isinstance(key, str) and len(key) == 64, "key must be a SHA-256 hex digest"
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            intent, created_at = entry
            if time.monotonic() - created_at > self._ttl_s:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return intent

    def put(self, key: str, intent: ClassifiedIntent) -> None:
        """Store a ClassifiedIntent, evicting LRU entries if over capacity."""
        assert isinstance(key, str) and len(key) == 64, "key must be a SHA-256 hex digest"
        assert isinstance(intent, ClassifiedIntent), "intent must be a ClassifiedIntent"
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = (intent, time.monotonic())
            self._entries.move_to_end(key)
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Remove oldest entries until at or below max_entries. Caller holds lock."""
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# Module-level singleton cache (reconfigurable via configure_cache)
_cache = _ClassificationCache()


def configure_cache(max_entries: int = 5000, ttl_s: float = 300.0) -> None:
    """Replace the module-level cache with new parameters."""
    global _cache  # noqa: PLW0603
    assert max_entries > 0, f"max_entries must be positive, got {max_entries}"
    assert ttl_s >= 0, f"ttl_s must be non-negative, got {ttl_s}"
    _cache = _ClassificationCache(max_entries=max_entries, ttl_s=ttl_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_cache_key(operator_message: str) -> str:
    """Compute SHA-256 hex digest of operator_message for cache keying."""
    assert isinstance(operator_message, str), "operator_message must be a string"
    digest = hashlib.sha256(operator_message.encode("utf-8")).hexdigest()
    assert len(digest) == 64, "SHA-256 hex digest must be 64 characters"
    return digest


def _validate_classification(raw: dict[str, Any]) -> bool:
    """Validate that raw classification dict has legal values for all dimensions."""
    assert isinstance(raw, dict), "raw must be a dict"
    task_type = raw.get("task_type")
    domain = raw.get("domain")
    quality_speed = raw.get("quality_speed")
    confidence = raw.get("confidence")

    if task_type not in TASK_TYPES:
        return False
    if domain not in DOMAINS:
        return False
    if quality_speed not in QUALITY_SPEED:
        return False
    if not isinstance(confidence, (int, float)):
        return False
    return 0.0 <= float(confidence) <= 1.0


def _parse_response(text: str) -> dict[str, Any] | None:
    """Extract and parse JSON from classifier response text.

    Handles responses that may include markdown fencing or preamble text.
    Returns None if parsing fails.
    """
    assert isinstance(text, str), "text must be a string"
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        result: dict[str, Any] | None = json.loads(cleaned[start:end + 1])
        return result
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------


async def _call_classifier(
    operator_message: str,
    adapter: GenerativeBackend,
) -> ClassifiedIntent | None:
    """Send classification request to the adapter and parse the response.

    Returns ClassifiedIntent on success, None on parse/validation failure.
    """
    assert isinstance(operator_message, str), "operator_message must be a string"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _CLASSIFICATION_PROMPT},
        {"role": "user", "content": operator_message},
    ]

    start_ns = time.perf_counter_ns()
    collected: list[str] = []

    try:
        async for chunk in adapter.generate(
            messages,
            max_tokens=128,
            temperature=0.0,
            stream=False,
        ):
            collected.append(chunk)
    except Exception:
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.warning(
            "ibr_classification_adapter_error",
            latency_ms=round(latency_ms, 2),
        )
        return None

    latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    raw_text = "".join(collected)

    parsed = _parse_response(raw_text)
    if parsed is None:
        logger.warning(
            "ibr_classification_parse_error",
            raw_text=raw_text[:200],
            latency_ms=round(latency_ms, 2),
        )
        return None

    if not _validate_classification(parsed):
        logger.warning(
            "ibr_classification_validation_error",
            parsed=parsed,
            latency_ms=round(latency_ms, 2),
        )
        return None

    return ClassifiedIntent(
        task_type=parsed["task_type"],
        domain=parsed["domain"],
        quality_speed=parsed["quality_speed"],
        confidence=float(parsed["confidence"]),
        latency_ms=round(latency_ms, 2),
        from_cache=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def classify_intent(
    operator_message: str,
    adapter: GenerativeBackend,
    *,
    timeout_s: float = 0.1,
) -> ClassifiedIntent | None:
    """Classify an operator message into intent dimensions.

    Checks the in-memory cache first (SHA-256 of operator_message).
    On cache miss, calls the classification model via *adapter* with
    a hard timeout.  All failures are logged at warning level and
    return None -- this function never raises.

    Args:
        operator_message: The raw operator message to classify.
        adapter: A GenerativeBackend for the classification model.
        timeout_s: Hard timeout in seconds (default 0.1 = 100ms).

    Returns:
        ClassifiedIntent on success, None on timeout / parse error / failure.
    """
    assert isinstance(operator_message, str), "operator_message must be a string"
    assert timeout_s > 0, f"timeout_s must be positive, got {timeout_s}"

    cache_key = _compute_cache_key(operator_message)

    # --- cache hit path ---
    cached = _cache.get(cache_key)
    if cached is not None:
        start_ns = time.perf_counter_ns()
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.debug(
            "ibr_classification_cache_hit",
            task_type=cached.task_type,
            domain=cached.domain,
            quality_speed=cached.quality_speed,
        )
        return ClassifiedIntent(
            task_type=cached.task_type,
            domain=cached.domain,
            quality_speed=cached.quality_speed,
            confidence=cached.confidence,
            latency_ms=round(latency_ms, 2),
            from_cache=True,
        )

    # --- cache miss path: call classifier with hard timeout ---
    try:
        result = await asyncio.wait_for(
            _call_classifier(operator_message, adapter),
            timeout=timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "ibr_classification_timeout",
            timeout_s=timeout_s,
            operator_message_len=len(operator_message),
        )
        return None
    except Exception:
        logger.warning(
            "ibr_classification_unexpected_error",
            operator_message_len=len(operator_message),
            exc_info=True,
        )
        return None

    if result is not None:
        _cache.put(cache_key, result)
        logger.info(
            "ibr_classification",
            task_type=result.task_type,
            domain=result.domain,
            quality_speed=result.quality_speed,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            from_cache=False,
        )

    return result
