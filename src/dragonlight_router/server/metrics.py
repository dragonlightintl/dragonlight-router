"""In-memory metrics collection for the Dragonlight Router.

Tracks per-endpoint request counts, error counts, and latency percentiles
(p50/p95/p99). Also tracks router-level dispatch stats. No external
dependencies required — all data held in memory.
"""

from __future__ import annotations

import resource
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _EndpointStats:
    """Accumulated stats for a single endpoint (method + path)."""

    request_count: int = 0
    error_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class MetricsCollector:
    """Thread-safe in-memory metrics collector.

    Records per-endpoint request/error counts and latency samples,
    plus router-level dispatch counters.

    Latency samples are capped at ``max_latency_samples`` to bound
    memory usage. When the cap is reached, the oldest half is discarded.
    """

    def __init__(self, max_latency_samples: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._max_samples = max_latency_samples
        self._endpoints: dict[str, _EndpointStats] = {}
        self._total_dispatches: int = 0
        self._fallback_count: int = 0
        self._circuit_breaker_trips: int = 0

    # DEVIATION DCS-PARAM-001: record_request takes 5 params (excl. self).
    # Justification: HTTP request metrics require method, path, status, and duration;
    # all are scalar primitives. Approved by: architect. Scope: this method.
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record a completed HTTP request."""
        key = f"{method} {path}"
        with self._lock:
            stats = self._endpoints.setdefault(key, _EndpointStats())
            stats.request_count += 1
            if status_code >= 400:
                stats.error_count += 1
            stats.latencies_ms.append(duration_ms)
            if len(stats.latencies_ms) > self._max_samples:
                stats.latencies_ms = stats.latencies_ms[self._max_samples // 2 :]

    def record_dispatch(self, was_fallback: bool = False) -> None:
        """Record a dispatch event (called from dispatch route handler)."""
        with self._lock:
            self._total_dispatches += 1
            if was_fallback:
                self._fallback_count += 1

    def record_circuit_breaker_trip(self) -> None:
        """Record a circuit breaker trip event."""
        with self._lock:
            self._circuit_breaker_trips += 1

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable metrics summary."""
        with self._lock:
            uptime_seconds = round(time.monotonic() - self._start_time, 2)

            endpoints: dict[str, dict[str, object]] = {}
            for key, stats in self._endpoints.items():
                entry: dict[str, object] = {
                    "request_count": stats.request_count,
                    "error_count": stats.error_count,
                }
                if stats.latencies_ms:
                    sorted_lat = sorted(stats.latencies_ms)
                    entry["latency_ms"] = {
                        "p50": round(_percentile(sorted_lat, 50), 2),
                        "p95": round(_percentile(sorted_lat, 95), 2),
                        "p99": round(_percentile(sorted_lat, 99), 2),
                    }
                else:
                    entry["latency_ms"] = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
                endpoints[key] = entry

            # Memory usage via resource module (maxrss in KB on macOS, bytes on Linux)
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = round(rusage.ru_maxrss / (1024 * 1024), 2)
            # macOS reports in bytes; Linux in KB — detect via platform
            if memory_mb < 1.0:
                # Likely Linux (value in KB), re-derive
                memory_mb = round(rusage.ru_maxrss / 1024, 2)

            return {
                "uptime_seconds": uptime_seconds,
                "memory_mb": memory_mb,
                "endpoints": endpoints,
                "router": {
                    "total_dispatches": self._total_dispatches,
                    "fallback_count": self._fallback_count,
                    "circuit_breaker_trips": self._circuit_breaker_trips,
                },
            }

    def reset(self) -> None:
        """Reset all metrics. Primarily for testing."""
        with self._lock:
            self._endpoints.clear()
            self._total_dispatches = 0
            self._fallback_count = 0
            self._circuit_breaker_trips = 0
            self._start_time = time.monotonic()


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute the pct-th percentile from pre-sorted data.

    Uses the nearest-rank method. Returns 0.0 for empty input.
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    rank = max(0, min(n - 1, int(pct / 100.0 * n)))
    return sorted_data[rank]
