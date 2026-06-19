"""Tests for server/metrics.py — in-memory metrics collection.

Covers MetricsCollector: per-endpoint stats, router-level dispatch
counters, latency percentiles, memory/uptime reporting, and reset.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dragonlight_router.server.metrics import MetricsCollector, _percentile

pytestmark = pytest.mark.unit


class TestPercentile:
    def test_empty_list_returns_zero(self):
        """_percentile returns 0.0 for empty input."""
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        """_percentile returns the sole element for any percentile."""
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_p50_of_sorted_data(self):
        """_percentile computes p50 correctly from sorted data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _percentile(data, 50)
        assert result == 6.0  # rank = int(0.5 * 10) = 5 => data[5] = 6.0

    def test_p99_of_sorted_data(self):
        """_percentile computes p99 correctly."""
        data = list(range(1, 101))
        floats = [float(x) for x in data]
        result = _percentile(floats, 99)
        assert result == 100.0  # rank = min(99, int(0.99 * 100)) = 99 => data[99]

    def test_p95_of_sorted_data(self):
        """_percentile computes p95 correctly."""
        data = [float(i) for i in range(1, 21)]  # 1..20
        result = _percentile(data, 95)
        # rank = int(0.95 * 20) = 19 => data[19] = 20.0
        assert result == 20.0


class TestMetricsCollectorRecordRequest:
    def test_records_single_request(self):
        """A single recorded request shows up in snapshot."""
        mc = MetricsCollector()
        mc.record_request("GET", "/v1/health", 200, 5.0)
        snap = mc.snapshot()
        key = "GET /v1/health"
        assert key in snap["endpoints"]
        assert snap["endpoints"][key]["request_count"] == 1
        assert snap["endpoints"][key]["error_count"] == 0

    def test_counts_errors_for_4xx_and_5xx(self):
        """Status codes >= 400 increment error_count."""
        mc = MetricsCollector()
        mc.record_request("POST", "/v1/dispatch", 400, 1.0)
        mc.record_request("POST", "/v1/dispatch", 500, 2.0)
        mc.record_request("POST", "/v1/dispatch", 200, 3.0)
        snap = mc.snapshot()
        stats = snap["endpoints"]["POST /v1/dispatch"]
        assert stats["request_count"] == 3
        assert stats["error_count"] == 2

    def test_latency_percentiles(self):
        """Latency percentiles are computed from recorded durations."""
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record_request("GET", "/v1/health", 200, float(i))
        snap = mc.snapshot()
        lat = snap["endpoints"]["GET /v1/health"]["latency_ms"]
        assert lat["p50"] > 0
        assert lat["p95"] >= lat["p50"]
        assert lat["p99"] >= lat["p95"]

    def test_latency_defaults_to_zero_when_empty(self):
        """Snapshot with no requests returns zero latencies."""
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["endpoints"] == {}

    def test_multiple_endpoints_tracked_independently(self):
        """Different method+path combos get separate stats."""
        mc = MetricsCollector()
        mc.record_request("GET", "/v1/health", 200, 1.0)
        mc.record_request("POST", "/v1/dispatch", 200, 2.0)
        mc.record_request("POST", "/v1/dispatch", 200, 3.0)
        snap = mc.snapshot()
        assert snap["endpoints"]["GET /v1/health"]["request_count"] == 1
        assert snap["endpoints"]["POST /v1/dispatch"]["request_count"] == 2

    def test_latency_sample_cap(self):
        """When latency samples exceed max_latency_samples, oldest half is trimmed."""
        mc = MetricsCollector(max_latency_samples=10)
        for i in range(15):
            mc.record_request("GET", "/v1/health", 200, float(i))
        snap = mc.snapshot()
        # After 15 inserts with cap 10, trim triggers when > 10, keeping last 5
        # Then 5 more are added => total should be <=10
        stats = snap["endpoints"]["GET /v1/health"]
        assert stats["request_count"] == 15  # count is always accurate


class TestMetricsCollectorRouterLevel:
    def test_record_dispatch(self):
        """record_dispatch increments total_dispatches."""
        mc = MetricsCollector()
        mc.record_dispatch()
        mc.record_dispatch()
        snap = mc.snapshot()
        assert snap["router"]["total_dispatches"] == 2
        assert snap["router"]["fallback_count"] == 0

    def test_record_dispatch_with_fallback(self):
        """record_dispatch with was_fallback=True increments fallback_count."""
        mc = MetricsCollector()
        mc.record_dispatch(was_fallback=False)
        mc.record_dispatch(was_fallback=True)
        mc.record_dispatch(was_fallback=True)
        snap = mc.snapshot()
        assert snap["router"]["total_dispatches"] == 3
        assert snap["router"]["fallback_count"] == 2

    def test_record_circuit_breaker_trip(self):
        """record_circuit_breaker_trip increments the counter."""
        mc = MetricsCollector()
        mc.record_circuit_breaker_trip()
        mc.record_circuit_breaker_trip()
        mc.record_circuit_breaker_trip()
        snap = mc.snapshot()
        assert snap["router"]["circuit_breaker_trips"] == 3


class TestMetricsCollectorSnapshot:
    def test_snapshot_includes_uptime(self):
        """Snapshot includes a non-negative uptime_seconds."""
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert "uptime_seconds" in snap
        assert snap["uptime_seconds"] >= 0

    def test_snapshot_includes_memory(self):
        """Snapshot includes a positive memory_mb value."""
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert "memory_mb" in snap
        assert snap["memory_mb"] > 0

    def test_snapshot_memory_linux_fallback(self):
        """When maxrss yields < 1 MB (Linux path), memory is recomputed from KB."""
        mc = MetricsCollector()

        class FakeRusage:
            ru_maxrss = 500  # 500 bytes on macOS => 0.0004 MB => triggers Linux path

        rusage_path = "dragonlight_router.server.metrics.resource.getrusage"
        with patch(rusage_path, return_value=FakeRusage()):
            snap = mc.snapshot()
        # Linux path: 500 / 1024 ≈ 0.49
        assert snap["memory_mb"] > 0

    def test_snapshot_router_defaults(self):
        """Fresh collector has zero dispatch/fallback/circuit counters."""
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["router"]["total_dispatches"] == 0
        assert snap["router"]["fallback_count"] == 0
        assert snap["router"]["circuit_breaker_trips"] == 0


class TestMetricsCollectorEdgeCases:
    def test_snapshot_empty_latencies_for_endpoint(self):
        """Endpoint with no latency samples returns zero percentiles."""
        mc = MetricsCollector()
        # Manually create an endpoint entry with empty latencies
        from dragonlight_router.server.metrics import _EndpointStats

        mc._endpoints["GET /edge"] = _EndpointStats(request_count=1, error_count=0)
        snap = mc.snapshot()
        lat = snap["endpoints"]["GET /edge"]["latency_ms"]
        assert lat == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


class TestMetricsCollectorReset:
    def test_reset_clears_all_state(self):
        """reset() clears endpoints and router counters."""
        mc = MetricsCollector()
        mc.record_request("GET", "/v1/health", 200, 1.0)
        mc.record_dispatch(was_fallback=True)
        mc.record_circuit_breaker_trip()

        mc.reset()
        snap = mc.snapshot()
        assert snap["endpoints"] == {}
        assert snap["router"]["total_dispatches"] == 0
        assert snap["router"]["fallback_count"] == 0
        assert snap["router"]["circuit_breaker_trips"] == 0
