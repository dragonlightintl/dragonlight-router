"""Tests for server security hardening: rate limiting, sanitization, output validation.

Covers QA-022, QA-023, QA-024.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from dragonlight_router.server.middleware import (
    CORSMiddleware,
    RateLimitMiddleware,
    _TokenBucket,
    get_cors_config,
)
from dragonlight_router.server.routes import _sanitize_prompt, _validate_llm_response


# ---------------------------------------------------------------------------
# QA-023: Rate-limiting middleware
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_allows_requests_under_limit(self):
        """Bucket with capacity 5 allows 5 consecutive requests."""
        bucket = _TokenBucket(capacity=5.0, tokens=5.0)
        results = [bucket.consume(refill_rate=1.0) for _ in range(5)]
        assert all(results)

    def test_blocks_requests_over_limit(self):
        """After exhausting capacity, the next request is blocked."""
        bucket = _TokenBucket(capacity=3.0, tokens=3.0)
        for _ in range(3):
            bucket.consume(refill_rate=1.0)
        assert bucket.consume(refill_rate=1.0) is False

    def test_resets_after_window(self):
        """Tokens refill over time, allowing requests again."""
        now = 1000.0
        bucket = _TokenBucket(capacity=2.0, tokens=0.0, last_refill=now)
        # No tokens, should be blocked
        assert bucket.consume(refill_rate=1.0, now=now) is False
        # Advance 3 seconds with refill_rate=1.0 => 3 tokens refilled, capped at 2
        assert bucket.consume(refill_rate=1.0, now=now + 3.0) is True

    def test_partial_refill(self):
        """Partial time elapsed refills partial tokens."""
        now = 1000.0
        bucket = _TokenBucket(capacity=10.0, tokens=0.0, last_refill=now)
        # 0.5 seconds at rate 2.0 => 1.0 token
        assert bucket.consume(refill_rate=2.0, now=now + 0.5) is True
        # Should have 0 tokens left now
        assert bucket.consume(refill_rate=2.0, now=now + 0.5) is False


class TestRateLimitMiddleware:
    def test_middleware_default_configuration(self):
        """Middleware initializes with default 60 req/60s."""
        mw = RateLimitMiddleware(app=None)
        assert mw.max_requests == 60
        assert mw.window_seconds == 60
        assert mw.refill_rate == 1.0

    def test_middleware_custom_configuration(self):
        """Middleware accepts custom rate limit parameters."""
        mw = RateLimitMiddleware(app=None, max_requests=10, window_seconds=30)
        assert mw.max_requests == 10
        assert mw.window_seconds == 30
        assert mw.refill_rate == pytest.approx(10.0 / 30.0)

    def test_get_bucket_creates_new(self):
        """_get_bucket creates a new bucket for unknown IPs."""
        mw = RateLimitMiddleware(app=None, max_requests=5)
        bucket = mw._get_bucket("192.168.1.1")
        assert bucket.capacity == 5.0
        assert bucket.tokens == 5.0

    def test_get_bucket_returns_existing(self):
        """_get_bucket returns the same bucket for repeated calls."""
        mw = RateLimitMiddleware(app=None)
        b1 = mw._get_bucket("10.0.0.1")
        b2 = mw._get_bucket("10.0.0.1")
        assert b1 is b2

    def test_separate_buckets_per_ip(self):
        """Different IPs get independent buckets."""
        mw = RateLimitMiddleware(app=None, max_requests=2)
        b1 = mw._get_bucket("10.0.0.1")
        b2 = mw._get_bucket("10.0.0.2")
        assert b1 is not b2


# ---------------------------------------------------------------------------
# QA-022: Prompt sanitization
# ---------------------------------------------------------------------------


class TestSanitizePrompt:
    def test_preserves_normal_text(self):
        """Normal text passes through unchanged."""
        text = "Hello, how are you?"
        assert _sanitize_prompt(text) == text

    def test_preserves_newlines_and_tabs(self):
        """Newlines and tabs are kept."""
        text = "Line 1\nLine 2\tindented\r\nLine 3"
        assert _sanitize_prompt(text) == text

    def test_preserves_unicode(self):
        """Unicode characters (emoji, CJK, etc.) are preserved."""
        text = "Hello 世界! 🌍 Ñoño café"
        assert _sanitize_prompt(text) == text

    def test_strips_null_bytes(self):
        """Null bytes are removed."""
        text = "hello\x00world"
        assert _sanitize_prompt(text) == "helloworld"

    def test_strips_control_characters(self):
        """Control characters (except \\n, \\r, \\t) are removed."""
        # \x01 = SOH, \x07 = BEL, \x0b = VT, \x1f = US, \x7f = DEL
        text = "clean\x01\x07\x0b\x1f\x7ftext"
        result = _sanitize_prompt(text)
        assert result == "cleantext"

    def test_truncates_long_input(self):
        """Input exceeding 100K chars is truncated."""
        text = "a" * 150_000
        result = _sanitize_prompt(text)
        assert len(result) == 100_000

    def test_empty_string(self):
        """Empty string passes through."""
        assert _sanitize_prompt("") == ""

    def test_only_control_chars(self):
        """String of only control chars becomes empty."""
        text = "\x00\x01\x02\x03"
        assert _sanitize_prompt(text) == ""


# ---------------------------------------------------------------------------
# QA-024: LLM output validation
# ---------------------------------------------------------------------------


class TestValidateLlmResponse:
    def test_valid_content_unchanged(self):
        """Normal response content passes through."""
        content = "This is a valid LLM response."
        assert _validate_llm_response(content) == content

    def test_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        assert _validate_llm_response("") == ""

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert _validate_llm_response(None) == ""  # type: ignore[arg-type]

    def test_non_string_returns_empty(self):
        """Non-string input returns empty string."""
        assert _validate_llm_response(123) == ""  # type: ignore[arg-type]

    def test_strips_null_bytes(self):
        """Null bytes in response are stripped."""
        content = "hello\x00world\x00!"
        assert _validate_llm_response(content) == "helloworld!"

    def test_truncates_long_response(self):
        """Responses over 500K chars are truncated."""
        content = "x" * 600_000
        result = _validate_llm_response(content)
        assert len(result) == 500_000

    def test_preserves_unicode_in_response(self):
        """Unicode content in responses is preserved."""
        content = "Réponse en français avec des émojis 🎉"
        assert _validate_llm_response(content) == content

    def test_null_bytes_then_truncate(self):
        """Null bytes are stripped before length check."""
        # 500_001 chars + 10 null bytes => after strip = 500_001 => truncated to 500_000
        content = "y" * 500_001 + "\x00" * 10
        result = _validate_llm_response(content)
        assert len(result) == 500_000
        assert "\x00" not in result


# ---------------------------------------------------------------------------
# QA-023: Middleware dispatch — rate-limit exceeded path and IP extraction
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> Path:
    """Create a minimal router config for middleware integration tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    import json as _json
    (state_dir / "model_role_matrix.json").write_text(_json.dumps({}))
    config = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [],
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestRateLimitMiddlewareDispatch:
    def test_rate_limit_exceeded_returns_429(self, tmp_path: Path):
        """[TM-008 AC-1] Exhausted bucket causes middleware to return 429."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient as TC

        async def _dummy(request):
            return JSONResponse({"ok": True})

        inner = Starlette(routes=[Route("/v1/health", _dummy, methods=["GET"])])
        mw = RateLimitMiddleware(inner, max_requests=5, window_seconds=60)
        bucket = mw._get_bucket("testclient")
        bucket.tokens = 0.0
        client = TC(mw)
        response = client.get("/v1/health")
        assert response.status_code == 429
        assert "Too many requests" in response.json()["error"]

    def test_rate_limit_exceeded_via_direct_dispatch(self):
        """[TM-008 AC-1] Middleware dispatch returns 429 JSON when bucket is empty."""
        import asyncio
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse

        async def _dummy(request):
            return JSONResponse({"ok": True})

        inner_app = Starlette(routes=[Route("/ping", _dummy, methods=["GET"])])
        mw = RateLimitMiddleware(inner_app, max_requests=2, window_seconds=60)

        from starlette.testclient import TestClient as TC
        client = TC(mw)

        bucket = mw._get_bucket("testclient")
        bucket.tokens = 0.0

        response = client.get("/ping")
        assert response.status_code == 429
        data = response.json()
        assert "Too many requests" in data["error"]

    def test_get_client_ip_no_client_returns_unknown(self):
        """[TM-008 AC-2] _get_client_ip returns 'unknown' when request.client is None."""
        from starlette.requests import Request as StarletteRequest

        mw = RateLimitMiddleware(app=None)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "client": None,
        }
        req = StarletteRequest(scope=scope)
        assert req.client is None
        ip = mw._get_client_ip(req)
        assert ip == "unknown"


# ---------------------------------------------------------------------------
# CORS middleware configuration
# ---------------------------------------------------------------------------


class TestCORSConfig:
    def test_default_cors_config(self, monkeypatch):
        """Default CORS config allows all origins."""
        monkeypatch.delenv("DRAGONLIGHT_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_METHODS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_HEADERS", raising=False)
        config = get_cors_config()
        assert config["allow_origins"] == ["*"]
        assert config["allow_credentials"] is True
        assert "GET" in config["allow_methods"]
        assert "POST" in config["allow_methods"]
        assert "OPTIONS" in config["allow_methods"]
        assert config["allow_headers"] == ["*"]

    def test_custom_cors_origins(self, monkeypatch):
        """Custom origins are parsed from comma-separated env var."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "https://app.example.com,https://admin.example.com")
        monkeypatch.delenv("DRAGONLIGHT_CORS_METHODS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_HEADERS", raising=False)
        config = get_cors_config()
        assert config["allow_origins"] == ["https://app.example.com", "https://admin.example.com"]

    def test_custom_cors_methods(self, monkeypatch):
        """Custom methods are parsed from comma-separated env var."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_METHODS", "GET,POST,PUT,DELETE")
        monkeypatch.delenv("DRAGONLIGHT_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_HEADERS", raising=False)
        config = get_cors_config()
        assert config["allow_methods"] == ["GET", "POST", "PUT", "DELETE"]

    def test_custom_cors_headers(self, monkeypatch):
        """Custom headers are parsed from comma-separated env var."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_HEADERS", "Content-Type,Authorization,X-Request-ID")
        monkeypatch.delenv("DRAGONLIGHT_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_METHODS", raising=False)
        config = get_cors_config()
        assert config["allow_headers"] == ["Content-Type", "Authorization", "X-Request-ID"]

    def test_cors_preflight_request(self, tmp_path):
        """CORS preflight OPTIONS request receives correct headers."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient

        async def _dummy(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/v1/health", _dummy, methods=["GET"])])
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
        client = TestClient(app)

        response = client.options(
            "/v1/health",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_actual_request(self, tmp_path):
        """CORS actual request receives Access-Control-Allow-Origin header."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient

        async def _dummy(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/v1/health", _dummy, methods=["GET"])])
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
        client = TestClient(app)

        response = client.get("/v1/health", headers={"Origin": "https://example.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

    def test_cors_restricted_origin(self, tmp_path):
        """CORS with restricted origins rejects unlisted origins."""
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient

        async def _dummy(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/v1/health", _dummy, methods=["GET"])])
        app.add_middleware(CORSMiddleware, allow_origins=["https://allowed.com"], allow_methods=["GET"], allow_headers=["*"])
        client = TestClient(app)

        response = client.get("/v1/health", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
