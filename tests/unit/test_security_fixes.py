"""Tests for security audit fixes: SEC-001, SEC-003, SEC-005, SEC-006, SEC-007.

Covers:
- SEC-001: CORS defaults to disabled (no origins = no CORS headers)
- SEC-003: SSRF validation rejects private IPs, allows known providers
- SEC-005: Admin auth rate limiting (5 failures = 429)
- SEC-006: Admin API key startup warning
- SEC-007: CORS allow_credentials defaults to False
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from starlette.testclient import TestClient

from dragonlight_router.core.validation import validate_provider_url
from dragonlight_router.server.app import create_app
from dragonlight_router.server.middleware import get_cors_config
from dragonlight_router.server.routes import (
    _admin_auth_failures,
    _is_admin_auth_rate_limited,
    _record_admin_auth_failure,
    _reset_admin_auth_failures,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, admin_api_key: str | None = None) -> Path:
    """Create a minimal router config for security tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "model_role_matrix.json").write_text(json.dumps({}))

    config: dict = {
        "state_dir": str(state_dir),
        "catalog_ttl_hours": 24,
        "default_top_n": 12,
        "max_consecutive_same_provider": 2,
        "providers": [],
    }
    if admin_api_key is not None:
        config["admin_api_key"] = admin_api_key

    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


# ---------------------------------------------------------------------------
# SEC-001: CORS defaults to disabled
# ---------------------------------------------------------------------------


class TestCORSDefaultDisabled:
    """SEC-001: With no DRAGONLIGHT_CORS_ORIGINS set, CORS is disabled."""

    def test_no_origins_returns_none(self, monkeypatch):
        """get_cors_config returns None when no origins are configured."""
        monkeypatch.delenv("DRAGONLIGHT_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_METHODS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_HEADERS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_CREDENTIALS", raising=False)
        assert get_cors_config() is None

    def test_no_cors_headers_in_response(self, tmp_path, monkeypatch):
        """When CORS is disabled, responses have no Access-Control-Allow-Origin header."""
        monkeypatch.delenv("DRAGONLIGHT_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("DRAGONLIGHT_CORS_CREDENTIALS", raising=False)
        config_path = _make_config(tmp_path)
        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get(
            "/v1/health",
            headers={"Origin": "https://evil.com"},
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_explicit_origin_enables_cors(self, monkeypatch):
        """Setting DRAGONLIGHT_CORS_ORIGINS explicitly re-enables CORS."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "https://app.example.com")
        monkeypatch.delenv("DRAGONLIGHT_CORS_CREDENTIALS", raising=False)
        config = get_cors_config()
        assert config is not None
        assert config["allow_origins"] == ["https://app.example.com"]


# ---------------------------------------------------------------------------
# SEC-007: CORS allow_credentials defaults to False
# ---------------------------------------------------------------------------


class TestCORSCredentialsDefault:
    """SEC-007: allow_credentials defaults to False, configurable via env var."""

    def test_credentials_default_false(self, monkeypatch):
        """Without DRAGONLIGHT_CORS_CREDENTIALS, credentials are disabled."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "*")
        monkeypatch.delenv("DRAGONLIGHT_CORS_CREDENTIALS", raising=False)
        config = get_cors_config()
        assert config is not None
        assert config["allow_credentials"] is False

    def test_credentials_env_true(self, monkeypatch):
        """Setting DRAGONLIGHT_CORS_CREDENTIALS=true enables credentials."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "*")
        monkeypatch.setenv("DRAGONLIGHT_CORS_CREDENTIALS", "true")
        config = get_cors_config()
        assert config is not None
        assert config["allow_credentials"] is True

    def test_credentials_env_false(self, monkeypatch):
        """Setting DRAGONLIGHT_CORS_CREDENTIALS=false disables credentials."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "*")
        monkeypatch.setenv("DRAGONLIGHT_CORS_CREDENTIALS", "false")
        config = get_cors_config()
        assert config is not None
        assert config["allow_credentials"] is False

    def test_credentials_env_1(self, monkeypatch):
        """Setting DRAGONLIGHT_CORS_CREDENTIALS=1 enables credentials."""
        monkeypatch.setenv("DRAGONLIGHT_CORS_ORIGINS", "*")
        monkeypatch.setenv("DRAGONLIGHT_CORS_CREDENTIALS", "1")
        config = get_cors_config()
        assert config is not None
        assert config["allow_credentials"] is True


# ---------------------------------------------------------------------------
# SEC-005: Admin auth rate limiting
# ---------------------------------------------------------------------------


class TestAdminAuthRateLimiting:
    """SEC-005: 5 failed admin auth attempts within 60s triggers 429."""

    def test_under_limit_allows_401(self):
        """Fewer than 5 failures still returns 401 (not 429)."""
        for _ in range(4):
            _record_admin_auth_failure("10.0.0.1")
        assert not _is_admin_auth_rate_limited("10.0.0.1")

    def test_at_limit_triggers_rate_limit(self):
        """Exactly 5 failures triggers rate limiting."""
        for _ in range(5):
            _record_admin_auth_failure("10.0.0.2")
        assert _is_admin_auth_rate_limited("10.0.0.2")

    def test_over_limit_stays_rate_limited(self):
        """More than 5 failures keeps rate limiting active."""
        for _ in range(10):
            _record_admin_auth_failure("10.0.0.3")
        assert _is_admin_auth_rate_limited("10.0.0.3")

    def test_different_ips_independent(self):
        """Rate limiting is per-IP."""
        for _ in range(5):
            _record_admin_auth_failure("10.0.0.4")
        assert _is_admin_auth_rate_limited("10.0.0.4")
        assert not _is_admin_auth_rate_limited("10.0.0.5")

    def test_old_failures_expire(self, monkeypatch):
        """Failures older than 60 seconds are pruned and don't count."""
        now = time.monotonic()
        # Inject old failure timestamps (65 seconds ago)
        _admin_auth_failures["10.0.0.6"] = [now - 65.0 for _ in range(5)]
        assert not _is_admin_auth_rate_limited("10.0.0.6")

    def test_reset_clears_state(self):
        """_reset_admin_auth_failures clears all tracked failures."""
        for _ in range(5):
            _record_admin_auth_failure("10.0.0.7")
        assert _is_admin_auth_rate_limited("10.0.0.7")
        _reset_admin_auth_failures()
        assert not _is_admin_auth_rate_limited("10.0.0.7")

    def test_http_429_after_5_failures(self, tmp_path):
        """Integration: 5 wrong-key attempts then a 6th returns 429."""
        config_path = _make_config(tmp_path, admin_api_key="correct-key")
        app = create_app(config_path=config_path)
        client = TestClient(app)

        # 5 failed attempts
        for _ in range(5):
            resp = client.post(
                "/v1/retire",
                json={"backend": "x"},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401

        # 6th attempt should be rate-limited
        resp = client.post(
            "/v1/retire",
            json={"backend": "x"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 429
        assert "Too many failed" in resp.json()["error"]

    def test_valid_auth_not_affected_by_other_ip_failures(self, tmp_path):
        """Valid auth succeeds even if another IP is rate-limited."""
        config_path = _make_config(tmp_path, admin_api_key="my-key")
        app = create_app(config_path=config_path)

        # Manually inject failures for a different IP
        for _ in range(10):
            _record_admin_auth_failure("1.2.3.4")

        client = TestClient(app)
        resp = client.post(
            "/v1/retire",
            json={"backend": "x"},
            headers={"Authorization": "Bearer my-key"},
        )
        # Should get 404 (backend not found), not 429
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# SEC-006: Admin API key startup warning
# ---------------------------------------------------------------------------


class TestAdminKeyWarning:
    """SEC-006: Warning logged when admin_api_key is not configured."""

    def test_warning_logged_when_no_admin_key(self, tmp_path):
        """create_app logs admin_endpoints_unprotected when no key is set."""
        config_path = _make_config(tmp_path, admin_api_key=None)
        with patch("dragonlight_router.server.app.structlog.get_logger") as mock_get:
            mock_logger = mock_get.return_value
            create_app(config_path=config_path)
            # Check that warning was called with the expected event
            mock_logger.warning.assert_any_call(
                "admin_endpoints_unprotected",
                detail="No admin_api_key configured — admin endpoints are open to all callers.",
            )

    def test_no_warning_when_admin_key_set(self, tmp_path):
        """create_app does NOT log the warning when admin_api_key is configured."""
        config_path = _make_config(tmp_path, admin_api_key="my-secret")
        with patch("dragonlight_router.server.app.structlog.get_logger") as mock_get:
            mock_logger = mock_get.return_value
            create_app(config_path=config_path)
            # Ensure no admin_endpoints_unprotected warning was emitted
            for call in mock_logger.warning.call_args_list:
                assert call.args[0] != "admin_endpoints_unprotected"


# ---------------------------------------------------------------------------
# SEC-003: SSRF validation
# ---------------------------------------------------------------------------


class TestSSRFValidation:
    """SEC-003: validate_provider_url rejects private IPs and unsafe schemes."""

    # --- Should pass ---

    def test_allows_https_public_url(self):
        """Standard HTTPS provider URLs pass validation."""
        validate_provider_url("https://api.openai.com/v1")

    def test_allows_https_groq(self):
        """Groq API URL passes validation."""
        validate_provider_url("https://api.groq.com/openai/v1")

    def test_allows_https_anthropic(self):
        """Anthropic API URL passes validation."""
        validate_provider_url("https://api.anthropic.com/v1")

    def test_allows_http_localhost(self):
        """http://localhost is allowed for local providers like Ollama."""
        validate_provider_url("http://localhost:11434")

    def test_allows_http_127_0_0_1(self):
        """http://127.0.0.1 is allowed for local providers."""
        validate_provider_url("http://127.0.0.1:11434")

    # --- Should reject ---

    def test_rejects_http_non_localhost(self):
        """HTTP to non-localhost addresses is rejected."""
        with pytest.raises(ValueError, match="https"):
            validate_provider_url("http://api.example.com/v1")

    def test_rejects_private_10_x(self):
        """Private 10.x.x.x addresses are rejected."""
        with pytest.raises(ValueError, match="private"):
            validate_provider_url("https://10.0.0.1/v1")

    def test_rejects_private_172_16(self):
        """Private 172.16.x.x addresses are rejected."""
        with pytest.raises(ValueError, match="private"):
            validate_provider_url("https://172.16.0.1/v1")

    def test_rejects_private_192_168(self):
        """Private 192.168.x.x addresses are rejected."""
        with pytest.raises(ValueError, match="private"):
            validate_provider_url("https://192.168.1.1/v1")

    def test_rejects_metadata_endpoint(self):
        """Cloud metadata endpoints are rejected."""
        with pytest.raises(ValueError, match="metadata"):
            validate_provider_url("https://169.254.169.254/latest/meta-data/")

    def test_rejects_metadata_google(self):
        """Google metadata hostname is rejected."""
        with pytest.raises(ValueError, match="metadata"):
            validate_provider_url("https://metadata.google.internal/")

    def test_rejects_empty_hostname(self):
        """URLs with no hostname are rejected."""
        with pytest.raises(ValueError, match="no hostname"):
            validate_provider_url("https:///v1")

    def test_rejects_ftp_scheme(self):
        """FTP scheme is rejected."""
        with pytest.raises(ValueError, match="https"):
            validate_provider_url("ftp://example.com/v1")

    def test_rejects_file_scheme(self):
        """file:// scheme is rejected (no hostname)."""
        with pytest.raises(ValueError):
            validate_provider_url("file:///etc/passwd")

    def test_allows_known_provider_urls(self):
        """All standard provider URLs from the router config pass validation."""
        known_urls = [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
            "https://api.groq.com/openai/v1",
            "https://generativelanguage.googleapis.com/v1beta",
            "https://api.together.xyz/v1",
            "https://api.mistral.ai/v1",
            "https://integrate.api.nvidia.com/v1",
            "https://api.cerebras.ai/v1",
            "https://api.cohere.com/v2",
            "https://openrouter.ai/api/v1",
            "http://localhost:11434",
        ]
        for url in known_urls:
            validate_provider_url(url)  # Should not raise


# ---------------------------------------------------------------------------
# Coverage for _is_private_ip DNS resolution branch (line 47)
# ---------------------------------------------------------------------------


class TestIsPrivateIpDnsResolution:
    """SEC-003: _is_private_ip detects private IPs returned by DNS resolution."""

    def test_hostname_resolving_to_private_ip_returns_true(self):
        """[SEC-003] Hostname that DNS-resolves to a private IP is detected (line 47)."""
        from dragonlight_router.core.validation import _is_private_ip

        # Mock getaddrinfo to return a private IP for a non-IP hostname
        fake_info = [(2, 1, 6, "", ("10.0.0.1", 0))]
        with patch("dragonlight_router.core.validation.socket.getaddrinfo", return_value=fake_info):
            result = _is_private_ip("evil-internal.example.com")
        assert result is True

    def test_hostname_resolving_to_public_ip_returns_false(self):
        """[SEC-003] Hostname that DNS-resolves to a public IP returns False."""
        from dragonlight_router.core.validation import _is_private_ip

        fake_info = [(2, 1, 6, "", ("8.8.8.8", 0))]
        with patch("dragonlight_router.core.validation.socket.getaddrinfo", return_value=fake_info):
            result = _is_private_ip("safe.example.com")
        assert result is False

    def test_hostname_dns_failure_returns_false(self):
        """[SEC-003] DNS resolution failure → treat as non-private (line 48-50)."""
        import socket

        from dragonlight_router.core.validation import _is_private_ip

        with patch(
            "dragonlight_router.core.validation.socket.getaddrinfo",
            side_effect=socket.gaierror("DNS failed"),
        ):
            result = _is_private_ip("unknown.example.com")
        assert result is False
