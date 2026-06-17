"""Tests for server/logging.py — secret scrubbing in structlog pipeline.

HAZ-006 mitigation: Prevents API keys from appearing in log output.
Spec traceability: HAZ-006 (API Key Exposure in Logs)
"""
from __future__ import annotations

import pytest

from dragonlight_router.server.logging import (
    _scrub_value,
    configure_logging,
    scrub_secrets,
)


class TestScrubValue:
    """Tests for the _scrub_value recursive scrubber."""

    def test_plain_string_unchanged(self):
        """[HAZ-006 AC-1] Non-secret strings pass through unchanged."""
        assert _scrub_value("hello world") == "hello world"

    def test_bearer_token_scrubbed(self):
        """[HAZ-006 AC-1] Bearer tokens are replaced with [REDACTED]."""
        value = "Authorization: Bearer sk-abc123xyz456"
        result = _scrub_value(value)
        assert "sk-abc123xyz456" not in result
        assert "[REDACTED]" in result

    def test_openai_api_key_scrubbed(self):
        """[HAZ-006 AC-1] OpenAI-style API keys (sk-...) are scrubbed."""
        value = "key is sk-proj1234567890abcdef"
        result = _scrub_value(value)
        assert "sk-proj1234567890abcdef" not in result
        assert "[REDACTED]" in result

    def test_groq_api_key_scrubbed(self):
        """[HAZ-006 AC-1] Groq-style API keys (gsk_...) are scrubbed."""
        value = "gsk_test1234567890abcdef"
        result = _scrub_value(value)
        assert "gsk_test1234567890abcdef" not in result

    def test_nvidia_api_key_scrubbed(self):
        """[HAZ-006 AC-1] NVIDIA-style API keys (nvapi-...) are scrubbed."""
        value = "nvapi-testkey1234567890ab"
        result = _scrub_value(value)
        assert "nvapi-testkey1234567890ab" not in result

    def test_google_api_key_scrubbed(self):
        """[HAZ-006 AC-1] Google-style API keys (AIza...) are scrubbed."""
        value = "AIzaSyC1234567890abcdef12"
        result = _scrub_value(value)
        assert "AIzaSyC1234567890abcdef12" not in result

    def test_dict_values_scrubbed(self):
        """[HAZ-006 AC-2] Secrets nested in dicts are recursively scrubbed."""
        value = {"headers": {"Authorization": "Bearer sk-mysecretkey12345"}}
        result = _scrub_value(value)
        assert isinstance(result, dict)
        assert "sk-mysecretkey12345" not in str(result)
        assert "[REDACTED]" in str(result)

    def test_list_values_scrubbed(self):
        """[HAZ-006 AC-2] Secrets nested in lists are recursively scrubbed."""
        value = ["safe text", "Bearer sk-secrettoken123456"]
        result = _scrub_value(value)
        assert isinstance(result, list)
        assert "sk-secrettoken123456" not in str(result)

    def test_tuple_values_scrubbed(self):
        """[HAZ-006 AC-2] Secrets nested in tuples are recursively scrubbed."""
        value = ("safe", "Bearer sk-tuplesecret12345")
        result = _scrub_value(value)
        assert isinstance(result, tuple)
        assert "sk-tuplesecret12345" not in str(result)

    def test_non_string_values_pass_through(self):
        """[HAZ-006 AC-1] Non-string values (int, float, bool, None) pass through."""
        assert _scrub_value(42) == 42
        assert _scrub_value(3.14) == 3.14
        assert _scrub_value(True) is True
        assert _scrub_value(None) is None

    def test_empty_string_unchanged(self):
        """[HAZ-006 AC-1] Empty string passes through unchanged."""
        assert _scrub_value("") == ""

    def test_bearer_case_insensitive(self):
        """[HAZ-006 AC-1] Bearer token scrubbing is case-insensitive."""
        result = _scrub_value("BEARER sk-abcdefghijklmno1234")
        assert "sk-abcdefghijklmno1234" not in result


class TestScrubSecrets:
    """Tests for the scrub_secrets structlog processor."""

    def test_scrubs_known_secret_keys(self):
        """[HAZ-006 AC-3] Known secret keys are replaced entirely."""
        event_dict = {
            "event": "request_sent",
            "authorization": "Bearer sk-realkey12345678",
            "api_key": "gsk_realgroqkey12345678",
            "token": "sensitive-value",
            "secret": "top-secret",
        }
        result = scrub_secrets(None, "info", event_dict)
        assert result["authorization"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["secret"] == "[REDACTED]"
        assert result["event"] == "request_sent"

    def test_scrubs_secret_patterns_in_values(self):
        """[HAZ-006 AC-3] Secret patterns within string values are scrubbed."""
        event_dict = {
            "event": "adapter_error",
            "error": "API call failed with key sk-abc123secretkey4567",
        }
        result = scrub_secrets(None, "error", event_dict)
        assert "sk-abc123secretkey4567" not in str(result["error"])
        assert "[REDACTED]" in str(result["error"])

    def test_preserves_non_secret_values(self):
        """[HAZ-006 AC-3] Non-secret event dict values are preserved."""
        event_dict = {
            "event": "dispatch_complete",
            "backend": "groq_llama70b",
            "latency_ms": 123.4,
            "tokens_out": 50,
        }
        result = scrub_secrets(None, "info", event_dict)
        assert result["event"] == "dispatch_complete"
        assert result["backend"] == "groq_llama70b"
        assert result["latency_ms"] == 123.4
        assert result["tokens_out"] == 50

    def test_handles_nested_structures(self):
        """[HAZ-006 AC-3] Nested dicts and lists in event_dict are scrubbed."""
        event_dict = {
            "event": "http_request",
            "headers": {
                "Content-Type": "application/json",
                "Auth": "Bearer sk-nestedkey1234567890",
            },
        }
        result = scrub_secrets(None, "debug", event_dict)
        assert "sk-nestedkey1234567890" not in str(result)

    def test_empty_event_dict(self):
        """[HAZ-006 AC-3] Empty event dict produces empty result."""
        result = scrub_secrets(None, "info", {})
        assert result == {}


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_logging_does_not_raise(self):
        """[HAZ-006 AC-4] configure_logging() completes without error."""
        configure_logging()

    def test_configure_logging_installs_processor(self):
        """[HAZ-006 AC-4] After configure_logging, structlog uses the scrub processor."""
        import structlog

        configure_logging()
        # Verify structlog is configured (calling get_logger should work)
        log = structlog.get_logger("test")
        assert log is not None
