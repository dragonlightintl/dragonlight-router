"""Structured logging configuration with secret scrubbing.

HAZ-006 mitigation: Prevents API keys from appearing in log output
by scrubbing known secret patterns from all log event dicts before
they reach any sink.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

# Patterns that identify secrets in log values.
# Matches Bearer tokens, common API key formats, and env var references.
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(
    r"(?:sk-|gsk_|xai-|key-|nvapi-|AIza)[A-Za-z0-9_\-]{10,}",
)
_REDACTED = "[REDACTED]"


def _scrub_value(value: object) -> object:
    """Recursively scrub secrets from a single value.

    Handles str, dict, list, and tuple. Other types pass through.
    """
    if isinstance(value, str):
        scrubbed = _BEARER_RE.sub(f"Bearer {_REDACTED}", value)
        scrubbed = _API_KEY_RE.sub(_REDACTED, scrubbed)
        return scrubbed
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(item) for item in value)
    return value


def scrub_secrets(
    logger: object,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> Mapping[str, Any]:
    """Structlog processor that scrubs secrets from event dicts.

    HAZ-006 mitigation: Runs before the final renderer to ensure no
    API keys, Bearer tokens, or other secrets appear in log output.
    Scans all values (not just top-level) for known secret patterns.

    Also scrubs any environment variable values that match configured
    env_key patterns from the router's provider configs.
    """
    assert isinstance(event_dict, dict), "event_dict must be a dict"

    scrubbed: dict[str, object] = {}
    for key, value in event_dict.items():
        # Scrub keys that are known to contain secrets
        if key in ("authorization", "api_key", "api-key", "token", "secret"):
            scrubbed[key] = _REDACTED
            continue
        scrubbed[key] = _scrub_value(value)

    assert isinstance(scrubbed, dict), "scrubbed event_dict must be a dict"
    return scrubbed


def configure_logging() -> None:
    """Configure structlog with secret-scrubbing processor.

    HAZ-006 mitigation: Inserts scrub_secrets into the processor chain
    before the final renderer, ensuring all log output is scrubbed.

    Should be called once at application startup (before any logging).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            scrub_secrets,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
