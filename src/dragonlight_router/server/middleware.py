"""Security middleware for the Dragonlight Router HTTP server.

Provides rate-limiting middleware using a token-bucket algorithm
per client IP address, request correlation ID middleware for
structured request logging with optional metrics collection,
and CORS middleware for cross-origin requests.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware  # re-export
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from dragonlight_router.server.metrics import MetricsCollector

logger = structlog.get_logger()

_REQUEST_ID_HEADER = "X-Request-ID"


@dataclass
class _TokenBucket:
    """Per-client token bucket for rate limiting."""

    capacity: float
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, refill_rate: float, now: float | None = None) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = now if now is not None else time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces per-IP rate limits.

    Uses a token-bucket algorithm: each IP gets a bucket with
    ``max_requests`` capacity that refills at ``max_requests / window_seconds``
    tokens per second.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    max_requests:
        Maximum burst size / bucket capacity. Default 60.
    window_seconds:
        Time window for the rate limit. Default 60 (i.e. 60 req/min).
    """

    def __init__(
        self,
        app: object,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.refill_rate = max_requests / window_seconds
        self._buckets: dict[str, _TokenBucket] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from the request."""
        if request.client:
            return request.client.host
        return "unknown"

    def _get_bucket(self, client_ip: str) -> _TokenBucket:
        """Get or create a token bucket for the given client IP."""
        if client_ip not in self._buckets:
            self._buckets[client_ip] = _TokenBucket(
                capacity=float(self.max_requests),
                tokens=float(self.max_requests),
            )
        return self._buckets[client_ip]

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Check rate limit before forwarding the request."""
        client_ip = self._get_client_ip(request)
        bucket = self._get_bucket(client_ip)

        if not bucket.consume(self.refill_rate):
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                max_requests=self.max_requests,
                window_seconds=self.window_seconds,
            )
            return JSONResponse(
                {"error": "Too many requests. Please try again later."},
                status_code=429,
            )

        response: Response = await call_next(request)  # type: ignore[operator]
        return response


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that adds request correlation IDs and structured logging.

    For every incoming request:
    - Reads ``X-Request-ID`` from the client, or generates a UUID4 if absent.
    - Binds the request_id to structlog's contextvars so all downstream log
      lines automatically include it.
    - Logs method, path, status_code, duration_ms, and request_id after
      the response is produced.
    - Sets the ``X-Request-ID`` header on the response.
    - Optionally records per-request metrics via a MetricsCollector.
    """

    def __init__(
        self,
        app: object,
        metrics: MetricsCollector | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._metrics = metrics

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Attach correlation ID, call downstream, log request summary."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Bind to structlog contextvars so all log lines in this request include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.monotonic()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_ms = round((time.monotonic() - start) * 1000, 2)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )

        # Record metrics if a collector is attached
        if self._metrics is not None:
            self._metrics.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        structlog.contextvars.clear_contextvars()
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


def get_cors_config() -> dict[str, Any]:
    """Build CORS middleware configuration from environment variables.

    Environment variables:
        DRAGONLIGHT_CORS_ORIGINS: Comma-separated list of allowed origins.
            Default: ``*`` (allow all origins, suitable for development).
            For production: set to specific origin(s), e.g. ``https://app.example.com``.
        DRAGONLIGHT_CORS_METHODS: Comma-separated list of allowed HTTP methods.
            Default: ``GET,POST,OPTIONS``.
        DRAGONLIGHT_CORS_HEADERS: Comma-separated list of allowed request headers.
            Default: ``*`` (allow all headers).

    Returns:
        Dict of keyword arguments for ``CORSMiddleware``.
    """
    origins_env = os.environ.get("DRAGONLIGHT_CORS_ORIGINS", "*")
    methods_env = os.environ.get("DRAGONLIGHT_CORS_METHODS", "GET,POST,OPTIONS")
    headers_env = os.environ.get("DRAGONLIGHT_CORS_HEADERS", "*")

    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    allow_methods = [m.strip() for m in methods_env.split(",") if m.strip()]
    allow_headers = [h.strip() for h in headers_env.split(",") if h.strip()]

    return {
        "allow_origins": allow_origins,
        "allow_credentials": True,
        "allow_methods": allow_methods,
        "allow_headers": allow_headers,
    }
