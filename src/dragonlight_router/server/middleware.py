"""Security middleware for the Dragonlight Router HTTP server.

Provides rate-limiting middleware using a token-bucket algorithm
per client IP address.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger()


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

        response = await call_next(request)  # type: ignore[operator]
        return response
