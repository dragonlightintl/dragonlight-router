# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir ".[all]"

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and config
COPY src/ src/
COPY config/ config/

# Create non-root user
RUN groupadd -r router && useradd -r -g router -d /app router
RUN mkdir -p /app/state && chown -R router:router /app
USER router

# State directory for budget/health persistence across restarts
VOLUME ["/app/state"]

ENV DRAGONLIGHT_HOST=0.0.0.0
ENV DRAGONLIGHT_PORT=8100
ENV DRAGONLIGHT_STATE_DIR=/app/state

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/v1/health')" || exit 1

ENTRYPOINT ["python3", "-m", "dragonlight_router.server.app"]
