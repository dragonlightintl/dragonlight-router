# =============================================================================
# Security Posture (SEC-004, SEC-008)
# -----------------------------------------------------------------------------
# - Multi-stage build: build dependencies never ship in the runtime image.
# - Hash-verified dependencies: requirements-hashed.txt (pip --require-hashes)
#   ensures supply-chain integrity. Regenerate with:
#       pip-compile --generate-hashes --output-file=requirements-hashed.txt pyproject.toml
# - Non-root user: the runtime stage runs as uid/gid "router" (never root).
# - Read-only root filesystem: enforced via docker-compose security_opt /
#   Kubernetes securityContext. /tmp is a tmpfs mount.
# - Capability drops: all Linux capabilities dropped (cap_drop: ALL in compose).
# - No new privileges: no-new-privileges:true prevents setuid escalation.
# =============================================================================

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies — use hash-verified lockfile when present
COPY pyproject.toml .
COPY requirements-hashed.txt* ./
COPY src/ src/
COPY config/ config/

RUN if [ -f requirements-hashed.txt ]; then \
        pip install --no-cache-dir --require-hashes -r requirements-hashed.txt && \
        pip install --no-cache-dir --no-deps ".[all]"; \
    else \
        pip install --no-cache-dir ".[all]"; \
    fi

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
RUN mkdir -p /app/state /tmp && chown -R router:router /app /tmp
USER router

# State directory for budget/health persistence across restarts
VOLUME ["/app/state"]

# Default to loopback; operators must override to 0.0.0.0 for container networking
ENV DRAGONLIGHT_HOST=127.0.0.1
ENV DRAGONLIGHT_PORT=8100
ENV DRAGONLIGHT_STATE_DIR=/app/state

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/v1/health')" || exit 1

ENTRYPOINT ["python3", "-m", "dragonlight_router.server.app"]
