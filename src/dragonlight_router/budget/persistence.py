"""Budget state persistence — atomic file I/O.

Writes state via .tmp → rename pattern to prevent corruption.
Reads return None for missing or corrupt files (fresh start).
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()


def save_budget_state(state: dict, path: Path) -> Result[None, StatePersistenceError]:
    """Atomically write budget state to disk.

    Creates parent directories if needed. Writes to a .tmp file
    then renames to avoid partial writes on crash.
    """
    assert isinstance(state, dict), "state must be a dictionary"
    assert isinstance(path, Path), "path must be a Path object"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".budget_state_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(path))
        return Ok(None)
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        # Clean up temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        logger.error("budget_state_save_failed", path=str(path), error=str(exc))
        return Err(StatePersistenceError(
            path=str(path),
            message=f"Failed to save budget state: {exc}",
            operation="write"
        ))


def load_budget_state(path: Path) -> Result[dict | None, StatePersistenceError]:
    """Load budget state from disk.

    Returns Ok(dict) if successful, Ok(None) if missing/empty/corrupt (fresh start),
    or Err(StatePersistenceError) on unexpected read errors.
    """
    assert isinstance(path, Path), "path must be a Path object"
    if path.exists():
        assert not path.is_dir(), "path must not be a directory"
    if not path.exists():
        return Ok(None)

    try:
        text = path.read_text()
        if not text.strip():
            return Ok(None)
        return Ok(json.loads(text))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("budget_state_load_failed", path=str(path), error=str(exc))
        return Err(StatePersistenceError(
            path=str(path),
            message=f"Failed to load budget state: {exc}",
            operation="read"
        ))
