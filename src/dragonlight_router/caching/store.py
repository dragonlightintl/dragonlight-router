"""Shared SQLite store for caching modules.

Provides schema initialization and connection management.
Both SimpleCache and SemanticCache share the same database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode enabled."""
    assert isinstance(db_path, Path), "db_path must be a Path instance"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    assert conn is not None, "connection must be established"
    return conn


def init_simple_cache_schema(conn: sqlite3.Connection) -> None:
    """Create the simple_cache table if it doesn't exist."""
    assert conn is not None, "connection must not be None"
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simple_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_simple_cache_created
        ON simple_cache(created_at)
    """)
    conn.commit()


def init_semantic_cache_schema(conn: sqlite3.Connection) -> None:
    """Create the semantic_cache table if it doesn't exist."""
    assert conn is not None, "connection must not be None"
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            response TEXT NOT NULL,
            ngrams TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
