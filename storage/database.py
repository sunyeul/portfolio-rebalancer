"""SQLite connection and Toss-only schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from storage.policy_store import ensure_default_policy
from storage.schema import migrate


DEFAULT_DB_PATH = Path("data") / "portfolio_rebalancer.sqlite3"


class DatabaseIntegrityError(RuntimeError):
    """Raised when a migrated database fails SQLite integrity checks."""


def db_path() -> Path:
    """Return the configured SQLite database path."""
    return Path(os.getenv("PORTFOLIO_DB_PATH", str(DEFAULT_DB_PATH)))


def connect() -> sqlite3.Connection:
    """Open a SQLite connection with application defaults."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _assert_integrity(conn: sqlite3.Connection) -> None:
    """Fail closed when SQLite reports structural corruption."""
    result = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
    if result != "ok" or foreign_keys:
        raise DatabaseIntegrityError(
            f"database integrity failed: integrity={result}, "
            f"foreign_key_errors={len(foreign_keys)}"
        )


def initialize_database() -> None:
    """Migrate, seed, validate, and vacuum the Toss-only local database."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        migrate(conn)
        ensure_default_policy(conn)
        _assert_integrity(conn)
