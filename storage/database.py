"""SQLite connection and Toss-only schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from storage.policy_store import ensure_default_policy
from storage.schema import LATEST_SCHEMA_VERSION, migrate, schema_version


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
    source_version = _migration_source_version(path)
    is_real_upgrade = (
        source_version is not None and source_version < LATEST_SCHEMA_VERSION
    )

    with connect() as conn:
        if is_real_upgrade:
            conn.execute("PRAGMA secure_delete = ON")
        migrate(conn)
        ensure_default_policy(conn)
        _assert_integrity(conn)

    if is_real_upgrade:
        with connect() as conn:
            conn.execute("VACUUM")
        with connect() as conn:
            _assert_integrity(conn)


def _migration_source_version(path: Path) -> int | None:
    """Return a migratable source version only when a real schema exists."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with sqlite3.connect(path) as conn:
        version = schema_version(conn)
        object_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger', 'view')
                  AND name NOT LIKE 'sqlite_%'
                """
            ).fetchone()[0]
        )
    if object_count == 0 or version >= LATEST_SCHEMA_VERSION:
        return None
    return version
