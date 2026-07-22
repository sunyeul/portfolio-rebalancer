"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.schema import LATEST_SCHEMA_VERSION, migrate, schema_version


DEFAULT_DB_PATH = Path("data") / "portfolio_rebalancer.sqlite3"

THESIS_STATUS_SEEDS = [
    ("unknown", "미정", 0),
    ("valid", "유효", 10),
    ("watch", "관찰", 20),
    ("broken", "훼손", 30),
]

TARGET_ALLOCATION_SEEDS = [
    ("core", 0.70, 0.80, 0.90),
    ("satellite", 0.10, 0.20, 0.30),
    ("experiment", 0.00, 0.00, 0.05),
]


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


def initialize_database() -> None:
    """Back up, migrate, and seed the local persistence database."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    source_version = _migration_source_version(path)
    if source_version is not None:
        _create_migration_backup(path, source_version)

    with connect() as conn:
        migrate(conn)
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)
        _seed_target_allocations(conn)


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


def _create_migration_backup(path: Path, source_version: int) -> Path:
    """Create a SQLite-consistent backup before a forward migration."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(
        f"{path.name}.pre-v{source_version}-to-v{LATEST_SCHEMA_VERSION}-{stamp}.bak"
    )
    with sqlite3.connect(path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


def _seed_lookup(
    conn: sqlite3.Connection,
    table: str,
    rows: list[tuple[str, str, int]],
) -> None:
    for code, label, sort_order in rows:
        conn.execute(
            f"""
            INSERT INTO {table} (code, label, sort_order, is_active)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(code) DO UPDATE SET
                label = excluded.label,
                sort_order = excluded.sort_order,
                is_active = 1
            """,
            (code, label, sort_order),
        )


def _seed_target_allocations(conn: sqlite3.Connection) -> None:
    active_layers = [layer for layer, _, _, _ in TARGET_ALLOCATION_SEEDS]
    placeholders = ",".join("?" for _ in active_layers)
    conn.execute(
        f"DELETE FROM ips_target_allocations WHERE layer NOT IN ({placeholders})",
        active_layers,
    )
    for layer, min_value, target_value, max_value in TARGET_ALLOCATION_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_target_allocations (layer, min, target, max)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(layer) DO NOTHING
            """,
            (layer, min_value, target_value, max_value),
        )
