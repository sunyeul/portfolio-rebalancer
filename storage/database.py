"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


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
    """Create all persistence tables and seed lookup values."""
    with connect() as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS analysis_metrics;
            DROP TABLE IF EXISTS evaluation_runs;
            DROP TABLE IF EXISTS analysis_runs;

            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE,
                display_name TEXT,
                asset_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS thesis_statuses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 999,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
                name TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio_current_states (
                portfolio_id INTEGER PRIMARY KEY REFERENCES portfolios(id) ON DELETE CASCADE,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS snapshot_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                asset_id INTEGER NOT NULL REFERENCES assets(id),
                allocation REAL NOT NULL,
                weight REAL NOT NULL,
                return_total REAL,
                layer TEXT NOT NULL DEFAULT 'core',
                thesis_status_id INTEGER NOT NULL REFERENCES thesis_statuses(id),
                position_order INTEGER NOT NULL DEFAULT 0,
                UNIQUE(snapshot_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS snapshot_evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                settings_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                engine_version TEXT NOT NULL,
                ips_config_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                superseded_by_run_id INTEGER REFERENCES snapshot_evaluation_runs(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_snapshot_evaluation_runs_snapshot_status
                ON snapshot_evaluation_runs(snapshot_id, status, id);

            CREATE TABLE IF NOT EXISTS ips_target_allocations (
                layer TEXT PRIMARY KEY,
                min REAL NOT NULL,
                target REAL NOT NULL,
                max REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ips_action_priorities (
                action_code TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                priority INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS ips_rules (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL UNIQUE REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                date TEXT NOT NULL,
                decision_context TEXT NOT NULL,
                playbook_code TEXT,
                review_items_json TEXT NOT NULL DEFAULT '[]',
                decision_note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)
        _seed_target_allocations(conn)


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
