"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "portfolio_rebalancer.sqlite3"

THESIS_STATUS_SEEDS = [
    ("unknown", "미정", 0),
    ("intact", "유효", 10),
    ("watch", "관찰", 20),
    ("broken", "훼손", 30),
]

TARGET_ALLOCATION_SEEDS = [
    ("core", 0.70, 0.80, 0.90),
    ("satellite_ai_infra", 0.00, 0.08, 0.15),
    ("satellite_ai_software", 0.00, 0.04, 0.10),
    ("satellite_nextgen", 0.00, 0.08, 0.15),
]

ACTION_PRIORITY_SEEDS = [
    ("block_action", "행동 보류", 1),
    ("rebalance_sell_review", "리밸런싱 매도 검토", 2),
    ("risk_control_review", "위험 관리 점검", 3),
    ("review_before_action", "실행 전 점검", 4),
    ("reduce_or_pause_dca", "정기매수 축소·중단 후보", 5),
    ("increase_dca", "정기매수 증액 후보", 6),
    ("hold_observe", "유지·관찰", 7),
]

IPS_RULE_SEEDS = [
    ("default_when_uncertain", '"core"'),
    ("immediate_buy_is_exception", "true"),
    ("prefer_dca_over_sell", "true"),
    ("min_trade_pct", "0.01"),
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
                "group" TEXT NOT NULL DEFAULT 'core',
                dca_enabled INTEGER NOT NULL DEFAULT 1,
                thesis_status_id INTEGER NOT NULL REFERENCES thesis_statuses(id),
                position_order INTEGER NOT NULL DEFAULT 0,
                UNIQUE(snapshot_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL UNIQUE REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                period TEXT,
                rf REAL,
                bench TEXT,
                portfolio_metrics_json TEXT,
                benchmark_metrics_json TEXT,
                missing_tickers_json TEXT,
                returns_smooth_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS analysis_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
                asset_id INTEGER REFERENCES assets(id),
                ticker TEXT NOT NULL,
                cagr REAL,
                volatility REAL,
                sharpe REAL,
                max_drawdown REAL,
                information_ratio REAL,
                beta REAL,
                alpha REAL,
                risk_contribution REAL,
                return_contribution REAL,
                weight REAL,
                efficiency_score REAL,
                return_total REAL,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL UNIQUE REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                rc_over_thresh_pct REAL,
                e_thresh REAL,
                target_weights_json TEXT,
                ips_config_snapshot_json TEXT,
                playbook_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ips_target_allocations (
                "group" TEXT PRIMARY KEY,
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

            CREATE TABLE IF NOT EXISTS evaluation_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
                asset_id INTEGER REFERENCES assets(id),
                ticker TEXT NOT NULL,
                current_weight_pct REAL,
                target_weight_pct REAL,
                gap_pct REAL,
                adjusted_gap_pct REAL,
                rc_over_pct REAL,
                rc_target_pct REAL,
                should_execute INTEGER,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ips_action_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
                ticker TEXT,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_summary_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
                "group" TEXT,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rc_violation_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
                ticker TEXT,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL UNIQUE REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                date TEXT NOT NULL,
                decision_context TEXT NOT NULL,
                playbook_code TEXT,
                dca_changes_considered_json TEXT NOT NULL DEFAULT '[]',
                review_items_json TEXT NOT NULL DEFAULT '[]',
                decision_note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        _ensure_column(conn, "evaluation_runs", "ips_config_snapshot_json", "TEXT")
        _ensure_column(conn, "evaluation_runs", "playbook_json", "TEXT")
        _ensure_target_allocation_table(conn)
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)
        _seed_target_allocations(conn)
        _seed_action_priorities(conn)
        _seed_ips_rules(conn)


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> bool:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        return True
    return False


def _ensure_target_allocation_table(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(ips_target_allocations)")
    }
    if "group" in columns:
        return
    conn.execute("DROP TABLE IF EXISTS ips_target_allocations")
    conn.execute(
        """
        CREATE TABLE ips_target_allocations (
            "group" TEXT PRIMARY KEY,
            min REAL NOT NULL,
            target REAL NOT NULL,
            max REAL NOT NULL
        )
        """
    )


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
            ON CONFLICT(code) DO NOTHING
            """,
            (code, label, sort_order),
        )


def _seed_target_allocations(conn: sqlite3.Connection) -> None:
    active_groups = [group for group, _, _, _ in TARGET_ALLOCATION_SEEDS]
    placeholders = ",".join("?" for _ in active_groups)
    conn.execute(
        f'DELETE FROM ips_target_allocations WHERE "group" NOT IN ({placeholders})',
        active_groups,
    )
    for group, min_value, target_value, max_value in TARGET_ALLOCATION_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_target_allocations ("group", min, target, max)
            VALUES (?, ?, ?, ?)
            ON CONFLICT("group") DO NOTHING
            """,
            (group, min_value, target_value, max_value),
        )


def _seed_action_priorities(conn: sqlite3.Connection) -> None:
    active_codes = [action_code for action_code, _, _ in ACTION_PRIORITY_SEEDS]
    placeholders = ",".join("?" for _ in active_codes)
    conn.execute(
        f"DELETE FROM ips_action_priorities WHERE action_code NOT IN ({placeholders})",
        active_codes,
    )
    for action_code, label, priority in ACTION_PRIORITY_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_action_priorities (action_code, label, priority, is_active)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(action_code) DO UPDATE SET
                label = excluded.label,
                priority = excluded.priority,
                is_active = 1
            """,
            (action_code, label, priority),
        )


def _seed_ips_rules(conn: sqlite3.Connection) -> None:
    for key, value_json in IPS_RULE_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_rules (key, value_json)
            VALUES (?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (key, value_json),
        )
