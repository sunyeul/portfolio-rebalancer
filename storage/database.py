"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "portfolio_rebalancer.sqlite3"

GROUP_SEEDS = [
    ("ungrouped", "미분류", "unknown", 0),
    ("core", "핵심 자산", "core", 10),
    ("satellite_ai_infra", "위성: AI 인프라", "satellite", 20),
    ("satellite_ai_software", "위성: AI 소프트웨어", "satellite", 30),
    ("satellite_space", "위성: 우주/항공", "satellite", 40),
    ("satellite_quantum", "위성: 양자", "satellite", 50),
    ("korea_equity", "한국 주식", "satellite", 60),
    ("bond_mixed", "채권/혼합", "defensive", 70),
    ("cash", "현금", "cash", 80),
]

THESIS_STATUS_SEEDS = [
    ("unknown", "미정", 0),
    ("intact", "유효", 10),
    ("watch", "관찰", 20),
    ("broken", "훼손", 30),
]

TARGET_ALLOCATION_SEEDS = [
    ("core", 0.70, 0.80, 0.90),
    ("satellite", 0.10, 0.20, 0.30),
]

ACTION_PRIORITY_SEEDS = [
    ("increase_dca", "정기매수 증액 후보", 1),
    ("decrease_dca", "정기매수 감액/중단 후보", 2),
    ("review_thesis", "투자 논리 점검", 3),
    ("consider_rebalance_sell", "예외적 리밸런싱 매도 검토", 4),
    ("hold_observe", "유지·관찰", 5),
    ("block_action", "행동 보류", 6),
]

IPS_RULE_SEEDS = [
    ("default_when_uncertain", '"core"'),
    ("immediate_buy_is_exception", "true"),
    ("prefer_dca_over_sell", "true"),
    ("use_momentum_as_dca_intensity_only", "true"),
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

            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                group_type TEXT NOT NULL DEFAULT 'unknown',
                sort_order INTEGER NOT NULL DEFAULT 999,
                is_active INTEGER NOT NULL DEFAULT 1
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
                group_id INTEGER NOT NULL REFERENCES groups(id),
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
                dca_intensity_score REAL,
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ips_target_allocations (
                group_type TEXT PRIMARY KEY,
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
                group_type TEXT,
                group_code TEXT,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rc_violation_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
                ticker TEXT,
                record_json TEXT NOT NULL
            );
            """
        )
        group_type_added = _ensure_column(
            conn, "groups", "group_type", "TEXT NOT NULL DEFAULT 'unknown'"
        )
        _ensure_column(conn, "evaluation_runs", "ips_config_snapshot_json", "TEXT")
        _seed_groups(conn, update_group_type=group_type_added)
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


def _seed_groups(conn: sqlite3.Connection, update_group_type: bool = False) -> None:
    for code, label, group_type, sort_order in GROUP_SEEDS:
        conn.execute(
            """
            INSERT INTO groups (code, label, group_type, sort_order, is_active)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(code) DO NOTHING
            """,
            (code, label, group_type, sort_order),
        )
        if update_group_type:
            conn.execute(
                "UPDATE groups SET group_type = ? WHERE code = ?",
                (group_type, code),
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
    for group_type, min_value, target_value, max_value in TARGET_ALLOCATION_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_target_allocations (group_type, min, target, max)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(group_type) DO NOTHING
            """,
            (group_type, min_value, target_value, max_value),
        )


def _seed_action_priorities(conn: sqlite3.Connection) -> None:
    for action_code, label, priority in ACTION_PRIORITY_SEEDS:
        conn.execute(
            """
            INSERT INTO ips_action_priorities (action_code, label, priority, is_active)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(action_code) DO NOTHING
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
