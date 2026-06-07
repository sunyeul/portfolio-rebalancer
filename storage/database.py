"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "portfolio_rebalancer.sqlite3"

GROUP_SEEDS = [
    ("ungrouped", "미분류", 0),
    ("core", "핵심 자산", 10),
    ("satellite_ai_infra", "위성: AI 인프라", 20),
    ("satellite_ai_software", "위성: AI 소프트웨어", 30),
    ("satellite_space", "위성: 우주/항공", 40),
    ("satellite_quantum", "위성: 양자", 50),
    ("korea_equity", "한국 주식", 60),
    ("bond_mixed", "채권/혼합", 70),
    ("cash", "현금", 80),
]

ROLE_SEEDS = [
    ("unknown", "미정", 0),
    ("broad_etf", "광범위 ETF", 10),
    ("theme_etf", "테마 ETF", 20),
    ("individual", "개별 종목", 30),
    ("duplicate", "중복 포지션", 40),
    ("small_position", "소액 포지션", 50),
]

THESIS_STATUS_SEEDS = [
    ("unknown", "미정", 0),
    ("intact", "유효", 10),
    ("watch", "관찰", 20),
    ("broken", "훼손", 30),
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
                sort_order INTEGER NOT NULL DEFAULT 999,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
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

            CREATE TABLE IF NOT EXISTS snapshot_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
                asset_id INTEGER NOT NULL REFERENCES assets(id),
                allocation REAL NOT NULL,
                weight REAL NOT NULL,
                return_total REAL,
                group_id INTEGER NOT NULL REFERENCES groups(id),
                role_id INTEGER NOT NULL REFERENCES roles(id),
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
                momentum_weight REAL,
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
                efficiency_score_prime REAL,
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        _seed_lookup(conn, "groups", GROUP_SEEDS)
        _seed_lookup(conn, "roles", ROLE_SEEDS)
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)


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
