"""Forward-only SQLite schema migrations."""

from __future__ import annotations

import sqlite3


LATEST_SCHEMA_VERSION = 2


class SchemaVersionError(RuntimeError):
    """Raised when the database schema cannot be migrated safely."""


MIGRATION_1_SQL = """
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


MIGRATION_2_SQL = """
CREATE TABLE IF NOT EXISTS broker_account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL,
    sync_started_at TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('complete', 'partial', 'stale', 'failed')),
    is_current_evaluable INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT NOT NULL,
    source_timestamps_json TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    reconciliation_json TEXT NOT NULL,
    total_value_krw REAL,
    invested_value_krw REAL,
    cash_value_krw REAL,
    UNIQUE(account_alias, source_fingerprint)
);

CREATE TABLE IF NOT EXISTS broker_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    market_country TEXT NOT NULL,
    currency TEXT NOT NULL,
    quantity REAL NOT NULL,
    last_price REAL NOT NULL,
    average_purchase_price REAL NOT NULL,
    market_value_native REAL NOT NULL,
    market_value_krw REAL,
    cost_native REAL NOT NULL,
    cost_krw REAL,
    profit_loss_native REAL NOT NULL,
    profit_loss_krw REAL,
    daily_profit_loss_native REAL NOT NULL,
    daily_profit_loss_krw REAL,
    UNIQUE(snapshot_id, symbol)
);

CREATE TABLE IF NOT EXISTS broker_cash_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    buying_power_native REAL NOT NULL,
    buying_power_krw REAL,
    UNIQUE(snapshot_id, currency)
);

CREATE TABLE IF NOT EXISTS broker_exchange_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    rate REAL NOT NULL,
    mid_rate REAL,
    valid_from TEXT,
    valid_until TEXT
);

CREATE TABLE IF NOT EXISTS broker_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    currency TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    ordered_at TEXT,
    canceled_at TEXT,
    quantity REAL NOT NULL,
    order_price_native REAL,
    order_amount_native REAL,
    filled_quantity REAL NOT NULL,
    average_filled_price_native REAL,
    filled_amount_native REAL,
    commission_native REAL,
    tax_native REAL,
    filled_at TEXT,
    settlement_date TEXT,
    UNIQUE(snapshot_id, order_id)
);

CREATE INDEX IF NOT EXISTS idx_broker_snapshots_latest
    ON broker_account_snapshots(account_alias, synced_at, id);
CREATE INDEX IF NOT EXISTS idx_broker_holdings_snapshot
    ON broker_holdings(snapshot_id, symbol);
CREATE INDEX IF NOT EXISTS idx_broker_orders_snapshot
    ON broker_orders(snapshot_id, order_id);
"""


MIGRATIONS = {1: MIGRATION_1_SQL, 2: MIGRATION_2_SQL}


def schema_version(conn: sqlite3.Connection) -> int:
    """Return SQLite's application schema version."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every pending migration in order without destructive cleanup."""
    current = schema_version(conn)
    if current > LATEST_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Database schema {current} is newer than supported {LATEST_SCHEMA_VERSION}."
        )

    for target in range(current + 1, LATEST_SCHEMA_VERSION + 1):
        script = MIGRATIONS[target]
        try:
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;"
            )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
    return schema_version(conn)
