"""Forward-only SQLite schema migrations."""

from __future__ import annotations

import sqlite3


LATEST_SCHEMA_VERSION = 5


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


MIGRATION_3_SQL = """
CREATE TABLE IF NOT EXISTS account_tracking_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL UNIQUE,
    baseline_snapshot_id INTEGER NOT NULL UNIQUE
        REFERENCES broker_account_snapshots(id),
    tracking_started_at TEXT NOT NULL,
    initial_principal_krw REAL NOT NULL,
    baseline_fx_rate REAL,
    confirmed_at TEXT NOT NULL,
    confirmation_fingerprint TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS account_cash_flow_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER NOT NULL REFERENCES account_tracking_baselines(id),
    from_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    to_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    currency TEXT NOT NULL,
    observed_delta_native REAL NOT NULL,
    explained_trade_delta_native REAL NOT NULL,
    residual_native REAL NOT NULL,
    residual_krw REAL,
    materiality_threshold_krw REAL NOT NULL,
    bridge_basis TEXT NOT NULL CHECK(
        bridge_basis IN ('filled_at', 'settlement_date', 'none')
    ),
    candidate_fingerprint TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_cash_flow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES account_cash_flow_candidates(id),
    classification TEXT NOT NULL CHECK(classification IN (
        'external_deposit', 'external_withdrawal', 'investment_income',
        'fee_or_tax', 'internal_fx', 'rounding_or_false_positive',
        'other_non_external'
    )),
    confirmed_amount_native REAL,
    confirmed_amount_krw REAL,
    effective_at TEXT,
    note TEXT NOT NULL DEFAULT '',
    decided_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_performance_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER NOT NULL REFERENCES account_tracking_baselines(id),
    through_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    input_fingerprint TEXT NOT NULL UNIQUE,
    engine_version TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('complete', 'partial', 'blocked')),
    data_quality_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_performance_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES account_performance_runs(id),
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    previous_snapshot_id INTEGER REFERENCES broker_account_snapshots(id),
    point_at TEXT NOT NULL,
    evaluation_state TEXT NOT NULL CHECK(
        evaluation_state IN ('evaluable', 'non_evaluable')
    ),
    evaluation_reason TEXT,
    total_value_krw REAL,
    invested_value_krw REAL,
    cash_value_krw REAL,
    current_cost_basis_krw REAL,
    unrealized_pnl_krw REAL,
    tracking_principal_krw REAL,
    cumulative_external_flow_krw REAL,
    account_gain_krw REAL,
    simple_return REAL,
    interval_twr REAL,
    segment_id INTEGER,
    segment_twr REAL,
    tracked_realized_pnl_krw REAL,
    actual_realized_pnl_krw REAL,
    fx_remeasurement_krw REAL,
    UNIQUE(run_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS account_execution_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES account_performance_runs(id),
    source_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    currency TEXT NOT NULL,
    side TEXT NOT NULL,
    filled_at TEXT,
    settlement_date TEXT,
    filled_quantity_native REAL NOT NULL,
    filled_amount_native REAL,
    commission_native REAL,
    tax_native REAL,
    actual_basis_before_native REAL,
    tracking_basis_before_native REAL,
    actual_realized_pnl_native REAL,
    tracking_realized_pnl_native REAL,
    realized_pnl_krw REAL,
    krw_conversion_snapshot_id INTEGER
        REFERENCES broker_account_snapshots(id),
    UNIQUE(run_id, order_id)
);

CREATE INDEX IF NOT EXISTS idx_perf_points_run_snapshot
    ON account_performance_points(run_id, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_perf_candidates_baseline
    ON account_cash_flow_candidates(baseline_id, from_snapshot_id, to_snapshot_id);
"""


MIGRATION_4_SQL = """
PRAGMA secure_delete = ON;

DROP TABLE IF EXISTS journal_entries;
DROP TABLE IF EXISTS snapshot_evaluation_runs;
DROP TABLE IF EXISTS snapshot_positions;
DROP TABLE IF EXISTS portfolio_current_states;
DROP TABLE IF EXISTS portfolio_snapshots;
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS ips_action_priorities;
DROP TABLE IF EXISTS ips_rules;
DROP TABLE IF EXISTS ips_target_allocations;
DROP TABLE IF EXISTS thesis_statuses;
DROP TABLE IF EXISTS analysis_runs;

CREATE TABLE ips_instrument_profiles (
    account_alias TEXT NOT NULL,
    market_country TEXT NOT NULL,
    symbol TEXT NOT NULL,
    layer TEXT NOT NULL CHECK(layer IN ('core', 'satellite', 'experiment')),
    thesis_status TEXT NOT NULL
        CHECK(thesis_status IN ('unknown', 'valid', 'watch', 'broken')),
    thesis_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(account_alias, market_country, symbol)
);

CREATE TABLE ips_policy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL,
    version INTEGER NOT NULL,
    policy_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL UNIQUE,
    superseded_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_alias, version)
);

CREATE UNIQUE INDEX idx_ips_policy_one_active
    ON ips_policy_versions(account_alias)
    WHERE superseded_at IS NULL;
"""


MIGRATION_5_SQL = """
CREATE TABLE IF NOT EXISTS ips_evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    performance_run_id INTEGER REFERENCES account_performance_runs(id),
    policy_version_id INTEGER NOT NULL REFERENCES ips_policy_versions(id),
    source_fingerprint TEXT NOT NULL,
    performance_fingerprint TEXT,
    policy_hash TEXT NOT NULL,
    profile_snapshot_json TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('complete', 'not_evaluable', 'failed')),
    non_evaluable_reason TEXT,
    result_json TEXT NOT NULL,
    evaluation_fingerprint TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ips_evaluation_runs_account_created
    ON ips_evaluation_runs(account_alias, created_at, id);
CREATE INDEX IF NOT EXISTS idx_ips_evaluation_runs_snapshot
    ON ips_evaluation_runs(snapshot_id, id);
"""


MIGRATIONS = {
    1: MIGRATION_1_SQL,
    2: MIGRATION_2_SQL,
    3: MIGRATION_3_SQL,
    4: MIGRATION_4_SQL,
    5: MIGRATION_5_SQL,
}


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
