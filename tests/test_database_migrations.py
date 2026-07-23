import sqlite3

import pytest

import storage.schema as schema_module
from storage.database import connect, initialize_database
from storage.schema import (
    LATEST_SCHEMA_VERSION,
    MIGRATION_1_SQL,
    MIGRATION_2_SQL,
    MIGRATION_3_SQL,
    SchemaVersionError,
)


GENERIC_TABLES = {
    "portfolios",
    "assets",
    "thesis_statuses",
    "portfolio_snapshots",
    "portfolio_current_states",
    "snapshot_positions",
    "snapshot_evaluation_runs",
    "ips_target_allocations",
    "ips_action_priorities",
    "ips_rules",
    "journal_entries",
    "analysis_runs",
}


def _set_database_path(monkeypatch, path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))


def _schema_version(path):
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _create_v3_fixture(path):
    with sqlite3.connect(path) as conn:
        for script in (MIGRATION_1_SQL, MIGRATION_2_SQL, MIGRATION_3_SQL):
            conn.executescript(script)
        conn.execute("PRAGMA user_version = 3")


def _insert_v3_generic_and_toss_rows(path):
    with sqlite3.connect(path) as conn:
        portfolio_id = int(
            conn.execute("INSERT INTO portfolios (name) VALUES ('legacy')").lastrowid
        )
        conn.execute(
            "INSERT INTO portfolio_snapshots (portfolio_id, name) VALUES (?, 'legacy')",
            (portfolio_id,),
        )
        snapshot_id = int(
            conn.execute(
                """
                INSERT INTO broker_account_snapshots (
                    account_alias, sync_started_at, synced_at, state,
                    is_current_evaluable, source_fingerprint,
                    source_timestamps_json, data_quality_json,
                    reconciliation_json, total_value_krw, invested_value_krw,
                    cash_value_krw
                )
                VALUES (
                    'toss-brokerage', '2026-07-22T00:00:00Z',
                    '2026-07-22T00:01:00Z', 'complete', 1,
                    'snapshot-fingerprint', '{}', '{}', '{}', 100.0, 90.0,
                    10.0
                )
                """
            ).lastrowid
        )
        baseline_id = int(
            conn.execute(
                """
                INSERT INTO account_tracking_baselines (
                    account_alias, baseline_snapshot_id, tracking_started_at,
                    initial_principal_krw, baseline_fx_rate, confirmed_at,
                    confirmation_fingerprint
                )
                VALUES (
                    'toss-brokerage', ?, '2026-07-22T00:02:00Z', 100.0,
                    1480.0, '2026-07-22T00:03:00Z', 'baseline-fingerprint'
                )
                """,
                (snapshot_id,),
            ).lastrowid
        )
        run_id = int(
            conn.execute(
                """
                INSERT INTO account_performance_runs (
                    baseline_id, through_snapshot_id, input_fingerprint,
                    engine_version, state, data_quality_json
                )
                VALUES (?, ?, 'run-fingerprint', 'test', 'complete', '{}')
                """,
                (baseline_id, snapshot_id),
            ).lastrowid
        )
    return {
        "snapshot_id": snapshot_id,
        "baseline_id": baseline_id,
        "run_id": run_id,
        "snapshot_fingerprint": "snapshot-fingerprint",
        "baseline_fingerprint": "baseline-fingerprint",
        "run_fingerprint": "run-fingerprint",
    }


def _table_names(path):
    with sqlite3.connect(path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def test_fresh_database_uses_latest_schema_without_generic_tables(
    monkeypatch, tmp_path
):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION == 5
    names = _table_names(path)
    assert GENERIC_TABLES.isdisjoint(names)
    assert {
        "broker_account_snapshots",
        "broker_holdings",
        "broker_cash_observations",
        "broker_exchange_rates",
        "broker_orders",
        "account_tracking_baselines",
        "account_cash_flow_candidates",
        "account_cash_flow_decisions",
        "account_performance_runs",
        "account_performance_points",
        "account_execution_ledger",
        "ips_instrument_profiles",
        "ips_policy_versions",
        "ips_evaluation_runs",
    }.issubset(names)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v3_to_v4_drops_generic_tables_and_preserves_toss_evidence(
    monkeypatch, tmp_path
):
    path = tmp_path / "v3.sqlite3"
    _create_v3_fixture(path)
    expected = _insert_v3_generic_and_toss_rows(path)
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == 5
    with sqlite3.connect(path) as conn:
        assert GENERIC_TABLES.isdisjoint(_table_names(path))
        assert (
            conn.execute(
                "SELECT source_fingerprint FROM broker_account_snapshots WHERE id = ?",
                (expected["snapshot_id"],),
            ).fetchone()[0]
            == expected["snapshot_fingerprint"]
        )
        assert (
            conn.execute(
                "SELECT confirmation_fingerprint FROM account_tracking_baselines "
                "WHERE id = ?",
                (expected["baseline_id"],),
            ).fetchone()[0]
            == expected["baseline_fingerprint"]
        )
        assert (
            conn.execute(
                "SELECT input_fingerprint FROM account_performance_runs WHERE id = ?",
                (expected["run_id"],),
            ).fetchone()[0]
            == expected["run_fingerprint"]
        )


def test_failed_v4_migration_rolls_back_all_drops(monkeypatch, tmp_path):
    path = tmp_path / "rollback.sqlite3"
    _create_v3_fixture(path)
    _set_database_path(monkeypatch, path)
    broken = schema_module.MIGRATION_4_SQL + "\nINSERT INTO missing_table VALUES (1);"
    monkeypatch.setitem(schema_module.MIGRATIONS, 4, broken)

    with pytest.raises(sqlite3.OperationalError), connect() as conn:
        schema_module.migrate(conn)

    assert _schema_version(path) == 3
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolios'"
            ).fetchone()
            is not None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='ips_instrument_profiles'"
            ).fetchone()
            is None
        )


def test_v4_migration_creates_no_adjacent_backup(monkeypatch, tmp_path):
    path = tmp_path / "portfolio.sqlite3"
    _create_v3_fixture(path)
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert list(tmp_path.glob("*.bak")) == []


def test_database_from_newer_schema_is_rejected_without_mutation(monkeypatch, tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('unchanged')")
        conn.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 1}")
    _set_database_path(monkeypatch, path)

    with pytest.raises(SchemaVersionError, match="newer than supported"):
        initialize_database()

    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "unchanged"
        assert (
            int(conn.execute("PRAGMA user_version").fetchone()[0])
            == LATEST_SCHEMA_VERSION + 1
        )
