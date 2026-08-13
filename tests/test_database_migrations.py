import sqlite3

import pytest

import storage.schema as schema_module
from storage.database import connect, initialize_database
from storage.schema import LATEST_SCHEMA_VERSION, SchemaVersionError


ACTIVE_TABLES = {
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
    "ips_policy_versions",
    "ips_policy_candidates",
    "toss_market_candles",
    "ips_evaluation_runs",
    "ips_retrospective_cases",
    "ips_retrospective_reviews",
}


def _set_database_path(monkeypatch, path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))


def _schema_version(path):
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _table_names(path):
    with sqlite3.connect(path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        }


def _insert_snapshot(path):
    with sqlite3.connect(path) as conn:
        return int(
            conn.execute(
                """
                INSERT INTO broker_account_snapshots (
                    account_alias, sync_started_at, synced_at, state,
                    is_current_evaluable, source_fingerprint,
                    source_timestamps_json, data_quality_json,
                    reconciliation_json, total_value_krw,
                    invested_value_krw, cash_value_krw
                ) VALUES (
                    'toss-brokerage', '2026-07-23T00:00:00Z',
                    '2026-07-23T00:01:00Z', 'complete', 1,
                    'preservation-source', '{}', '{}', '{}', 100.0, 90.0,
                    10.0
                )
                """
            ).lastrowid
        )


def test_fresh_database_uses_current_schema(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION == 11
    assert _table_names(path) == ACTIVE_TABLES
    with sqlite3.connect(path) as conn:
        evaluation_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(ips_evaluation_runs)")
        }
        assert {"market_evidence_fingerprint", "market_evidence_json"}.issubset(
            evaluation_columns
        )
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_existing_v10_database_preserves_user_data(monkeypatch, tmp_path):
    path = tmp_path / "current.sqlite3"
    _set_database_path(monkeypatch, path)
    initialize_database()
    snapshot_id = _insert_snapshot(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE ips_retrospective_reviews")
        conn.execute("DROP TABLE ips_retrospective_cases")
        conn.execute("PRAGMA user_version = 10")

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT source_fingerprint FROM broker_account_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()[0]
            == "preservation-source"
        )
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ips_retrospective_cases'"
        ).fetchone()


def test_empty_unversioned_database_is_initialized(monkeypatch, tmp_path):
    path = tmp_path / "empty.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    assert _table_names(path) == ACTIVE_TABLES


def test_nonempty_unversioned_database_is_rejected_without_mutation(
    monkeypatch, tmp_path
):
    path = tmp_path / "unversioned.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('unchanged')")
    _set_database_path(monkeypatch, path)

    with pytest.raises(SchemaVersionError, match="unversioned"):
        initialize_database()

    assert _schema_version(path) == 0
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "unchanged"


@pytest.mark.parametrize(
    "version", [LATEST_SCHEMA_VERSION - 1, LATEST_SCHEMA_VERSION + 1]
)
def test_unsupported_schema_is_rejected_without_mutation(
    monkeypatch, tmp_path, version
):
    path = tmp_path / f"schema-{version}.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('unchanged')")
        conn.execute(f"PRAGMA user_version = {version}")
    _set_database_path(monkeypatch, path)

    with pytest.raises(SchemaVersionError, match="Unsupported.*schema"):
        initialize_database()

    assert _schema_version(path) == version
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "unchanged"


def test_incomplete_v10_database_is_rejected_before_retrospective_ddl(monkeypatch, tmp_path):
    path = tmp_path / "incomplete-v10.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE broker_account_snapshots (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE ips_evaluation_runs (id INTEGER PRIMARY KEY)")
        conn.execute("PRAGMA user_version = 10")
    _set_database_path(monkeypatch, path)

    with pytest.raises(SchemaVersionError, match="Unsupported schema 10"):
        initialize_database()

    assert _schema_version(path) == 10
    assert "ips_retrospective_cases" not in _table_names(path)


def test_failed_baseline_creation_rolls_back(monkeypatch, tmp_path):
    path = tmp_path / "rollback.sqlite3"
    _set_database_path(monkeypatch, path)
    broken = (
        schema_module.CURRENT_SCHEMA_SQL + "\nINSERT INTO missing_table VALUES (1);"
    )
    monkeypatch.setattr(schema_module, "CURRENT_SCHEMA_SQL", broken)

    with pytest.raises(sqlite3.OperationalError):
        with connect() as conn:
            schema_module.migrate(conn)

    assert _schema_version(path) == 0
    assert _table_names(path) == set()
