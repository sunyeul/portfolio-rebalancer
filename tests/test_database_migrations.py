import sqlite3

import pytest

from storage.database import connect, initialize_database
from storage.schema import LATEST_SCHEMA_VERSION, SchemaVersionError


def _set_database_path(monkeypatch, path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))


def _schema_version(path):
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def test_fresh_database_uses_latest_schema_without_legacy_tables(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with sqlite3.connect(path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "portfolios" in names
    assert "snapshot_evaluation_runs" in names
    assert "analysis_runs" not in names


def test_existing_version_zero_database_is_adopted_without_data_loss(
    monkeypatch, tmp_path
):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE analysis_runs (id INTEGER PRIMARY KEY, payload TEXT);
            INSERT INTO portfolios (name) VALUES ('Existing account');
            INSERT INTO analysis_runs (id, payload) VALUES (7, 'preserve me');
            """
        )
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with connect() as conn:
        assert (
            conn.execute("SELECT name FROM portfolios WHERE id = 1").fetchone()[0]
            == "Existing account"
        )
        assert (
            conn.execute("SELECT payload FROM analysis_runs WHERE id = 7").fetchone()[0]
            == "preserve me"
        )


def test_migration_preserves_snapshots_evaluations_ips_config_and_journal(
    monkeypatch, tmp_path
):
    path = tmp_path / "real-prior-schema.sqlite3"
    _set_database_path(monkeypatch, path)
    initialize_database()

    with connect() as conn:
        portfolio_id = int(
            conn.execute(
                "INSERT INTO portfolios (name) VALUES ('Existing portfolio')"
            ).lastrowid
        )
        asset_id = int(
            conn.execute(
                "INSERT INTO assets (ticker, display_name) VALUES ('VOO', 'S&P 500')"
            ).lastrowid
        )
        thesis_status_id = int(
            conn.execute(
                "SELECT id FROM thesis_statuses WHERE code = 'valid'"
            ).fetchone()["id"]
        )
        snapshot_id = int(
            conn.execute(
                """
                INSERT INTO portfolio_snapshots (portfolio_id, name, note)
                VALUES (?, 'Existing snapshot', 'keep snapshot')
                """,
                (portfolio_id,),
            ).lastrowid
        )
        conn.execute(
            """
            INSERT INTO portfolio_current_states (portfolio_id, state_json)
            VALUES (?, '{"source":"existing"}')
            """,
            (portfolio_id,),
        )
        conn.execute(
            """
            INSERT INTO snapshot_positions (
                snapshot_id, asset_id, allocation, weight, return_total,
                layer, thesis_status_id, position_order
            )
            VALUES (?, ?, 100.0, 1.0, 0.12, 'core', ?, 0)
            """,
            (snapshot_id, asset_id, thesis_status_id),
        )
        conn.execute(
            """
            INSERT INTO snapshot_evaluation_runs (
                snapshot_id, settings_json, result_json, schema_version,
                engine_version, ips_config_hash, status
            )
            VALUES (?, '{}', '{"status":"OK"}', 2, 'existing-engine',
                    'existing-hash', 'active')
            """,
            (snapshot_id,),
        )
        conn.execute(
            "INSERT INTO ips_rules (key, value_json) VALUES ('existing-rule', 'true')"
        )
        conn.execute(
            """
            INSERT INTO ips_action_priorities (action_code, label, priority)
            VALUES ('existing-action', 'Existing', 77)
            """
        )
        conn.execute(
            "UPDATE ips_target_allocations SET target = 0.77 WHERE layer = 'core'"
        )
        conn.execute(
            """
            INSERT INTO journal_entries (
                snapshot_id, date, decision_context, review_items_json,
                decision_note
            )
            VALUES (?, '2026-07-22', 'existing context', '[]', 'keep journal')
            """,
            (snapshot_id,),
        )
        conn.execute("PRAGMA user_version = 0")

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with connect() as conn:
        assert conn.execute(
            "SELECT note FROM portfolio_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()["note"] == "keep snapshot"
        assert conn.execute(
            "SELECT state_json FROM portfolio_current_states WHERE portfolio_id = ?",
            (portfolio_id,),
        ).fetchone()["state_json"] == '{"source":"existing"}'
        assert conn.execute(
            "SELECT return_total FROM snapshot_positions WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["return_total"] == pytest.approx(0.12)
        assert conn.execute(
            "SELECT engine_version FROM snapshot_evaluation_runs WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["engine_version"] == "existing-engine"
        assert conn.execute(
            "SELECT value_json FROM ips_rules WHERE key = 'existing-rule'"
        ).fetchone()["value_json"] == "true"
        assert conn.execute(
            "SELECT priority FROM ips_action_priorities WHERE action_code = 'existing-action'"
        ).fetchone()["priority"] == 77
        assert conn.execute(
            "SELECT target FROM ips_target_allocations WHERE layer = 'core'"
        ).fetchone()["target"] == pytest.approx(0.77)
        assert conn.execute(
            "SELECT decision_note FROM journal_entries WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["decision_note"] == "keep journal"


def test_database_from_newer_schema_is_rejected_without_mutation(
    monkeypatch, tmp_path
):
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


def test_existing_database_is_backed_up_once_before_first_migration(
    monkeypatch, tmp_path
):
    path = tmp_path / "portfolio.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('original')")
    _set_database_path(monkeypatch, path)

    initialize_database()

    backups = list(tmp_path.glob("portfolio.sqlite3.pre-v0-to-v1-*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "original"
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 0

    initialize_database()

    assert list(tmp_path.glob("portfolio.sqlite3.pre-v0-to-v1-*.bak")) == backups


def test_fresh_database_does_not_create_a_backup(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert list(tmp_path.glob("*.bak")) == []
