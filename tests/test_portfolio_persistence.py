import sqlite3

import pytest

from storage.database import connect, initialize_database
from storage.portfolio_store import (
    _state_payload,
    create_portfolio,
    create_snapshot,
    get_snapshot,
    update_snapshot,
)


@pytest.fixture()
def portfolio_db(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()


def test_state_payload_uses_evaluation_v2_only():
    state, analysis, evaluation = _state_payload(
        {
            "asset_df": [{"ticker": "VOO", "allocation": 100}],
            "metrics_df": [{"ticker": "VOO"}],
            "portfolio_metrics": {"cagr": 0.1},
            "benchmark_metrics": None,
            "missing_tickers": [],
            "evaluation_v2": {
                "evaluation_period": {"label": "3M"},
                "layer_evaluations": [],
                "asset_evaluations": [],
                "review_queue": [],
                "journal_draft": [],
                "warnings": [],
                "guardrails": {"not_investment_advice": True},
            },
        }
    )

    assert state["asset_df"]
    assert analysis["metrics_df"]
    assert evaluation["evaluation_period"]["label"] == "3M"


def test_state_payload_ignores_unrecognized_evaluation_keys():
    _state, _analysis, evaluation = _state_payload(
        {
            "asset_df": [{"ticker": "VOO", "allocation": 100}],
            "other_evaluation": {"rows": [{"ticker": "VOO"}]},
        }
    )

    assert evaluation is None


def test_snapshot_persists_metadata_without_as_of_date_contract(portfolio_db):
    portfolio = create_portfolio("Snapshot metadata account")

    snapshot = create_snapshot(
        portfolio["id"],
        "June review",
        "monthly checkpoint",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "category": "core_market",
                    "dca_enabled": True,
                    "thesis_status": "valid",
                }
            ]
        },
    )

    assert snapshot["name"] == "June review"
    assert snapshot["note"] == "monthly checkpoint"
    assert snapshot["created_at"]
    assert snapshot["updated_at"]
    assert "as_of_date" not in snapshot
    loaded = get_snapshot(snapshot["id"])
    assert loaded is not None
    assert loaded["summary"]["created_at"]
    assert loaded["summary"]["updated_at"]
    assert "as_of_date" not in loaded["summary"]


def test_initialize_database_backfills_existing_snapshot_updated_at(monkeypatch, tmp_path):
    db_path = tmp_path / "portfolio_rebalancer.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            as_of_date TEXT,
            updated_at TEXT
        );
        INSERT INTO portfolio_snapshots (
            portfolio_id,
            name,
            note,
            created_at,
            updated_at
        )
        VALUES (
            1,
            'Legacy snapshot',
            '',
            '2026-06-04 09:00:00',
            NULL
        );
        """
    )
    conn.close()

    initialize_database()

    with connect() as conn:
        row = conn.execute(
            "SELECT created_at, updated_at FROM portfolio_snapshots WHERE id = 1"
        ).fetchone()

    assert row["updated_at"] == row["created_at"]


def test_snapshot_update_persists_editable_metadata(portfolio_db):
    portfolio = create_portfolio("Editable snapshot account")
    snapshot = create_snapshot(
        portfolio["id"],
        "Before edit",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "category": "core_market",
                    "dca_enabled": True,
                    "thesis_status": "valid",
                }
            ]
        },
    )

    stale_timestamp = "2000-01-01 00:00:00"
    with connect() as conn:
        conn.execute(
            "UPDATE portfolio_snapshots SET updated_at = ? WHERE id = ?",
            (stale_timestamp, snapshot["id"]),
        )

    updated = update_snapshot(
        snapshot["id"],
        name="After edit",
        note="메모 수정",
    )

    assert updated["name"] == "After edit"
    assert updated["note"] == "메모 수정"
    assert updated["created_at"] == snapshot["created_at"]
    assert updated["updated_at"] != stale_timestamp
    assert "as_of_date" not in updated


def test_initialize_database_migrates_legacy_intact_thesis_code(monkeypatch, tmp_path):
    db_path = tmp_path / "portfolio_rebalancer.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE thesis_statuses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 999,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        INSERT INTO thesis_statuses (code, label, sort_order, is_active)
        VALUES ('intact', '유효', 10, 1);
        """
    )
    conn.close()

    initialize_database()

    with connect() as conn:
        codes = [
            row["code"]
            for row in conn.execute(
                "SELECT code FROM thesis_statuses ORDER BY sort_order ASC, code ASC"
            ).fetchall()
        ]

    assert codes == ["unknown", "valid", "watch", "broken"]


def test_snapshot_defaults_missing_thesis_status_to_valid(portfolio_db):
    portfolio = create_portfolio("Default thesis account")

    snapshot = create_snapshot(
        portfolio["id"],
        "Default thesis",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "category": "core_market",
                    "dca_enabled": True,
                }
            ]
        },
    )

    loaded = get_snapshot(snapshot["id"])

    assert loaded is not None
    assert loaded["session_state"]["asset_df"][0]["thesis_status"] == "valid"
