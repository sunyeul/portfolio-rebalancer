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
                    "thesis_status": "intact",
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
                    "thesis_status": "intact",
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
