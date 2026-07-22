import pytest

from storage.database import connect, initialize_database
import storage.portfolio_store as portfolio_store
from storage.portfolio_store import (
    _state_payload,
    create_snapshot_evaluation_run,
    create_portfolio,
    create_snapshot,
    get_snapshot,
    get_active_snapshot_evaluation_run,
    list_snapshot_evaluation_runs,
    delete_snapshot,
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
    assert "has_analysis" not in loaded["summary"]
    assert "has_evaluation" not in loaded["summary"]


def test_snapshot_does_not_persist_analysis_or_evaluation_payloads(portfolio_db):
    portfolio = create_portfolio("Snapshot without run payloads")

    snapshot = create_snapshot(
        portfolio["id"],
        "Position-only snapshot",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ],
            "metrics_df": [{"ticker": "VOO", "CAGR": 0.1}],
            "portfolio_metrics": {"cagr": 0.1},
            "benchmark_metrics": None,
            "missing_tickers": [],
            "returns_smooth": [{"Date": "2026-01-01", "VOO": 0.0}],
            "analysis_settings": {"period": "3M", "rf": 0.025, "bench": "SPY"},
            "evaluation_v2": {
                "evaluation_period": {"label": "3M"},
                "layer_evaluations": [],
                "asset_evaluations": [],
                "review_queue": [],
                "journal_draft": [],
                "warnings": [],
                "guardrails": {"not_investment_advice": True},
            },
            "evaluation_settings": {"period": "3M", "bench": "SPY"},
        },
    )

    loaded = get_snapshot(snapshot["id"])

    assert loaded is not None
    assert loaded["analysis"] is None
    assert loaded["evaluation"] is None
    assert loaded["session_state"] == {
        "asset_df": [
            {
                "ticker": "VOO",
                "allocation": 100.0,
                "return_total": None,
                "layer": "core",
                "thesis_status": "valid",
                "weight": 1.0,
            }
        ]
    }


def test_initialize_database_preserves_unversioned_legacy_run_tables(portfolio_db):
    with connect() as conn:
        conn.execute("CREATE TABLE analysis_runs (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE analysis_metrics (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE evaluation_runs (id INTEGER PRIMARY KEY)")

    initialize_database()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('analysis_runs', 'analysis_metrics', 'evaluation_runs')
            ORDER BY name
            """
        ).fetchall()

    assert [row["name"] for row in rows] == [
        "analysis_metrics",
        "analysis_runs",
        "evaluation_runs",
    ]


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


def test_snapshot_update_rejects_position_changes(portfolio_db):
    portfolio = create_portfolio("Immutable snapshot account")
    snapshot = create_snapshot(
        portfolio["id"],
        "Before rejected edit",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ]
        },
    )

    with pytest.raises(portfolio_store.StorageError, match="새 보유현황 스냅샷"):
        update_snapshot(
            snapshot["id"],
            asset_rows=[
                {
                    "ticker": "QQQ",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ],
        )


def test_snapshot_evaluation_runs_are_append_only_with_one_active_run(portfolio_db):
    portfolio = create_portfolio("Evaluation run account")
    snapshot = create_snapshot(
        portfolio["id"],
        "Run snapshot",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ]
        },
    )

    first = create_snapshot_evaluation_run(
        snapshot["id"],
        {"period": "3M", "bench": "SPY"},
        {"evaluation_period": {"label": "3M"}, "layer_evaluations": [], "asset_evaluations": []},
    )
    second = create_snapshot_evaluation_run(
        snapshot["id"],
        {"period": "1Y", "bench": "SPY"},
        {"evaluation_period": {"label": "1Y"}, "layer_evaluations": [], "asset_evaluations": []},
    )

    runs = list_snapshot_evaluation_runs(snapshot["id"])
    active = get_active_snapshot_evaluation_run(snapshot["id"])

    assert [run["id"] for run in runs] == [second["id"], first["id"]]
    assert active["id"] == second["id"]
    assert runs[0]["status"] == "active"
    assert runs[1]["status"] == "superseded"
    assert runs[1]["superseded_by_run_id"] == second["id"]


def test_snapshot_evaluation_run_stale_detection(portfolio_db, monkeypatch):
    portfolio = create_portfolio("Stale evaluation account")
    snapshot = create_snapshot(
        portfolio["id"],
        "Stale snapshot",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ]
        },
    )
    create_snapshot_evaluation_run(
        snapshot["id"],
        {"period": "3M", "bench": "SPY"},
        {"evaluation_period": {"label": "3M"}, "layer_evaluations": [], "asset_evaluations": []},
    )

    assert get_active_snapshot_evaluation_run(snapshot["id"])["is_stale"] is False

    monkeypatch.setattr(portfolio_store, "EVALUATION_ENGINE_VERSION", "changed-engine")

    assert get_active_snapshot_evaluation_run(snapshot["id"])["is_stale"] is True


def test_snapshot_delete_cascades_evaluation_runs(portfolio_db):
    portfolio = create_portfolio("Cascade evaluation account")
    snapshot = create_snapshot(
        portfolio["id"],
        "Cascade snapshot",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ]
        },
    )
    create_snapshot_evaluation_run(
        snapshot["id"],
        {"period": "3M", "bench": "SPY"},
        {"evaluation_period": {"label": "3M"}, "layer_evaluations": [], "asset_evaluations": []},
    )

    delete_snapshot(snapshot["id"])

    with connect() as conn:
        rows = conn.execute("SELECT id FROM snapshot_evaluation_runs").fetchall()

    assert rows == []


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
                }
            ]
        },
    )

    loaded = get_snapshot(snapshot["id"])

    assert loaded is not None
    assert loaded["session_state"]["asset_df"][0]["thesis_status"] == "valid"
