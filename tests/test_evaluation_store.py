import pytest

from storage.database import connect, initialize_database
from storage.evaluation_store import (
    EvaluationStorageError,
    insert_evaluation_run,
    latest_evaluation_run,
    list_evaluation_runs,
)
from storage.policy_store import get_active_policy


def _evaluation():
    return {
        "account_alias": "toss-brokerage",
        "snapshot_id": 1,
        "performance_run_id": None,
        "policy_version_id": 1,
        "source_fingerprint": "source",
        "performance_fingerprint": None,
        "policy_hash": "policy",
        "engine_version": "phase5-v2",
        "state": "not_evaluable",
        "non_evaluable_reason": "missing_source",
        "result": {
            "engine_version": "phase5-v2",
            "source": {},
            "allocation_state": "not_evaluable",
            "account": {},
            "layers": [],
            "instruments": [],
            "review_queue": [],
        },
        "market_evidence_fingerprint": "market-v1",
        "market_evidence": {
            "US/AAA": {
                "state": "insufficient_history",
                "history_points": 0,
                "source_fingerprint": "range-a",
            }
        },
        "evaluation_fingerprint": "evaluation",
    }


def test_evaluation_store_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "evaluation.sqlite3"))
    initialize_database()
    with connect() as conn:
        snapshot_id = int(
            conn.execute(
                "INSERT INTO broker_account_snapshots (account_alias, sync_started_at, synced_at, state, is_current_evaluable, source_fingerprint, source_timestamps_json, data_quality_json, reconciliation_json) VALUES ('toss-brokerage','a','b','failed',0,'source','{}','{}','{}')"
            ).lastrowid
        )
    evaluation = _evaluation()
    active = get_active_policy()
    evaluation["snapshot_id"] = snapshot_id
    evaluation["policy_version_id"] = active["id"]
    evaluation["policy_hash"] = active["policy_hash"]
    first = insert_evaluation_run(evaluation)
    second = insert_evaluation_run(evaluation)
    assert first == second
    assert latest_evaluation_run()["id"] == first["id"]
    assert first["result"] == evaluation["result"]
    assert first["market_evidence_fingerprint"] == "market-v1"
    assert first["market_evidence"]["US/AAA"]["source_fingerprint"] == "range-a"


def test_market_evidence_is_part_of_evaluation_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "fingerprint.sqlite3"))
    initialize_database()
    with connect() as conn:
        snapshot_id = int(
            conn.execute(
                "INSERT INTO broker_account_snapshots (account_alias, sync_started_at, synced_at, state, is_current_evaluable, source_fingerprint, source_timestamps_json, data_quality_json, reconciliation_json) VALUES ('toss-brokerage','a','b','failed',0,'source','{}','{}','{}')"
            ).lastrowid
        )
    first = _evaluation()
    first["snapshot_id"] = snapshot_id
    active = get_active_policy()
    first["policy_version_id"] = active["id"]
    first["policy_hash"] = active["policy_hash"]
    first["evaluation_fingerprint"] = "evaluation-a"
    second = dict(first)
    second["market_evidence_fingerprint"] = "market-v2"
    second["evaluation_fingerprint"] = "evaluation-b"

    first_persisted = insert_evaluation_run(first)
    persisted = insert_evaluation_run(second)

    assert persisted["id"] != first_persisted["id"]
    assert persisted["market_evidence_fingerprint"] == "market-v2"


def test_evaluation_history_returns_newest_runs_first(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "history.sqlite3"))
    initialize_database()
    with connect() as conn:
        snapshot_id = int(
            conn.execute(
                "INSERT INTO broker_account_snapshots (account_alias, sync_started_at, synced_at, state, is_current_evaluable, source_fingerprint, source_timestamps_json, data_quality_json, reconciliation_json) VALUES ('toss-brokerage','a','b','failed',0,'source','{}','{}','{}')"
            ).lastrowid
        )
    active = get_active_policy()
    first = _evaluation()
    first.update(
        snapshot_id=snapshot_id,
        policy_version_id=active["id"],
        policy_hash=active["policy_hash"],
        evaluation_fingerprint="history-a",
    )
    second = dict(first)
    second["evaluation_fingerprint"] = "history-b"

    first_persisted = insert_evaluation_run(first)
    second_persisted = insert_evaluation_run(second)

    assert [item["id"] for item in list_evaluation_runs(limit=2)] == [
        second_persisted["id"],
        first_persisted["id"],
    ]


def test_evaluation_store_rejects_malformed_phase5_v2_result(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "evaluation.sqlite3"))
    initialize_database()
    with connect() as conn:
        snapshot_id = int(
            conn.execute(
                "INSERT INTO broker_account_snapshots (account_alias, sync_started_at, synced_at, state, is_current_evaluable, source_fingerprint, source_timestamps_json, data_quality_json, reconciliation_json) VALUES ('toss-brokerage','a','b','failed',0,'source','{}','{}','{}')"
            ).lastrowid
        )
    active = get_active_policy()
    evaluation = _evaluation()
    evaluation.update(
        snapshot_id=snapshot_id,
        policy_version_id=active["id"],
        policy_hash=active["policy_hash"],
        result={"status": "Review"},
    )

    with pytest.raises(EvaluationStorageError, match="result.account"):
        insert_evaluation_run(evaluation)
