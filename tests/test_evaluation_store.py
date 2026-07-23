from storage.database import connect, initialize_database
from storage.evaluation_store import insert_evaluation_run, latest_evaluation_run
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
        "profile_snapshot": [{"symbol": "AAA"}],
        "profile_hash": "profiles",
        "engine_version": "test",
        "state": "not_evaluable",
        "non_evaluable_reason": "missing_source",
        "result": {"status": "Review"},
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
    assert first["result"] == {"status": "Review"}
