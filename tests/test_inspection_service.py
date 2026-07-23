from services.inspection_service import run_inspection
from integrations.toss.observation import SyncState
from storage.account_observation_store import insert_snapshot, latest_complete
from storage.database import initialize_database
from tests.test_account_observation_store import _snapshot


def test_failed_latest_attempt_is_persisted_as_non_evaluable(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "inspection.sqlite3"))
    initialize_database()
    complete = insert_snapshot(_snapshot())
    failed = insert_snapshot(
        _snapshot(
            state=SyncState.FAILED,
            fingerprint="failed-attempt",
            synced_at="2026-07-23T00:05:00+00:00",
        )
    )
    assert latest_complete() is None
    evaluation = run_inspection()
    assert evaluation["snapshot_id"] == failed["id"]
    assert evaluation["state"] == "not_evaluable"
    assert evaluation["result"]["source"]["snapshot_id"] == failed["id"]
    assert evaluation["result"]["review_queue"][0]["kind"] == "source"
    assert complete["id"] != failed["id"]
