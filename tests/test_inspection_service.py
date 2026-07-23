from services.inspection_service import (
    _select_market_candles,
    preview_inspection,
    run_inspection,
)
from integrations.toss.observation import SyncState
from storage.account_observation_store import insert_snapshot, latest_complete
from storage.database import initialize_database
from storage.database import connect
from storage.policy_store import DEFAULT_POLICY
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


def test_market_candle_selection_uses_only_held_identities(monkeypatch):
    calls = []

    def fake_selector(**kwargs):
        calls.append(kwargs)
        return [{"symbol": kwargs["symbol"]}]

    monkeypatch.setattr(
        "services.inspection_service.list_adjusted_stock_candles", fake_selector
    )
    projection = {
        "synced_at": "2026-07-22T16:01:04+00:00",
        "positions": [
            {"market_country": "US", "symbol": "BBB"},
            {"market_country": "KR", "symbol": "005930"},
            {"market_country": "US", "symbol": "BBB"},
        ],
    }
    risk_policy = {"lookback_sessions": 252}

    selected = _select_market_candles(projection, risk_policy)

    assert list(selected) == [("KR", "005930"), ("US", "BBB")]
    assert calls == [
        {
            "market_country": "KR",
            "symbol": "005930",
            "through_at": "2026-07-22T16:01:04+00:00",
            "limit": 252,
        },
        {
            "market_country": "US",
            "symbol": "BBB",
            "through_at": "2026-07-22T16:01:04+00:00",
            "limit": 252,
        },
    ]


def test_preview_does_not_persist_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "preview.sqlite3"))
    initialize_database()
    with connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM ips_evaluation_runs").fetchone()[0]

    preview = preview_inspection(DEFAULT_POLICY)

    with connect() as conn:
        after = conn.execute("SELECT COUNT(*) FROM ips_evaluation_runs").fetchone()[0]
    assert preview["persisted"] is False
    assert preview["policy_version_id"] is None
    assert preview["market_evidence_fingerprint"]
    assert before == after == 0
