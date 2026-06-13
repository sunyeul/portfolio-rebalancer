import pytest

from storage.database import initialize_database
from storage.journal_store import get_journal, update_journal, upsert_journal
from storage.portfolio_store import StorageError, create_portfolio, create_snapshot


def _create_snapshot() -> dict:
    portfolio = create_portfolio("Journal store account")
    return create_snapshot(
        portfolio["id"],
        "Monthly review",
        "",
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "weight": 1.0,
                    "return_total": None,
                    "group": "core",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                }
            ]
        },
    )


@pytest.fixture()
def journal_db(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()


def test_journal_store_upserts_and_patches_latest_snapshot_entry(journal_db):
    snapshot = _create_snapshot()

    assert get_journal(snapshot["id"]) is None

    created = upsert_journal(
        snapshot["id"],
        {
            "date": "2026-06-14",
            "decision_context": "regular_review",
            "playbook_code": "regular_review",
            "dca_changes_considered": [{"ticker": "VOO", "candidate": "increase"}],
            "review_items": [{"ticker": "QQQ", "queue": "thesis_review"}],
            "decision_note": "정기 점검 기록",
        },
    )

    assert created["snapshot_id"] == snapshot["id"]
    assert created["dca_changes_considered"] == [
        {"ticker": "VOO", "candidate": "increase"}
    ]
    assert created["review_items"] == [{"ticker": "QQQ", "queue": "thesis_review"}]
    assert created["decision_note"] == "정기 점검 기록"

    patched = update_journal(
        snapshot["id"],
        {
            "decision_note": "검토 큐 확인 완료",
            "review_items": [],
        },
    )

    assert patched["date"] == "2026-06-14"
    assert patched["decision_context"] == "regular_review"
    assert patched["dca_changes_considered"] == [
        {"ticker": "VOO", "candidate": "increase"}
    ]
    assert patched["review_items"] == []
    assert patched["decision_note"] == "검토 큐 확인 완료"

    replaced = upsert_journal(
        snapshot["id"],
        {
            "date": "2026-06-15",
            "decision_context": "rebalance_review",
            "playbook_code": None,
            "dca_changes_considered": [],
            "review_items": [{"ticker": "GLD", "queue": "risk_review"}],
            "decision_note": "",
        },
    )

    assert replaced["id"] == created["id"]
    assert replaced["date"] == "2026-06-15"
    assert replaced["decision_context"] == "rebalance_review"
    assert replaced["playbook_code"] is None
    assert replaced["review_items"] == [{"ticker": "GLD", "queue": "risk_review"}]


def test_journal_store_rejects_missing_snapshot_or_entry(journal_db):
    with pytest.raises(StorageError, match="스냅샷을 찾을 수 없습니다"):
        upsert_journal(
            999,
            {
                "date": "2026-06-14",
                "decision_context": "regular_review",
                "playbook_code": "regular_review",
                "dca_changes_considered": [],
                "review_items": [],
                "decision_note": "",
            },
        )

    with pytest.raises(StorageError, match="수정할 저널을 찾을 수 없습니다"):
        update_journal(999, {"decision_note": "없는 저널"})
