import pytest

from services.market_context import evaluate_market_context
from storage.database import initialize_database
from storage.market_store import (
    insert_candles,
    insert_policy_candidate,
    list_candles,
    latest_policy_candidate,
    MarketStoreError,
)


def test_market_candles_and_policy_candidates_are_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "market.sqlite3"))
    initialize_database()
    candle = {
        "source_kind": "market_indicator",
        "market_country": "KR",
        "symbol": "KOSPI",
        "interval": "1d",
        "candle_at": "2026-07-23T09:00:00+09:00",
        "currency": "KRW",
        "open_price": 2500.0,
        "high_price": 2520.0,
        "low_price": 2480.0,
        "close_price": 2510.0,
        "volume": 100.0,
        "adjusted": True,
    }
    first = insert_candles([candle])
    second = insert_candles([candle])
    assert len(first) == len(second) == 1
    assert (
        len(
            list_candles(
                symbol="KOSPI", source_kind="market_indicator", market_country="KR"
            )
        )
        == 1
    )

    candidate = insert_policy_candidate(
        {
            "account_alias": "toss-brokerage",
            "base_policy_version_id": 1,
            "candidate_json": evaluate_market_context([candle]),
        }
    )
    again = insert_policy_candidate(
        {
            "account_alias": "toss-brokerage",
            "base_policy_version_id": 1,
            "candidate_json": evaluate_market_context([candle]),
        }
    )
    assert candidate["id"] == again["id"]
    assert latest_policy_candidate()["id"] == candidate["id"]


def test_policy_candidate_identity_includes_account_and_base_policy(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "candidate.sqlite3"))
    initialize_database()
    candidate_json = {"status": "Review", "candidate_state": "candidate"}

    first = insert_policy_candidate(
        {
            "account_alias": "account-a",
            "base_policy_version_id": 1,
            "candidate_json": candidate_json,
        }
    )
    second = insert_policy_candidate(
        {
            "account_alias": "account-b",
            "base_policy_version_id": 1,
            "candidate_json": candidate_json,
        }
    )

    assert second["id"] != first["id"]
    assert second["account_alias"] == "account-b"
    assert second["base_policy_version_id"] == 1


def test_conflicting_candle_revision_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "revision.sqlite3"))
    initialize_database()
    candle = {
        "source_kind": "market_indicator",
        "market_country": "KR",
        "symbol": "KOSPI",
        "interval": "1d",
        "candle_at": "2026-07-23T09:00:00+09:00",
        "currency": "",
        "open_price": 2500.0,
        "high_price": 2520.0,
        "low_price": 2480.0,
        "close_price": 2510.0,
        "volume": 100.0,
        "adjusted": False,
        "adjusted_supported": False,
    }
    insert_candles([candle])
    revised = dict(candle, close_price=2511.0)

    with pytest.raises(MarketStoreError, match="conflicting candle"):
        insert_candles([revised])
