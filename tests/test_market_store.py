import pytest

from services.market_context import evaluate_market_context
from storage.database import initialize_database
from storage.market_store import (
    insert_candles,
    insert_policy_candidate,
    list_adjusted_stock_candles,
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


def test_adjusted_stock_selector_is_bounded_and_chronological(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "bounded.sqlite3"))
    initialize_database()
    candles = [
        {
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "AAA",
            "interval": "1d",
            "candle_at": "2026-07-20T00:00:00+00:00",
            "currency": "USD",
            "open_price": 100.0,
            "high_price": 100.0,
            "low_price": 100.0,
            "close_price": 100.0,
            "volume": 1.0,
            "adjusted": True,
            "adjusted_supported": True,
        },
        {
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "AAA",
            "interval": "1d",
            "candle_at": "2026-07-21T00:00:00+00:00",
            "currency": "USD",
            "open_price": 101.0,
            "high_price": 101.0,
            "low_price": 101.0,
            "close_price": 101.0,
            "volume": 1.0,
            "adjusted": False,
            "adjusted_supported": False,
        },
        {
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "AAA",
            "interval": "1d",
            "candle_at": "2026-07-22T00:00:00+00:00",
            "currency": "USD",
            "open_price": 102.0,
            "high_price": 102.0,
            "low_price": 102.0,
            "close_price": 102.0,
            "volume": 1.0,
            "adjusted": True,
            "adjusted_supported": True,
        },
        {
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "AAA",
            "interval": "1d",
            "candle_at": "2026-07-23T00:00:00+00:00",
            "currency": "USD",
            "open_price": 103.0,
            "high_price": 103.0,
            "low_price": 103.0,
            "close_price": 103.0,
            "volume": 1.0,
            "adjusted": True,
            "adjusted_supported": True,
        },
    ]
    insert_candles(candles)

    selected = list_adjusted_stock_candles(
        market_country="US",
        symbol="AAA",
        through_at="2026-07-22T16:01:04+00:00",
        limit=2,
    )

    assert [row["candle_at"] for row in selected] == [
        "2026-07-20T00:00:00+00:00",
        "2026-07-22T00:00:00+00:00",
    ]
    assert all(row["adjusted"] == 1 for row in selected)
