from storage.database import initialize_database
from storage.market_store import insert_candles, list_adjusted_stock_candles


def _candle(fingerprint: str) -> dict[str, object]:
    return {
        "source_kind": "stock",
        "market_country": "US",
        "symbol": "AAA",
        "interval": "1d",
        "candle_at": "2026-08-01T00:00:00+00:00",
        "currency": "USD",
        "open_price": 100.0,
        "high_price": 101.0,
        "low_price": 99.0,
        "close_price": 100.0,
        "volume": 1000.0,
        "adjusted": True,
        "adjusted_supported": True,
        "source_fingerprint": fingerprint,
    }


def test_candle_replay_uses_full_identity_and_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "market.sqlite3"))
    initialize_database()

    first = insert_candles([_candle("a")])[0]
    same_fingerprint = insert_candles([_candle("a")])[0]
    revised = insert_candles([_candle("b")])[0]
    replay = insert_candles([_candle("a")])[0]

    assert same_fingerprint["id"] == first["id"]
    assert revised["id"] != first["id"]
    assert replay["id"] == first["id"]


def test_latest_candle_selector_uses_latest_fingerprint_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "market.sqlite3"))
    initialize_database()

    first = insert_candles([_candle("a")])[0]
    revised = insert_candles([_candle("b")])[0]
    replay = insert_candles([_candle("a")])[0]

    selected = list_adjusted_stock_candles(
        market_country="US",
        symbol="AAA",
        through_at="2026-08-02T00:00:00+00:00",
        limit=1,
    )

    assert replay["id"] == first["id"]
    assert selected == [revised]
