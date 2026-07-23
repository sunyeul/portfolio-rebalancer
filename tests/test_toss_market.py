from dataclasses import dataclass

import pytest

from integrations.toss.market import MarketObservationError, TossMarketDataService


@dataclass
class FakeMarketReader:
    def __post_init__(self):
        self.calls = []

    def get_json(self, path, *, params=None, include_account_header=True):
        self.calls.append((path, params, include_account_header))
        before = (params or {}).get("before")
        if path.endswith("/market-indicators/KOSPI/candles"):
            if before == "2026-01-01T09:00:00+09:00":
                return {
                    "result": {
                        "candles": [
                            {
                                "timestamp": "2025-12-31T09:00:00+09:00",
                                "openPrice": "2500",
                                "highPrice": "2520",
                                "lowPrice": "2480",
                                "closePrice": "2510",
                                "volume": "100",
                                "currency": "KRW",
                            }
                        ],
                        "nextBefore": None,
                    }
                }
            return {
                "result": {
                    "candles": [
                        {
                            "timestamp": "2026-01-01T09:00:00+09:00",
                            "openPrice": "2600",
                            "highPrice": "2610",
                            "lowPrice": "2590",
                            "closePrice": "2605",
                            "volume": "200",
                            "currency": "KRW",
                        }
                    ],
                    "nextBefore": "2026-01-01T09:00:00+09:00",
                }
            }
        raise AssertionError(path)


def test_collects_market_indicator_candles_with_bounded_pagination():
    reader = FakeMarketReader()
    candles = TossMarketDataService(reader).collect_history(
        symbol="KOSPI", source_kind="market_indicator", market_country="KR"
    )

    assert [item.close_price for item in candles] == [2510, 2605]
    assert reader.calls[0][0] == "/api/v1/market-indicators/KOSPI/candles"
    assert set(reader.calls[0][1]) == {"interval", "count"}
    assert all(call[2] is False for call in reader.calls)


def test_stock_candles_send_stock_only_parameters():
    class StockReader:
        def __init__(self):
            self.params = None

        def get_json(self, path, *, params=None, include_account_header=True):
            self.params = params
            return {
                "result": {
                    "candles": [
                        {
                            "timestamp": "2026-01-01T09:00:00+09:00",
                            "openPrice": "10",
                            "highPrice": "11",
                            "lowPrice": "9",
                            "closePrice": "10",
                            "volume": "1",
                            "currency": "KRW",
                        }
                    ],
                    "nextBefore": None,
                }
            }

    reader = StockReader()
    candle, _ = TossMarketDataService(reader).read_page(
        symbol="005930", source_kind="stock", market_country="KR"
    )

    assert set(reader.params or {}) == {"symbol", "interval", "count", "adjusted"}
    assert candle[0].adjusted_supported is True


def test_market_indicator_does_not_claim_adjustment_support():
    reader = FakeMarketReader()
    candles, _ = TossMarketDataService(reader).read_page(
        symbol="KOSPI", source_kind="market_indicator", market_country="KR"
    )

    assert candles[0].adjusted is False
    assert candles[0].adjusted_supported is False


def test_rejects_invalid_ohlc_invariant():
    class BadReader:
        def get_json(self, path, *, params=None, include_account_header=True):
            return {
                "result": {
                    "candles": [
                        {
                            "timestamp": "2026-01-01T09:00:00+09:00",
                            "openPrice": "10",
                            "highPrice": "9",
                            "lowPrice": "8",
                            "closePrice": "10",
                            "volume": "1",
                            "currency": "KRW",
                        }
                    ],
                    "nextBefore": None,
                }
            }

    with pytest.raises(MarketObservationError, match="OHLC invariant"):
        TossMarketDataService(BadReader()).read_page(
            symbol="KOSPI", source_kind="market_indicator", market_country="KR"
        )


def test_rejects_unsupported_market_indicator_symbol():
    with pytest.raises(MarketObservationError, match="unsupported"):
        TossMarketDataService(FakeMarketReader()).read_page(
            symbol="NOT_REAL", source_kind="market_indicator", market_country="KR"
        )


def test_rejects_history_that_exceeds_page_bound():
    class EndlessReader:
        def get_json(self, path, *, params=None, include_account_header=True):
            return {
                "result": {
                    "candles": [
                        {
                            "timestamp": "2026-01-01T09:00:00+09:00",
                            "openPrice": "10",
                            "highPrice": "11",
                            "lowPrice": "9",
                            "closePrice": "10",
                            "volume": "1",
                        }
                    ],
                    "nextBefore": "next-cursor",
                }
            }

    with pytest.raises(MarketObservationError, match="exceeds max_pages"):
        TossMarketDataService(EndlessReader()).collect_history(
            symbol="KOSPI", source_kind="market_indicator", max_pages=1
        )
