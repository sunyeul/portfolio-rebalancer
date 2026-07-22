from dataclasses import dataclass

import pytest

from integrations.toss.observation import (
    ObservationError,
    SyncState,
    TossObservationService,
)


@dataclass
class FakeReader:
    pages: list[dict]
    repeated_cursor: bool = False
    fail_holdings: bool = False

    def __post_init__(self):
        self.calls = []

    def get_json(self, path, *, params=None, include_account_header=True):
        self.calls.append((path, params, include_account_header))
        if path == "/api/v1/accounts":
            return {
                "result": [
                    {
                        "accountNo": "12345678901",
                        "accountSeq": 1,
                        "accountType": "BROKERAGE",
                    }
                ]
            }
        if path == "/api/v1/holdings":
            if self.fail_holdings:
                raise ObservationError("holdings unavailable")
            return _holdings_response()
        if path == "/api/v1/buying-power":
            currency = params["currency"]
            return {
                "result": {
                    "currency": currency,
                    "cashBuyingPower": "1000000" if currency == "KRW" else "100",
                }
            }
        if path == "/api/v1/exchange-rate":
            return {
                "result": {
                    "baseCurrency": "USD",
                    "quoteCurrency": "KRW",
                    "rate": "1400",
                    "midRate": "1390",
                    "validFrom": "2026-07-23T09:00:00+09:00",
                    "validUntil": "2026-07-23T15:30:00+09:00",
                }
            }
        if path == "/api/v1/orders":
            cursor = (params or {}).get("cursor")
            if cursor == "next" and self.repeated_cursor:
                return {
                    "result": {
                        "orders": [_order("o1")],
                        "nextCursor": "next",
                        "hasNext": True,
                    }
                }
            if cursor == "next":
                return {
                    "result": {
                        "orders": [_order("o1"), _order("o2")],
                        "nextCursor": None,
                        "hasNext": False,
                    }
                }
            return {
                "result": {
                    "orders": [_order("o1")],
                    "nextCursor": "next" if self.repeated_cursor else None,
                    "hasNext": self.repeated_cursor,
                }
            }
        raise AssertionError(path)


def _order(order_id):
    return {
        "orderId": order_id,
        "symbol": "005930",
        "currency": "KRW",
        "side": "BUY",
        "orderType": "LIMIT",
        "status": "FILLED",
        "orderedAt": "2026-07-22T09:00:00+09:00",
        "canceledAt": None,
        "quantity": "10",
        "price": "65000",
        "orderAmount": "650000",
        "execution": {
            "filledQuantity": "10",
            "averageFilledPrice": "65000",
            "filledAmount": "650000",
            "commission": "100",
            "tax": "0",
            "filledAt": "2026-07-22T09:01:00+09:00",
            "settlementDate": "2026-07-24",
        },
    }


def _holdings_response(market_value="7200000"):
    return {
        "result": {
            "totalPurchaseAmount": {"krw": "6500000", "usd": None},
            "marketValue": {
                "amount": {"krw": market_value, "usd": None},
                "amountAfterCost": {"krw": market_value, "usd": None},
            },
            "profitLoss": {
                "krw": "700000",
                "usd": None,
                "rate": "0.1076",
                "rateAfterCost": "0.105",
            },
            "dailyProfitLoss": {
                "krw": "10000",
                "usd": None,
                "rate": "0.001",
                "rateAfterCost": "0.001",
            },
            "items": [
                {
                    "symbol": "005930",
                    "name": "Samsung Electronics",
                    "marketCountry": "KR",
                    "currency": "KRW",
                    "quantity": "100",
                    "lastPrice": "72000",
                    "averagePurchasePrice": "65000",
                    "marketValue": {
                        "purchaseAmount": "6500000",
                        "amount": "7200000",
                        "amountAfterCost": "7190000",
                    },
                    "cost": {
                        "commission": "0",
                        "tax": "0",
                    },
                    "profitLoss": {
                        "amount": "700000",
                        "amountAfterCost": "690000",
                        "rate": "0.1076",
                        "rateAfterCost": "0.106",
                    },
                    "dailyProfitLoss": {
                        "amount": "10000",
                        "amountAfterCost": "9000",
                        "rate": "0.001",
                        "rateAfterCost": "0.001",
                    },
                }
            ],
        }
    }


def _service(reader, now=1753228800.0):
    return TossObservationService(
        config=type("Config", (), {"account_seq": 1})(),
        reader=reader,
        clock=lambda: now,
    )


def test_collect_normalizes_holdings_cash_fx_and_orders():
    reader = FakeReader([])

    snapshot = _service(reader).collect()

    assert snapshot.state == SyncState.COMPLETE
    assert snapshot.account_alias == "toss-brokerage"
    assert snapshot.holdings[0].market_value_native == pytest.approx(7200000)
    assert snapshot.holdings[0].market_value_krw == pytest.approx(7200000)
    assert snapshot.holdings[0].cost_native == pytest.approx(6500000)
    assert snapshot.cash_by_currency["KRW"] == pytest.approx(1000000)
    assert snapshot.cash_by_currency["USD"] == pytest.approx(100)
    assert snapshot.cash_value_krw == pytest.approx(1140000)
    assert snapshot.invested_value_krw == pytest.approx(7200000)
    assert snapshot.total_value_krw == pytest.approx(8340000)
    assert snapshot.orders[0].order_id == "o1"
    assert snapshot.orders[0].filled_amount_native == pytest.approx(650000)
    assert snapshot.reconciliation["cash_currency_check"] == (
        "verified_by_currency_labels"
    )
    assert snapshot.fingerprint
    assert all(
        forbidden not in repr(snapshot)
        for forbidden in ("12345678901", "accountSeq", "access_token")
    )
    account_call = next(call for call in reader.calls if call[0] == "/api/v1/accounts")
    assert account_call[2] is False


def test_collect_marks_partial_when_holdings_do_not_reconcile():
    reader = FakeReader([])
    reader.get_json = lambda path, **kwargs: (
        _holdings_response("7200001")
        if path == "/api/v1/holdings"
        else FakeReader.get_json(reader, path, **kwargs)
    )

    snapshot = _service(reader).collect()

    assert snapshot.state == SyncState.PARTIAL
    assert snapshot.reconciliation["holdings"]["KRW"]["within_tolerance"] is False


def test_collect_marks_failed_when_holdings_are_unavailable():
    reader = FakeReader([], fail_holdings=True)

    snapshot = _service(reader).collect()

    assert snapshot.state == SyncState.FAILED
    assert snapshot.holdings == ()
    assert snapshot.data_quality["failed_stage"] == "holdings"


def test_collect_deduplicates_orders_and_repeated_cursor_is_partial():
    reader = FakeReader([], repeated_cursor=True)

    snapshot = _service(reader).collect()

    assert snapshot.state == SyncState.PARTIAL
    assert [order.order_id for order in snapshot.orders] == ["o1"]
    assert snapshot.data_quality["orders"]["repeated_cursor"] == "next"


def test_fingerprint_excludes_sync_time():
    first = _service(FakeReader([]), now=1753228800.0).collect()
    second = _service(FakeReader([]), now=1753229900.0).collect()

    assert first.fingerprint == second.fingerprint


def test_health_checks_oauth_account_discovery_and_match():
    result = _service(FakeReader([])).health()

    assert result["ok"] is True
    assert result["checks"]["oauth"] == "ok"
    assert result["checks"]["account_match"] == "ok"
    assert result["account_count"] == 1
    assert "12345678901" not in repr(result)


def test_expired_exchange_rate_marks_snapshot_stale():
    reader = FakeReader([])
    original = reader.get_json

    def expired(path, **kwargs):
        if path == "/api/v1/exchange-rate":
            return {
                "result": {
                    "baseCurrency": "USD",
                    "quoteCurrency": "KRW",
                    "rate": "1400",
                    "midRate": "1390",
                    "validFrom": "2026-07-20T09:00:00+00:00",
                    "validUntil": "2026-07-20T10:00:00+00:00",
                }
            }
        return original(path, **kwargs)

    reader.get_json = expired

    snapshot = _service(reader, now=1784734877.0).collect()

    assert snapshot.state == SyncState.STALE
    assert snapshot.data_quality["stale"] is True
