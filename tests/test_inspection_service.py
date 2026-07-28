from copy import deepcopy
from datetime import date, timedelta

from services.inspection_service import (
    _select_market_candles,
    preview_inspection,
    run_inspection,
)
from integrations.toss.observation import (
    NormalizedCash,
    NormalizedFxRate,
    NormalizedHolding,
    NormalizedSnapshot,
    SyncState,
)
from storage.account_observation_store import insert_snapshot, latest_complete
from storage.database import connect, initialize_database
from storage.market_store import insert_candles
from storage.performance_store import create_baseline, refresh_performance
from services.policy_validation import validate_policy
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
    assert evaluation["result"]["review_queue"][0]["kind"] == "allocation"
    assert evaluation["result"]["review_queue"][0]["queue_class"] == "blocking"
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


def _phase5_snapshot(*, synced_at: str, fingerprint: str, kr_price: float, us_price: float):
    holdings = (
        NormalizedHolding(
            symbol="005930",
            name="Samsung Electronics",
            market_country="KR",
            currency="KRW",
            quantity=50.0,
            last_price=kr_price,
            average_purchase_price=100000.0,
            market_value_native=50.0 * kr_price,
            market_value_krw=50.0 * kr_price,
            cost_native=5000000.0,
            cost_krw=5000000.0,
            profit_loss_native=50.0 * kr_price - 5000000.0,
            profit_loss_krw=50.0 * kr_price - 5000000.0,
            daily_profit_loss_native=0.0,
            daily_profit_loss_krw=0.0,
        ),
        NormalizedHolding(
            symbol="NBIS",
            name="Nebius Group",
            market_country="US",
            currency="USD",
            quantity=50.0,
            last_price=us_price,
            average_purchase_price=50.0,
            market_value_native=50.0 * us_price,
            market_value_krw=50.0 * us_price * 1400.0,
            cost_native=2500.0,
            cost_krw=2500.0 * 1400.0,
            profit_loss_native=50.0 * us_price - 2500.0,
            profit_loss_krw=(50.0 * us_price - 2500.0) * 1400.0,
            daily_profit_loss_native=0.0,
            daily_profit_loss_krw=0.0,
        ),
    )
    return NormalizedSnapshot(
        account_alias="toss-brokerage",
        sync_started_at=synced_at,
        synced_at=synced_at,
        state=SyncState.COMPLETE,
        holdings=holdings,
        cash=(NormalizedCash("KRW", 1000000.0, 1000000.0),),
        fx_rate=NormalizedFxRate("USD", "KRW", 1400.0, 1390.0, None, None),
        orders=(),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": True}},
        total_value_krw=(50.0 * kr_price) + (50.0 * us_price * 1400.0) + 1000000.0,
        invested_value_krw=(50.0 * kr_price) + (50.0 * us_price * 1400.0),
        cash_value_krw=1000000.0,
        fingerprint=fingerprint,
    )


def _phase5_candles(symbol: str, country: str, through: date, *, base: float) -> list[dict[str, object]]:
    return [
        {
            "source_kind": "stock",
            "market_country": country,
            "symbol": symbol,
            "interval": "1d",
            "candle_at": (through - timedelta(days=offset)).isoformat(),
            "currency": "KRW" if country == "KR" else "USD",
            "open_price": base,
            "high_price": base + 1.0,
            "low_price": base - 1.0,
            "close_price": base,
            "volume": 1000.0,
            "adjusted": True,
            "adjusted_supported": True,
        }
        for offset in range(251, -1, -1)
    ]


def _assert_no_direct_action_semantics(value: object) -> None:
    forbidden_keys = {
        "buy",
        "sell",
        "execute",
        "order_quantity",
        "transaction_value",
        "stop_loss",
        "take_profit",
    }
    forbidden_phrases = {"buy now", "sell now", "즉시 매수", "즉시 매도", "주문 수량"}
    allowed = {"future regular-purchase policy", "향후 정기매수 정책"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert not forbidden_keys.intersection(node)
            for key, child in node.items():
                if isinstance(child, str):
                    lowered = child.lower()
                    assert not any(phrase in lowered for phrase in forbidden_phrases if phrase not in allowed)
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)


def test_phase5_offline_preview_is_deterministic_and_read_only(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "phase5.sqlite3"))
    initialize_database()
    first = insert_snapshot(
        _phase5_snapshot(
            synced_at="2026-01-02T16:00:00+00:00",
            fingerprint="phase5-first",
            kr_price=120000.0,
            us_price=60.0,
        )
    )
    create_baseline(first["id"], first["total_value_krw"])
    second = insert_snapshot(
        _phase5_snapshot(
            synced_at="2026-07-22T16:00:00+00:00",
            fingerprint="phase5-second",
            kr_price=140000.0,
            us_price=100.0,
        )
    )
    performance = refresh_performance()
    assert performance["state"] == "complete"

    insert_candles(_phase5_candles("005930", "KR", date(2026, 7, 22), base=100.0))
    insert_candles(_phase5_candles("NBIS", "US", date(2026, 7, 22), base=100.0))
    policy = deepcopy(DEFAULT_POLICY)
    policy["layers"] = {
        "core": {"minimum": 0.7, "target": 0.8, "maximum": 0.9},
        "satellite": {"minimum": 0.1, "target": 0.2, "maximum": 0.3},
        "experiment": {"minimum": 0.0, "target": 0.0, "maximum": 0.05},
    }
    policy["instruments"] = [
        {"market_country": "KR", "symbol": "005930", "layer": "core", "minimum": 0.7, "target": 0.8, "maximum": 0.9},
        {"market_country": "US", "symbol": "NBIS", "layer": "satellite", "minimum": 0.1, "target": 0.2, "maximum": 0.3},
    ]
    policy = validate_policy(policy, [("KR", "005930"), ("US", "NBIS")])

    with connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM ips_evaluation_runs").fetchone()[0]
    first_preview = preview_inspection(policy, snapshot_id=second["id"])
    second_preview = preview_inspection(policy, snapshot_id=second["id"])
    with connect() as conn:
        after = conn.execute("SELECT COUNT(*) FROM ips_evaluation_runs").fetchone()[0]

    assert first_preview == second_preview
    assert first_preview["persisted"] is False
    assert first_preview["snapshot_id"] == second["id"]
    assert first_preview["evaluation_fingerprint"]
    assert first_preview["market_evidence_fingerprint"]
    assert before == after == 0
    assert all(item["status"] != "Action" for item in first_preview["evaluation"]["review_queue"])
    _assert_no_direct_action_semantics(first_preview)
