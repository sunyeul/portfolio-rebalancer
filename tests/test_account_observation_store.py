from integrations.toss.observation import (
    NormalizedCash,
    NormalizedFxRate,
    NormalizedHolding,
    NormalizedOrder,
    NormalizedSnapshot,
    SyncState,
)
from storage.account_observation_store import (
    get_snapshot,
    insert_snapshot,
    latest_complete,
    list_snapshots,
)
from storage.database import initialize_database


def _snapshot(
    state=SyncState.COMPLETE,
    fingerprint="fingerprint-1",
    synced_at="2026-07-23T00:00:00+00:00",
):
    return NormalizedSnapshot(
        account_alias="toss-brokerage",
        sync_started_at=synced_at,
        synced_at=synced_at,
        state=state,
        holdings=(
            NormalizedHolding(
                symbol="005930",
                name="Samsung Electronics",
                market_country="KR",
                currency="KRW",
                quantity=100.0,
                last_price=72000.0,
                average_purchase_price=65000.0,
                market_value_native=7200000.0,
                market_value_krw=7200000.0,
                cost_native=6500000.0,
                cost_krw=6500000.0,
                profit_loss_native=700000.0,
                profit_loss_krw=700000.0,
                daily_profit_loss_native=10000.0,
                daily_profit_loss_krw=10000.0,
            ),
        ),
        cash=(NormalizedCash("KRW", 1000000.0, 1000000.0),),
        fx_rate=NormalizedFxRate("USD", "KRW", 1400.0, 1390.0, None, None),
        orders=(
            NormalizedOrder(
                order_id="order-1",
                symbol="005930",
                currency="KRW",
                side="BUY",
                order_type="LIMIT",
                status="FILLED",
                ordered_at="2026-07-22T09:00:00+00:00",
                canceled_at=None,
                quantity=10.0,
                order_price_native=65000.0,
                order_amount_native=650000.0,
                filled_quantity=10.0,
                average_filled_price_native=65000.0,
                filled_amount_native=650000.0,
                commission_native=100.0,
                tax_native=0.0,
                filled_at="2026-07-22T09:01:00+00:00",
                settlement_date="2026-07-24",
            ),
        ),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": True}},
        total_value_krw=8200000.0,
        invested_value_krw=7200000.0,
        cash_value_krw=1000000.0,
        fingerprint=fingerprint,
    )


def test_insert_snapshot_is_idempotent_and_returns_normalized_children(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "observations.sqlite3"))
    initialize_database()

    first = insert_snapshot(_snapshot())
    second = insert_snapshot(_snapshot())

    assert first["id"] == second["id"]
    assert first["state"] == "complete"
    assert first["holdings"][0]["symbol"] == "005930"
    assert first["cash"][0]["currency"] == "KRW"
    assert first["orders"][0]["order_id"] == "order-1"
    assert "accountNo" not in repr(first)
    assert "account_seq" not in repr(first)
    assert len(list_snapshots()) == 1


def test_partial_snapshot_does_not_replace_latest_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "observations.sqlite3"))
    initialize_database()

    complete = insert_snapshot(_snapshot())
    partial = insert_snapshot(
        _snapshot(
            state=SyncState.PARTIAL,
            fingerprint="fingerprint-2",
            synced_at="2026-07-23T00:05:00+00:00",
        )
    )

    assert partial["state"] == "partial"
    assert latest_complete()["id"] == complete["id"]
    assert get_snapshot(partial["id"])["state"] == "partial"


def test_new_complete_snapshot_replaces_current_evaluable(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "observations.sqlite3"))
    initialize_database()

    first = insert_snapshot(_snapshot())
    second = insert_snapshot(
        _snapshot(
            fingerprint="fingerprint-2",
            synced_at="2026-07-23T00:05:00+00:00",
        )
    )

    assert latest_complete()["id"] == second["id"]
    assert get_snapshot(first["id"])["is_current_evaluable"] is False
