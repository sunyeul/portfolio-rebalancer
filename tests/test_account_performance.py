from decimal import Decimal

import pytest

from services.account_performance import (
    PerformanceCalculationError,
    TrackingBaseline,
    apply_execution,
    build_projection,
    canonical_executions,
    detect_cash_candidates,
)


def _holding(quantity=100.0, market_value=7200000.0):
    return {
        "symbol": "005930",
        "name": "Samsung Electronics",
        "market_country": "KR",
        "currency": "KRW",
        "quantity": quantity,
        "last_price": market_value / quantity,
        "average_purchase_price": 65000.0,
        "market_value_native": market_value,
        "market_value_krw": market_value,
        "cost_native": quantity * 65000.0,
        "cost_krw": quantity * 65000.0,
        "profit_loss_native": market_value - quantity * 65000.0,
        "profit_loss_krw": market_value - quantity * 65000.0,
        "daily_profit_loss_native": 0.0,
        "daily_profit_loss_krw": 0.0,
    }


def _snapshot(
    snapshot_id,
    *,
    total=8200000.0,
    cash=1000000.0,
    fingerprint=None,
    synced_at=None,
    holdings=None,
    orders=None,
):
    return {
        "id": snapshot_id,
        "account_alias": "toss-brokerage",
        "synced_at": synced_at or f"2026-07-23T00:0{snapshot_id}:00+00:00",
        "state": "complete",
        "is_current_evaluable": True,
        "source_fingerprint": fingerprint or f"snapshot-{snapshot_id}",
        "total_value_krw": total,
        "invested_value_krw": total - cash,
        "cash_value_krw": cash,
        "holdings": holdings if holdings is not None else [_holding()],
        "cash": [
            {"currency": "KRW", "buying_power_native": cash, "buying_power_krw": cash}
        ],
        "fx_rate": {
            "base_currency": "USD",
            "quote_currency": "KRW",
            "rate": 1400.0,
        },
        "orders": orders or [],
    }


def _baseline():
    return TrackingBaseline(
        id=1,
        account_alias="toss-brokerage",
        baseline_snapshot_id=1,
        tracking_started_at="2026-07-23T00:01:00+00:00",
        initial_principal_krw=Decimal("8200000"),
        baseline_fx_rate=Decimal("1400"),
    )


def _order(
    order_id="order-1",
    side="BUY",
    quantity=10.0,
    amount=650000.0,
    filled_at="2026-07-23T00:02:00+00:00",
):
    return {
        "order_id": order_id,
        "symbol": "005930",
        "currency": "KRW",
        "side": side,
        "status": "FILLED",
        "filled_quantity": quantity,
        "average_filled_price_native": amount / quantity,
        "filled_amount_native": amount,
        "commission_native": 100.0,
        "tax_native": 0.0,
        "filled_at": filled_at,
        "ordered_at": filled_at,
        "settlement_date": "2026-07-24",
    }


def test_baseline_and_no_flow_projection_have_zero_then_positive_twr():
    baseline = _baseline()
    snapshots = [_snapshot(1), _snapshot(2, total=8400000.0)]

    projection = build_projection(baseline, snapshots, {})

    assert projection.state == "complete"
    assert projection.points[0]["account_gain_krw"] == pytest.approx(0.0)
    assert projection.points[0]["interval_twr"] == pytest.approx(0.0)
    assert projection.points[1]["account_gain_krw"] == pytest.approx(200000.0)
    assert projection.points[1]["interval_twr"] == pytest.approx(200000 / 8200000)


def test_material_cash_residual_creates_non_evaluable_candidate():
    baseline = _baseline()
    snapshots = [_snapshot(1), _snapshot(2, total=8215000.0, cash=1015000.0)]

    candidates = detect_cash_candidates(baseline.id, snapshots[0], snapshots[1], [])
    projection = build_projection(baseline, snapshots, {})

    assert len(candidates) == 1
    assert candidates[0].residual_native == Decimal("15000")
    assert projection.state == "partial"
    assert projection.points[1]["evaluation_state"] == "non_evaluable"


def test_tracking_basis_excludes_prebaseline_appreciation():
    state = {
        ("005930", "KRW"): {
            "quantity": Decimal("100"),
            "actual_total_basis": Decimal("6500000"),
            "tracking_total_basis": Decimal("7200000"),
        }
    }

    row = apply_execution(state, _order(side="SELL", quantity=10, amount=750000))

    assert row["actual_realized_pnl_native"] == pytest.approx(99900.0)
    assert row["tracking_realized_pnl_native"] == pytest.approx(29900.0)
    assert state[("005930", "KRW")]["quantity"] == Decimal("90")


def test_duplicate_execution_is_collapsed_and_conflict_is_rejected():
    first = _snapshot(1, orders=[_order()])
    second = _snapshot(2, orders=[_order()])
    assert len(canonical_executions([first, second])) == 1

    conflict = _snapshot(2, orders=[_order(amount=660000)])
    with pytest.raises(PerformanceCalculationError, match="conflicting execution"):
        canonical_executions([first, conflict])
