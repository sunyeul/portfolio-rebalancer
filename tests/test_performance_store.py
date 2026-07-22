from types import SimpleNamespace

import pytest

from integrations.toss.observation import (
    NormalizedCash,
    NormalizedFxRate,
    NormalizedHolding,
    NormalizedSnapshot,
    SyncState,
)
from storage.account_observation_store import insert_snapshot
from storage.database import initialize_database
from storage.performance_store import (
    PerformanceStorageError,
    append_cash_flow_decision,
    create_baseline,
    get_baseline,
    get_performance_run,
    insert_cash_flow_candidate,
    insert_performance_run,
    latest_cash_flow_decisions,
    latest_performance_run,
    list_cash_flow_candidates,
)


def _snapshot(*, fingerprint="snapshot-1", synced_at="2026-07-23T00:00:00+00:00"):
    return NormalizedSnapshot(
        account_alias="toss-brokerage",
        sync_started_at=synced_at,
        synced_at=synced_at,
        state=SyncState.COMPLETE,
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
        orders=(),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": True}},
        total_value_krw=8200000.0,
        invested_value_krw=7200000.0,
        cash_value_krw=1000000.0,
        fingerprint=fingerprint,
    )


@pytest.fixture()
def database(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "performance.sqlite3"))
    initialize_database()


def test_baseline_requires_exact_current_complete_snapshot(database):
    snapshot = insert_snapshot(_snapshot())

    baseline = create_baseline(snapshot["id"], 8200000.0)

    assert baseline["baseline_snapshot_id"] == snapshot["id"]
    assert baseline["initial_principal_krw"] == pytest.approx(8200000.0)
    assert baseline["baseline_fx_rate"] == pytest.approx(1400.0)
    assert get_baseline()["id"] == baseline["id"]
    with pytest.raises(PerformanceStorageError, match="already exists"):
        create_baseline(snapshot["id"], 8200000.0)


def test_baseline_rejects_mismatched_principal(database):
    snapshot = insert_snapshot(_snapshot())

    with pytest.raises(PerformanceStorageError, match="does not match"):
        create_baseline(snapshot["id"], 8200001.0)


def test_candidate_and_decision_are_append_only(database):
    snapshot = insert_snapshot(_snapshot())
    baseline = create_baseline(snapshot["id"], 8200000.0)
    candidate = {
        "baseline_id": baseline["id"],
        "from_snapshot_id": snapshot["id"],
        "to_snapshot_id": snapshot["id"],
        "currency": "KRW",
        "observed_delta_native": 15000.0,
        "explained_trade_delta_native": 0.0,
        "residual_native": 15000.0,
        "residual_krw": 15000.0,
        "materiality_threshold_krw": 10000.0,
        "bridge_basis": "none",
        "candidate_fingerprint": "candidate-1",
    }

    first = insert_cash_flow_candidate(candidate)
    same = insert_cash_flow_candidate(candidate)
    decision1 = append_cash_flow_decision(
        first["id"], classification="internal_fx", note="first review"
    )
    decision2 = append_cash_flow_decision(
        first["id"], classification="other_non_external", note="corrected review"
    )

    assert first["id"] == same["id"]
    assert len(list_cash_flow_candidates(baseline["id"])) == 1
    assert (
        latest_cash_flow_decisions([first["id"]])[first["id"]]["id"] == decision2["id"]
    )
    assert decision1["id"] != decision2["id"]


def test_performance_run_is_idempotent_and_hides_execution_side(database):
    snapshot = insert_snapshot(_snapshot())
    baseline = create_baseline(snapshot["id"], 8200000.0)
    projection = SimpleNamespace(
        baseline_id=baseline["id"],
        through_snapshot_id=snapshot["id"],
        input_fingerprint="run-1",
        engine_version="phase2-test",
        state="complete",
        data_quality={"issues": []},
        points=(
            {
                "snapshot_id": snapshot["id"],
                "previous_snapshot_id": None,
                "point_at": snapshot["synced_at"],
                "evaluation_state": "evaluable",
                "evaluation_reason": None,
                "total_value_krw": 8200000.0,
                "invested_value_krw": 7200000.0,
                "cash_value_krw": 1000000.0,
                "current_cost_basis_krw": 6500000.0,
                "unrealized_pnl_krw": 700000.0,
                "tracking_principal_krw": 8200000.0,
                "cumulative_external_flow_krw": 0.0,
                "account_gain_krw": 0.0,
                "simple_return": 0.0,
                "interval_twr": 0.0,
                "segment_id": 0,
                "segment_twr": 0.0,
                "tracked_realized_pnl_krw": 0.0,
                "actual_realized_pnl_krw": 0.0,
                "fx_remeasurement_krw": 0.0,
            },
        ),
        executions=(
            {
                "source_snapshot_id": snapshot["id"],
                "order_id": "order-1",
                "symbol": "005930",
                "currency": "KRW",
                "side": "BUY",
                "filled_at": None,
                "settlement_date": None,
                "filled_quantity_native": 1.0,
                "filled_amount_native": 65000.0,
                "commission_native": 100.0,
                "tax_native": 0.0,
                "actual_basis_before_native": 0.0,
                "tracking_basis_before_native": 0.0,
                "actual_realized_pnl_native": 0.0,
                "tracking_realized_pnl_native": 0.0,
                "realized_pnl_krw": 0.0,
                "krw_conversion_snapshot_id": snapshot["id"],
            },
        ),
    )

    first = insert_performance_run(projection)
    same = insert_performance_run(projection)
    hydrated = get_performance_run(first["id"])

    assert first["id"] == same["id"] == latest_performance_run(baseline["id"])["id"]
    assert hydrated["execution_count"] == 1
    assert "side" not in hydrated["points"][0]
