import pytest

from integrations.toss.observation import (
    NormalizedCash,
    NormalizedHolding,
    NormalizedSnapshot,
    SyncState,
)
from services.account_projection import AccountProjectionError, build_account_projection
from storage.account_observation_store import insert_snapshot
from storage.database import initialize_database


def _holding(
    symbol: str,
    market_country: str,
    market_value_krw: float | None,
) -> NormalizedHolding:
    return NormalizedHolding(
        symbol=symbol,
        name=symbol,
        market_country=market_country,
        currency="USD" if market_country == "US" else "KRW",
        quantity=1.0,
        last_price=market_value_krw or 100.0,
        average_purchase_price=90.0,
        market_value_native=market_value_krw or 0.0,
        market_value_krw=market_value_krw,
        cost_native=90.0,
        cost_krw=90.0,
        profit_loss_native=10.0,
        profit_loss_krw=10.0,
        daily_profit_loss_native=1.0,
        daily_profit_loss_krw=1.0,
    )


def _snapshot(
    *,
    state: SyncState = SyncState.COMPLETE,
    fingerprint: str = "projection-1",
    synced_at: str = "2026-07-23T00:00:00+00:00",
    holdings: tuple[NormalizedHolding, ...] | None = None,
    total: float = 100_000.0,
    invested: float = 90_000.0,
    cash_value: float = 10_000.0,
) -> NormalizedSnapshot:
    return NormalizedSnapshot(
        account_alias="toss-brokerage",
        sync_started_at=synced_at,
        synced_at=synced_at,
        state=state,
        holdings=holdings
        if holdings is not None
        else (_holding("AAPL", "US", 60_000.0), _holding("005930", "KR", 30_000.0)),
        cash=(NormalizedCash("KRW", cash_value, cash_value),),
        fx_rate=None,
        orders=(),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": True}},
        total_value_krw=total,
        invested_value_krw=invested,
        cash_value_krw=cash_value,
        fingerprint=fingerprint,
    )


@pytest.fixture
def complete_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    return insert_snapshot(_snapshot())


def test_projection_uses_explicit_gross_and_invested_denominators(
    complete_snapshot,
):
    result = build_account_projection(
        snapshot_id=complete_snapshot["id"],
        layer_map={("US", "AAPL"): "core"},
    )

    assert result["snapshot_id"] == complete_snapshot["id"]
    assert result["cash_weight_gross"] == pytest.approx(0.1)
    aapl = next(item for item in result["positions"] if item["symbol"] == "AAPL")
    assert aapl["gross_weight"] == pytest.approx(
        aapl["market_value_krw"] / result["total_value_krw"]
    )
    assert aapl["invested_weight"] == pytest.approx(
        aapl["market_value_krw"] / result["invested_value_krw"]
    )
    assert result["unclassified"] == [{"market_country": "KR", "symbol": "005930"}]
    assert result["layer_weights_invested"]["core"] == pytest.approx(2 / 3)


def test_latest_projection_reports_the_actual_snapshot_id(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    first = insert_snapshot(_snapshot())
    second = insert_snapshot(
        _snapshot(fingerprint="projection-2", synced_at="2026-07-23T00:05:00+00:00")
    )

    assert build_account_projection()["snapshot_id"] == second["id"]
    assert first["id"] != second["id"]


def test_historical_complete_projection_can_skip_currentness_requirement(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    first = insert_snapshot(_snapshot())
    insert_snapshot(
        _snapshot(fingerprint="projection-2", synced_at="2026-07-23T00:05:00+00:00")
    )

    result = build_account_projection(
        snapshot_id=first["id"], require_current_evaluable=False
    )

    assert result["snapshot_id"] == first["id"]


@pytest.mark.parametrize(
    "state", [SyncState.PARTIAL, SyncState.STALE, SyncState.FAILED]
)
def test_non_complete_snapshot_is_rejected(monkeypatch, tmp_path, state):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    snapshot = insert_snapshot(_snapshot(state=state, fingerprint=f"{state}-1"))

    with pytest.raises(AccountProjectionError, match="not complete"):
        build_account_projection(snapshot_id=snapshot["id"])


def test_missing_krw_value_is_rejected(complete_snapshot):
    from storage.account_observation_store import get_snapshot

    snapshot = get_snapshot(complete_snapshot["id"])
    assert snapshot is not None
    snapshot["holdings"][0]["market_value_krw"] = None

    with pytest.raises(AccountProjectionError, match="market_value_krw"):
        from services.account_projection import _project_complete_snapshot

        _project_complete_snapshot(snapshot, {})


def test_inconsistent_account_totals_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    snapshot = insert_snapshot(
        _snapshot(total=100_000.0, invested=80_000.0, cash_value=10_000.0)
    )

    with pytest.raises(AccountProjectionError, match="account totals"):
        build_account_projection(snapshot_id=snapshot["id"])


def test_all_cash_account_has_no_invented_invested_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    snapshot = insert_snapshot(
        _snapshot(
            fingerprint="cash-only",
            holdings=(),
            total=100_000.0,
            invested=0.0,
            cash_value=100_000.0,
        )
    )

    result = build_account_projection(snapshot_id=snapshot["id"])

    assert result["cash_weight_gross"] == pytest.approx(1.0)
    assert result["invested_weights_evaluable"] is False
    assert result["classification_coverage_invested"] is None
    assert result["positions"] == []


def test_zero_invested_value_with_positive_holding_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "projection.sqlite3"))
    initialize_database()
    snapshot = insert_snapshot(
        _snapshot(
            fingerprint="bad-cash-only",
            invested=0.0,
            total=100_000.0,
            cash_value=100_000.0,
        )
    )

    with pytest.raises(AccountProjectionError, match="holding totals"):
        build_account_projection(snapshot_id=snapshot["id"])


def test_policy_layer_weights_and_unclassified_list_are_deterministic(
    complete_snapshot,
):
    result = build_account_projection(
        snapshot_id=complete_snapshot["id"],
        layer_map={("KR", "005930"): "satellite", ("US", "AAPL"): "core"},
    )

    assert result["unclassified"] == []
    assert result["classification_coverage_invested"] == pytest.approx(1.0)
    assert result["layer_weights_invested"] == {
        "core": pytest.approx(2 / 3),
        "satellite": pytest.approx(1 / 3),
        "experiment": pytest.approx(0.0),
    }
