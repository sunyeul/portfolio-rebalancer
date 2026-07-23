from datetime import datetime, timedelta, timezone

import pytest

from services.risk_evidence import build_account_profit_loss, build_risk_evidence


RISK_POLICY = {
    "lookback_sessions": 252,
    "minimum_history_points": 200,
    "max_data_age_days": 7,
    "max_gap_days": 7,
    "account_drawdown_review": -0.15,
    "instrument_drawdown_review": {
        "core": -0.25,
        "satellite": -0.20,
        "experiment": -0.15,
    },
}


def _point(point_at: str, interval_twr: float, *, state: str = "evaluable"):
    return {
        "point_at": point_at,
        "interval_twr": interval_twr,
        "evaluation_state": state,
        "current_cost_basis_krw": 1000.0,
        "unrealized_pnl_krw": 100.0,
        "tracked_realized_pnl_krw": 20.0,
        "actual_realized_pnl_krw": 30.0,
    }


def _candle(point_at: datetime, close: float, fingerprint: str):
    return {
        "source_kind": "stock",
        "market_country": "US",
        "symbol": "AAA",
        "interval": "1d",
        "candle_at": point_at.isoformat(),
        "currency": "USD",
        "open_price": close,
        "high_price": close,
        "low_price": close,
        "close_price": close,
        "volume": 100.0,
        "adjusted": 1,
        "adjusted_supported": 1,
        "source_fingerprint": fingerprint,
    }


def _projection():
    return {
        "snapshot_id": 7,
        "synced_at": "2026-07-22T16:01:04+00:00",
        "positions": [
            {
                "market_country": "US",
                "symbol": "AAA",
                "market_value_krw": 1200.0,
                "cost_krw": 1000.0,
                "profit_loss_krw": 200.0,
            }
        ],
    }


def test_account_profit_loss_drawdown_uses_compounded_twr_curve():
    evidence = build_account_profit_loss(
        snapshot_id=7,
        performance_run={
            "id": 3,
            "state": "complete",
            "execution_count": 1,
            "points": [
                _point("2026-01-01T00:00:00Z", 0.00),
                _point("2026-02-01T00:00:00Z", 0.20),
                _point("2026-03-01T00:00:00Z", -0.25),
                _point("2026-04-01T00:00:00Z", 0.10),
            ],
        },
    )

    assert evidence["unrealized_return"] == pytest.approx(0.1)
    assert evidence["realized_pnl_supported"] is True
    assert evidence["tracked_realized_pnl_krw"] == 20.0
    assert evidence["actual_realized_pnl_krw"] == 30.0
    assert evidence["drawdown"]["maximum"] == pytest.approx(-0.25)
    assert evidence["drawdown"]["current"] == pytest.approx(-0.175)


def test_account_realized_zero_is_unsupported_without_execution_evidence():
    evidence = build_account_profit_loss(
        snapshot_id=7,
        performance_run={
            "id": 3,
            "state": "complete",
            "execution_count": 0,
            "points": [_point("2026-01-01T00:00:00Z", 0.0)],
        },
    )

    assert evidence["realized_pnl_supported"] is False
    assert evidence["tracked_realized_pnl_krw"] is None
    assert evidence["actual_realized_pnl_krw"] is None
    assert evidence["drawdown"]["state"] == "insufficient_history"


def test_account_drawdown_does_not_bridge_non_evaluable_boundary():
    evidence = build_account_profit_loss(
        snapshot_id=7,
        performance_run={
            "id": 3,
            "state": "partial",
            "execution_count": 0,
            "points": [
                _point("2026-01-01T00:00:00Z", 0.0),
                _point(
                    "2026-02-01T00:00:00Z", 0.0, state="non_evaluable"
                ),
                _point("2026-03-01T00:00:00Z", 0.2),
            ],
        },
    )

    assert evidence["drawdown"]["state"] == "boundary_unresolved"
    assert evidence["drawdown"]["current"] is None


def test_risk_evidence_calculates_instrument_drawdown_and_market_fingerprint():
    through = datetime(2026, 7, 22, 16, 1, 4, tzinfo=timezone.utc)
    candles = [
        _candle(
            through - timedelta(days=199 - index),
            120.0 if index == 1 else 84.0 if index == 2 else 100.0,
            f"candle-{index}",
        )
        for index in range(200)
    ]
    candles = sorted(candles, key=lambda item: item["candle_at"])

    evidence = build_risk_evidence(
        _projection(),
        {
            "id": 3,
            "state": "complete",
            "execution_count": 0,
            "points": [_point("2026-01-01T00:00:00Z", 0.0)],
        },
        {("US", "AAA"): candles},
        RISK_POLICY,
    )

    instrument = evidence["instruments"]["US/AAA"]
    assert instrument["unrealized_return"] == pytest.approx(0.2)
    assert instrument["drawdown"]["state"] == "complete"
    assert instrument["drawdown"]["history_points"] == 200
    assert instrument["drawdown"]["maximum"] == pytest.approx(-0.3)
    assert instrument["drawdown"]["current"] == pytest.approx(-1 / 6)
    assert evidence["market_evidence_fingerprint"]
    assert evidence["market_evidence"]["US/AAA"]["source_fingerprint"]


@pytest.mark.parametrize(
    "mutator, expected_state",
    [
        (lambda rows: rows.__setitem__(0, dict(rows[0], adjusted=0)), "unadjusted"),
        (
            lambda rows: rows.__setitem__(0, dict(rows[0], close_price=0)),
            "invalid_candle",
        ),
        (
            lambda rows: rows.append(dict(rows[0])),
            "duplicate_timestamp",
        ),
    ],
)
def test_instrument_drawdown_rejects_invalid_candle_quality(mutator, expected_state):
    through = datetime(2026, 7, 22, tzinfo=timezone.utc)
    rows = [
        _candle(through - timedelta(days=200 - index), 100.0, str(index))
        for index in range(200)
    ]
    mutator(rows)
    evidence = build_risk_evidence(
        dict(_projection(), synced_at=through.isoformat()),
        None,
        {("US", "AAA"): rows},
        RISK_POLICY,
    )

    assert evidence["instruments"]["US/AAA"]["drawdown"]["state"] == expected_state


@pytest.mark.parametrize(
    ("extra_day", "expected_state"),
    [(8, "stale"), (1, "gap")],
)
def test_instrument_drawdown_rejects_stale_or_gapped_history(extra_day, expected_state):
    through = datetime(2026, 7, 22, tzinfo=timezone.utc)
    if expected_state == "gap":
        days = [day for day in range(207, -1, -1) if not 93 <= day <= 100]
        rows = [
            _candle(through - timedelta(days=day), 100.0, str(day))
            for day in days
        ]
    else:
        rows = [
            _candle(
                through - timedelta(days=extra_day + 199 - index),
                100.0,
                str(index),
            )
            for index in range(200)
        ]
    evidence = build_risk_evidence(
        dict(_projection(), synced_at=through.isoformat()),
        None,
        {("US", "AAA"): rows},
        RISK_POLICY,
    )

    assert evidence["instruments"]["US/AAA"]["drawdown"]["state"] == expected_state


def test_missing_projection_and_performance_are_explicit_not_zero():
    evidence = build_risk_evidence(None, None, {}, RISK_POLICY)

    assert evidence["account_profit_loss"]["state"] == "source_unavailable"
    assert evidence["account_profit_loss"]["unrealized_pnl_krw"] is None
    assert evidence["market_evidence"] == {}
    assert evidence["market_evidence_fingerprint"]
