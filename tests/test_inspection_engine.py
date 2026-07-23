import pytest

from services.inspection_engine import evaluate_inspection


def _policy(measurement="ytd_twr"):
    return {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": measurement,
            "minimum_history_days": 365,
        },
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 0.7, "target": 0.8, "maximum": 0.9},
            "satellite": {"minimum": 0.1, "target": 0.2, "maximum": 0.3},
            "experiment": {"minimum": 0, "target": 0, "maximum": 0.05},
        },
        "instruments": [
            {
                "market_country": "US",
                "symbol": "AAA",
                "layer": "core",
                "minimum": 0.7,
                "target": 0.8,
                "maximum": 0.9,
            },
            {
                "market_country": "US",
                "symbol": "BBB",
                "layer": "satellite",
                "minimum": 0.1,
                "target": 0.2,
                "maximum": 0.3,
            },
        ],
    }


def _projection(cash=0.15, aaa=0.8, bbb=0.2):
    return {
        "snapshot_id": 7,
        "account_alias": "toss-brokerage",
        "source_fingerprint": "source-7",
        "cash_weight_gross": cash,
        "cash_value_krw": cash * 1000,
        "layer_weights_invested": {
            "core": aaa / (aaa + bbb),
            "satellite": bbb / (aaa + bbb),
            "experiment": 0.0,
        },
        "positions": [
            {
                "market_country": "US",
                "symbol": "AAA",
                "layer": "core",
                "invested_weight": aaa,
                "profit_loss_krw": 100,
            },
            {
                "market_country": "US",
                "symbol": "BBB",
                "layer": "satellite",
                "invested_weight": bbb,
                "profit_loss_krw": -100,
            },
        ],
        "reconciliation": {"holdings": {"all_within_tolerance": True}},
    }


def test_ytd_without_year_start_point_is_watch_and_cumulative_return_is_separate():
    result = evaluate_inspection(
        _projection(),
        {
            "state": "complete",
            "input_fingerprint": "perf",
            "points": [
                {
                    "point_at": "2026-01-15T00:00:00Z",
                    "evaluation_state": "evaluable",
                    "interval_twr": 0.05,
                }
            ],
        },
        _policy(),
        {
            ("US", "AAA"): {"layer": "core", "thesis_status": "valid"},
            ("US", "BBB"): {"layer": "satellite", "thesis_status": "valid"},
        },
    )
    assert result["performance"]["cumulative_twr"] == pytest.approx(0.05)
    assert result["performance"]["annual_twr"] is None
    assert result["performance"]["ytd_twr"] is None
    assert result["performance"]["status"] == "Watch"
    assert result["performance"]["triggers"] == ["ytd_return_history_insufficient"]
    aaa = next(item for item in result["instruments"] if item["symbol"] == "AAA")
    bbb = next(item for item in result["instruments"] if item["symbol"] == "BBB")
    assert aaa["layer"] == "core"
    assert bbb["layer"] == "satellite"


def test_broken_thesis_and_hard_maximum_is_action_without_order_semantics():
    projection = _projection(cash=0.15, aaa=0.95, bbb=0.05)
    result = evaluate_inspection(
        projection,
        {"state": "complete", "input_fingerprint": "perf", "points": []},
        _policy(),
        {
            ("US", "AAA"): {"layer": "core", "thesis_status": "broken"},
            ("US", "BBB"): {"layer": "satellite", "thesis_status": "valid"},
        },
    )
    aaa = next(item for item in result["instruments"] if item["symbol"] == "AAA")
    assert aaa["status"] == "Action"
    encoded = str(result).lower()
    assert "buy" not in encoded and "sell" not in encoded and "execute" not in encoded


def test_missing_profile_is_review_and_cash_uses_gross_denominator():
    result = evaluate_inspection(_projection(cash=0.05), None, _policy(), {})
    assert result["state"] == "not_evaluable"
    assert result["cash"] is None
    assert result["instruments"] == []
    assert result["review_queue"][0]["kind"] == "performance"


def test_missing_profile_keeps_instrument_gaps_null():
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": []},
        _policy(),
        {},
    )
    assert result["state"] == "not_evaluable"
    assert all(
        item["current"] is None and item["gap"] is None
        for item in result["instruments"]
    )
    assert all(
        item["current"] is None and item["gap"] is None for item in result["layers"]
    )


def test_trailing_return_does_not_compound_history_before_window():
    profiles = {
        ("US", "AAA"): {"layer": "core", "thesis_status": "valid"},
        ("US", "BBB"): {"layer": "satellite", "thesis_status": "valid"},
    }
    points = [
        {
            "point_at": "2024-12-31T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.50,
        },
        {
            "point_at": "2025-12-31T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.10,
        },
        {
            "point_at": "2026-12-31T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.20,
        },
    ]
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": points},
        _policy("trailing_12_month_twr"),
        profiles,
    )
    assert result["performance"]["annual_twr"] == pytest.approx(0.20)


def test_ytd_return_compounds_from_the_calendar_year_anchor():
    profiles = {
        ("US", "AAA"): {"layer": "core", "thesis_status": "valid"},
        ("US", "BBB"): {"layer": "satellite", "thesis_status": "valid"},
    }
    points = [
        {
            "point_at": "2025-12-31T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.03,
        },
        {
            "point_at": "2026-01-01T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.02,
        },
        {
            "point_at": "2026-07-22T00:00:00Z",
            "evaluation_state": "evaluable",
            "interval_twr": 0.10,
        },
    ]
    result = evaluate_inspection(
        _projection(), {"state": "complete", "points": points}, _policy(), profiles
    )
    assert result["performance"]["ytd_twr"] == pytest.approx(0.10)
    assert result["performance"]["trailing_12m_twr"] is None
    assert result["performance"]["annual_twr"] == pytest.approx(0.10)
    assert result["performance"]["measurement"] == "ytd_twr"
