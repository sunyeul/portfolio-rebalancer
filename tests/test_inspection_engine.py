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
        "risk_review": {
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


def _risk_evidence(
    *,
    aaa_drawdown=-0.05,
    bbb_drawdown=-0.05,
    account_drawdown=-0.05,
    aaa_pnl=100.0,
    bbb_pnl=-100.0,
):
    drawdown = lambda current: {
        "state": "complete",
        "current": current,
        "maximum": current,
        "history_points": 252,
    }
    return {
        "account_profit_loss": {
            "state": "complete",
            "snapshot_id": 7,
            "performance_run_id": 3,
            "cost_basis_krw": 1000.0,
            "unrealized_pnl_krw": aaa_pnl + bbb_pnl,
            "unrealized_return": 0.0,
            "realized_pnl_supported": False,
            "drawdown": drawdown(account_drawdown),
        },
        "instruments": {
            "US/AAA": {
                "snapshot_id": 7,
                "cost_basis_krw": 1000.0,
                "unrealized_pnl_krw": aaa_pnl,
                "unrealized_return": aaa_pnl / 1000.0,
                "drawdown": drawdown(aaa_drawdown),
            },
            "US/BBB": {
                "snapshot_id": 7,
                "cost_basis_krw": 1000.0,
                "unrealized_pnl_krw": bbb_pnl,
                "unrealized_return": bbb_pnl / 1000.0,
                "drawdown": drawdown(bbb_drawdown),
            },
        },
        "market_evidence_fingerprint": "market-test",
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


def test_account_result_exposes_investment_principal_profit_and_return():
    projection = _projection()
    projection.update(
        total_value_krw=1120.0,
        invested_value_krw=952.0,
        cash_value_krw=168.0,
    )
    result = evaluate_inspection(
        projection,
        {
            "state": "complete",
            "input_fingerprint": "principal-return",
            "points": [
                {
                    "point_at": "2026-07-23T00:00:00Z",
                    "evaluation_state": "evaluable",
                    "investment_principal_krw": 1000.0,
                    "account_gain_krw": 120.0,
                    "simple_return": 0.12,
                }
            ],
        },
        _policy(),
        {
            ("US", "AAA"): {"layer": "core", "thesis_status": "valid"},
            ("US", "BBB"): {"layer": "satellite", "thesis_status": "valid"},
        },
    )

    assert result["account"]["investment_principal_krw"] == 1000.0
    assert result["account"]["account_profit_krw"] == 120.0
    assert result["account"]["account_return"] == pytest.approx(0.12)
    assert "tracking_principal_krw" not in result["account"]


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


def test_account_and_core_drawdown_are_reviewed_without_action():
    profiles = {
        ("US", "AAA"): {
            "layer": "core",
            "thesis_status": "valid",
            "overlap_status": "unknown",
            "management_burden_status": "unknown",
            "holdability_status": "unknown",
            "etf_substitution_status": "unknown",
        },
        ("US", "BBB"): {
            "layer": "satellite",
            "thesis_status": "valid",
            "overlap_status": "clear",
            "management_burden_status": "clear",
            "holdability_status": "clear",
            "etf_substitution_status": "not_applicable",
        },
    }
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": []},
        _policy(),
        profiles,
        _risk_evidence(aaa_drawdown=-0.26, account_drawdown=-0.16),
    )

    aaa = next(item for item in result["instruments"] if item["symbol"] == "AAA")
    assert result["account_profit_loss"]["status"] == "Review"
    assert "account_drawdown_review_threshold" in result["account_profit_loss"]["triggers"]
    assert aaa["status"] == "Watch"
    assert "core_drawdown_review_threshold" in aaa["triggers"]
    assert all(item["status"] != "Action" for item in result["instruments"])


def test_satellite_drawdown_and_unknown_factors_are_one_review_unit():
    profiles = {
        ("US", "AAA"): {
            "layer": "core",
            "thesis_status": "valid",
        },
        ("US", "BBB"): {
            "layer": "satellite",
            "thesis_status": "valid",
            "overlap_status": "unknown",
            "management_burden_status": "review",
            "holdability_status": "unknown",
            "etf_substitution_status": "unknown",
        },
    }
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": []},
        _policy(),
        profiles,
        _risk_evidence(bbb_drawdown=-0.21),
    )

    bbb = next(item for item in result["instruments"] if item["symbol"] == "BBB")
    assert bbb["status"] == "Review"
    assert bbb["triggers"].count("management_burden_review") == 1
    assert "overlap_unknown" in bbb["triggers"]
    assert "holdability_unknown" in bbb["triggers"]
    assert "etf_substitution_unknown" in bbb["triggers"]
    queue_items = [item for item in result["review_queue"] if item["identity"] == "US/BBB"]
    assert len(queue_items) == 1


def test_broken_thesis_requires_same_instrument_hard_maximum_for_action():
    profiles = {
        ("US", "AAA"): {
            "layer": "core",
            "thesis_status": "broken",
            "overlap_status": "clear",
            "management_burden_status": "clear",
            "holdability_status": "clear",
            "etf_substitution_status": "not_applicable",
        },
        ("US", "BBB"): {
            "layer": "satellite",
            "thesis_status": "valid",
            "overlap_status": "clear",
            "management_burden_status": "clear",
            "holdability_status": "clear",
            "etf_substitution_status": "not_applicable",
        },
    }
    within_instrument_max = _policy()
    within_instrument_max["instruments"][0]["maximum"] = 0.99
    within = evaluate_inspection(
        _projection(aaa=0.95, bbb=0.05),
        {"state": "complete", "points": []},
        within_instrument_max,
        profiles,
        _risk_evidence(aaa_pnl=-100.0),
    )
    aaa_within = next(item for item in within["instruments"] if item["symbol"] == "AAA")
    assert aaa_within["status"] == "Review"
    assert "broken_thesis_and_hard_maximum_breach" not in aaa_within["triggers"]

    hard_breach = _policy()
    action = evaluate_inspection(
        _projection(aaa=0.95, bbb=0.05),
        {"state": "complete", "points": []},
        hard_breach,
        profiles,
        _risk_evidence(aaa_pnl=-100.0),
    )
    aaa_action = next(item for item in action["instruments"] if item["symbol"] == "AAA")
    assert aaa_action["status"] == "Action"
    assert "broken_thesis_and_hard_maximum_breach" in aaa_action["triggers"]
    encoded = str(action).lower()
    assert all(term not in encoded for term in ("buy", "sell", "execute"))
