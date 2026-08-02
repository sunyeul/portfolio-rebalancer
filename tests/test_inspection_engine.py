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
        "total_value_krw": 1000.0,
        "invested_value_krw": 1000.0 * (1.0 - cash),
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


def _layer_map():
    return {("US", "AAA"): "core", ("US", "BBB"): "satellite"}


def _risk_evidence(
    *,
    aaa_drawdown=-0.05,
    bbb_drawdown=-0.05,
    account_drawdown=-0.05,
    aaa_pnl=100.0,
    bbb_pnl=-100.0,
):
    def drawdown(current):
        return {
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
        _layer_map(),
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
        _layer_map(),
    )

    assert result["account"]["investment_principal_krw"] == 1000.0
    assert result["account"]["account_profit_krw"] == 120.0
    assert result["account"]["account_return"] == pytest.approx(0.12)
    assert "tracking_principal_krw" not in result["account"]


def test_metadata_does_not_create_action_without_allocation_or_risk_breach():
    projection = _projection(cash=0.15, aaa=0.95, bbb=0.05)
    result = evaluate_inspection(
        projection,
        {"state": "complete", "input_fingerprint": "perf", "points": []},
        _policy(),
        _layer_map(),
    )
    aaa = next(item for item in result["instruments"] if item["symbol"] == "AAA")
    assert aaa["status"] == "Review"
    encoded = str(result).lower()
    assert "buy" not in encoded and "sell" not in encoded and "execute" not in encoded


def test_review_queue_adds_red_team_without_changing_allocation_decision():
    result = evaluate_inspection(
        _projection(cash=0.15, aaa=0.95, bbb=0.05),
        {"state": "complete", "input_fingerprint": "red-team", "points": []},
        _policy(),
        _layer_map(),
    )

    item = next(item for item in result["review_queue"] if item["identity"] == "US/AAA")

    assert item["status"] == "Review"
    assert item["priority"] == "P2"
    assert item["queue_class"] == "adjustment"
    assert item["suggestion"] == {
        "code": "review_overweight_normalization",
        "label": "초과비중 완화: 향후 정기매수 축소·중단 검토",
    }
    assert item["red_team"] == {
        "counterargument": "비중 범위 이탈만으로 거래나 예외 개입을 확정할 수 없습니다.",
        "evidence_needed": item["verification_task"],
    }


def test_review_queue_red_team_explains_blocking_source_and_performance_limits():
    incomplete_projection = _projection()
    incomplete_projection["positions"][0]["layer"] = None
    blocked = evaluate_inspection(incomplete_projection, None, _policy(), {})
    blocking_item = blocked["review_queue"][0]

    assert blocking_item["queue_class"] == "blocking"
    assert blocking_item["red_team"] == {
        "counterargument": "현재 Toss 원천과 정책 커버리지가 평가 가능한 상태인지 확인하기 전에는 비중 판단을 확정할 수 없습니다.",
        "evidence_needed": blocking_item["verification_task"],
    }

    performance = evaluate_inspection(_projection(), None, _policy(), _layer_map())
    performance_item = next(
        item for item in performance["review_queue"] if item["kind"] == "performance"
    )

    assert performance_item["status"] == "Watch"
    assert performance_item["priority"] == "P4"
    assert performance_item["red_team"] == {
        "counterargument": "수익률·손익·drawdown만으로 비중 조정이나 예외 개입을 확정할 수 없습니다.",
        "evidence_needed": performance_item["verification_task"],
    }


def test_missing_policy_layer_is_review_and_cash_uses_gross_denominator():
    projection = _projection(cash=0.05)
    projection["positions"][0]["layer"] = None
    result = evaluate_inspection(projection, None, _policy(), {})
    assert result["allocation_state"] == "not_evaluable"
    assert result["cash"] is None
    assert result["instruments"] == []
    assert result["review_queue"][0]["queue_class"] == "blocking"


def test_missing_policy_layer_keeps_instrument_gaps_null():
    projection = _projection()
    projection["positions"][0]["layer"] = None
    result = evaluate_inspection(
        projection,
        {"state": "complete", "points": []},
        _policy(),
        {},
    )
    assert result["allocation_state"] == "not_evaluable"
    assert all(
        item["current"] is None and item["gap"] is None
        for item in result["instruments"]
    )
    assert all(
        item["current"] is None and item["gap"] is None for item in result["layers"]
    )


def test_trailing_return_does_not_compound_history_before_window():
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
        _layer_map(),
    )
    assert result["performance"]["annual_twr"] == pytest.approx(0.20)


def test_ytd_return_compounds_from_the_calendar_year_anchor():
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
        _projection(), {"state": "complete", "points": points}, _policy(), _layer_map()
    )
    assert result["performance"]["ytd_twr"] == pytest.approx(0.10)
    assert result["performance"]["trailing_12m_twr"] is None
    assert result["performance"]["annual_twr"] == pytest.approx(0.10)
    assert result["performance"]["measurement"] == "ytd_twr"


def test_annual_return_target_is_descriptive_not_a_status_trigger():
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
            "interval_twr": 0.05,
        },
    ]

    result = evaluate_inspection(
        _projection(), {"state": "complete", "points": points}, _policy(), _layer_map()
    )

    assert result["performance"]["annual_twr"] == pytest.approx(0.05)
    assert result["performance"]["annual_target"] == pytest.approx(0.10)
    assert result["performance"]["status"] == "OK"
    assert result["performance"]["triggers"] == []


def test_account_and_core_drawdown_are_reviewed_without_action():
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": []},
        _policy(),
        _layer_map(),
        _risk_evidence(aaa_drawdown=-0.26, account_drawdown=-0.16),
    )

    aaa = next(item for item in result["instruments"] if item["symbol"] == "AAA")
    assert result["account_profit_loss"]["status"] == "Review"
    assert (
        "account_drawdown_review_threshold" in result["account_profit_loss"]["triggers"]
    )
    assert aaa["status"] == "Watch"
    assert "core_drawdown_review_threshold" in aaa["triggers"]
    assert all(item["status"] != "Action" for item in result["instruments"])


def test_satellite_drawdown_is_review():
    result = evaluate_inspection(
        _projection(),
        {"state": "complete", "points": []},
        _policy(),
        _layer_map(),
        _risk_evidence(bbb_drawdown=-0.21),
    )

    bbb = next(item for item in result["instruments"] if item["symbol"] == "BBB")
    assert bbb["status"] == "Review"
    assert bbb["triggers"] == ["strict_layer_drawdown_review_threshold"]
    encoded = str(bbb).lower()
    assert all(
        term not in encoded
        for term in ("overlap", "burden", "holdability", "substitution", "thesis")
    )
    queue_items = [
        item for item in result["review_queue"] if item["identity"] == "US/BBB"
    ]
    assert len(queue_items) == 1
    assert queue_items[0]["priority"] == "P2"
    assert queue_items[0]["suggestion"] == {
        "code": "hold_and_observe",
        "label": "관찰 유지",
    }
    assert queue_items[0]["queue_class"] == "observation"
    assert all(
        item["identity"] != "US/BBB" for item in result["adjustment_suggestions"]
    )


def test_overweight_never_creates_action_without_explicit_backend_status():
    within_instrument_max = _policy()
    within_instrument_max["instruments"][0]["maximum"] = 0.99
    within = evaluate_inspection(
        _projection(aaa=0.95, bbb=0.05),
        {"state": "complete", "points": []},
        within_instrument_max,
        _layer_map(),
        _risk_evidence(aaa_pnl=-100.0),
    )
    aaa_within = next(item for item in within["instruments"] if item["symbol"] == "AAA")
    assert aaa_within["status"] == "OK"

    hard_breach = _policy()
    action = evaluate_inspection(
        _projection(aaa=0.95, bbb=0.05),
        {"state": "complete", "points": []},
        hard_breach,
        _layer_map(),
        _risk_evidence(aaa_pnl=-100.0),
    )
    aaa_action = next(item for item in action["instruments"] if item["symbol"] == "AAA")
    assert aaa_action["status"] == "Review"
    encoded = str(action).lower()
    assert all(term not in encoded for term in ("buy", "sell", "execute"))


def test_missing_performance_does_not_suppress_cash_or_allocation_suggestions():
    policy = _policy()
    result = evaluate_inspection(_projection(cash=0.05), None, policy, _layer_map())
    assert result["allocation_state"] == "complete"
    assert result["performance"]["status"] == "Watch"
    assert result["account"]["investment_principal_krw"] is None
    assert result["adjustment_suggestions"][0]["priority"] == "P1"
    assert result["adjustment_suggestions"][0]["suggestion"]["code"] == (
        "review_reduce_or_pause_regular_purchase_pace"
    )


def test_all_cash_is_partial_and_only_cash_is_evaluable():
    projection = {
        "snapshot_id": 8,
        "source_fingerprint": "source-8",
        "cash_weight_gross": 1.0,
        "cash_value_krw": 1000.0,
        "total_value_krw": 1000.0,
        "invested_value_krw": 0.0,
        "invested_weights_evaluable": False,
        "layer_weights_invested": {},
        "positions": [],
        "reconciliation": {"holdings": {"all_within_tolerance": True}},
    }
    result = evaluate_inspection(projection, None, _policy(), {})
    assert result["allocation_state"] == "partial"
    assert result["allocation_reason"] == "invested_denominator_unavailable"
    assert result["cash"] is not None
    assert result["layers"] == []
    assert result["instruments"] == []
    assert result["adjustment_suggestions"]
    assert all(item["kind"] == "cash" for item in result["adjustment_suggestions"])


def test_stale_source_is_blocking_and_never_emits_adjustment_suggestion():
    result = evaluate_inspection(
        _projection(cash=0.01),
        None,
        _policy(),
        {},
        source_error="snapshot is stale",
    )
    assert result["allocation_state"] == "not_evaluable"
    assert result["allocation_reason"] == "source_not_current_evaluable"
    assert result["adjustment_suggestions"] == []
    assert result["review_queue"][0]["queue_class"] == "blocking"
    assert result["review_queue"][0]["priority"] is None
    assert result["review_queue"][0]["suggestion"] is None
    assert len(result["review_queue"]) == 1


def test_currentness_block_preserves_no_action_or_secondary_queue_items():
    result = evaluate_inspection(
        _projection(cash=0.01),
        {"state": "complete", "points": []},
        _policy(),
        _layer_map(),
        risk_evidence=_risk_evidence(aaa_drawdown=-0.5, bbb_drawdown=-0.5),
        source_currentness={
            "is_current": False,
            "reasons": ["snapshot_stale"],
            "snapshot_age_seconds": 259201,
        },
    )

    assert result["allocation_state"] == "not_evaluable"
    assert result["allocation_reason"] == "snapshot_stale"
    assert result["adjustment_suggestions"] == []
    assert [item["queue_class"] for item in result["review_queue"]] == ["blocking"]
    assert all(item["status"] != "Action" for item in result["review_queue"])


def test_configured_absent_instrument_uses_zero_current_weight():
    policy = _policy()
    policy["instruments"].append(
        {
            "market_country": "US",
            "symbol": "CCC",
            "layer": "core",
            "minimum": 0.05,
            "target": 0.10,
            "maximum": 0.20,
        }
    )
    result = evaluate_inspection(
        _projection(), {"state": "complete", "points": []}, policy, _layer_map()
    )
    ccc = next(item for item in result["instruments"] if item["identity"] == "US/CCC")
    assert ccc["current"] == 0.0
    assert ccc["priority"] == "P3"
    assert ccc["suggestion"]["code"] == "review_increase_regular_purchase_allocation"


def test_overweight_precedes_secondary_review_for_non_action_unit():
    policy = _policy()
    policy["instruments"][0]["maximum"] = 0.85
    result = evaluate_inspection(
        _projection(aaa=0.95, bbb=0.05),
        {"state": "complete", "points": []},
        policy,
        _layer_map(),
    )
    aaa = next(item for item in result["instruments"] if item["identity"] == "US/AAA")
    assert aaa["status"] == "Review"
    assert aaa["priority"] == "P2"
    assert aaa["suggestion"]["code"] == "review_overweight_normalization"


def test_cash_shortfall_blocks_layer_and_instrument_p3_suggestions():
    policy = _policy()
    policy["instruments"][0]["minimum"] = 0.9
    result = evaluate_inspection(
        _projection(cash=0.05, aaa=0.75, bbb=0.25),
        {"state": "complete", "points": []},
        policy,
        _layer_map(),
    )
    assert result["cash"]["priority"] == "P1"
    assert (
        result["cash"]["suggestion"]["code"]
        == "review_reduce_or_pause_regular_purchase_pace"
    )
    assert all(item["priority"] != "P3" for item in result["adjustment_suggestions"])


def test_underweight_instrument_review_uses_policy_layer_only():
    policy = _policy()
    policy["instruments"][0]["minimum"] = 0.9
    result = evaluate_inspection(
        _projection(cash=0.15, aaa=0.75, bbb=0.25),
        {"state": "complete", "points": []},
        policy,
        _layer_map(),
    )
    aaa = next(item for item in result["instruments"] if item["identity"] == "US/AAA")
    assert aaa["priority"] == "P3"
    assert aaa["suggestion"]["code"] == "review_increase_regular_purchase_allocation"
    queue_item = next(
        item for item in result["review_queue"] if item["identity"] == "US/AAA"
    )
    assert "decision_detail" not in aaa
    assert "decision_detail" not in queue_item


def test_missing_configured_policy_instrument_blocks_invested_universe_coverage():
    policy = _policy()
    policy["instruments"] = [
        item for item in policy["instruments"] if item["symbol"] != "AAA"
    ]
    result = evaluate_inspection(
        _projection(), {"state": "complete", "points": []}, policy
    )
    assert result["allocation_state"] == "not_evaluable"
    assert result["allocation_reason"] == "policy_coverage_incomplete"
    assert result["adjustment_suggestions"] == []
    assert result["review_queue"][0]["queue_class"] == "blocking"


def test_invalid_configured_target_blocks_invested_universe_coverage():
    policy = _policy()
    policy["instruments"].append(
        {
            "market_country": "US",
            "symbol": "CCC",
            "layer": "core",
            "minimum": 0.10,
            "target": None,
            "maximum": 0.20,
        }
    )
    result = evaluate_inspection(
        _projection(), {"state": "complete", "points": []}, policy, _layer_map()
    )
    assert result["allocation_state"] == "not_evaluable"
    assert result["allocation_reason"] == "policy_coverage_incomplete"
    assert result["adjustment_suggestions"] == []
