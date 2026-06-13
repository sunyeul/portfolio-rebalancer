"""Shared IPS operating-view builders for API, CLI, and saved snapshots."""

from __future__ import annotations

from typing import Any


GUARDRAILS = {
    "not_investment_advice": True,
    "no_immediate_order_instruction": True,
}
NOT_ADVICE_NOTICE = "CLI/API 결과는 IPS 점검 자료이며 즉시 매수/매도 주문 지시가 아닙니다."
STATUS_LABELS = {
    "data_or_policy_blocked": "데이터/정책 확인 필요",
    "review_required": "검토 필요",
    "dca_adjustment_available": "정기매수 조정 후보 있음",
    "ok": "특이사항 없음",
}
STATUS_DESCRIPTIONS = {
    "data_or_policy_blocked": "데이터 누락, 품질 저하, 정책 차단 항목을 먼저 확인해야 합니다.",
    "review_required": "투자 논리, 위험, 예외적 리밸런싱 검토 항목이 있습니다.",
    "dca_adjustment_available": "다음 정기매수에서 배분 조정을 검토할 후보가 있습니다.",
    "ok": "표준 운용 큐에서 즉시 점검할 항목이 없습니다.",
}
RISK_FLAG_TYPE_LABELS = {
    "risk_contribution_limit": "위험기여도 한도",
    "risk_over": "위험기여도 초과",
    "data_quality_low": "데이터 품질 낮음",
    "missing_ticker": "가격 데이터 누락",
}
RISK_SEVERITY_LABELS = {
    "warning": "주의",
    "review": "점검",
}


def empty_operating_view() -> dict[str, Any]:
    """Return the stable empty IPS operating-view shape."""
    return {
        "ips_status": None,
        "dca_plan": {
            "increase": [],
            "reduce_or_pause": [],
            "hold": [],
        },
        "review_queue": {
            "thesis_review": [],
            "risk_review": [],
            "sell_review": [],
            "blocked": [],
        },
        "risk_flags": [],
        "playbook": None,
        "guardrails": GUARDRAILS.copy(),
        "not_advice_notice": NOT_ADVICE_NOTICE,
    }


def _value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _brief_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": _value(row, "ticker"),
        "ips_action": _value(row, "ips_action"),
        "label": _value(row, "action_label"),
        "family": _value(row, "action_family"),
        "execution_type": _value(row, "execution_type"),
        "current_weight_pct": _value(row, "current_weight_pct", "현재%"),
        "target_weight_pct": _value(row, "target_weight_pct", "목표%"),
        "gap_pct": _value(row, "gap_pct", "갭%"),
        "group": _value(row, "group"),
        "ips_fit_score": _value(row, "ips_fit_score", "IPS적합도"),
        "ips_fit_band": _value(row, "ips_fit_band", "IPS등급"),
        "risk_over": _value(row, "risk_over"),
        "data_quality_low": _value(row, "data_quality_low"),
        "decision_summary": _value(row, "decision_summary", "action_reason", "판단사유"),
        "decision_reasons": _value(row, "decision_reasons") or [],
        "risk_notes": _value(row, "risk_notes") or [],
        "next_step": _value(row, "next_step"),
        "blocked_reason": _value(row, "blocked_reason"),
    }


def _risk_flag(type_code: str, severity_code: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": type_code,
        "type_code": type_code,
        "type_label": RISK_FLAG_TYPE_LABELS.get(type_code, type_code),
        "severity": severity_code,
        "severity_code": severity_code,
        "severity_label": RISK_SEVERITY_LABELS.get(severity_code, severity_code),
        **fields,
    }


def build_evaluation_view(
    *,
    metrics: list[dict[str, Any]],
    proposal: list[dict[str, Any]],
    ips_actions: list[dict[str, Any]],
    group_summary: list[dict[str, Any]],
    rc_violations: list[dict[str, Any]],
    missing_tickers: list[str] | None = None,
    playbook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build stable IPS status, DCA Plan, Review Queue, and Risk Flags."""
    missing_tickers = missing_tickers or []
    proposal_by_ticker = {_value(row, "ticker"): row for row in proposal}
    action_rows = []
    for action in ips_actions:
        ticker = _value(action, "ticker")
        action_rows.append({**proposal_by_ticker.get(ticker, {}), **action})

    dca_plan = {
        "increase": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "increase_dca"
        ],
        "reduce_or_pause": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "reduce_or_pause_dca"
        ],
        "hold": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "hold_observe"
        ],
    }
    review_queue = {
        "thesis_review": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "review_before_action"
        ],
        "risk_review": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "risk_control_review"
        ],
        "sell_review": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "rebalance_sell_review"
        ],
        "blocked": [
            _brief_item(row)
            for row in action_rows
            if _value(row, "ips_action") == "block_action"
        ],
    }

    risk_flags: list[dict[str, Any]] = []
    for row in rc_violations:
        risk_flags.append(
            _risk_flag(
                "risk_contribution_limit",
                "warning",
                ticker=_value(row, "ticker"),
                message=_value(row, "status", "상태") or "위험기여도 상한 초과 위험",
                current_rc_pct=_value(row, "current_rc_pct", "현재RC%"),
                rc_cap_pct=_value(row, "rc_cap_pct", "RC상한%"),
            )
        )
    for row in action_rows:
        if _value(row, "data_quality_low"):
            risk_flags.append(
                _risk_flag(
                    "data_quality_low",
                    "warning",
                    ticker=_value(row, "ticker"),
                    message="데이터 품질이 낮아 실행 판단보다 점검이 우선입니다.",
                    missing_ratio=_value(row, "missing_ratio"),
                    observation_count=_value(row, "observation_count"),
                )
            )
        if _value(row, "risk_over"):
            risk_flags.append(
                _risk_flag(
                    "risk_over",
                    "review",
                    ticker=_value(row, "ticker"),
                    message="위험기여도가 기준보다 높아 위험 점검이 필요합니다.",
                    rc_over_pct=_value(row, "rc_over_pct", "RC_Over%"),
                )
            )
    for ticker in missing_tickers:
        risk_flags.append(
            _risk_flag(
                "missing_ticker",
                "warning",
                ticker=ticker,
                message="가격 데이터를 조회하지 못했습니다.",
            )
        )

    review_count = sum(len(items) for items in review_queue.values())
    dca_count = len(dca_plan["increase"]) + len(dca_plan["reduce_or_pause"])
    if review_queue["blocked"] or missing_tickers:
        status = "data_or_policy_blocked"
    elif risk_flags or review_count:
        status = "review_required"
    elif dca_count:
        status = "dca_adjustment_available"
    else:
        status = "ok"

    top_risk_contributors = sorted(
        metrics,
        key=lambda row: _value(row, "risk_contribution", "위험기여도") or 0,
        reverse=True,
    )[:5]

    view = empty_operating_view()
    view.update(
        {
            "ips_status": {
                "status": status,
                "status_code": status,
                "status_label": STATUS_LABELS.get(status, status),
                "status_description": STATUS_DESCRIPTIONS.get(status, status),
                "summary": {
                    "dca_increase_count": len(dca_plan["increase"]),
                    "dca_reduce_or_pause_count": len(dca_plan["reduce_or_pause"]),
                    "hold_count": len(dca_plan["hold"]),
                    "review_count": review_count,
                    "risk_flag_count": len(risk_flags),
                },
                "group_summary": group_summary,
                "top_risk_contributors": top_risk_contributors,
            },
            "dca_plan": dca_plan,
            "review_queue": review_queue,
            "risk_flags": risk_flags,
            "playbook": playbook,
        }
    )
    return view
