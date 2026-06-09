"""IPS 기반 그룹 요약 및 액션 분류."""

from __future__ import annotations

import pandas as pd

from core.asset import DEFAULT_GROUP, VALID_GROUPS


ACTION_LABELS = {
    "increase_dca": "정기매수 증액 후보",
    "reduce_or_pause_dca": "정기매수 축소·중단 후보",
    "hold_observe": "유지·관찰",
    "review_before_action": "실행 전 점검",
    "risk_control_review": "위험 관리 점검",
    "rebalance_sell_review": "리밸런싱 매도 검토",
    "block_action": "행동 보류",
}

NEXT_STEPS = {
    "increase_dca": "다음 정기매수에서 현재 비중이 목표보다 낮은 자산 또는 그룹의 배분을 늘립니다.",
    "reduce_or_pause_dca": "신규 매수 중단 또는 정기매수 배분 축소를 우선 적용합니다.",
    "hold_observe": "매매하지 않고 다음 점검까지 관찰합니다.",
    "review_before_action": "투자 논리, 역할, 중복성, ETF 대체 가능성을 확인한 뒤 다음 정기매수 반영 여부를 결정합니다.",
    "risk_control_review": "위험기여도, 변동성, 낙폭, 베타, 집중도를 확인하고 신규매수 축소 또는 한도 조정을 검토합니다.",
    "rebalance_sell_review": "정기매수 조정만으로 목표 비중과 위험기여도 초과를 낮추기 어려운 경우에만 부분 매도 여부를 검토합니다.",
    "block_action": "데이터, 분류, 투자 논리 문제를 해소하기 전까지 실행 판단을 보류합니다.",
}

DEFAULT_ACTION_PRIORITIES = {
    "block_action": 1,
    "rebalance_sell_review": 2,
    "risk_control_review": 3,
    "review_before_action": 4,
    "reduce_or_pause_dca": 5,
    "increase_dca": 6,
    "hold_observe": 7,
}

EXECUTION_TYPES = {
    "increase_dca": "dca_adjustment",
    "reduce_or_pause_dca": "pause_or_reduce_dca",
    "hold_observe": "observe",
    "review_before_action": "review_required",
    "risk_control_review": "risk_review",
    "rebalance_sell_review": "sell_review",
    "block_action": "blocked",
}

ACTION_FAMILIES = {
    "increase_dca": "buy_adjustment",
    "reduce_or_pause_dca": "buy_adjustment",
    "hold_observe": "hold",
    "review_before_action": "thesis_review",
    "risk_control_review": "risk_review",
    "rebalance_sell_review": "sell_review",
    "block_action": "blocked",
}

DECISION_CONTEXTS = {
    "regular_review",
    "market_correction",
    "sharp_drop_review",
    "rebalance_review",
}

DECISION_SUMMARIES = {
    "increase_dca": "현재 비중이 목표보다 낮고, 주요 위험·효율·투자 논리 게이트를 통과해 정기매수 보강이 가능합니다.",
    "reduce_or_pause_dca": "현재 비중이 목표보다 높거나 위험 부담이 있어 신규 매수 축소 또는 중단이 우선입니다.",
    "hold_observe": "수치 조정 기준을 넘지 않아 다음 점검까지 관찰합니다.",
    "review_before_action": "투자 논리, 역할, 중복 노출, ETF 대체 가능성을 확인한 뒤 실행 여부를 정해야 합니다.",
    "risk_control_review": "비중보다 위험기여도, 변동성, 낙폭, 베타 또는 집중도 점검이 먼저입니다.",
    "rebalance_sell_review": "정기매수 축소만으로 위험기여도 또는 과대 비중을 낮추기 어려워 부분 리밸런싱 검토가 필요합니다.",
    "block_action": "데이터나 행동 동기가 불충분해 실행 판단을 차단합니다.",
}

REASON_TEXT = {
    "within_hysteresis_or_below_min_trade": "히스테리시스 범위이거나 최소 거래 기준에 미달합니다.",
    "hysteresis_blocked": "히스테리시스 범위 안입니다.",
    "min_trade_blocked": "최소 거래 기준에 미달합니다.",
    "data_quality_low": "데이터 신뢰도가 낮습니다.",
    "missing_ratio_high": "가격 데이터 결측 비율이 높습니다.",
    "observation_too_low": "관측치 수가 부족합니다.",
    "unclassified_group": "자산 그룹이 미분류 상태입니다.",
    "role_unclassified": "자산 역할이 미분류 상태입니다.",
    "risk_ok": "위험기여도가 기준 범위 안에 있습니다.",
    "risk_over": "위험기여도가 기준보다 높습니다.",
    "rc_cap_exceeded": "개별 위험기여도 상한을 초과했습니다.",
    "rc_cap_near": "개별 위험기여도 상한에 근접했습니다.",
    "volatility_high": "변동성이 높습니다.",
    "mdd_high": "최대 낙폭이 큽니다.",
    "beta_high": "베타가 높습니다.",
    "concentration_high": "포트폴리오 집중도가 높습니다.",
    "ips_fit_high": "정책 기준에 잘 맞습니다.",
    "ips_fit_medium": "정책 기준상 조건부 확인이 필요합니다.",
    "ips_fit_low": "정책 기준에 잘 맞지 않습니다.",
    "efficiency_warning": "효율 점수가 기준보다 낮아 위험 대비 보상이 약합니다.",
    "efficiency_low": "효율 점수가 매우 낮습니다.",
    "efficiency_borderline": "효율 점수가 경계선에 있습니다.",
    "sharpe_negative": "샤프 지수가 음수입니다.",
    "sharpe_weak": "샤프 지수가 약합니다.",
    "ir_negative": "정보비율이 음수입니다.",
    "momentum_too_hot": "최근 성과가 과열되어 추격매수 위험이 있습니다.",
    "momentum_weak": "최근 성과가 약합니다.",
    "positive_gap": "현재 비중이 목표보다 낮습니다.",
    "negative_gap": "현재 비중이 목표보다 높습니다.",
    "allocation_under_target": "현재 비중이 목표보다 낮습니다.",
    "allocation_over_target": "현재 비중이 목표보다 높습니다.",
    "group_satellite_over_max": "위성 그룹 비중이 IPS 상한을 초과했습니다.",
    "group_satellite_under_target": "위성 그룹 비중이 목표보다 낮습니다.",
    "group_core_under_target": "코어 그룹 비중이 목표보다 낮습니다.",
    "group_core_over_max": "코어 그룹 비중이 IPS 상한을 초과했습니다.",
    "dca_disabled": "정기매수가 비활성화되어 있습니다.",
    "avoid_immediate_increase": "즉시 증액보다 관찰을 우선합니다.",
    "sell_gate_passed": "비중 초과 또는 논리 훼손으로 리밸런싱 매도 검토 조건에 걸립니다.",
    "sell_gate_blocked": "비중 초과만으로는 매도 검토 조건이 충분하지 않습니다.",
    "buy_gate_blocked": "매수 게이트를 통과하지 못했습니다.",
    "thesis_broken": "투자 논리가 훼손되었습니다.",
    "thesis_watch": "투자 논리가 관찰 상태입니다.",
    "thesis_unknown": "투자 논리가 미정입니다.",
    "thesis_intact": "투자 논리가 유효합니다.",
    "thesis_not_broken": "투자 논리 훼손이 확인되지 않았습니다.",
    "prefer_dca_over_sell": "매도보다 정기매수 조정을 우선합니다.",
    "satellite_requires_review": "위성 자산은 증액 전 장기 보유 가능성과 대체 가능성을 점검해야 합니다.",
    "hedge_role_needs_confirmation": "방어·헤지 역할이 명확한지 확인해야 합니다.",
    "sell_requires_stronger_gate": "매도 검토에는 더 강한 위험 또는 논리 훼손 신호가 필요합니다.",
    "core_priority_context": "현재 판단 모드에서는 목표보다 낮은 코어 비중을 먼저 보강합니다.",
    "satellite_downgraded_for_core_priority": "코어 비중이 낮은 하락장에서는 위성 증액 전 보유 가능성을 먼저 점검합니다.",
    "correction_core_reinforcement": "하락장에서는 목표보다 낮은 코어 비중을 정기매수로 우선 보강합니다.",
    "satellite_correction_requires_review": "하락장 위성 추가매수 전에는 수익률 지속성, 변동성, 장기 보유 가능성을 먼저 점검합니다.",
    "sharp_drop_buy_caution": "단기 급락은 단독 매수 사유가 아닙니다.",
    "unclassified": "분류되지 않은 판단 조합입니다.",
}


def fixed_group(value: object) -> str:
    """입력 그룹을 앱 고정 분류값으로 정규화합니다."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in VALID_GROUPS else DEFAULT_GROUP


def classify_range(value: float, cfg: dict) -> str:
    """비중이 IPS 범위 대비 어느 상태인지 분류합니다."""
    if value < cfg["min"]:
        return "under_min"
    if value < cfg["target"]:
        return "under_target"
    if value <= cfg["max"]:
        return "in_range"
    return "over_max"


def compute_group_summary(metrics_df: pd.DataFrame, ips_config: dict) -> pd.DataFrame:
    """자산별 metrics_df에서 IPS 그룹 요약을 계산합니다."""
    df = metrics_df.copy()
    df["group"] = df.get("group", DEFAULT_GROUP)
    df["group"] = df["group"].fillna(DEFAULT_GROUP).map(fixed_group)
    if "missing_ratio" not in df.columns:
        df["missing_ratio"] = 0.0
    if "observation_count" not in df.columns:
        df["observation_count"] = pd.NA

    return (
        df.groupby("group", as_index=False)
        .agg(
            weight=("가중치", "sum"),
            risk_contribution=("위험기여도", "sum"),
            avg_efficiency=("E", "mean"),
            avg_missing_ratio=("missing_ratio", "mean"),
            min_observation_count=("observation_count", "min"),
        )
        .sort_values("group")
    )


def compute_ips_allocation_status(
    group_summary: pd.DataFrame, ips_config: dict
) -> dict:
    """코어/위성 비중이 IPS 범위 안에 있는지 계산합니다."""
    target_cfg = ips_config.get("target_allocation", {})
    core_cfg = target_cfg.get("core", {"min": 0.70, "target": 0.80, "max": 0.90})
    sat_cfg = target_cfg.get("satellite", {"min": 0.10, "target": 0.20, "max": 0.30})

    core_weight = group_summary.loc[
        group_summary["group"] == "core", "weight"
    ].sum()
    satellite_weight = group_summary.loc[
        group_summary["group"] == "satellite", "weight"
    ].sum()

    return {
        "core_weight": float(core_weight),
        "satellite_weight": float(satellite_weight),
        "core_status": classify_range(float(core_weight), core_cfg),
        "satellite_status": classify_range(float(satellite_weight), sat_cfg),
    }


def _action_result(
    ips_action: str,
    reason_codes: list[str],
    ips_config: dict,
    decision_context: str = "regular_review",
    blocked_reason: str | None = None,
    next_step: str | None = None,
    decision_summary: str | None = None,
    decision_reasons: list[str] | None = None,
    risk_notes: list[str] | None = None,
) -> dict:
    priorities = ips_config.get("action_priority", {})
    fallback_priority = DEFAULT_ACTION_PRIORITIES.get(ips_action, 99)
    return {
        "ips_action": ips_action,
        "action_label": ACTION_LABELS[ips_action],
        "action_family": ACTION_FAMILIES[ips_action],
        "action_priority": priorities.get(ips_action, fallback_priority),
        "execution_type": EXECUTION_TYPES[ips_action],
        "decision_context": decision_context,
        "decision_summary": decision_summary or DECISION_SUMMARIES[ips_action],
        "decision_reasons": decision_reasons
        if decision_reasons is not None
        else [REASON_TEXT.get(code, code) for code in reason_codes],
        "risk_notes": risk_notes or [],
        "reason_codes": reason_codes,
        "next_step": next_step or NEXT_STEPS[ips_action],
        "blocked_reason": blocked_reason,
    }


def _normalize_decision_context(decision_context: str | None) -> str:
    context = str(decision_context or "regular_review").strip()
    return context if context in DECISION_CONTEXTS else "regular_review"


def _with_action_metadata(
    action: dict,
    ips_action: str,
    ips_config: dict,
    decision_context: str,
    reason_codes: list[str] | None = None,
    next_step: str | None = None,
    decision_summary: str | None = None,
    decision_reasons: list[str] | None = None,
    risk_notes: list[str] | None = None,
    blocked_reason: str | None = None,
) -> dict:
    updated = action.copy()
    updated["ips_action"] = ips_action
    updated["action_label"] = ACTION_LABELS[ips_action]
    updated["action_family"] = ACTION_FAMILIES[ips_action]
    updated["action_priority"] = ips_config.get("action_priority", {}).get(
        ips_action, DEFAULT_ACTION_PRIORITIES.get(ips_action, 99)
    )
    updated["execution_type"] = EXECUTION_TYPES[ips_action]
    updated["decision_context"] = decision_context
    updated["decision_summary"] = decision_summary or DECISION_SUMMARIES[ips_action]
    if reason_codes is not None:
        updated["reason_codes"] = reason_codes
    updated["decision_reasons"] = (
        decision_reasons
        if decision_reasons is not None
        else [REASON_TEXT.get(code, code) for code in updated.get("reason_codes", [])]
    )
    updated["risk_notes"] = risk_notes if risk_notes is not None else updated.get("risk_notes", [])
    updated["next_step"] = next_step or NEXT_STEPS[ips_action]
    updated["blocked_reason"] = blocked_reason
    return updated


def apply_contextual_ips_overlay(
    action: dict,
    row: pd.Series | dict,
    allocation_status: dict,
    decision_context: str,
    ips_config: dict,
) -> dict:
    """판단 모드에 따른 IPS 상황 보정을 기존 액션 위에 적용합니다."""
    context = _normalize_decision_context(decision_context)
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    gap = float(row.get("갭%", 0) or 0)
    core_under_target = allocation_status.get("core_status") in {
        "under_min",
        "under_target",
    }
    correction_context = context in {"market_correction", "sharp_drop_review"}
    risk_notes = list(action.get("risk_notes", []))

    if context == "sharp_drop_review" and gap > 0:
        risk_notes.append("하루 급락, 프리마켓 급락, 평단 방어는 단독 매수 사유가 아닙니다.")
        risk_notes.append("FOMO 가능성을 점검하고 정기매수 증액으로 대응 가능한지 먼저 확인합니다.")

    is_correction_core_reinforcement = (
        "correction_core_reinforcement" in action.get("reason_codes", [])
    )

    if (
        correction_context
        and core_under_target
        and group == "core"
        and action["ips_action"] == "increase_dca"
        and is_correction_core_reinforcement
    ):
        return _with_action_metadata(
            action,
            "increase_dca",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "core_priority_context"],
            decision_summary="하락장에는 목표보다 낮은 코어 비중을 정기매수로 보강합니다.",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 목표보다 낮아 장기 시장 노출이 부족합니다.",
                "최근 효율 점수는 낮지만, 코어 비중 미달이 더 중요한 조정 신호입니다.",
                "즉시매수가 아니라 다음 정기매수 배분 조정으로 처리합니다.",
            ],
            risk_notes=risk_notes
            or [
                "효율 점수 미달은 확인하되, 하락장에서는 목표보다 낮은 코어 비중을 정기매수로 보강하는 쪽을 우선했습니다."
            ],
            next_step="다음 정기매수에서 현재 비중이 목표보다 낮은 코어 자산의 배분을 늘립니다.",
        )

    if correction_context and core_under_target and group == "core" and action["ips_action"] == "increase_dca":
        return _with_action_metadata(
            action,
            "increase_dca",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "core_priority_context"],
            decision_summary="목표보다 낮은 코어 비중을 정기매수로 보강합니다.",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 목표보다 낮아 장기 시장 노출이 부족합니다.",
                "하락장 또는 급락 검토에서는 위성 확대보다 목표보다 낮은 코어 비중 보강을 우선합니다.",
            ],
            risk_notes=risk_notes or ["즉시매수가 아니라 다음 정기매수 배분 조정으로 처리합니다."],
            next_step="다음 정기매수에서 코어 자산의 배분을 늘립니다.",
        )

    if correction_context and core_under_target and group == "satellite" and action["ips_action"] == "increase_dca":
        return _with_action_metadata(
            action,
            "review_before_action",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "satellite_downgraded_for_core_priority"],
            decision_summary="하락장 위성 자산은 증액 전 수익률과 변동성을 점검합니다.",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 목표보다 낮아 기본 노출이 부족합니다.",
                "위성 자산 증액 전 수익률 지속성, 변동성 부담, 장기 보유 가능성을 먼저 점검합니다.",
            ],
            risk_notes=risk_notes,
            next_step="위성 자산의 투자 논리와 장기 보유 가능성을 확인한 뒤 다음 정기매수 반영 여부를 결정합니다.",
        )

    if correction_context and core_under_target and group == "unclassified" and action["ips_action"] == "review_before_action":
        next_step = f"{action['next_step']} 판단이 어려운 자산보다 코어 정기매수 증액을 우선합니다."
        return _with_action_metadata(
            action,
            "review_before_action",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "core_priority_context"],
            decision_summary="미분류 자산은 비중 조정 전 그룹을 먼저 확인합니다.",
            decision_reasons=[
                "자산 그룹이 미분류 상태입니다.",
                "코어 비중이 목표보다 낮아 판단이 어려운 자산보다 코어 보강을 우선합니다.",
            ],
            risk_notes=risk_notes,
            next_step=next_step,
        )

    if context == "sharp_drop_review" and risk_notes:
        reason_codes = action["reason_codes"]
        if "sharp_drop_buy_caution" not in reason_codes:
            reason_codes = [*reason_codes, "sharp_drop_buy_caution"]
        return _with_action_metadata(
            action,
            action["ips_action"],
            ips_config,
            context,
            reason_codes=reason_codes,
            decision_reasons=[*action.get("decision_reasons", []), "단기 급락은 단독 매수 사유가 아닙니다."],
            risk_notes=risk_notes,
            next_step=action["next_step"],
            blocked_reason=action.get("blocked_reason"),
        )

    action["decision_context"] = context
    return action


def _as_float(row: pd.Series | dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if value is not None and pd.notna(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _signal_codes(
    row: pd.Series | dict,
    allocation_status: dict,
    e_thresh: float = 0.5,
) -> tuple[list[str], dict[str, bool]]:
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    gap = _as_float(row, "갭%", "gap_pct")
    thesis_status = str(row.get("thesis_status", "unknown") or "unknown").lower()
    efficiency = _as_float(row, "E", "efficiency_score", default=0.5)
    sharpe = _as_float(row, "샤프", "sharpe", default=float("nan"))
    ir = _as_float(row, "IR", "information_ratio", default=float("nan"))
    volatility = _as_float(row, "변동성", "volatility", default=0.0)
    max_drawdown = _as_float(row, "최대낙폭", "MDD", "max_drawdown", default=0.0)
    beta = _as_float(row, "베타", "beta", default=0.0)
    weight = _as_float(row, "현재%", default=0.0) / 100.0
    risk_contribution = _as_float(row, "위험기여도", "risk_contribution", default=0.0)
    rc_target_pct = _as_float(row, "RC_Target%", "rc_target_pct", default=0.0)
    rc_cap_pct = min(12.0, rc_target_pct * 1.5) if rc_target_pct > 0 else 12.0
    current_rc_pct = risk_contribution * 100.0
    return_pct = _as_float(row, "return_total%", "return_total_pct", default=0.0)
    if "return_total%" not in row and "return_total_pct" not in row:
        return_pct *= 100.0
    missing_ratio = _as_float(row, "missing_ratio", default=0.0)
    observations = _as_float(row, "observation_count", default=9999.0)
    data_quality_low = bool(row.get("data_quality_low", False))
    risk_over = bool(row.get("risk_over", False))
    efficiency_warning = bool(row.get("efficiency_warning", efficiency < e_thresh))

    rc_cap_exceeded = current_rc_pct > rc_cap_pct if current_rc_pct > 0 else risk_over
    rc_cap_near = not rc_cap_exceeded and current_rc_pct > (rc_cap_pct * 0.8)
    volatility_high = volatility >= (0.35 if group == "satellite" else 0.25)
    mdd_high = max_drawdown <= (-0.20 if group == "satellite" else -0.15)
    beta_high = beta >= (2.0 if group == "satellite" else 1.5)
    concentration_high = weight >= (0.08 if group == "satellite" else 0.50)
    efficiency_low = efficiency < max(0.2, e_thresh * 0.4)
    efficiency_borderline = not efficiency_low and efficiency_warning
    sharpe_negative = pd.notna(sharpe) and sharpe < 0
    sharpe_weak = pd.notna(sharpe) and 0 <= sharpe < 1.0
    ir_negative = pd.notna(ir) and ir < 0
    momentum_too_hot = return_pct >= (50.0 if group == "satellite" else 40.0)
    momentum_weak = return_pct <= -5.0

    codes: list[str] = []
    if gap > 0:
        codes.extend(["positive_gap", "allocation_under_target"])
    elif gap < 0:
        codes.extend(["negative_gap", "allocation_over_target"])

    if allocation_status.get("satellite_status") == "over_max":
        codes.append("group_satellite_over_max")
    elif allocation_status.get("satellite_status") in {"under_min", "under_target"}:
        codes.append("group_satellite_under_target")
    if allocation_status.get("core_status") in {"under_min", "under_target"}:
        codes.append("group_core_under_target")
    elif allocation_status.get("core_status") == "over_max":
        codes.append("group_core_over_max")

    thesis_code = {
        "intact": "thesis_intact",
        "watch": "thesis_watch",
        "unknown": "thesis_unknown",
        "broken": "thesis_broken",
    }.get(thesis_status)
    if thesis_code:
        codes.append(thesis_code)
    if group == "unclassified":
        codes.extend(["unclassified_group", "role_unclassified"])
    if group == "satellite":
        codes.append("satellite_requires_review")

    if risk_over:
        codes.append("risk_over")
    else:
        codes.append("risk_ok")
    if rc_cap_exceeded:
        codes.append("rc_cap_exceeded")
    elif rc_cap_near:
        codes.append("rc_cap_near")
    if volatility_high:
        codes.append("volatility_high")
    if mdd_high:
        codes.append("mdd_high")
    if beta_high:
        codes.append("beta_high")
    if concentration_high:
        codes.append("concentration_high")

    if efficiency_warning:
        codes.append("efficiency_warning")
    if efficiency_low:
        codes.append("efficiency_low")
    elif efficiency_borderline:
        codes.append("efficiency_borderline")
    if sharpe_negative:
        codes.append("sharpe_negative")
    elif sharpe_weak:
        codes.append("sharpe_weak")
    if ir_negative:
        codes.append("ir_negative")
    if momentum_too_hot:
        codes.append("momentum_too_hot")
    elif momentum_weak:
        codes.append("momentum_weak")

    if data_quality_low:
        codes.append("data_quality_low")
    if missing_ratio > 0.2:
        codes.append("missing_ratio_high")
    if observations < 60:
        codes.append("observation_too_low")

    flags = {
        "rc_cap_exceeded": rc_cap_exceeded,
        "strong_market_risk": volatility_high or mdd_high or beta_high,
        "risk_signal": rc_cap_exceeded or volatility_high or mdd_high or beta_high,
        "efficiency_low": efficiency_low,
        "efficiency_warning": efficiency_warning,
        "sharpe_negative": sharpe_negative,
        "thesis_watch": thesis_status == "watch",
        "thesis_unknown": thesis_status == "unknown",
        "thesis_broken": thesis_status == "broken",
        "data_quality_low": data_quality_low,
        "momentum_too_hot": momentum_too_hot,
    }
    return list(dict.fromkeys(codes)), flags


def _sell_gate_allows(row: pd.Series | dict, allocation_status: dict) -> bool:
    thesis_status = str(row.get("thesis_status", "unknown") or "unknown").lower()
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    gap = float(row.get("갭%", 0) or 0)
    reason_codes, flags = _signal_codes(row, allocation_status)
    strong_sell_risk = (
        flags["rc_cap_exceeded"]
        and (
            flags["strong_market_risk"]
            or "concentration_high" in reason_codes
            or flags["momentum_too_hot"]
        )
    )

    return (
        thesis_status == "broken"
        or (
            group == "satellite"
            and allocation_status.get("satellite_status") == "over_max"
            and gap < 0
            and abs(gap) >= 3.0
            and strong_sell_risk
        )
    )


def classify_ips_action(
    row: pd.Series | dict,
    allocation_status: dict,
    ips_config: dict | None = None,
    decision_context: str = "regular_review",
) -> dict:
    """IPS 원천 신호와 메타데이터로 최종 액션을 분류합니다."""
    ips_config = ips_config or {}
    decision_context = _normalize_decision_context(decision_context)
    gap = float(row.get("갭%", 0) or 0)
    ips_band = str(row.get("IPS등급", row.get("ips_fit_band", "low")) or "low")
    ips_score = float(row.get("IPS적합도", row.get("ips_fit_score", 0)) or 0)
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    dca_enabled = bool(row.get("dca_enabled", True))
    thesis_status = str(row.get("thesis_status", "unknown") or "unknown").lower()
    should_execute = bool(row.get("수치후보", row.get("실행", False)))
    fit_reason = {
        "high": "ips_fit_high",
        "medium": "ips_fit_medium",
        "low": "ips_fit_low",
    }.get(ips_band, "ips_fit_low")
    signal_codes, flags = _signal_codes(row, allocation_status)
    base_reasons = list(dict.fromkeys([fit_reason, *signal_codes]))
    non_trade_reasons = [fit_reason]
    if bool(row.get("히스테리시스제외", False)):
        non_trade_reasons.append("hysteresis_blocked")
    if bool(row.get("최소거래미만", False)):
        non_trade_reasons.append("min_trade_blocked")

    if group == "unclassified":
        next_step = NEXT_STEPS["review_before_action"]
        if allocation_status.get("core_status") in {"under_min", "under_target"}:
            next_step += " 판단이 어려운 자산보다 코어 정기매수 증액을 우선합니다."
        action = _action_result(
            "review_before_action",
            [*base_reasons, "role_unclassified"],
            ips_config,
            decision_context,
            next_step=next_step,
            decision_summary="미분류 자산은 비중 조정 전 그룹과 역할을 먼저 확인합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if flags["thesis_broken"]:
        if gap < 0 and _sell_gate_allows(row, allocation_status):
            action = _action_result(
                "rebalance_sell_review",
                [*base_reasons, "sell_gate_passed"],
                ips_config,
                decision_context,
                decision_summary="투자 논리가 훼손되어 부분 리밸런싱 매도 검토가 필요합니다.",
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        action = _action_result(
            "block_action",
            [*base_reasons, "buy_gate_blocked"],
            ips_config,
            decision_context,
            blocked_reason="투자 논리가 훼손된 자산은 매수 또는 증액 판단을 차단합니다.",
            decision_summary="투자 논리가 훼손되어 실행보다 원인 확인과 보유 여부 검토가 먼저입니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if flags["data_quality_low"] and should_execute:
        action = _action_result(
            "block_action",
            base_reasons,
            ips_config,
            decision_context,
            blocked_reason="데이터 신뢰도가 낮아 실행 판단을 보류합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if not should_execute:
        reason_codes = list(dict.fromkeys([*non_trade_reasons, *signal_codes]))
        hold_action = "risk_control_review" if flags["rc_cap_exceeded"] else "hold_observe"
        action = _action_result(
            hold_action,
            reason_codes,
            ips_config,
            decision_context,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if gap < 0 and _sell_gate_allows(row, allocation_status):
        action = _action_result(
            "rebalance_sell_review",
            [*base_reasons, "sell_gate_passed"],
            ips_config,
            decision_context,
            decision_summary="현재 비중 초과와 위험 신호가 함께 커 부분 리밸런싱 매도 검토가 필요합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    correction_context = decision_context in {"market_correction", "sharp_drop_review"}

    if correction_context and group == "satellite" and gap > 0:
        action = _action_result(
            "review_before_action",
            [*base_reasons, "satellite_correction_requires_review"],
            ips_config,
            decision_context,
            decision_summary="하락장 위성 자산은 증액 전 수익률과 변동성을 점검합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if group == "satellite" and flags["rc_cap_exceeded"]:
        action = _action_result(
            "risk_control_review",
            base_reasons,
            ips_config,
            decision_context,
            decision_summary="비중 조정보다 위험기여도와 집중도 점검이 먼저입니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if gap < 0:
        action = _action_result(
            "reduce_or_pause_dca",
            [*base_reasons, "prefer_dca_over_sell"] if dca_enabled else [*base_reasons, "dca_disabled", "sell_gate_blocked"],
            ips_config,
            decision_context,
            decision_summary="현재 비중이 목표보다 높아 매도보다 신규 매수 축소 또는 중단이 우선입니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if gap > 0 and not dca_enabled:
        action = _action_result(
            "hold_observe",
            [*base_reasons, "dca_disabled"],
            ips_config,
            decision_context,
            decision_summary="현재 비중은 목표보다 낮지만 정기매수가 꺼져 있어 증액을 보류합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if flags["thesis_unknown"]:
        action = _action_result(
            "review_before_action",
            base_reasons,
            ips_config,
            decision_context,
            decision_summary="투자 논리가 미정이라 실행 전 분류와 보유 논리 확인이 먼저입니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if group == "satellite" and (
        flags["thesis_watch"]
        and (
            flags["efficiency_low"]
            or flags["sharpe_negative"]
            or flags["efficiency_warning"]
            or flags["momentum_too_hot"]
        )
    ):
        action = _action_result(
            "review_before_action",
            [*base_reasons, "buy_gate_blocked"],
            ips_config,
            decision_context,
            decision_summary="위성 자산은 관찰 상태와 효율 경고가 있어 증액 전 투자 논리 확인이 먼저입니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if group == "core" and gap > 0 and (flags["efficiency_low"] or flags["sharpe_negative"]):
        action = _action_result(
            "review_before_action",
            [*base_reasons, "hedge_role_needs_confirmation", "buy_gate_blocked"],
            ips_config,
            decision_context,
            decision_summary="코어 자산이라도 효율 지표가 크게 약해 역할 확인 후 보강 여부를 정해야 합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if group == "core" and gap > 0 and dca_enabled and correction_context and ips_score >= 45:
        action = _action_result(
            "increase_dca",
            [*base_reasons, "correction_core_reinforcement"],
            ips_config,
            decision_context,
            decision_summary="하락장에는 목표보다 낮은 코어 비중을 정기매수로 보강합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if gap > 0 and dca_enabled and ips_band in {"high", "medium"}:
        action = _action_result(
            "increase_dca",
            base_reasons,
            ips_config,
            decision_context,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    action = _action_result(
        "review_before_action",
        [*base_reasons, "buy_gate_blocked"],
        ips_config,
        decision_context,
        decision_summary="정책 점수, 효율, 위험, 투자 논리 중 하나 이상이 약해 실행 전 점검이 필요합니다.",
    )
    return apply_contextual_ips_overlay(
        action, row, allocation_status, decision_context, ips_config
    )


def classify_ips_actions(
    proposal_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    allocation_status: dict,
    ips_config: dict,
    decision_context: str = "regular_review",
) -> pd.DataFrame:
    """proposal_df와 metrics_df를 합쳐 전체 IPS 액션 테이블을 생성합니다."""
    metrics_cols = [
        "group",
        "dca_enabled",
        "thesis_status",
        "missing_ratio",
        "observation_count",
        "샤프",
        "IR",
        "변동성",
        "최대낙폭",
        "베타",
        "위험기여도",
        "가중치",
    ]
    meta = metrics_df[[col for col in metrics_cols if col in metrics_df.columns]].copy()

    df = proposal_df.copy()
    meta_df = meta.reset_index().rename(columns={"index": "ticker"})
    for col in ["ticker", *metrics_cols]:
        if col not in df.columns and col in meta_df.columns:
            df = df.merge(meta_df[["ticker", col]], on="ticker", how="left")
    df["group"] = df["group"].fillna(DEFAULT_GROUP).map(fixed_group)
    df["dca_enabled"] = df["dca_enabled"].fillna(True).astype(bool)
    df["thesis_status"] = df["thesis_status"].fillna("unknown")
    df["data_quality_low"] = (
        df.get("missing_ratio", pd.Series(0, index=df.index)).fillna(0).astype(float) > 0.2
    ) | (
        df.get("observation_count", pd.Series(9999, index=df.index)).fillna(9999).astype(float) < 60
    )

    action_rows = [
        classify_ips_action(row, allocation_status, ips_config, decision_context)
        for _, row in df.iterrows()
    ]
    actions = pd.DataFrame(action_rows)
    result = pd.concat([df.reset_index(drop=True), actions], axis=1)
    result["reason_codes_text"] = result["reason_codes"].map(
        lambda codes: ", ".join(codes)
    )
    return result.sort_values(["action_priority", "ticker"]).reset_index(drop=True)
