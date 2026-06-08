"""IPS 기반 그룹 요약 및 액션 분류."""

from __future__ import annotations

import pandas as pd

from core.asset import DEFAULT_GROUP, VALID_GROUPS


ACTION_LABELS = {
    "increase_dca": "정기매수 증액 후보",
    "decrease_dca": "정기매수 감액/중단 후보",
    "hold_observe": "유지·관찰",
    "review_thesis": "투자 논리 점검",
    "exceptional_buy_review": "예외적 즉시매수 검토",
    "consider_rebalance_sell": "예외적 리밸런싱 매도 검토",
    "block_action": "행동 보류",
}

NEXT_STEPS = {
    "increase_dca": "다음 정기매수 배분에서 해당 자산 또는 그룹의 비중을 늘립니다.",
    "decrease_dca": "신규 매수 중단 또는 정기매수 배분 축소를 우선 검토합니다.",
    "hold_observe": "매매하지 않고 다음 점검까지 관찰합니다.",
    "review_thesis": "투자 논리, 중복성, ETF 대체 가능성을 점검합니다.",
    "exceptional_buy_review": "투자 논리, 목표 비중, 정기매수로 대응 불가 여부를 모두 확인한 뒤에만 즉시매수를 검토합니다.",
    "consider_rebalance_sell": "정기매수 조정으로 해결하기 어려운 경우에만 매도 여부를 검토합니다.",
    "block_action": "FOMO, 단기 급락, 평단 방어성 행동을 보류합니다.",
}

DEFAULT_ACTION_PRIORITIES = {
    "increase_dca": 1,
    "decrease_dca": 2,
    "review_thesis": 3,
    "exceptional_buy_review": 4,
    "consider_rebalance_sell": 5,
    "hold_observe": 6,
    "block_action": 7,
}

EXECUTION_TYPES = {
    "increase_dca": "dca_adjustment",
    "decrease_dca": "dca_adjustment",
    "hold_observe": "observe",
    "review_thesis": "review_required",
    "exceptional_buy_review": "exceptional_buy_review",
    "consider_rebalance_sell": "exceptional_sell_review",
    "block_action": "blocked",
}

DECISION_CONTEXTS = {
    "regular_review",
    "market_correction",
    "sharp_drop_review",
    "rebalance_review",
}

DECISION_SUMMARIES = {
    "increase_dca": "목표 대비 부족하고 위험·효율 조건이 정기매수 보강을 허용합니다.",
    "decrease_dca": "목표 대비 초과 또는 위험 상승으로 신규 매수 축소가 우선입니다.",
    "hold_observe": "수치 조정 기준을 넘지 않아 다음 점검까지 관찰합니다.",
    "review_thesis": "수치만으로 증액하기 어려워 보유 논리 확인이 먼저입니다.",
    "exceptional_buy_review": "정기매수로 부족분을 해소하기 어려운 예외 조건인지 확인해야 합니다.",
    "consider_rebalance_sell": "정기매수 조정으로 낮추기 어려운 초과 위험인지 확인해야 합니다.",
    "block_action": "데이터나 행동 동기가 불충분해 실행 판단을 차단합니다.",
}

REASON_TEXT = {
    "within_hysteresis_or_below_min_trade": "히스테리시스 범위이거나 최소 거래 기준에 미달합니다.",
    "data_quality_low": "데이터 신뢰도가 낮습니다.",
    "unclassified_group": "자산 그룹이 미분류 상태입니다.",
    "risk_ok": "위험기여도가 허용 범위 안에 있습니다.",
    "risk_over": "위험기여도가 기준을 초과했습니다.",
    "efficiency_good": "효율 점수가 기준 이상입니다.",
    "efficiency_low": "효율 점수가 기준보다 낮습니다.",
    "positive_gap": "목표 비중 대비 부족합니다.",
    "negative_gap": "목표 비중 대비 초과 상태입니다.",
    "dca_disabled": "정기매수가 비활성화되어 있습니다.",
    "avoid_immediate_increase": "즉시 증액보다 관찰을 우선합니다.",
    "sell_gate_passed": "예외적 매도 검토 조건을 통과했습니다.",
    "sell_gate_blocked": "매도 게이트 조건을 충족하지 못했습니다.",
    "thesis_broken": "투자 논리가 훼손되었습니다.",
    "thesis_not_broken": "투자 논리 훼손이 확인되지 않았습니다.",
    "prefer_dca_over_sell": "매도보다 정기매수 조정을 우선합니다.",
    "core_priority_context": "현재 판단 모드에서는 코어 보강을 우선합니다.",
    "satellite_downgraded_for_core_priority": "코어가 부족한 하락장에서는 위성 증액 전 보유 가능성을 먼저 점검합니다.",
    "correction_core_reinforcement": "하락장에서는 목표 비중보다 부족한 코어를 정기매수로 우선 보강합니다.",
    "satellite_correction_requires_review": "하락장 위성 추가매수 전에는 투자 논리와 장기 보유 가능성을 먼저 점검합니다.",
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
            decision_summary="하락장 코어 정기매수 증액 후보",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 IPS 목표보다 낮습니다.",
                "최근 효율 점수는 낮지만 하락장 코어 보강 원칙을 우선합니다.",
                "즉시매수가 아니라 다음 정기매수 배분 조정으로 처리합니다.",
            ],
            risk_notes=risk_notes
            or [
                "최근 효율 점수는 낮지만, 하락장 코어 보강 원칙에 따라 정기매수 증액 후보로 분류했습니다."
            ],
            next_step="다음 정기매수에서 부족한 코어 자산의 배분을 늘립니다.",
        )

    if correction_context and core_under_target and group == "core" and action["ips_action"] == "increase_dca":
        return _with_action_metadata(
            action,
            "increase_dca",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "core_priority_context"],
            decision_summary="코어 정기매수 증액 우선",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 IPS 목표보다 낮습니다.",
                "IPS상 하락장 또는 급락 검토에서는 위성보다 코어 보강을 우선합니다.",
            ],
            risk_notes=risk_notes or ["즉시매수가 아니라 다음 정기매수 배분 조정으로 처리합니다."],
            next_step="다음 정기매수에서 코어 자산의 배분을 늘립니다.",
        )

    if correction_context and core_under_target and group == "satellite" and action["ips_action"] == "increase_dca":
        return _with_action_metadata(
            action,
            "review_thesis",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "satellite_downgraded_for_core_priority"],
            decision_summary="하락장 위성 증액 전 점검",
            decision_reasons=[
                f"현재 판단 모드가 {context}입니다.",
                "코어 비중이 IPS 목표보다 낮습니다.",
                "하락장에서는 위성 증액보다 코어 보강과 위성 보유 가능성 점검을 우선합니다.",
            ],
            risk_notes=risk_notes,
            next_step="위성 자산의 투자 논리와 장기 보유 가능성을 확인한 뒤 다음 정기매수 반영 여부를 결정합니다.",
        )

    if correction_context and core_under_target and group == "unclassified" and action["ips_action"] == "review_thesis":
        next_step = f"{action['next_step']} 판단이 어려운 자산보다 코어 정기매수 증액을 우선합니다."
        return _with_action_metadata(
            action,
            "review_thesis",
            ips_config,
            context,
            reason_codes=[*action["reason_codes"], "core_priority_context"],
            decision_summary="미분류 자산 점검",
            decision_reasons=[
                "자산 그룹이 미분류 상태입니다.",
                "코어 비중이 IPS 목표보다 낮아 판단이 어려운 자산보다 코어 보강을 우선합니다.",
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


def _sell_gate_allows(row: pd.Series | dict, allocation_status: dict) -> bool:
    thesis_status = row.get("thesis_status", "unknown")
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    gap = float(row.get("갭%", 0) or 0)

    return (
        thesis_status == "broken"
        or (
            group == "satellite"
            and allocation_status.get("satellite_status") == "over_max"
        )
        or (gap < 0 and abs(gap) >= 1.0 and thesis_status == "broken")
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
    risk_over = bool(row.get("risk_over", False))
    efficiency_good = bool(row.get("efficiency_good", False))
    group = fixed_group(row.get("group", DEFAULT_GROUP))
    dca_enabled = bool(row.get("dca_enabled", True))
    thesis_status = row.get("thesis_status", "unknown")
    should_execute = bool(row.get("수치후보", row.get("실행", False)))
    low_data_quality = bool(row.get("data_quality_low", False))

    if not should_execute:
        reason_codes = ["within_hysteresis_or_below_min_trade"]
        if low_data_quality:
            reason_codes.append("data_quality_low")
        action = _action_result(
            "hold_observe",
            reason_codes,
            ips_config,
            decision_context,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if group == "unclassified":
        next_step = NEXT_STEPS["review_thesis"]
        if allocation_status.get("core_status") in {"under_min", "under_target"}:
            next_step += " 판단이 어려운 자산보다 코어 정기매수 증액을 우선합니다."
        action = _action_result(
            "review_thesis",
            ["unclassified_group"],
            ips_config,
            decision_context,
            next_step=next_step,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if low_data_quality:
        action = _action_result(
            "block_action",
            ["data_quality_low"],
            ips_config,
            decision_context,
            blocked_reason="데이터 신뢰도가 낮아 실행 판단을 보류합니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    data_reasons = ["data_quality_low"] if low_data_quality else []

    if not risk_over and efficiency_good:
        if gap > 0 and dca_enabled:
            action = _action_result(
                "increase_dca",
                ["risk_ok", "efficiency_good", "positive_gap", *data_reasons],
                ips_config,
                decision_context,
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        reason_codes = ["risk_ok", "efficiency_good", *data_reasons]
        if gap > 0:
            reason_codes.extend(["positive_gap", "dca_disabled"])
        action = _action_result("hold_observe", reason_codes, ips_config, decision_context)
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if risk_over and efficiency_good:
        if gap < 0 and dca_enabled:
            action = _action_result(
                "decrease_dca",
                ["risk_over", "efficiency_good", "negative_gap", *data_reasons],
                ips_config,
                decision_context,
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        reason_codes = ["risk_over", "efficiency_good", "avoid_immediate_increase", *data_reasons]
        if gap < 0:
            reason_codes.extend(["negative_gap", "dca_disabled"])
        action = _action_result(
            "hold_observe",
            reason_codes,
            ips_config,
            decision_context,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if not risk_over and not efficiency_good:
        correction_context = decision_context in {"market_correction", "sharp_drop_review"}
        if (
            correction_context
            and group == "core"
            and gap > 0
            and dca_enabled
            and thesis_status != "broken"
        ):
            action = _action_result(
                "increase_dca",
                [
                    "risk_ok",
                    "efficiency_low",
                    "positive_gap",
                    "correction_core_reinforcement",
                    *data_reasons,
                ],
                ips_config,
                decision_context,
                decision_summary="하락장 코어 정기매수 증액 후보",
                risk_notes=[
                    "최근 효율 점수는 낮지만, 하락장 코어 보강 원칙에 따라 정기매수 증액 후보로 분류했습니다."
                ],
                next_step="다음 정기매수에서 부족한 코어 자산의 배분을 늘립니다.",
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )

        if correction_context and group == "satellite" and gap > 0:
            action = _action_result(
                "review_thesis",
                [
                    "risk_ok",
                    "efficiency_low",
                    "positive_gap",
                    "satellite_correction_requires_review",
                    *data_reasons,
                ],
                ips_config,
                decision_context,
                decision_summary="하락장 위성 추가매수 전 점검",
                next_step="위성 자산의 투자 논리와 장기 보유 가능성을 확인한 뒤 다음 정기매수 반영 여부를 결정합니다.",
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )

        action = _action_result(
            "review_thesis",
            ["risk_ok", "efficiency_low", *data_reasons],
            ips_config,
            decision_context,
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    if risk_over and not efficiency_good:
        prefer_dca_over_sell = bool(
            ips_config.get("rules", {}).get("prefer_dca_over_sell", True)
        )
        if (
            prefer_dca_over_sell
            and thesis_status != "broken"
            and gap < 0
            and dca_enabled
        ):
            action = _action_result(
                "decrease_dca",
                [
                    "risk_over",
                    "efficiency_low",
                    "negative_gap",
                    "thesis_not_broken",
                    "prefer_dca_over_sell",
                    *data_reasons,
                ],
                ips_config,
                decision_context,
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        if _sell_gate_allows(row, allocation_status):
            reason_codes = ["risk_over", "efficiency_low", "sell_gate_passed", *data_reasons]
            if thesis_status == "broken":
                reason_codes.append("thesis_broken")
            else:
                reason_codes.append("thesis_not_broken")
            action = _action_result(
                "consider_rebalance_sell",
                reason_codes,
                ips_config,
                decision_context,
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        if gap < 0 and dca_enabled:
            action = _action_result(
                "decrease_dca",
                [
                    "risk_over",
                    "efficiency_low",
                    "negative_gap",
                    "thesis_not_broken",
                    "sell_gate_blocked",
                    *data_reasons,
                ],
                ips_config,
                decision_context,
            )
            return apply_contextual_ips_overlay(
                action, row, allocation_status, decision_context, ips_config
            )
        action = _action_result(
            "review_thesis",
            ["risk_over", "efficiency_low", "thesis_not_broken", "sell_gate_blocked", *data_reasons],
            ips_config,
            decision_context,
            blocked_reason="매도 게이트 조건을 충족하지 못했습니다.",
        )
        return apply_contextual_ips_overlay(
            action, row, allocation_status, decision_context, ips_config
        )

    action = _action_result("review_thesis", ["unclassified"], ips_config, decision_context)
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
    ]
    meta = metrics_df[metrics_cols].copy()

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
