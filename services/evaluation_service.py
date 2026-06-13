"""평가 및 제안 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

from typing import NamedTuple

import pandas as pd

from utils.metrics import compute_rc_target, risk_contributions
from utils.optimization import calculate_orders_with_constraints
from utils.ips_config import load_ips_config
from utils.ips import (
    DEFAULT_GROUP,
    classify_ips_actions,
    compute_group_summary,
    compute_ips_allocation_status,
    fixed_group,
    group_role,
)


IPS_FIT_SCORING_DEFAULTS = {
    "enabled": True,
    "bands": {"high": 70.0, "medium": 50.0},
    "weights": {
        "role": 20.0,
        "allocation": 25.0,
        "thesis": 20.0,
        "risk": 15.0,
        "action": 10.0,
        "efficiency": 5.0,
        "data_quality": 5.0,
    },
    "thesis_score": {
        "intact": 1.0,
        "watch": 0.65,
        "unknown": 0.5,
        "broken": 0.0,
    },
    "role_score": {
        "core": 1.0,
        "satellite_ai_infra": 0.85,
        "satellite_ai_software": 0.85,
        "satellite_nextgen": 0.85,
    },
}

FINAL_EXECUTABLE_ACTIONS = {
    "increase_dca",
}


class EvaluationResult(NamedTuple):
    """평가 결과 데이터 구조."""

    proposal_df: pd.DataFrame
    ips_action_df: pd.DataFrame
    group_summary_df: pd.DataFrame
    sell_list: pd.DataFrame
    buy_list: pd.DataFrame
    fine_tune_list: pd.DataFrame
    rc_violations: pd.DataFrame
    ips_config_snapshot: dict | None = None
    playbook: dict | None = None


class EvaluationError(Exception):
    """평가 처리 중 발생하는 오류."""

    pass


def _ips_fit_scoring_config(ips_config: dict) -> dict:
    cfg = IPS_FIT_SCORING_DEFAULTS.copy()
    cfg["bands"] = IPS_FIT_SCORING_DEFAULTS["bands"].copy()
    cfg["weights"] = IPS_FIT_SCORING_DEFAULTS["weights"].copy()
    cfg["thesis_score"] = IPS_FIT_SCORING_DEFAULTS["thesis_score"].copy()
    cfg["role_score"] = IPS_FIT_SCORING_DEFAULTS["role_score"].copy()

    user_cfg = ips_config.get("ips_fit_scoring", {})
    cfg.update(
        {
            key: value
            for key, value in user_cfg.items()
            if key not in {"bands", "weights", "thesis_score", "role_score"}
        }
    )
    cfg["bands"].update(user_cfg.get("bands", {}))
    cfg["weights"].update(user_cfg.get("weights", {}))
    cfg["thesis_score"].update(user_cfg.get("thesis_score", {}))
    cfg["role_score"].update(user_cfg.get("role_score", {}))
    return cfg


def compute_ips_fit_breakdown(
    proposal_df: pd.DataFrame,
    ips_config: dict,
    e_thresh: float,
) -> pd.DataFrame:
    """IPS 정성 의도를 0~100 적합도와 구성 점수로 변환합니다."""
    cfg = _ips_fit_scoring_config(ips_config)
    weights = {
        key: max(0.0, float(value)) for key, value in cfg.get("weights", {}).items()
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    df = proposal_df.copy()
    index = df.index
    group = (
        df.get("group", pd.Series(DEFAULT_GROUP, index=index))
        .fillna(DEFAULT_GROUP)
        .map(fixed_group)
    )
    role = group.map(group_role)
    gap_pct = pd.to_numeric(df.get("갭%", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    risk_over = df.get("risk_over", pd.Series(False, index=index)).fillna(False).astype(bool)
    rc_over_pct = pd.to_numeric(
        df.get("RC_Over%", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    dca_enabled = df.get("dca_enabled", pd.Series(True, index=index)).fillna(True).astype(bool)
    thesis_status = (
        df.get("thesis_status", pd.Series("unknown", index=index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )
    numeric_candidate = (
        df.get("수치후보", pd.Series(False, index=index)).fillna(False).astype(bool)
    )
    efficiency = pd.to_numeric(df.get("E", pd.Series(0.5, index=index)), errors="coerce").fillna(0.5)
    missing_ratio = pd.to_numeric(
        df.get("missing_ratio", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    observations = pd.to_numeric(
        df.get("observation_count", pd.Series(9999, index=index)), errors="coerce"
    ).fillna(9999)
    data_quality_low = (
        df.get("data_quality_low", pd.Series(False, index=index)).fillna(False).astype(bool)
    )

    role_score = group.map(
        lambda value: float(
            cfg["role_score"].get(
                value,
                cfg["role_score"].get(group_role(value), cfg["role_score"].get("core", 1.0)),
            )
        )
    )

    allocation_score = pd.Series(0.65, index=index, dtype=float)
    allocation_score.loc[(role == "core") & (gap_pct > 0)] = 1.0
    allocation_score.loc[(role == "satellite") & (gap_pct > 0)] = 0.6
    allocation_score.loc[gap_pct < 0] = 0.75
    allocation_score.loc[(gap_pct < 0) & risk_over] = 0.9
    allocation_score.loc[gap_pct.abs() < 1.0] = 0.7

    thesis_score = thesis_status.map(
        lambda value: float(cfg["thesis_score"].get(value, cfg["thesis_score"].get("unknown", 0.5)))
    )

    risk_score = (1.0 - (rc_over_pct / 10.0)).clip(lower=0.2, upper=1.0)
    risk_score.loc[~risk_over] = 1.0

    action_score = pd.Series(0.6, index=index, dtype=float)
    action_score.loc[numeric_candidate & dca_enabled] = 1.0
    action_score.loc[numeric_candidate & (~dca_enabled)] = 0.25
    action_score.loc[(gap_pct > 0) & (role == "satellite")] *= 0.75

    efficiency_score = efficiency.clip(lower=0.0, upper=1.0)
    data_quality_score = ((1.0 - missing_ratio).clip(lower=0.0, upper=1.0)) * (
        (observations / 60.0).clip(lower=0.0, upper=1.0)
    )
    data_quality_score.loc[data_quality_low] = data_quality_score.loc[data_quality_low].clip(upper=0.35)

    components = {
        "IPS점수_역할": role_score.clip(lower=0.0, upper=1.0),
        "IPS점수_비중": allocation_score.clip(lower=0.0, upper=1.0),
        "IPS점수_논리": thesis_score.clip(lower=0.0, upper=1.0),
        "IPS점수_위험": risk_score.clip(lower=0.0, upper=1.0),
        "IPS점수_실행": action_score.clip(lower=0.0, upper=1.0),
        "IPS점수_E": efficiency_score.clip(lower=0.0, upper=1.0),
        "IPS점수_데이터": data_quality_score.clip(lower=0.0, upper=1.0),
    }
    result = pd.DataFrame(components, index=index)
    weighted = (
        result["IPS점수_역할"] * weights.get("role", 0.0)
        + result["IPS점수_비중"] * weights.get("allocation", 0.0)
        + result["IPS점수_논리"] * weights.get("thesis", 0.0)
        + result["IPS점수_위험"] * weights.get("risk", 0.0)
        + result["IPS점수_실행"] * weights.get("action", 0.0)
        + result["IPS점수_E"] * weights.get("efficiency", 0.0)
        + result["IPS점수_데이터"] * weights.get("data_quality", 0.0)
    )
    result["IPS적합도"] = ((weighted / total_weight) * 100.0).clip(lower=0.0, upper=100.0)

    high = float(cfg.get("bands", {}).get("high", 70.0))
    medium = float(cfg.get("bands", {}).get("medium", 50.0))
    result["IPS등급"] = "low"
    result.loc[result["IPS적합도"] >= medium, "IPS등급"] = "medium"
    result.loc[result["IPS적합도"] >= high, "IPS등급"] = "high"
    result["efficiency_warning"] = efficiency < float(e_thresh)

    score_columns = [column for column in result.columns if column.startswith("IPS점수_")]
    result[score_columns] = result[score_columns].round(3)
    result["IPS적합도"] = result["IPS적합도"].round(1)
    return result


PLAYBOOK_LABELS = {
    "regular_review": "일반 점검",
    "market_correction": "시장 조정 대응",
    "sharp_drop_review": "급락 후 추매 검토",
    "rebalance_review": "비중 리밸런싱 점검",
}

PLAYBOOK_STEPS = {
    "regular_review": [
        "데이터 품질과 분류가 충분한지 먼저 확인합니다.",
        "코어/위성 목표 비중에서 벗어난 자산을 점검합니다.",
        "정기매수 증액, 감액, 보류 중 정책에 맞는 다음 행동만 남깁니다.",
    ],
    "market_correction": [
        "코어 비중이 IPS 목표보다 낮은지 확인합니다.",
        "코어 자산은 투자 논리가 유지되는 범위에서 정기매수 배분을 늘립니다.",
        "위성 자산은 가격 하락보다 thesis와 장기 보유 가능성을 먼저 점검합니다.",
        "즉시매수는 정기매수 조정으로 부족할 때만 예외적으로 검토합니다.",
    ],
    "sharp_drop_review": [
        "급락 자체를 단독 매수 사유로 보지 않습니다.",
        "위성 자산은 thesis 훼손, 변동성, 관리 부담을 먼저 확인합니다.",
        "코어 부족분은 즉시매수보다 정기매수 조정으로 대응 가능한지 점검합니다.",
        "판단 근거가 약하면 보류하거나 다음 리뷰로 넘깁니다.",
    ],
    "rebalance_review": [
        "초과 비중과 위험기여도 초과 자산을 먼저 확인합니다.",
        "매도보다 신규 정기매수 감액 또는 중단으로 해결 가능한지 점검합니다.",
        "매도 검토는 thesis 훼손, 과도한 집중, 포트폴리오 단순화 필요가 있을 때만 남깁니다.",
    ],
}


def _playbook_result(
    code: str,
    confidence: str,
    reasons: list[str],
    manual_context: str,
) -> dict:
    unique_reasons = list(dict.fromkeys(reasons))[:4]
    return {
        "code": code,
        "label": PLAYBOOK_LABELS[code],
        "confidence": confidence,
        "reasons": unique_reasons,
        "steps": PLAYBOOK_STEPS[code][:5],
        "manual_context": manual_context,
        "is_manual_override": code != manual_context,
    }


def recommend_playbook(
    proposal_df: pd.DataFrame,
    ips_action_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    allocation_status: dict,
    decision_context: str = "regular_review",
) -> dict:
    """현재 평가 결과를 사람이 읽기 쉬운 IPS 플레이북으로 분류합니다."""
    context = decision_context if decision_context in PLAYBOOK_LABELS else "regular_review"
    proposal = proposal_df.copy()
    actions = ips_action_df.copy()
    groups = group_summary_df.copy()

    data_quality_low = bool(
        actions.get("data_quality_low", pd.Series(False, index=actions.index))
        .fillna(False)
        .astype(bool)
        .any()
    )
    unknown_count = int(
        (actions.get("thesis_status", pd.Series("", index=actions.index)).fillna("") == "unknown").sum()
    )
    if data_quality_low or unknown_count >= max(1, len(actions) // 2):
        reasons = []
        if data_quality_low:
            reasons.append("일부 자산의 데이터 품질이 낮아 강한 행동 판단보다 기본 점검이 먼저입니다.")
        if unknown_count > 0:
            reasons.append("투자 논리가 미정인 자산이 있어 thesis 확인이 우선입니다.")
        return _playbook_result("regular_review", "high", reasons, context)

    if context == "sharp_drop_review":
        return _playbook_result(
            "sharp_drop_review",
            "high",
            [
                "사용자가 급락 후 추매 검토 모드를 선택했습니다.",
                "단기 급락은 단독 매수 사유가 아니므로 논리와 정기매수 가능성을 먼저 확인합니다.",
            ],
            context,
        )

    proposal_roles = proposal.get("group", pd.Series("core", index=proposal.index)).fillna("core").map(group_role)
    group_roles = groups.get("group", pd.Series("core", index=groups.index)).fillna("core").map(group_role)
    satellite = proposal_roles == "satellite"
    return_total_pct = pd.to_numeric(
        proposal.get("return_total%", pd.Series(0.0, index=proposal.index)),
        errors="coerce",
    ).fillna(0.0)
    if bool((satellite & (return_total_pct <= -15.0)).any()):
        return _playbook_result(
            "sharp_drop_review",
            "medium",
            [
                "위성 자산 중 손실 폭이 큰 후보가 있어 급락 대응 프레임이 필요합니다.",
                "위성 자산은 추가 매수보다 thesis와 장기 보유 가능성 확인이 먼저입니다.",
            ],
            context,
        )

    core_under = allocation_status.get("core_status") in {"under_min", "under_target"}
    satellite_buy_candidate = bool(
        (
            satellite
            & (pd.to_numeric(proposal.get("갭%", pd.Series(0.0, index=proposal.index)), errors="coerce").fillna(0.0) > 0)
            & proposal.get("수치후보", pd.Series(False, index=proposal.index)).fillna(False).astype(bool)
        ).any()
    )
    correction_signal = bool((return_total_pct < 0).any()) or context == "market_correction"
    if core_under and (satellite_buy_candidate or correction_signal):
        reasons = ["코어 비중이 IPS 목표보다 낮아 기본 시장 노출 보강이 우선입니다."]
        if satellite_buy_candidate:
            reasons.append("위성 증액 후보가 있어 코어 우선 원칙으로 한 번 걸러야 합니다.")
        if correction_signal:
            reasons.append("하락 또는 조정 신호가 있어 정기매수 조정 중심으로 대응합니다.")
        return _playbook_result("market_correction", "high", reasons, context)

    action_codes = actions.get("ips_action", pd.Series("", index=actions.index)).fillna("")
    rebalance_action_count = int(action_codes.isin(["rebalance_sell_review", "reduce_or_pause_dca"]).sum())
    risk_over_count = int(
        actions.get("risk_over", pd.Series(False, index=actions.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    negative_gap_count = int(
        (pd.to_numeric(proposal.get("갭%", pd.Series(0.0, index=proposal.index)), errors="coerce").fillna(0.0) < 0).sum()
    )
    satellite_over = (
        allocation_status.get("satellite_status") == "over_max"
        or bool(
            (
                (group_roles == "satellite")
                & (pd.to_numeric(groups.get("weight", pd.Series(0.0, index=groups.index)), errors="coerce").fillna(0.0) > 0.30)
            ).any()
        )
    )
    if satellite_over or rebalance_action_count >= 2 or risk_over_count > 0 or negative_gap_count >= 2:
        reasons = []
        if satellite_over:
            reasons.append("위성 비중이 IPS 상한을 넘거나 상한에 가까워졌습니다.")
        if rebalance_action_count > 0:
            reasons.append("정기매수 감액 또는 리밸런싱 검토 대상이 있습니다.")
        if risk_over_count > 0:
            reasons.append("위험기여도 초과 자산이 있어 위험 점검이 필요합니다.")
        if negative_gap_count > 0:
            reasons.append("목표보다 높은 비중의 자산이 여럿 있습니다.")
        return _playbook_result("rebalance_review", "medium", reasons, context)

    return _playbook_result(
        "regular_review",
        "medium",
        ["강한 조정, 급락, 리밸런싱 신호가 두드러지지 않아 일반 점검으로 충분합니다."],
        context,
    )


def build_ips_target_weights(metrics_df: pd.DataFrame, ips_config: dict) -> pd.Series:
    """고정 그룹 분류와 IPS 목표로 자산별 목표 비중을 자동 생성합니다."""
    current = metrics_df["가중치"].fillna(0).astype(float)
    group_series = metrics_df.get("group", pd.Series(DEFAULT_GROUP, index=metrics_df.index))
    group_series = group_series.fillna(DEFAULT_GROUP).map(fixed_group)
    target = current.copy()

    locked_mask = pd.Series(False, index=group_series.index)
    locked_weight = float(current[locked_mask].sum())
    remaining_weight = max(0.0, 1.0 - locked_weight)
    target_cfg = ips_config.get("target_allocation", {})
    adjustable_group_values = [
        group
        for group in target_cfg
        if (group_series == group).any()
    ]

    if not adjustable_group_values:
        return target / target.sum() if target.sum() > 0 else target

    desired = {
        group: float(target_cfg.get(group, {}).get("target", 0.0))
        for group in adjustable_group_values
    }
    desired_total = sum(desired.values())
    if desired_total <= 0:
        desired = {group: 1.0 for group in adjustable_group_values}
        desired_total = float(len(adjustable_group_values))

    for group in adjustable_group_values:
        group_mask = group_series == group
        group_target = remaining_weight * desired[group] / desired_total
        current_in_group = current[group_mask]
        current_group_total = float(current_in_group.sum())
        if current_group_total > 0:
            current_share = current_in_group / current_group_total
        else:
            current_share = pd.Series(
                1.0 / int(group_mask.sum()),
                index=current_in_group.index,
                dtype=float,
            )

        target[group_mask] = current_share * group_target

    if target.sum() > 0:
        target = target / target.sum()
    return target


def _action_reason(row: pd.Series | dict) -> str:
    """제안 행의 실행/보류 이유를 사람이 읽는 짧은 문구로 변환합니다."""
    within_hysteresis = bool(row.get("히스테리시스제외", False))
    below_min_trade = bool(row.get("최소거래미만", False))
    should_execute = bool(row.get("실행", False))

    if not should_execute:
        if bool(row.get("data_quality_low", False)):
            return "데이터 신뢰도 낮음"
        if within_hysteresis and below_min_trade:
            return "히스테리시스 범위 및 최소 거래 미만"
        if within_hysteresis:
            return "히스테리시스 범위"
        if below_min_trade:
            return "최소 거래 미만"
        return "보류"

    risk_over = bool(row.get("risk_over", False))
    efficiency_warning = bool(row.get("efficiency_warning", False))
    gap_pct = float(row.get("갭%", 0) or 0)
    return_total = row.get("return_total%", row.get("return_total"))
    signals: list[str] = []

    if gap_pct > 0:
        signals.append("비중 목표 미달")
    elif gap_pct < 0:
        signals.append("비중 목표 초과")
    if risk_over:
        signals.append("위험기여도 초과")
    if efficiency_warning:
        signals.append("효율 점수 미달")
    if return_total is not None and not pd.isna(return_total):
        return_pct = float(return_total)
        if "return_total%" not in row:
            return_pct *= 100
        if return_pct <= -5:
            signals.append("수익률 부진")
        elif return_pct >= 15:
            signals.append("수익률 양호")
    if bool(row.get("data_quality_low", False)):
        signals.append("데이터 신뢰도 낮음")

    ips_band = str(row.get("IPS등급", "") or "")
    policy_signal = {
        "high": "정책 적합",
        "medium": "정책 조건부",
        "low": "정책 부적합",
    }.get(ips_band, "실행 후보")
    signals.append(policy_signal)
    return " · ".join(signals)


def _apply_ips_execution_gate(
    proposal: pd.DataFrame,
    ips_action_df: pd.DataFrame,
) -> pd.DataFrame:
    """IPS 액션을 최종 실행 게이트로 적용합니다."""
    gated = proposal.copy()
    action_meta = ips_action_df.set_index("ticker")

    for idx, row in gated.iterrows():
        ticker = row["ticker"]
        action = action_meta.loc[ticker] if ticker in action_meta.index else None
        ips_action = action.get("ips_action") if action is not None else "hold_observe"
        gap_pct = float(row.get("갭%", 0) or 0)
        numeric_candidate = bool(row.get("수치후보", row.get("실행", False)))

        sign_allowed = ips_action == "increase_dca" and gap_pct > 0
        final_execute = numeric_candidate and ips_action in FINAL_EXECUTABLE_ACTIONS and sign_allowed

        if final_execute:
            gated.at[idx, "실행"] = True
            gated.at[idx, "제안조정%"] = round(gap_pct, 2)
            reason_codes = action.get("reason_codes", []) if action is not None else []
            if "correction_core_reinforcement" in reason_codes:
                gated.at[idx, "판단사유"] = action.get(
                    "decision_summary",
                    "하락장에는 목표보다 낮은 코어 비중을 정기매수로 보강합니다.",
                )
            elif "core_priority_context" in reason_codes:
                gated.at[idx, "판단사유"] = action.get(
                    "decision_summary",
                    "목표보다 낮은 코어 비중을 정기매수로 보강합니다.",
                )
            else:
                gated.at[idx, "판단사유"] = _action_reason(gated.loc[idx])
        else:
            gated.at[idx, "실행"] = False
            gated.at[idx, "제안조정%"] = 0.0
            if numeric_candidate and action is not None:
                gated.at[idx, "판단사유"] = action.get("action_label", "보류")
            else:
                gated.at[idx, "판단사유"] = _action_reason(gated.loc[idx])

    return gated


def _post_trade_rc(
    mdf: pd.DataFrame,
    proposal: pd.DataFrame,
    cov_matrix: pd.DataFrame | None,
) -> pd.Series:
    """최종 제안조정 반영 후 예상 RC를 계산합니다."""
    if cov_matrix is None:
        return mdf["위험기여도"].copy()

    current = mdf["가중치"].astype(float).copy()
    orders = proposal.set_index("ticker")["제안조정%"].reindex(current.index).fillna(0) / 100.0
    expected = (current + orders).clip(lower=0)
    if expected.sum() > 0:
        expected = expected / expected.sum()

    common = cov_matrix.index.intersection(expected.index)
    if len(common) == 0:
        return mdf["위험기여도"].copy()
    rc = risk_contributions(expected.reindex(common), cov_matrix.loc[common, common])
    return rc.reindex(mdf.index).fillna(mdf["위험기여도"])


def run_evaluation(
    metrics_df: pd.DataFrame,
    target_weights: pd.Series | dict[str, float] | None,
    rc_over_thresh_pct: float,
    e_thresh: float,
    cov_matrix: pd.DataFrame | None = None,
    decision_context: str = "regular_review",
) -> EvaluationResult:
    """평가 & 실행 계획 제안을 실행합니다.

    Args:
        metrics_df: 분석 결과 메트릭 데이터프레임
        target_weights: 목표 가중치 (Series 또는 dict, None이면 현재 가중치 사용)
        rc_over_thresh_pct: RC_Over 임계값 (%)
        e_thresh: 효율 점수 E 임계값
        cov_matrix: 연율화된 공분산 행렬 (None이면 단순 비중 기반 RC_Target 사용)
        decision_context: IPS 판단 모드

    Returns:
        EvaluationResult: 평가 결과

    Raises:
        EvaluationError: 평가 실패 시
    """
    mdf = metrics_df.copy()
    ips_config_snapshot = load_ips_config()
    ips_config_snapshot["decision_context"] = decision_context
    mdf["group"] = (
        mdf.get("group", pd.Series(DEFAULT_GROUP, index=mdf.index))
        .fillna(DEFAULT_GROUP)
        .map(fixed_group)
    )
    mdf["dca_enabled"] = (
        mdf.get("dca_enabled", pd.Series(True, index=mdf.index))
        .fillna(True)
        .astype(bool)
    )
    mdf["thesis_status"] = mdf.get("thesis_status", pd.Series(index=mdf.index)).fillna(
        "unknown"
    )
    mdf["return_total"] = mdf.get("return_total", pd.Series(index=mdf.index))
    mdf["missing_ratio"] = mdf.get("missing_ratio", pd.Series(0, index=mdf.index)).fillna(0)
    mdf["observation_count"] = (
        mdf.get("observation_count", pd.Series(9999, index=mdf.index))
        .fillna(9999)
    )
    mdf["data_quality_low"] = (
        mdf["missing_ratio"].astype(float) > 0.2
    ) | (mdf["observation_count"].astype(float) < 60)

    # 목표 가중치 시리즈 구축
    if target_weights is None:
        tgt = build_ips_target_weights(mdf, ips_config_snapshot)
    elif isinstance(target_weights, dict):
        tgt = pd.Series(target_weights, index=mdf.index).fillna(0)
    else:
        tgt = target_weights.reindex(mdf.index).fillna(0)

    # RC 타깃 계산: 공분산 행렬 기반 기하학적 계산 또는 단순 비중 기반
    # AIDEV-NOTE: geometric-rc-target; 공분산 행렬을 고려한 기하학적 RC_Target 계산
    if cov_matrix is not None:
        # 공분산 행렬이 제공되면 기하학적 계산
        try:
            rc_target = compute_rc_target(tgt.fillna(0), cov_matrix)
            rc_target = rc_target.reindex(mdf.index).fillna(0)
        except Exception:
            # 계산 실패 시 폴백: 단순 비중 기반
            rc_target = tgt.fillna(0)
    else:
        # 공분산 행렬이 없으면 단순 비중 기반 RC 목표를 사용합니다.
        rc_target = tgt.fillna(0)
    mdf["RC_Target"] = rc_target
    rc_gap = mdf["위험기여도"] - mdf["RC_Target"]
    mdf["RC_Over"] = rc_gap.clip(lower=0)

    # E는 위험 대비 수익 효율 신호로 노출하고, 실행 방식은 IPS 게이트로 제한합니다.
    mdf["효율E"] = mdf["E"]

    rc_over_pct = mdf["RC_Over"] * 100  # 백분율로 표시
    mdf["risk_over"] = rc_over_pct > rc_over_thresh_pct
    mdf["efficiency_warning"] = mdf["효율E"] < e_thresh

    # 갭 분석
    current_w = mdf["가중치"]
    gap = tgt - current_w

    proposal = pd.DataFrame(
        {
            "ticker": mdf.index,
            "현재%": (current_w * 100).round(2).values,
            "목표%": (tgt * 100).round(2).values,
            "갭%": (gap * 100).round(2).values,
            "E": mdf["E"].round(2).values,
            "RC_Gap%": (rc_gap * 100).round(2).values,
            "RC_Over%": rc_over_pct.round(2).values,
            "RC_Target%": (rc_target * 100).round(2).values,
            "return_total%": (mdf["return_total"] * 100).round(2).values,
            "group": mdf["group"].values,
            "dca_enabled": mdf["dca_enabled"].values,
            "thesis_status": mdf["thesis_status"].values,
            "missing_ratio": mdf["missing_ratio"].values,
            "observation_count": mdf["observation_count"].values,
            "data_quality_low": mdf["data_quality_low"].values,
            "risk_over": mdf["risk_over"].values,
            "efficiency_warning": mdf["efficiency_warning"].values,
        }
    )

    # AIDEV-NOTE: trade-filtering-rules; 히스테리시스(affinity-based), 최소거래(1.0%p) 적용하여 과잉거래 방지

    # 히스테리시스 밴드: Affinity 기반 (상수항 + 비례항)
    # AIDEV-NOTE: affinity-hysteresis-band; 작은 자산도 최소한의 허용 범위 확보
    hysteresis_constant = 0.005  # 0.5%p 상수항
    hysteresis_factor = 0.15  # 15% 비례항
    max_gap_pct = hysteresis_constant + (tgt * hysteresis_factor).clip(
        lower=hysteresis_constant
    )
    within_band = gap.abs() <= max_gap_pct

    # 최소 거래 단위: 1.0%p 이상만 처리
    min_trade_pct = 1.0 / 100.0
    above_min_trade = gap.abs() >= min_trade_pct

    # 거래 대상 필터링
    should_trade = above_min_trade & (~within_band)

    proposal["히스테리시스제외"] = within_band.values
    proposal["최소거래미만"] = (~above_min_trade).values
    proposal["수치후보"] = should_trade.values
    proposal["실행"] = should_trade.values
    proposal["제안조정%"] = 0.0
    proposal["참고조정%"] = 0.0
    ips_fit = compute_ips_fit_breakdown(proposal, ips_config_snapshot, e_thresh)
    proposal = proposal.drop(columns=["efficiency_warning"], errors="ignore")
    proposal = pd.concat([proposal, ips_fit], axis=1)
    proposal["판단사유"] = proposal.apply(_action_reason, axis=1)

    # 실행 규칙: 우선순위 정의
    sell_list = proposal[(proposal["갭%"] < 0) & proposal["실행"]].copy()
    sell_list = sell_list.sort_values(["IPS적합도", "RC_Over%"], ascending=[True, False])

    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["IPS적합도", "갭%"], ascending=[False, False])

    # AIDEV-NOTE: constrained-scaling; 현금 중립성과 RC 상한을 동시에 만족하는 반복적 스케일링
    # 제약 조건을 고려한 주문 조정
    if cov_matrix is not None and len(buy_list) > 0 and len(sell_list) > 0:
        # 현재 가중치와 목표 가중치 준비
        current_w = mdf["가중치"]

        # 초기 주문 (갭 기반, 실행 대상 자산만)
        initial_orders = pd.Series(0.0, index=mdf.index)
        for _, row in buy_list.iterrows():
            ticker = row["ticker"]
            initial_orders[ticker] = row["갭%"] / 100.0  # 양수 (매수)
        for _, row in sell_list.iterrows():
            ticker = row["ticker"]
            initial_orders[ticker] = row["갭%"] / 100.0  # 음수 (매도)

        # RC 상한선 계산
        rc_cap_single = 0.12  # 단일 자산 최대 12%
        rc_cap_target_ratio = 1.5  # RC_Target의 1.5배
        rc_cap = pd.Series(index=mdf.index, dtype=float)
        for ticker in mdf.index:
            rc_cap[ticker] = min(rc_cap_single, rc_target[ticker] * rc_cap_target_ratio)

        # 현재 RC
        current_rc = mdf["위험기여도"]

        # 제약 조건을 만족하도록 주문 조정
        adjusted_orders, conv_info = calculate_orders_with_constraints(
            current_w,
            current_w + initial_orders,  # 목표 가중치 = 현재 + 주문
            current_rc,
            rc_cap,
            cov_matrix,
            max_iterations=10,
            tolerance=0.001,
        )

        # 조정된 주문을 buy_list와 sell_list에 반영
        buy_list = buy_list.copy()
        sell_list = sell_list.copy()
        buy_list["제안조정%"] = buy_list["갭%"].round(2)
        sell_list["제안조정%"] = sell_list["갭%"].round(2)

        for idx in buy_list.index:
            ticker = buy_list.loc[idx, "ticker"]
            if ticker in adjusted_orders.index:
                adjusted_gap = adjusted_orders[ticker] * 100.0
                buy_list.at[idx, "제안조정%"] = round(adjusted_gap, 2)

        for idx in sell_list.index:
            ticker = sell_list.loc[idx, "ticker"]
            if ticker in adjusted_orders.index:
                adjusted_gap = adjusted_orders[ticker] * 100.0
                sell_list.at[idx, "제안조정%"] = round(adjusted_gap, 2)
    else:
        # 공분산 행렬이 없거나 실행 대상이 없으면 갭 기준으로 단순 스케일링합니다.
        total_sell = sell_list["갭%"].abs().sum() if len(sell_list) > 0 else 0
        total_buy_before_scale = buy_list["갭%"].sum() if len(buy_list) > 0 else 0

        if total_buy_before_scale > 0 and total_sell > 0:
            scale_factor = (
                min(1.0, total_sell / total_buy_before_scale)
                if total_buy_before_scale > 0
                else 1.0
            )
            buy_list["제안조정%"] = (buy_list["갭%"] * scale_factor).round(2)
        else:
            buy_list["제안조정%"] = buy_list["갭%"].round(2)

        if len(sell_list) > 0:
            sell_list["제안조정%"] = sell_list["갭%"].round(2)

    for trade_list in (buy_list, sell_list):
        for idx, row in trade_list.iterrows():
            proposal.at[idx, "참고조정%"] = row["제안조정%"]
            proposal.at[idx, "제안조정%"] = row["제안조정%"]

    group_summary_df = compute_group_summary(mdf, ips_config_snapshot)
    allocation_status = compute_ips_allocation_status(group_summary_df, ips_config_snapshot)
    ips_action_df = classify_ips_actions(
        proposal_df=proposal,
        metrics_df=mdf,
        group_summary_df=group_summary_df,
        allocation_status=allocation_status,
        ips_config=ips_config_snapshot,
        decision_context=decision_context,
    )
    proposal = _apply_ips_execution_gate(proposal, ips_action_df)
    ips_action_df = classify_ips_actions(
        proposal_df=proposal,
        metrics_df=mdf,
        group_summary_df=group_summary_df,
        allocation_status=allocation_status,
        ips_config=ips_config_snapshot,
        decision_context=decision_context,
    )

    sell_list = proposal[(proposal["갭%"] < 0) & proposal["실행"]].copy()
    sell_list = sell_list.sort_values(["IPS적합도", "RC_Over%"], ascending=[True, False])
    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["IPS적합도", "갭%"], ascending=[False, False])
    fine_tune = proposal[
        (proposal["실행"]) & (proposal["갭%"].abs() <= 1.0)
    ].copy()

    post_trade_rc = _post_trade_rc(mdf, proposal, cov_matrix)
    rc_cap_single = 0.12  # 단일 자산 최대 12%
    rc_cap_target_ratio = 1.5  # RC_Target의 1.5배
    violations = []
    for ticker in mdf.index:
        rc_cap = min(rc_cap_single, rc_target[ticker] * rc_cap_target_ratio)
        if post_trade_rc[ticker] > rc_cap:
            violations.append(
                {
                    "ticker": ticker,
                    "현재RC%": (post_trade_rc[ticker] * 100),
                    "RC상한%": (rc_cap * 100),
                    "상태": "⚠️ 경고: RC 상한 초과 위험",
                }
            )
    rc_violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()
    playbook = recommend_playbook(
        proposal_df=proposal,
        ips_action_df=ips_action_df,
        group_summary_df=group_summary_df,
        allocation_status=allocation_status,
        decision_context=decision_context,
    )

    return EvaluationResult(
        proposal_df=proposal,
        ips_action_df=ips_action_df,
        group_summary_df=group_summary_df,
        sell_list=sell_list,
        buy_list=buy_list,
        fine_tune_list=fine_tune,
        rc_violations=rc_violations_df,
        ips_config_snapshot=ips_config_snapshot,
        playbook=playbook,
    )
