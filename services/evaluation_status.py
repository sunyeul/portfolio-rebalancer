"""Status classification for Evaluation Framework v2."""

from __future__ import annotations

from core.evaluation import EvaluationOutput, EvaluationStatus, EvaluationUnit


def classify_evaluation_status(
    unit: EvaluationUnit,
    output: EvaluationOutput,
) -> tuple[EvaluationStatus, list[str]]:
    """Classify v2 output into OK/Watch/Review/Action and reason codes."""
    hard: list[str] = []
    soft: list[str] = []

    if output.period_return is None and output.cagr is None:
        hard.append("insufficient_performance_data")

    if unit.max_weight is not None and output.current_weight > unit.max_weight:
        hard.append("max_weight_exceeded")

    if (
        unit.allowed_mdd is not None
        and output.mdd is not None
        and output.mdd < unit.allowed_mdd
    ):
        hard.append("mdd_exceeded")

    if (
        unit.allowed_volatility is not None
        and output.volatility is not None
        and output.volatility > unit.allowed_volatility
    ):
        hard.append("volatility_exceeded")

    if output.risk_contribution is not None and output.risk_contribution > 0.20:
        hard.append("risk_contribution_high")

    if output.thesis_status == "broken":
        hard.append("thesis_broken")
    elif output.thesis_status == "watch":
        soft.append("thesis_watch")
    elif output.thesis_status == "unknown":
        soft.append("thesis_unknown")

    if output.weight_gap is not None and abs(output.weight_gap) >= 0.01:
        soft.append("target_gap_outside_tolerance")

    min_efficiency = unit.min_efficiency
    if (
        min_efficiency is not None
        and output.cagr_mdd_ratio is not None
        and output.cagr_mdd_ratio < min_efficiency
    ):
        soft.append("efficiency_below_threshold")

    if output.burden == "high":
        soft.append("high_burden")

    if output.thesis_status == "broken" and any(code != "thesis_broken" for code in hard):
        return "Action", hard + soft
    if hard:
        return "Review", hard + soft
    if soft:
        return "Watch", soft
    return "OK", []
