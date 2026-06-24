"""Evaluation Framework v2 engine."""

from __future__ import annotations

from typing import Any, NamedTuple

import pandas as pd

from api.v1.serialization import json_safe
from core.evaluation import EvaluationOutput, EvaluationPeriod, EvaluationUnit, ReviewItem
from services.analysis_service import AnalysisResult
from services.analysis_service import parse_benchmark
from services.evaluation_status import classify_evaluation_status
from services.evaluation_units import build_evaluation_units, normalize_layer_category
from utils.efficiency_metrics import cagr_mdd_ratio, return_mdd_ratio
from utils.ips_config import load_ips_config
from utils.performance_metrics import benchmark_excess_return, cagr, period_return
from utils.risk_metrics import annualized_volatility, concentration, maximum_drawdown


GUARDRAILS = {
    "not_investment_advice": True,
    "no_immediate_order_instruction": True,
}


class EvaluationEngineResult(NamedTuple):
    """Serializable v2 engine result."""

    evaluation_period: EvaluationPeriod
    layer_evaluations: list[dict[str, Any]]
    asset_evaluations: list[dict[str, Any]]
    review_queue: list[dict[str, Any]]
    journal_draft: list[dict[str, Any]]
    warnings: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "evaluation_period": self.evaluation_period.model_dump(mode="json"),
            "layer_evaluations": self.layer_evaluations,
            "asset_evaluations": self.asset_evaluations,
            "review_queue": self.review_queue,
            "journal_draft": self.journal_draft,
            "warnings": self.warnings,
            "guardrails": GUARDRAILS.copy(),
        }


def _normalize_thesis_status(value: object) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"valid", "intact"}:
        return "valid"
    if normalized == "watch":
        return "watch"
    if normalized == "broken":
        return "broken"
    return "unknown"


def _burden_for_unit(unit: EvaluationUnit, metrics_rows: pd.DataFrame | None = None) -> str:
    if unit.level == "asset":
        if unit.parent_layer == "experiment":
            return "high"
        if unit.parent_layer == "satellite":
            return "medium"
        return "low"
    count = 0 if metrics_rows is None else int(len(metrics_rows))
    if unit.name == "experiment" and count > 0:
        return "high"
    if unit.name == "satellite" or count >= 6:
        return "medium"
    return "low"


def _unit_nav_from_returns(returns: pd.DataFrame, tickers: list[str], weights: pd.Series) -> pd.Series:
    common = [ticker for ticker in tickers if ticker in returns.columns]
    if not common:
        return pd.Series(dtype=float)
    local_weights = weights.reindex(common).fillna(0.0)
    if float(local_weights.sum()) <= 0:
        local_weights = pd.Series(1.0 / len(common), index=common)
    else:
        local_weights = local_weights / float(local_weights.sum())
    daily = (returns[common] * local_weights).sum(axis=1)
    return (1.0 + daily).cumprod()


def _output_record(unit: EvaluationUnit, output: EvaluationOutput) -> dict[str, Any]:
    return {
        "unit": unit.model_dump(mode="json"),
        "output": output.model_dump(mode="json"),
    }


def _benchmark_label(unit: EvaluationUnit) -> str | None:
    benchmark = unit.benchmark
    if benchmark is None:
        return None
    if hasattr(benchmark, "label"):
        return str(benchmark.label).strip().upper()
    spec = parse_benchmark(str(benchmark))
    return spec.label if spec is not None else None


def _benchmark_return_for_unit(
    unit: EvaluationUnit,
    analysis: AnalysisResult,
) -> float | None:
    label = _benchmark_label(unit)
    if label is None or label not in analysis.prices.columns:
        return None
    return period_return(analysis.prices[label])


def _review_item(unit: EvaluationUnit, output: EvaluationOutput) -> ReviewItem | None:
    if output.status == "OK":
        return None
    status = output.status
    if status == "Watch":
        next_step = "Keep this unit on the next review checklist; do not treat it as an order instruction."
    elif status == "Review":
        next_step = "Review data quality, thesis, risk limits, and regular-purchase policy before changing exposure."
    else:
        next_step = "Inspect thesis damage and breached limits; any intervention remains an exceptional human decision."
    return ReviewItem(
        level=unit.level,
        name=unit.name,
        parent_layer=unit.parent_layer,
        status=status,
        triggered_by=output.triggered_by,
        metrics_snapshot={
            "current_weight": output.current_weight,
            "weight_gap": output.weight_gap,
            "period_return": output.period_return,
            "cagr": output.cagr,
            "mdd": output.mdd,
            "volatility": output.volatility,
            "risk_contribution": output.risk_contribution,
            "cagr_mdd_ratio": output.cagr_mdd_ratio,
        },
        thesis=unit.thesis,
        counter_scenario=unit.counter_scenario,
        suggested_next_step=next_step,
    )


def _journal_item(review_item: ReviewItem) -> dict[str, Any]:
    return {
        "title": f"{review_item.level}:{review_item.name} {review_item.status}",
        "unit": review_item.name,
        "level": review_item.level,
        "status": review_item.status,
        "prompts": [
            "What changed in the thesis?",
            "Which threshold was triggered?",
            "Can this be addressed through regular-purchase policy instead of immediate trading?",
        ],
    }


def _asset_output(
    unit: EvaluationUnit,
    metrics: pd.DataFrame,
    analysis: AnalysisResult,
    benchmark_return: float | None,
) -> EvaluationOutput:
    row = metrics.loc[unit.name]
    current_weight = float(row.get("가중치", 0.0) or 0.0)
    layer = str(row.get("layer"))
    layer_weight = float(metrics.loc[metrics["layer"] == layer, "가중치"].sum())
    px = analysis.prices[unit.name].dropna() if unit.name in analysis.prices.columns else pd.Series(dtype=float)
    dr = (
        analysis.returns_smooth[unit.name].dropna()
        if unit.name in analysis.returns_smooth.columns
        else pd.Series(dtype=float)
    )
    unit_return = period_return(px)
    unit_cagr = cagr(px)
    unit_mdd = maximum_drawdown(px)
    unit_vol = annualized_volatility(dr)
    output = EvaluationOutput(
        current_weight=current_weight,
        weight_gap=(unit.target_weight - current_weight) if unit.target_weight is not None else None,
        layer_internal_weight=(current_weight / layer_weight) if layer_weight > 0 else None,
        period_return=unit_return,
        cagr=unit_cagr,
        benchmark_return=benchmark_return,
        benchmark_excess_return=benchmark_excess_return(unit_return, benchmark_return),
        mdd=unit_mdd,
        volatility=unit_vol,
        concentration=current_weight,
        risk_contribution=json_safe(row.get("위험기여도")),
        return_mdd_ratio=return_mdd_ratio(unit_return, unit_mdd),
        cagr_mdd_ratio=cagr_mdd_ratio(unit_cagr, unit_mdd),
        thesis_status=_normalize_thesis_status(row.get("thesis_status")),
        burden=_burden_for_unit(unit),
        status="OK",
    )
    status, triggered_by = classify_evaluation_status(unit, output)
    output.status = status
    output.triggered_by = triggered_by
    return output


def _layer_output(
    unit: EvaluationUnit,
    metrics: pd.DataFrame,
    analysis: AnalysisResult,
    benchmark_return: float | None,
) -> EvaluationOutput:
    rows = metrics.loc[metrics["layer"] == unit.name]
    weights = pd.to_numeric(rows.get("가중치", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    current_weight = float(weights.sum())
    tickers = [str(ticker) for ticker in rows.index]
    nav = _unit_nav_from_returns(analysis.returns_smooth, tickers, weights)
    daily_returns = nav.pct_change(fill_method=None).dropna()
    unit_return = period_return(nav)
    unit_cagr = cagr(nav)
    unit_mdd = maximum_drawdown(nav)
    unit_vol = annualized_volatility(daily_returns)
    thesis_values = [_normalize_thesis_status(value) for value in rows.get("thesis_status", pd.Series(dtype=object))]
    thesis_status = "valid"
    if "broken" in thesis_values:
        thesis_status = "broken"
    elif "watch" in thesis_values:
        thesis_status = "watch"
    elif "unknown" in thesis_values or not thesis_values:
        thesis_status = "unknown"
    output = EvaluationOutput(
        current_weight=current_weight,
        weight_gap=(unit.target_weight - current_weight) if unit.target_weight is not None else None,
        layer_internal_weight=None,
        period_return=unit_return,
        cagr=unit_cagr,
        benchmark_return=benchmark_return,
        benchmark_excess_return=benchmark_excess_return(unit_return, benchmark_return),
        mdd=unit_mdd,
        volatility=unit_vol,
        concentration=concentration(weights),
        risk_contribution=json_safe(rows.get("위험기여도", pd.Series(dtype=float)).sum()),
        return_mdd_ratio=return_mdd_ratio(unit_return, unit_mdd),
        cagr_mdd_ratio=cagr_mdd_ratio(unit_cagr, unit_mdd),
        thesis_status=thesis_status,
        burden=_burden_for_unit(unit, rows),
        status="OK",
    )
    status, triggered_by = classify_evaluation_status(unit, output)
    output.status = status
    output.triggered_by = triggered_by
    return output


def run_evaluation(
    *,
    analysis: AnalysisResult,
    evaluation_period: EvaluationPeriod,
    bench: str,
    layer_benchmarks: dict[str, str] | None = None,
    ips_config: dict | None = None,
) -> EvaluationEngineResult:
    """Run the v2 shared layer/asset evaluation engine."""
    config = ips_config or load_ips_config()
    metrics = normalize_layer_category(analysis.metrics_df)
    unit_set = build_evaluation_units(
        metrics,
        config,
        evaluation_period,
        bench,
        layer_benchmarks=layer_benchmarks,
    )

    layer_records: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    review_items: list[ReviewItem] = []

    for unit in unit_set.layer_units:
        bench_return = _benchmark_return_for_unit(unit, analysis)
        output = _layer_output(unit, metrics, analysis, bench_return)
        layer_records.append(_output_record(unit, output))
        item = _review_item(unit, output)
        if item is not None:
            review_items.append(item)

    for unit in unit_set.asset_units:
        bench_return = _benchmark_return_for_unit(unit, analysis)
        output = _asset_output(unit, metrics, analysis, bench_return)
        asset_records.append(_output_record(unit, output))
        item = _review_item(unit, output)
        if item is not None:
            review_items.append(item)

    review_payload = [item.model_dump(mode="json") for item in review_items]
    journal_draft = [_journal_item(item) for item in review_items]
    return EvaluationEngineResult(
        evaluation_period=evaluation_period,
        layer_evaluations=layer_records,
        asset_evaluations=asset_records,
        review_queue=review_payload,
        journal_draft=journal_draft,
        warnings=[],
    )
