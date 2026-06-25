"""Analysis JSON API."""

from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.v1.serialization import METRICS_COLUMNS, dataframe_records, safe_mapping
from core.evaluation import EvaluationPeriod
from middleware.session import session_manager
from services.analysis_service import DEFAULT_BENCH, DEFAULT_RF, AnalysisError, run_analysis
from services.evaluation_period import EvaluationPeriodError, resolve_evaluation_period
from services.evaluation_units import DEFAULT_LAYER_BENCHMARKS

router = APIRouter()

EVALUATION_SESSION_KEYS = (
    "evaluation_v2",
    "evaluation_settings",
)


class AnalysisRunRequest(BaseModel):
    period: int | str = Field(12, description="Month count, YTD, or Max")
    as_of_date: date | None = None
    rf: float = DEFAULT_RF
    bench: str = DEFAULT_BENCH
    layer_benchmarks: dict[str, str] | None = None


def _parse_period(period: int | str) -> int | str:
    if isinstance(period, int):
        return period
    normalized = str(period).strip()
    if normalized.isdigit():
        return int(normalized)
    if normalized.upper() == "YTD":
        return "YTD"
    if normalized.lower() == "max":
        return "Max"
    raise HTTPException(status_code=400, detail="period는 개월 수, YTD, Max 중 하나여야 합니다.")


def _period_for_request(payload: AnalysisRunRequest) -> int | str | EvaluationPeriod:
    parsed = _parse_period(payload.period)
    if payload.as_of_date is None:
        return parsed
    if isinstance(parsed, int):
        label = {1: "1M", 3: "3M", 6: "6M", 12: "1Y"}.get(parsed)
        if label is not None:
            try:
                return resolve_evaluation_period(
                    period=label,
                    as_of_date=payload.as_of_date,
                )
            except EvaluationPeriodError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        start = (pd.Timestamp(payload.as_of_date) - pd.DateOffset(months=parsed)).date()
        return EvaluationPeriod(
            label="custom",
            start_date=start,
            end_date=payload.as_of_date,
        )
    try:
        return resolve_evaluation_period(period=parsed, as_of_date=payload.as_of_date)
    except EvaluationPeriodError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run")
async def run_analysis_endpoint(payload: AnalysisRunRequest, request: Request):
    """Run price fetch and portfolio metric enrichment for the session portfolio."""
    session_id = request.state.session_id
    asset_df_data = session_manager.get(session_id, "asset_df")
    if asset_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 포트폴리오를 입력해주세요.")

    asset_df = pd.DataFrame(asset_df_data)
    layer_benchmarks = DEFAULT_LAYER_BENCHMARKS.copy()
    layer_benchmarks.update(
        {
            str(layer).strip().lower(): str(value).strip().upper()
            for layer, value in (payload.layer_benchmarks or {}).items()
            if str(layer).strip() and str(value).strip()
        }
    )
    analysis_period = _period_for_request(payload)
    try:
        result = run_analysis(
            asset_df,
            analysis_period,
            payload.rf,
            payload.bench.upper(),
            extra_benchmarks=list(layer_benchmarks.values()),
        )
    except AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_manager.set(
        session_id, "prices", result.prices.reset_index().to_dict(orient="records")
    )
    session_manager.set(
        session_id, "returns", result.returns.reset_index().to_dict(orient="records")
    )
    session_manager.set(
        session_id,
        "returns_smooth",
        result.returns_smooth.reset_index().to_dict(orient="records"),
    )
    session_manager.set(session_id, "weights_no_bench", result.weights_no_bench.to_dict())
    session_manager.set(
        session_id, "metrics_df", result.metrics_df.reset_index().to_dict(orient="records")
    )
    session_manager.set(session_id, "portfolio_metrics", result.portfolio_metrics)
    session_manager.set(session_id, "benchmark_metrics", result.benchmark_metrics)
    session_manager.set(session_id, "missing_tickers", result.missing_tickers)
    session_manager.set(
        session_id,
        "analysis_settings",
        {
            "period": (
                analysis_period.label
                if isinstance(analysis_period, EvaluationPeriod)
                else analysis_period
            ),
            "start_date": (
                analysis_period.start_date.isoformat()
                if isinstance(analysis_period, EvaluationPeriod)
                else None
            ),
            "end_date": (
                analysis_period.end_date.isoformat()
                if isinstance(analysis_period, EvaluationPeriod)
                else None
            ),
            "rf": payload.rf,
            "bench": payload.bench.upper(),
        },
    )
    for key in EVALUATION_SESSION_KEYS:
        session_manager.delete(session_id, key)

    return {
        "metrics": dataframe_records(
            result.metrics_df, METRICS_COLUMNS, include_index=True
        ),
        "portfolio_metrics": safe_mapping(result.portfolio_metrics),
        "benchmark_metrics": safe_mapping(result.benchmark_metrics),
        "missing_tickers": result.missing_tickers,
    }
