"""Evaluation Framework v2 JSON API."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.v1.serialization import METRICS_COLUMNS, dataframe_records, safe_mapping
from core.evaluation import EvaluationPeriod
from middleware.session import session_manager
from services.analysis_service import (
    DEFAULT_BENCH,
    DEFAULT_RF,
    AnalysisError,
    AnalysisResult,
    run_analysis,
)
from services.evaluation_engine import run_evaluation
from services.evaluation_period import EvaluationPeriodError, resolve_evaluation_period
from services.evaluation_units import DEFAULT_LAYER_BENCHMARKS

router = APIRouter()


class EvaluationRunRequest(BaseModel):
    period: str | None = Field(default=None, description="1M, 3M, 6M, YTD, 1Y, Max")
    start_date: date | None = None
    end_date: date | None = None
    as_of_date: date | None = None
    bench: str | None = None
    layer_benchmarks: dict[str, str] | None = None


def _period_for_request(
    payload: EvaluationRunRequest,
    session_id: str,
) -> EvaluationPeriod:
    if payload.start_date is not None or payload.end_date is not None:
        try:
            return resolve_evaluation_period(
                period=payload.period or "3M",
                start_date=payload.start_date,
                end_date=payload.end_date,
            )
        except EvaluationPeriodError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis_settings = session_manager.get(session_id, "analysis_settings") or {}
    raw_period = str(payload.period or analysis_settings.get("period") or "3M")
    raw_period = {"1": "1M", "3": "3M", "6": "6M", "12": "1Y"}.get(
        raw_period,
        raw_period,
    )
    try:
        return resolve_evaluation_period(period=raw_period, as_of_date=payload.as_of_date)
    except EvaluationPeriodError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def normalized_layer_benchmarks(payload: EvaluationRunRequest) -> dict[str, str]:
    layer_benchmarks = DEFAULT_LAYER_BENCHMARKS.copy()
    layer_benchmarks.update(
        {
            str(layer).strip().lower(): str(value).strip().upper()
            for layer, value in (payload.layer_benchmarks or {}).items()
            if str(layer).strip() and str(value).strip()
        }
    )
    return layer_benchmarks


def evaluation_settings_payload(
    payload: EvaluationRunRequest,
    evaluation_period: EvaluationPeriod,
    bench: str,
    layer_benchmarks: dict[str, str],
) -> dict[str, Any]:
    return {
        "period": evaluation_period.label,
        "start_date": evaluation_period.start_date.isoformat(),
        "end_date": evaluation_period.end_date.isoformat(),
        "as_of_date": payload.as_of_date.isoformat() if payload.as_of_date else None,
        "bench": bench,
        "layer_benchmarks": layer_benchmarks,
    }


def _store_analysis_result(session_id: str, result: AnalysisResult) -> None:
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


def analysis_response_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "metrics": dataframe_records(
            result.metrics_df,
            METRICS_COLUMNS,
            include_index=True,
        ),
        "portfolio_metrics": safe_mapping(result.portfolio_metrics),
        "benchmark_metrics": safe_mapping(result.benchmark_metrics),
        "missing_tickers": result.missing_tickers,
    }


def run_evaluation_for_asset_df(
    asset_df: pd.DataFrame,
    payload: EvaluationRunRequest,
    session_id: str | None = None,
) -> tuple[AnalysisResult, dict[str, Any], dict[str, Any]]:
    analysis_settings = (
        session_manager.get(session_id, "analysis_settings")
        if session_id is not None
        else {}
    ) or {}
    bench = str(payload.bench or analysis_settings.get("bench") or DEFAULT_BENCH).upper()
    layer_benchmarks = normalized_layer_benchmarks(payload)
    evaluation_period = (
        _period_for_request(payload, session_id)
        if session_id is not None
        else _period_for_request(payload, "")
    )
    try:
        analysis = run_analysis(
            asset_df,
            evaluation_period,
            DEFAULT_RF,
            bench,
            extra_benchmarks=list(layer_benchmarks.values()),
        )
    except AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = run_evaluation(
        analysis=analysis,
        evaluation_period=evaluation_period,
        bench=bench,
        layer_benchmarks=layer_benchmarks,
    )
    response_payload = result.to_payload()
    settings = evaluation_settings_payload(
        payload,
        evaluation_period,
        bench,
        layer_benchmarks,
    )
    return analysis, response_payload, settings


def _analysis_result_for_session(
    session_id: str,
    payload: EvaluationRunRequest,
) -> tuple[AnalysisResult, dict[str, Any], dict[str, Any]]:
    asset_df_data = session_manager.get(session_id, "asset_df")
    if asset_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 포트폴리오를 입력해주세요.")
    asset_df = pd.DataFrame(asset_df_data)
    analysis, response_payload, settings = run_evaluation_for_asset_df(
        asset_df,
        payload,
        session_id=session_id,
    )
    _store_analysis_result(session_id, analysis)
    return analysis, response_payload, settings


def _csv_from_records(records: list[dict[str, Any]]) -> str:
    return pd.DataFrame(records).to_csv(index=False)


@router.post("/run")
async def run_evaluation_endpoint(payload: EvaluationRunRequest, request: Request):
    """Run Evaluation Framework v2 for the current session."""
    session_id = request.state.session_id
    _analysis, response_payload, settings = _analysis_result_for_session(session_id, payload)
    session_manager.set(session_id, "evaluation_v2", response_payload)
    session_manager.set(session_id, "evaluation_settings", settings)
    return response_payload


@router.get("/download-csv")
async def download_csv(request: Request, type: str = "metrics"):
    """Download v2 session results as CSV."""
    session_id = request.state.session_id
    if type == "metrics":
        df = session_manager.get_dataframe(session_id, "metrics_df")
        if df is None:
            raise HTTPException(status_code=404, detail="다운로드할 결과가 없습니다.")
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        content = pd.DataFrame(
            dataframe_records(df, METRICS_COLUMNS, include_index=True)
        ).to_csv(index=False)
        filename = "metrics.csv"
    else:
        evaluation = session_manager.get(session_id, "evaluation_v2")
        if not evaluation:
            raise HTTPException(status_code=404, detail="다운로드할 평가 결과가 없습니다.")
        options = {
            "layer_evaluations": "layer_evaluations.csv",
            "asset_evaluations": "asset_evaluations.csv",
            "review_queue": "review_queue.csv",
        }
        if type not in options:
            raise HTTPException(status_code=400, detail="잘못된 타입입니다.")
        content = _csv_from_records(evaluation.get(type, []))
        filename = options[type]

    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
