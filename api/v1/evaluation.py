"""Evaluation Framework v2 JSON API."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.v1.serialization import METRICS_COLUMNS, dataframe_records
from core.evaluation import EvaluationPeriod
from middleware.session import session_manager
from services.analysis_service import DEFAULT_BENCH, AnalysisResult
from services.evaluation_engine import run_evaluation
from services.evaluation_period import EvaluationPeriodError, resolve_evaluation_period
from services.evaluation_units import DEFAULT_LAYER_BENCHMARKS

router = APIRouter()


class EvaluationRunRequest(BaseModel):
    period: str | None = Field(default=None, description="1M, 3M, 6M, YTD, 1Y, Max")
    start_date: date | None = None
    end_date: date | None = None
    bench: str | None = None
    layer_benchmarks: dict[str, str] | None = None


def _frame_from_session(session_id: str, key: str) -> pd.DataFrame:
    data = session_manager.get(session_id, key)
    if data is None:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "Date" in df.columns or "date" in df.columns:
        date_col = "Date" if "Date" in df.columns else "date"
        df = df.set_index(date_col)
    return df


def _metrics_df_for_session(session_id: str) -> pd.DataFrame:
    metrics_df_data = session_manager.get(session_id, "metrics_df")
    if metrics_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 데이터 분석을 실행해주세요.")
    metrics_df = pd.DataFrame(metrics_df_data)
    if "ticker" in metrics_df.columns:
        metrics_df = metrics_df.set_index("ticker")
    return metrics_df


def _period_for_request(
    payload: EvaluationRunRequest,
    session_id: str,
    prices: pd.DataFrame,
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
    raw_period = payload.period or str(analysis_settings.get("period") or "3M")
    if not prices.empty:
        try:
            start = pd.Timestamp(prices.index.min()).date()
            end = pd.Timestamp(prices.index.max()).date()
            label = raw_period.upper()
            if label == "12":
                label = "1Y"
            if label == "MAX":
                label = "Max"
            if label not in {"1M", "3M", "6M", "YTD", "1Y", "Max"}:
                label = "custom"
            return EvaluationPeriod(label=label, start_date=start, end_date=end)
        except Exception:
            pass
    try:
        return resolve_evaluation_period(period=raw_period)
    except EvaluationPeriodError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _analysis_result_for_session(
    session_id: str,
    payload: EvaluationRunRequest,
) -> tuple[AnalysisResult, EvaluationPeriod, str, dict[str, str]]:
    metrics_df = _metrics_df_for_session(session_id)
    prices = _frame_from_session(session_id, "prices")
    returns = _frame_from_session(session_id, "returns")
    returns_smooth = _frame_from_session(session_id, "returns_smooth")
    if prices.empty or returns_smooth.empty:
        raise HTTPException(
            status_code=400,
            detail="v2 평가는 가격/수익률 데이터가 필요합니다. 분석을 다시 실행해주세요.",
        )

    analysis_settings = session_manager.get(session_id, "analysis_settings") or {}
    bench = str(payload.bench or analysis_settings.get("bench") or DEFAULT_BENCH).upper()
    layer_benchmarks = DEFAULT_LAYER_BENCHMARKS.copy()
    layer_benchmarks.update(
        {
            str(layer).strip().lower(): str(value).strip().upper()
            for layer, value in (payload.layer_benchmarks or {}).items()
            if str(layer).strip() and str(value).strip()
        }
    )
    evaluation_period = _period_for_request(payload, session_id, prices)
    weights = metrics_df.get("가중치", pd.Series(dtype=float)).astype(float)
    bench_nav = None
    if bench in prices.columns:
        bench_prices = pd.to_numeric(prices[bench], errors="coerce").dropna()
        if not bench_prices.empty and float(bench_prices.iloc[0]) > 0:
            bench_nav = bench_prices / float(bench_prices.iloc[0])
    return (
        AnalysisResult(
            prices=prices,
            returns=returns,
            returns_smooth=returns_smooth,
            weights_no_bench=weights,
            metrics_df=metrics_df,
            port_nav=pd.Series(dtype=float),
            bench_nav=bench_nav,
            portfolio_metrics=session_manager.get(session_id, "portfolio_metrics") or {},
            benchmark_metrics=session_manager.get(session_id, "benchmark_metrics"),
            missing_tickers=session_manager.get(session_id, "missing_tickers") or [],
        ),
        evaluation_period,
        bench,
        layer_benchmarks,
    )


def _csv_from_records(records: list[dict[str, Any]]) -> str:
    return pd.DataFrame(records).to_csv(index=False)


@router.post("/run")
async def run_evaluation_endpoint(payload: EvaluationRunRequest, request: Request):
    """Run Evaluation Framework v2 for the current session."""
    session_id = request.state.session_id
    analysis, evaluation_period, bench, layer_benchmarks = _analysis_result_for_session(session_id, payload)
    result = run_evaluation(
        analysis=analysis,
        evaluation_period=evaluation_period,
        bench=bench,
        layer_benchmarks=layer_benchmarks,
    )
    response_payload = result.to_payload()
    session_manager.set(session_id, "evaluation_v2", response_payload)
    session_manager.set(
        session_id,
        "evaluation_settings",
        {
            "period": evaluation_period.label,
            "start_date": evaluation_period.start_date.isoformat(),
            "end_date": evaluation_period.end_date.isoformat(),
            "bench": bench,
            "layer_benchmarks": layer_benchmarks,
        },
    )
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
