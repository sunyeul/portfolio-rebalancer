"""Analysis JSON API."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.v1.serialization import METRICS_COLUMNS, dataframe_records, safe_mapping
from middleware.session import session_manager
from services.analysis_service import DEFAULT_BENCH, DEFAULT_RF, AnalysisError, run_analysis

router = APIRouter()

EVALUATION_SESSION_KEYS = (
    "proposal_df",
    "ips_action_df",
    "group_summary_df",
    "rc_violations",
    "ips_config_snapshot",
    "evaluation_settings",
)


class AnalysisRunRequest(BaseModel):
    period: int | str = Field(12, description="Month count, YTD, or Max")
    rf: float = DEFAULT_RF
    bench: str = DEFAULT_BENCH


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


@router.post("/run")
async def run_analysis_endpoint(payload: AnalysisRunRequest, request: Request):
    """Run price fetch and portfolio metric enrichment for the session portfolio."""
    session_id = request.state.session_id
    asset_df_data = session_manager.get(session_id, "asset_df")
    if asset_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 포트폴리오를 입력해주세요.")

    asset_df = pd.DataFrame(asset_df_data)
    try:
        result = run_analysis(
            asset_df,
            _parse_period(payload.period),
            payload.rf,
            payload.bench.upper(),
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
            "period": _parse_period(payload.period),
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
