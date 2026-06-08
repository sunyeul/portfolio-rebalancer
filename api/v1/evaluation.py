"""Evaluation JSON API."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.v1.serialization import (
    GROUP_SUMMARY_COLUMNS,
    METRICS_COLUMNS,
    PROPOSAL_COLUMNS,
    RC_VIOLATION_COLUMNS,
    dataframe_records,
)
from middleware.session import session_manager
from services.evaluation_service import EvaluationError, run_evaluation
from utils.metrics import annualize_cov

router = APIRouter()


class EvaluationRunRequest(BaseModel):
    rc_over_thresh_pct: float = 1.5
    e_thresh: float = 0.5
    target_weights: dict[str, float] | None = Field(default=None)
    decision_context: Literal[
        "regular_review",
        "market_correction",
        "sharp_drop_review",
        "rebalance_review",
    ] = "regular_review"


def _cov_matrix_for_session(session_id: str, metrics_df: pd.DataFrame):
    returns_smooth_data = session_manager.get(session_id, "returns_smooth")
    if returns_smooth_data is None:
        return None

    returns_smooth_df = pd.DataFrame(returns_smooth_data)
    if "Date" in returns_smooth_df.columns or "date" in returns_smooth_df.columns:
        date_col = "Date" if "Date" in returns_smooth_df.columns else "date"
        returns_smooth_df = returns_smooth_df.set_index(date_col)

    common_tickers = returns_smooth_df.columns.intersection(metrics_df.index)
    if len(common_tickers) == 0:
        return None
    return annualize_cov(returns_smooth_df[common_tickers])


@router.post("/run")
async def run_evaluation_endpoint(payload: EvaluationRunRequest, request: Request):
    """Run IPS evaluation and proposal generation for the current session."""
    session_id = request.state.session_id
    metrics_df_data = session_manager.get(session_id, "metrics_df")
    if metrics_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 데이터 분석을 실행해주세요.")

    metrics_df = pd.DataFrame(metrics_df_data)
    if "ticker" in metrics_df.columns:
        metrics_df = metrics_df.set_index("ticker")

    try:
        result = run_evaluation(
            metrics_df,
            payload.target_weights,
            payload.rc_over_thresh_pct,
            payload.e_thresh,
            cov_matrix=_cov_matrix_for_session(session_id, metrics_df),
            decision_context=payload.decision_context,
        )
    except EvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    proposal_records = result.proposal_df.to_dict(orient="records")
    ips_action_records = result.ips_action_df.to_dict(orient="records")
    group_summary_records = result.group_summary_df.to_dict(orient="records")
    session_manager.set(session_id, "proposal_df", proposal_records)
    session_manager.set(session_id, "ips_action_df", ips_action_records)
    session_manager.set(session_id, "group_summary_df", group_summary_records)
    session_manager.set(
        session_id, "rc_violations", result.rc_violations.to_dict(orient="records")
    )
    session_manager.set(session_id, "ips_config_snapshot", result.ips_config_snapshot)
    session_manager.set(
        session_id,
        "evaluation_settings",
        {
            "rc_over_thresh_pct": payload.rc_over_thresh_pct,
            "e_thresh": payload.e_thresh,
            "target_weights": payload.target_weights,
            "decision_context": payload.decision_context,
        },
    )

    return {
        "proposal": dataframe_records(result.proposal_df, PROPOSAL_COLUMNS),
        "ips_actions": dataframe_records(result.ips_action_df),
        "group_summary": dataframe_records(result.group_summary_df, GROUP_SUMMARY_COLUMNS),
        "sell_list": dataframe_records(result.sell_list, PROPOSAL_COLUMNS),
        "buy_list": dataframe_records(result.buy_list, PROPOSAL_COLUMNS),
        "fine_tune_list": dataframe_records(result.fine_tune_list, PROPOSAL_COLUMNS),
        "rc_violations": dataframe_records(
            result.rc_violations, RC_VIOLATION_COLUMNS
        ),
        "ips_config_snapshot": result.ips_config_snapshot,
    }


@router.get("/download-csv")
async def download_csv(request: Request, type: str = "metrics"):
    """Download session results as CSV."""
    session_id = request.state.session_id
    options = {
        "metrics": ("metrics_df", "enriched_metrics.csv"),
        "proposal": ("proposal_df", "proposal.csv"),
        "ips_actions": ("ips_action_df", "ips_actions.csv"),
        "group_summary": ("group_summary_df", "group_summary.csv"),
    }
    if type not in options:
        raise HTTPException(status_code=400, detail="잘못된 타입입니다.")

    key, filename = options[type]
    df = session_manager.get_dataframe(session_id, key)
    if df is None:
        raise HTTPException(status_code=404, detail="다운로드할 결과가 없습니다.")

    column_maps = {
        "metrics": METRICS_COLUMNS,
        "proposal": PROPOSAL_COLUMNS,
        "group_summary": GROUP_SUMMARY_COLUMNS,
    }
    if type in column_maps:
        df = pd.DataFrame(dataframe_records(df, column_maps[type]))

    return Response(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
