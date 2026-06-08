"""Simulation JSON API for IPS-safe counterfactuals and backtests."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.v1.evaluation import _cov_matrix_for_session
from api.v1.serialization import json_safe
from middleware.session import session_manager
from services.simulation_service import (
    BacktestStrategy,
    CounterfactualScenario,
    SimulationError,
    run_counterfactual_simulation,
    run_ips_backtest,
)

router = APIRouter()


DecisionContext = Literal[
    "regular_review",
    "market_correction",
    "sharp_drop_review",
    "rebalance_review",
]


class CounterfactualRunRequest(BaseModel):
    scenario: CounterfactualScenario = "current_proposal"
    rc_over_thresh_pct: float = 1.5
    e_thresh: float = 0.5
    decision_context: DecisionContext = "regular_review"


class BacktestRunRequest(BaseModel):
    strategies: list[BacktestStrategy] = Field(
        default_factory=lambda: [
            "current_ips",
            "core_first_dca",
            "pause_overweight_satellite",
            "return_chasing_reference",
        ]
    )
    frequency: Literal["monthly"] = "monthly"
    decision_context: DecisionContext = "regular_review"
    rf: float = 0.0


def _json_safe_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_nested(item) for item in value]
    return json_safe(value)


def _metrics_df_for_session(session_id: str) -> pd.DataFrame:
    metrics_df_data = session_manager.get(session_id, "metrics_df")
    if metrics_df_data is None:
        raise HTTPException(status_code=400, detail="먼저 데이터 분석을 실행해주세요.")
    metrics_df = pd.DataFrame(metrics_df_data)
    if "ticker" in metrics_df.columns:
        metrics_df = metrics_df.set_index("ticker")
    return metrics_df


@router.post("/counterfactual")
async def run_counterfactual_endpoint(payload: CounterfactualRunRequest, request: Request):
    """Run a preset counterfactual comparison against the current IPS evaluation."""
    session_id = request.state.session_id
    metrics_df = _metrics_df_for_session(session_id)
    try:
        result = run_counterfactual_simulation(
            metrics_df=metrics_df,
            scenario=payload.scenario,
            rc_over_thresh_pct=payload.rc_over_thresh_pct,
            e_thresh=payload.e_thresh,
            cov_matrix=_cov_matrix_for_session(session_id, metrics_df),
            decision_context=payload.decision_context,
        )
    except SimulationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe_nested(result)


@router.post("/backtest")
async def run_backtest_endpoint(payload: BacktestRunRequest, request: Request):
    """Run a limited monthly IPS policy backtest from the current analysis data."""
    session_id = request.state.session_id
    metrics_df = _metrics_df_for_session(session_id)
    returns_smooth_data = session_manager.get(session_id, "returns_smooth")
    if returns_smooth_data is None:
        raise HTTPException(status_code=400, detail="백테스트에 사용할 수익률 데이터가 없습니다.")
    returns_smooth = pd.DataFrame(returns_smooth_data)
    try:
        result = run_ips_backtest(
            returns_smooth=returns_smooth,
            metrics_df=metrics_df,
            strategies=payload.strategies,
            cov_matrix=_cov_matrix_for_session(session_id, metrics_df),
            frequency=payload.frequency,
            decision_context=payload.decision_context,
            rf=payload.rf,
        )
    except SimulationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe_nested(result)
