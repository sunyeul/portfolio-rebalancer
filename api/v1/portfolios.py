"""Saved portfolio and snapshot JSON API."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.v1.serialization import (
    GROUP_SUMMARY_COLUMNS,
    METRICS_COLUMNS,
    PROPOSAL_COLUMNS,
    RC_VIOLATION_COLUMNS,
    dataframe_records,
    safe_mapping,
)
from middleware.session import session_manager
from storage.portfolio_store import (
    StorageError,
    create_portfolio,
    create_snapshot,
    delete_snapshot,
    get_snapshot,
    list_portfolios,
    list_snapshots,
    update_portfolio,
    update_snapshot,
)

router = APIRouter()


class PortfolioCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""


class PortfolioUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class SnapshotCreateRequest(BaseModel):
    name: str = ""
    note: str = ""


class SnapshotUpdateRequest(BaseModel):
    name: str | None = None
    note: str | None = None


@router.get("")
async def list_saved_portfolios():
    """List saved portfolios with latest snapshot summaries."""
    return {"portfolios": list_portfolios()}


@router.post("")
async def create_saved_portfolio(payload: PortfolioCreateRequest):
    """Create a saved portfolio container."""
    try:
        return {"portfolio": create_portfolio(payload.name, payload.description)}
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{portfolio_id}")
async def update_saved_portfolio(
    portfolio_id: int,
    payload: PortfolioUpdateRequest,
):
    """Update a saved portfolio's name or description."""
    try:
        return {
            "portfolio": update_portfolio(
                portfolio_id,
                name=payload.name,
                description=payload.description,
            )
        }
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{portfolio_id}/snapshots")
async def list_saved_snapshots(portfolio_id: int):
    """List snapshots for a saved portfolio."""
    return {"snapshots": list_snapshots(portfolio_id)}


@router.post("/{portfolio_id}/snapshots")
async def create_saved_snapshot(
    portfolio_id: int,
    payload: SnapshotCreateRequest,
    request: Request,
):
    """Persist the current session state as a named portfolio snapshot."""
    session_id = request.state.session_id
    session_data = {
        key: session_manager.get(session_id, key)
        for key in [
            "asset_df",
            "metrics_df",
            "portfolio_metrics",
            "benchmark_metrics",
            "missing_tickers",
            "returns_smooth",
            "analysis_settings",
            "proposal_df",
            "ips_action_df",
            "group_summary_df",
            "rc_violations",
            "evaluation_settings",
        ]
    }
    try:
        snapshot = create_snapshot(
            portfolio_id,
            payload.name,
            payload.note,
            session_data,
        )
        return {"snapshot": snapshot}
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/snapshots/{snapshot_id}")
async def update_saved_snapshot(
    snapshot_id: int,
    payload: SnapshotUpdateRequest,
):
    """Update a saved snapshot's editable metadata."""
    try:
        return {
            "snapshot": update_snapshot(
                snapshot_id,
                name=payload.name,
                note=payload.note,
            )
        }
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/snapshots/{snapshot_id}")
async def delete_saved_snapshot(snapshot_id: int):
    """Delete a saved snapshot and its persisted analysis/evaluation data."""
    try:
        delete_snapshot(snapshot_id)
        return {"ok": True}
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/snapshots/{snapshot_id}")
async def get_saved_snapshot(snapshot_id: int):
    """Return a saved snapshot in the same shapes used by the workbench."""
    snapshot = get_snapshot(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="스냅샷을 찾을 수 없습니다.")
    return _snapshot_response(snapshot)


@router.post("/snapshots/{snapshot_id}/load")
async def load_saved_snapshot(snapshot_id: int, request: Request):
    """Restore a saved snapshot into the current signed-cookie session."""
    snapshot = get_snapshot(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="스냅샷을 찾을 수 없습니다.")

    session_id = request.state.session_id
    for key, value in snapshot["session_state"].items():
        if value is not None:
            session_manager.set(session_id, key, value)
    return _snapshot_response(snapshot)


def _snapshot_response(snapshot: dict) -> dict:
    session_state = snapshot["session_state"]
    asset_df = pd.DataFrame(session_state.get("asset_df") or [])
    response = {
        "snapshot": snapshot["summary"],
        "portfolio": {
            "assets": dataframe_records(asset_df),
            "warnings": [],
        },
        "analysis": None,
        "evaluation": None,
    }

    if snapshot["analysis"]:
        metrics_df = pd.DataFrame(snapshot["analysis"]["metrics_df"])
        if "ticker" in metrics_df.columns:
            metrics_df = metrics_df.set_index("ticker")
        response["analysis"] = {
            "metrics": dataframe_records(
                metrics_df,
                METRICS_COLUMNS,
                include_index=True,
            ),
            "portfolio_metrics": safe_mapping(snapshot["analysis"]["portfolio_metrics"]),
            "benchmark_metrics": safe_mapping(snapshot["analysis"]["benchmark_metrics"]),
            "missing_tickers": snapshot["analysis"]["missing_tickers"],
        }

    if snapshot["evaluation"]:
        proposal_df = pd.DataFrame(snapshot["evaluation"]["proposal_df"])
        ips_action_df = pd.DataFrame(snapshot["evaluation"]["ips_action_df"])
        group_summary_df = pd.DataFrame(snapshot["evaluation"]["group_summary_df"])
        rc_violations_df = pd.DataFrame(snapshot["evaluation"]["rc_violations"])
        gap = proposal_df.get("갭%", pd.Series(dtype=float))
        should_execute = proposal_df.get("실행", pd.Series(dtype=bool)).astype(bool)
        response["evaluation"] = {
            "proposal": dataframe_records(proposal_df, PROPOSAL_COLUMNS),
            "ips_actions": dataframe_records(ips_action_df),
            "group_summary": dataframe_records(group_summary_df, GROUP_SUMMARY_COLUMNS),
            "sell_list": dataframe_records(
                proposal_df[(gap < 0) & should_execute],
                PROPOSAL_COLUMNS,
            ),
            "buy_list": dataframe_records(
                proposal_df[(gap > 0) & should_execute],
                PROPOSAL_COLUMNS,
            ),
            "fine_tune_list": dataframe_records(
                proposal_df[should_execute & (gap.abs() <= 1.0)],
                PROPOSAL_COLUMNS,
            ),
            "rc_violations": dataframe_records(
                rc_violations_df,
                RC_VIOLATION_COLUMNS,
            ),
        }

    return response
