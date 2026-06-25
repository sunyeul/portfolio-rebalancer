"""Saved portfolio and snapshot JSON API."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.v1.serialization import (
    METRICS_COLUMNS,
    dataframe_records,
    safe_mapping,
)
from middleware.session import session_manager
from services.portfolio_service import (
    PortfolioInputError,
    normalize_and_validate_assets,
    parse_manual_edit_to_assets,
)
from storage.portfolio_store import (
    StorageError,
    create_portfolio,
    create_snapshot,
    delete_snapshot,
    get_current_state,
    get_snapshot,
    list_portfolios,
    list_snapshots,
    save_current_state,
    update_portfolio,
    update_snapshot,
)

router = APIRouter()

SESSION_KEYS = [
    "asset_df",
    "metrics_df",
    "portfolio_metrics",
    "benchmark_metrics",
    "missing_tickers",
    "returns_smooth",
    "analysis_settings",
    "prices",
    "returns",
    "evaluation_v2",
    "evaluation_settings",
]


class PortfolioCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""


class PortfolioUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class SnapshotPortfolioRowIn(BaseModel):
    ticker: str = ""
    allocation: float | str | None = None
    return_total: float | str | None = None
    layer: str | None = None
    thesis_status: str | None = None


class SnapshotCreateRequest(BaseModel):
    name: str = ""
    note: str = ""
    rows: list[SnapshotPortfolioRowIn] | None = None
    source_snapshot_id: int | None = None


class SnapshotUpdateRequest(BaseModel):
    name: str | None = None
    note: str | None = None
    rows: list[SnapshotPortfolioRowIn] | None = None


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


@router.get("/{portfolio_id}/current-state")
async def get_saved_current_state(portfolio_id: int, request: Request):
    """Return and restore the portfolio's auto-saved current workbench state."""
    current_state = get_current_state(portfolio_id)
    if current_state is None:
        raise HTTPException(status_code=404, detail="저장된 현재 상태가 없습니다.")

    session_id = request.state.session_id
    for key, value in current_state["session_state"].items():
        if value is not None:
            session_manager.set(session_id, key, value)
    return _state_response(current_state)


@router.post("/{portfolio_id}/current-state")
async def save_saved_current_state(portfolio_id: int, request: Request):
    """Persist the current session state as the portfolio's auto-saved draft."""
    session_id = request.state.session_id
    session_data = _session_data(session_id)
    try:
        return _state_response(save_current_state(portfolio_id, session_data))
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{portfolio_id}/snapshots")
async def create_saved_snapshot(
    portfolio_id: int,
    payload: SnapshotCreateRequest,
    request: Request,
):
    """Persist the current session state as a named portfolio snapshot."""
    session_id = request.state.session_id
    session_data = _session_data(session_id)
    should_save_current_state = True
    if payload.rows is not None:
        try:
            assets, _warnings = parse_manual_edit_to_assets(
                [row.model_dump() for row in payload.rows]
            )
            asset_df, _validation_warnings = normalize_and_validate_assets(assets)
        except PortfolioInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session_data = {key: None for key in SESSION_KEYS}
        session_data["asset_df"] = asset_df.to_dict(orient="records")
        session_manager.set(session_id, "asset_df", session_data["asset_df"])
        for key in SESSION_KEYS:
            if key != "asset_df":
                session_manager.delete(session_id, key)
    elif payload.source_snapshot_id is not None:
        source_snapshot = get_snapshot(payload.source_snapshot_id)
        if source_snapshot is None:
            raise HTTPException(status_code=404, detail="스냅샷을 찾을 수 없습니다.")
        if source_snapshot["summary"]["portfolio_id"] != portfolio_id:
            raise HTTPException(
                status_code=400,
                detail="같은 포트폴리오의 스냅샷만 복사할 수 있습니다.",
            )
        session_data = {key: None for key in SESSION_KEYS}
        session_data["asset_df"] = source_snapshot["session_state"].get("asset_df") or []
        should_save_current_state = False
    elif not session_data.get("asset_df"):
        current_state = get_current_state(portfolio_id)
        if current_state is not None:
            session_data = current_state["session_state"]
    try:
        snapshot = create_snapshot(
            portfolio_id,
            payload.name,
            payload.note,
            session_data,
        )
        if should_save_current_state:
            save_current_state(portfolio_id, session_data)
        return {"snapshot": snapshot}
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/snapshots/{snapshot_id}")
async def update_saved_snapshot(
    snapshot_id: int,
    payload: SnapshotUpdateRequest,
):
    """Update a saved snapshot's editable metadata."""
    asset_rows = None
    if payload.rows is not None:
        try:
            assets, _warnings = parse_manual_edit_to_assets(
                [row.model_dump() for row in payload.rows]
            )
            asset_df, _validation_warnings = normalize_and_validate_assets(assets)
            asset_rows = asset_df.to_dict(orient="records")
        except PortfolioInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return {
            "snapshot": update_snapshot(
                snapshot_id,
                name=payload.name,
                note=payload.note,
                asset_rows=asset_rows,
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
    try:
        save_current_state(snapshot["summary"]["portfolio_id"], snapshot["session_state"])
    except StorageError:
        pass
    return _snapshot_response(snapshot)


def _session_data(session_id: str) -> dict:
    return {key: session_manager.get(session_id, key) for key in SESSION_KEYS}


def _state_response(state: dict) -> dict:
    session_state = state["session_state"]
    asset_df = pd.DataFrame(session_state.get("asset_df") or [])
    response = {
        "portfolio": {
            "assets": dataframe_records(asset_df),
            "warnings": [],
        },
        "analysis": None,
        "evaluation": None,
        "updated_at": state["updated_at"],
    }

    if state["analysis"]:
        metrics_df = pd.DataFrame(state["analysis"]["metrics_df"])
        if "ticker" in metrics_df.columns:
            metrics_df = metrics_df.set_index("ticker")
        response["analysis"] = {
            "metrics": dataframe_records(
                metrics_df,
                METRICS_COLUMNS,
                include_index=True,
            ),
            "portfolio_metrics": safe_mapping(state["analysis"]["portfolio_metrics"]),
            "benchmark_metrics": safe_mapping(state["analysis"]["benchmark_metrics"]),
            "missing_tickers": state["analysis"]["missing_tickers"],
        }

    if state["evaluation"] and "layer_evaluations" in state["evaluation"]:
        response["evaluation"] = state["evaluation"]

    return response


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

    if snapshot["evaluation"] and "layer_evaluations" in snapshot["evaluation"]:
        response["evaluation"] = snapshot["evaluation"]

    return response
