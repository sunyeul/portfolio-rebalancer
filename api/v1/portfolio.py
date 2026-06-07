"""Portfolio JSON API."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from api.v1.serialization import dataframe_records
from middleware.session import session_manager
from services.portfolio_service import (
    PortfolioInputError,
    normalize_and_validate_assets,
    parse_csv_to_assets,
    parse_manual_edit_to_assets,
)

router = APIRouter()


class PortfolioRowIn(BaseModel):
    ticker: str = ""
    allocation: float | str | None = None
    return_total: float | str | None = None
    group: str | None = None
    role: str | None = None
    dca_enabled: bool | str | None = True
    thesis_status: str | None = None


class ManualPortfolioRequest(BaseModel):
    rows: list[PortfolioRowIn] = Field(default_factory=list)


def _store_portfolio_result(
    request: Request,
    asset_df: pd.DataFrame,
    warnings: list[str],
) -> dict:
    session_id = request.state.session_id
    session_manager.set(session_id, "asset_df", asset_df.to_dict(orient="records"))
    return {
        "assets": dataframe_records(asset_df),
        "warnings": warnings,
    }


@router.post("/manual")
async def parse_manual_portfolio(payload: ManualPortfolioRequest, request: Request):
    """Parse manually edited portfolio rows into normalized asset rows."""
    try:
        assets, warnings = parse_manual_edit_to_assets(
            [row.model_dump() for row in payload.rows]
        )
        asset_df, validation_warnings = normalize_and_validate_assets(assets)
        return _store_portfolio_result(request, asset_df, warnings + validation_warnings)
    except PortfolioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/csv")
async def parse_csv_portfolio(request: Request, file: UploadFile = File(...)):
    """Parse uploaded CSV/TSV data into normalized asset rows."""
    try:
        contents = await file.read()
        separator = "\t" if file.filename and file.filename.endswith(".tsv") else ","
        df = pd.read_csv(BytesIO(contents), sep=separator)
        assets, warnings = parse_csv_to_assets(df)
        asset_df, validation_warnings = normalize_and_validate_assets(assets)
        return _store_portfolio_result(request, asset_df, warnings + validation_warnings)
    except PortfolioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"CSV 파일 처리 중 오류 발생: {exc}"
        ) from exc

