"""포트폴리오 입력 API 라우터."""

import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from middleware.session import session_manager
from services.portfolio_service import (
    PortfolioInputError,
    normalize_and_validate_assets,
    parse_csv_to_assets,
    parse_manual_edit_to_assets,
    parse_text_to_assets_service,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/parse-text", response_class=HTMLResponse)
async def parse_text(
    request: Request,
    text: str = Form(...),
):
    """텍스트를 파싱하여 자산 목록을 생성합니다.

    Args:
        request: FastAPI 요청 객체
        text: 파싱할 텍스트

    Returns:
        HTML 부분 (asset_table.html)
    """
    session_id = request.state.session_id

    try:
        assets, warnings = parse_text_to_assets_service(text)
        if not assets:
            return templates.TemplateResponse(
                "partials/error_message.html",
                {
                    "request": request,
                    "message": "파싱된 자산이 없습니다. 입력 형식을 확인해주세요.",
                },
            )

        asset_df, _ = normalize_and_validate_assets(assets)

        # 세션에 저장
        session_manager.set(session_id, "assets", [a.model_dump() for a in assets])
        session_manager.set(session_id, "asset_df", asset_df.to_dict(orient="records"))

        return templates.TemplateResponse(
            "partials/asset_table.html",
            {
                "request": request,
                "asset_df": asset_df.to_dict(orient="records"),
                "warnings": warnings,
            },
        )
    except PortfolioInputError as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": str(e),
            },
        )


@router.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
):
    """CSV 파일을 업로드하여 자산 목록을 생성합니다.

    Args:
        request: FastAPI 요청 객체
        file: 업로드된 CSV 파일

    Returns:
        HTML 부분 (asset_table.html)
    """
    session_id = request.state.session_id

    try:
        # CSV 파일 읽기
        from io import BytesIO

        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        assets, warnings = parse_csv_to_assets(df)
        if not assets:
            return templates.TemplateResponse(
                "partials/error_message.html",
                {
                    "request": request,
                    "message": "CSV에서 파싱된 자산이 없습니다.",
                },
            )

        asset_df, _ = normalize_and_validate_assets(assets)

        # 세션에 저장
        session_manager.set(session_id, "assets", [a.model_dump() for a in assets])
        session_manager.set(session_id, "asset_df", asset_df.to_dict(orient="records"))

        return templates.TemplateResponse(
            "partials/asset_table.html",
            {
                "request": request,
                "asset_df": asset_df.to_dict(orient="records"),
                "warnings": warnings,
            },
        )
    except PortfolioInputError as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": str(e),
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": f"CSV 파일 처리 중 오류 발생: {e}",
            },
        )


@router.post("/manual-edit", response_class=HTMLResponse)
async def manual_edit(
    request: Request,
    edited_data: str = Form(...),  # JSON 문자열
):
    """수동 편집 데이터를 처리합니다.

    Args:
        request: FastAPI 요청 객체
        edited_data: 편집된 데이터 (JSON 문자열)

    Returns:
        HTML 부분 (asset_table.html)
    """
    session_id = request.state.session_id

    try:
        import json

        data = json.loads(edited_data)
        assets, warnings = parse_manual_edit_to_assets(data)
        if not assets:
            return templates.TemplateResponse(
                "partials/error_message.html",
                {
                    "request": request,
                    "message": "편집된 자산이 없습니다.",
                },
            )

        asset_df, _ = normalize_and_validate_assets(assets)

        # 세션에 저장
        session_manager.set(session_id, "assets", [a.model_dump() for a in assets])
        session_manager.set(session_id, "asset_df", asset_df.to_dict(orient="records"))

        return templates.TemplateResponse(
            "partials/asset_table.html",
            {
                "request": request,
                "asset_df": asset_df.to_dict(orient="records"),
                "warnings": warnings,
            },
        )
    except PortfolioInputError as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": str(e),
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": f"데이터 처리 중 오류 발생: {e}",
            },
        )
