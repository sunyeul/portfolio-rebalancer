"""평가 및 제안 API 라우터."""

import json

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from middleware.session import session_manager
from services.evaluation_service import run_evaluation, EvaluationError

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/run", response_class=HTMLResponse)
async def run_evaluation_endpoint(
    request: Request,
    rc_over_thresh_pct: float = Form(...),
    e_thresh: float = Form(...),
    target_weights: str = Form(None),  # JSON 문자열
):
    """평가 및 제안을 실행합니다.

    Args:
        request: FastAPI 요청 객체
        rc_over_thresh_pct: RC_Over 임계값 (%)
        e_thresh: 효율 점수 E′ 임계값
        target_weights: 목표 가중치 (JSON 문자열, None이면 현재 가중치 사용)

    Returns:
        HTML 부분 (action_plan.html, quadrant_chart.html)
    """
    session_id = request.state.session_id

    # 세션에서 metrics_df 가져오기
    metrics_df_data = session_manager.get(session_id, "metrics_df")
    if metrics_df_data is None:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": "먼저 데이터 분석을 실행해주세요.",
            },
        )

    # DataFrame 재구성
    import pandas as pd

    metrics_df = pd.DataFrame(metrics_df_data)
    if "ticker" in metrics_df.columns:
        metrics_df = metrics_df.set_index("ticker")

    # 공분산 행렬 계산을 위한 returns_smooth 가져오기
    returns_smooth_data = session_manager.get(session_id, "returns_smooth")
    cov_matrix = None
    if returns_smooth_data is not None:
        from utils.metrics import annualize_cov

        returns_smooth_df = pd.DataFrame(returns_smooth_data)
        if "Date" in returns_smooth_df.columns or "date" in returns_smooth_df.columns:
            date_col = "Date" if "Date" in returns_smooth_df.columns else "date"
            returns_smooth_df = returns_smooth_df.set_index(date_col)
        # metrics_df의 티커와 일치하는 컬럼만 선택
        common_tickers = returns_smooth_df.columns.intersection(metrics_df.index)
        if len(common_tickers) > 0:
            returns_smooth_subset = returns_smooth_df[common_tickers]
            cov_matrix = annualize_cov(returns_smooth_subset)

    # target_weights 파싱
    target_weights_dict = None
    if target_weights:
        try:
            target_weights_dict = json.loads(target_weights)
        except json.JSONDecodeError:
            pass

    try:
        result = run_evaluation(
            metrics_df,
            target_weights_dict,
            rc_over_thresh_pct,
            e_thresh,
            cov_matrix=cov_matrix,
        )

        return templates.TemplateResponse(
            "partials/evaluation_results.html",
            {
                "request": request,
                "proposal_df": result.proposal_df.to_dict(orient="records"),
                "sell_list": result.sell_list.to_dict(orient="records"),
                "buy_list": result.buy_list.to_dict(orient="records"),
                "fine_tune_list": result.fine_tune_list.to_dict(orient="records"),
                "rc_violations": result.rc_violations.to_dict(orient="records"),
                "quadrant_chart_json": result.quadrant_chart_json,
            },
        )
    except EvaluationError as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": str(e),
            },
        )


@router.get("/download-csv")
async def download_csv(
    request: Request,
    type: str = "metrics",  # 'metrics' or 'proposal'
):
    """CSV 파일을 다운로드합니다.

    Args:
        request: FastAPI 요청 객체
        type: 다운로드 타입 ('metrics' 또는 'proposal')

    Returns:
        CSV 파일 응답
    """
    session_id = request.state.session_id

    if type == "metrics":
        df = session_manager.get_dataframe(session_id, "metrics_df")
        if df is None:
            raise HTTPException(status_code=404, detail="분석 결과가 없습니다.")
        filename = "enriched_metrics.csv"
    elif type == "proposal":
        # 평가 결과는 세션에 저장되지 않으므로 다시 계산 필요
        # 여기서는 간단히 에러 반환
        raise HTTPException(
            status_code=501, detail="제안 다운로드는 아직 구현되지 않았습니다."
        )
    else:
        raise HTTPException(status_code=400, detail="잘못된 타입입니다.")

    csv_content = df.to_csv(index=False)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
