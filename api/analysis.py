"""데이터 분석 API 라우터."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from middleware.session import session_manager
from services.analysis_service import AnalysisError, run_analysis

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/run", response_class=HTMLResponse)
async def run_analysis_endpoint(
    request: Request,
    period: str = Form(...),
    rf: float = Form(...),
    bench: str = Form(...),
    momentum_weight: float = Form(0.2),
):
    """데이터 분석을 실행합니다.

    Args:
        request: FastAPI 요청 객체
        period: 평가 기간 (정수 문자열 또는 'YTD', 'Max')
        rf: 무위험 수익률 (연간, 소수)
        bench: 벤치마크 티커
        momentum_weight: 모멘텀 가중치

    Returns:
        HTML 부분 (metrics_table.html, quadrant_chart.html)
    """
    session_id = request.state.session_id

    # 세션에서 asset_df 가져오기
    asset_df_data = session_manager.get(session_id, "asset_df")
    if asset_df_data is None:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": "먼저 포트폴리오를 입력해주세요.",
            },
        )

    # DataFrame 재구성
    import pandas as pd

    asset_df = pd.DataFrame(asset_df_data)

    # period 파싱
    try:
        if period.isdigit():
            period_int = int(period)
        elif period == "YTD":
            period_int = "YTD"
        elif period == "Max":
            period_int = "Max"
        else:
            period_int = int(period)
    except ValueError:
        period_int = 12  # 기본값

    try:
        result = run_analysis(asset_df, period_int, rf, bench, momentum_weight)

        # 세션에 저장 (DataFrame은 인덱스를 포함하여 저장)
        metrics_df_dict = result.metrics_df.reset_index().to_dict(orient="records")
        session_manager.set(
            session_id, "prices", result.prices.reset_index().to_dict(orient="records")
        )
        session_manager.set(
            session_id,
            "returns",
            result.returns.reset_index().to_dict(orient="records"),
        )
        session_manager.set(
            session_id,
            "returns_smooth",
            result.returns_smooth.reset_index().to_dict(orient="records"),
        )
        session_manager.set(
            session_id, "weights_no_bench", result.weights_no_bench.to_dict()
        )
        session_manager.set(session_id, "metrics_df", metrics_df_dict)
        session_manager.set(session_id, "portfolio_metrics", result.portfolio_metrics)
        session_manager.set(session_id, "benchmark_metrics", result.benchmark_metrics)
        session_manager.set(
            session_id,
            "analysis_params",
            {
                "period": period_int,
                "rf": rf,
                "bench": bench,
                "momentum_weight": momentum_weight,
            },
        )

        # AIDEV-FIXME: ticker-column-missing; 템플릿 렌더링 시 인덱스를 컬럼으로 변환하여 티커 표시
        return templates.TemplateResponse(
            "partials/analysis_results.html",
            {
                "request": request,
                "metrics_df": result.metrics_df.reset_index().to_dict(orient="records"),
                "portfolio_metrics": result.portfolio_metrics,
                "benchmark_metrics": result.benchmark_metrics,
                "missing_tickers": result.missing_tickers,
            },
        )
    except AnalysisError as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "message": str(e),
            },
        )
