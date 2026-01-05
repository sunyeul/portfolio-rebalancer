"""FastAPI 메인 애플리케이션.

# AIDEV-NOTE: fastapi-entry-point; Streamlit에서 FastAPI로 전환
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api import analysis, evaluation, portfolio
from middleware.session import session_manager

app = FastAPI(title="포트폴리오 리밸런서", version="0.1.0")

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 라우터 등록
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """세션 미들웨어: 요청에 세션 ID 추가."""
    # 쿠키에서 세션 ID 읽기
    session_id_signed = request.cookies.get("session_id")
    if session_id_signed:
        session_id = session_manager.unsign_session_id(session_id_signed)
    else:
        session_id = None

    # 세션 ID가 없으면 새로 생성
    if not session_id:
        session_id = session_manager.create_session_id()

    # 요청 상태에 세션 ID 추가
    request.state.session_id = session_id

    # 응답 처리
    response = await call_next(request)

    # 세션 ID를 쿠키에 설정
    signed_session_id = session_manager.sign_session_id(session_id)
    response.set_cookie(
        key="session_id",
        value=signed_session_id,
        max_age=1800,  # 30분
        httponly=True,
        samesite="lax",
    )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
