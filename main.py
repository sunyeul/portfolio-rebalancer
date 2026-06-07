"""FastAPI entrypoint for the portfolio rebalancer JSON API."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.v1.router import router as api_v1_router
from middleware.session import session_manager

app = FastAPI(title="포트폴리오 리밸런서", version="0.1.0")
app.include_router(api_v1_router)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Attach a signed session id cookie to every request."""
    session_id_signed = request.cookies.get("session_id")
    session_id = (
        session_manager.unsign_session_id(session_id_signed)
        if session_id_signed
        else None
    )
    if not session_id:
        session_id = session_manager.create_session_id()

    request.state.session_id = session_id
    response = await call_next(request)
    response.set_cookie(
        key="session_id",
        value=session_manager.sign_session_id(session_id),
        max_age=1800,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/{path:path}", include_in_schema=False)
async def react_app(path: str):
    """Serve the React app in production builds."""
    index_file = FRONTEND_DIST / "index.html"
    requested_file = FRONTEND_DIST / path
    if path and requested_file.is_file():
        return FileResponse(requested_file)
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend build not found. Run `bun run build` in frontend/."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
