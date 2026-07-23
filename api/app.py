"""Loopback-only read API over the Toss inspection services."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import ApiEnvelope
from services.account_projection import AccountProjectionError, build_account_projection
from storage.account_observation_store import (
    latest_verified_complete,
    list_snapshots,
)
from storage.database import initialize_database
from storage.evaluation_store import latest_evaluation_run
from storage.instrument_profile_store import list_profiles
from storage.performance_store import latest_performance_run
from storage.policy_store import get_active_policy


def _error(message: str, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "data": None, "error": {"message": message}},
        status_code=status_code,
    )


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Expose source health without browser-visible order or execution facts."""
    return {
        "id": snapshot["id"],
        "account_alias": snapshot["account_alias"],
        "sync_started_at": snapshot["sync_started_at"],
        "synced_at": snapshot["synced_at"],
        "state": snapshot["state"],
        "is_current_evaluable": snapshot["is_current_evaluable"],
        "source_fingerprint": snapshot["source_fingerprint"],
        "source_timestamps": snapshot["source_timestamps"],
        "data_quality": snapshot["data_quality"],
        "reconciliation": snapshot["reconciliation"],
        "total_value_krw": snapshot["total_value_krw"],
        "invested_value_krw": snapshot["invested_value_krw"],
        "cash_value_krw": snapshot["cash_value_krw"],
        "holding_count": len(snapshot["holdings"]),
        "cash_currency_count": len(snapshot["cash"]),
    }


def _projection_view(projection: dict[str, Any]) -> dict[str, Any]:
    """Keep account facts useful to the dashboard without holding-size fields."""
    visible = dict(projection)
    visible["positions"] = [
        {key: value for key, value in position.items() if key != "quantity"}
        for position in projection.get("positions", [])
    ]
    return visible


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        initialize_database()
        yield

    app = FastAPI(
        title="IPS Pilot Toss API",
        version="0.1.0",
        docs_url="/api/docs",
        lifespan=lifespan,
    )
    session_secret = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def session_cookie(request: Request, call_next):
        try:
            initialize_database()
        except Exception as exc:
            return _error(str(exc), status_code=503)
        response = await call_next(request)
        if request.cookies.get("ips_pilot_session") != session_secret:
            response.set_cookie(
                "ips_pilot_session",
                session_secret,
                httponly=True,
                samesite="strict",
                secure=False,
                path="/",
            )
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(_, exc: Exception) -> JSONResponse:
        return _error(str(exc), status_code=500)

    @app.get("/api/health", response_model=ApiEnvelope)
    async def health() -> dict[str, Any]:
        try:
            initialize_database()
            snapshots = list_snapshots(limit=1)
            verified = latest_verified_complete()
            return {
                "ok": True,
                "data": {
                    "account_alias": "toss-brokerage",
                    "latest_attempt": _snapshot_summary(snapshots[0])
                    if snapshots
                    else None,
                    "last_verified_complete": _snapshot_summary(verified)
                    if verified
                    else None,
                },
                "error": None,
            }
        except Exception as exc:
            return {"ok": False, "data": None, "error": {"message": str(exc)}}

    @app.get("/api/snapshots", response_model=ApiEnvelope)
    async def snapshots(limit: int = 20) -> dict[str, Any]:
        return {
            "ok": True,
            "data": {
                "snapshots": [_snapshot_summary(item) for item in list_snapshots(limit)]
            },
            "error": None,
        }

    @app.get("/api/account", response_model=ApiEnvelope)
    async def account(snapshot_id: int | None = None) -> Any:
        try:
            return {
                "ok": True,
                "data": {
                    "projection": _projection_view(
                        build_account_projection(snapshot_id=snapshot_id)
                    )
                },
                "error": None,
            }
        except AccountProjectionError as exc:
            return _error(str(exc))

    @app.get("/api/performance", response_model=ApiEnvelope)
    async def performance() -> dict[str, Any]:
        return {"ok": True, "data": {"run": latest_performance_run()}, "error": None}

    @app.get("/api/policy", response_model=ApiEnvelope)
    async def policy() -> dict[str, Any]:
        return {"ok": True, "data": {"policy": get_active_policy()}, "error": None}

    @app.get("/api/profiles", response_model=ApiEnvelope)
    async def profiles() -> dict[str, Any]:
        return {"ok": True, "data": {"profiles": list_profiles()}, "error": None}

    @app.get("/api/inspection", response_model=ApiEnvelope)
    async def inspection() -> dict[str, Any]:
        return {
            "ok": True,
            "data": {"evaluation": latest_evaluation_run()},
            "error": None,
        }

    @app.get("/api/review-queue", response_model=ApiEnvelope)
    async def review_queue() -> dict[str, Any]:
        evaluation = latest_evaluation_run()
        return {
            "ok": True,
            "data": {
                "items": (evaluation or {}).get("result", {}).get("review_queue", []),
                "run_id": evaluation["id"] if evaluation else None,
            },
            "error": None,
        }

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
