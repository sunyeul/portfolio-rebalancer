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
from services.change_brief import build_change_brief
from services.currentness import evaluate_currentness
from services.policy_candidate_assessment import unavailable_policy_candidate_assessment
from storage.account_observation_store import (
    latest_verified_complete,
    list_snapshots,
)
from storage.database import initialize_database
from storage.evaluation_store import (
    ENGINE_VERSION,
    current_v2_result,
    list_evaluation_runs,
    latest_evaluation_run,
)
from storage.performance_store import get_performance_run, latest_performance_run
from storage.policy_store import get_active_policy


def _contract_supported(evaluation: dict[str, Any] | None) -> bool:
    """Return whether an evaluation has the approved persisted v2 result."""
    return (
        isinstance(evaluation, dict)
        and evaluation.get("engine_version") == ENGINE_VERSION
        and current_v2_result(evaluation.get("result"))
    )


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


def _latest_account_attempt() -> dict[str, Any] | None:
    """Select the newest Toss attempt for the same source-currentness rule."""
    snapshots = list_snapshots(limit=1, account_alias="toss-brokerage")
    return snapshots[0] if snapshots else None


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

    @app.get("/api/performance", response_model=ApiEnvelope)
    async def performance(run_id: int | None = None) -> dict[str, Any]:
        run = (
            get_performance_run(run_id)
            if run_id is not None
            else latest_performance_run()
        )
        return {"ok": True, "data": {"run": run}, "error": None}

    @app.get("/api/inspection", response_model=ApiEnvelope)
    async def inspection() -> dict[str, Any]:
        evaluation = latest_evaluation_run()
        active_policy = get_active_policy()
        currentness = evaluate_currentness(
            evaluation=evaluation,
            snapshot=_latest_account_attempt(),
            active_policy=active_policy,
            require_evaluation=True,
        )
        return {
            "ok": True,
            "data": {
                "evaluation": evaluation,
                "contract_supported": _contract_supported(evaluation),
                "currentness": currentness,
                "policy_candidate_assessment": unavailable_policy_candidate_assessment(
                    active_policy
                ),
            },
            "error": None,
        }

    @app.get("/api/change-brief", response_model=ApiEnvelope)
    async def change_brief() -> dict[str, Any]:
        evaluations = list_evaluation_runs(limit=2)
        current = evaluations[0] if evaluations else latest_evaluation_run()
        previous = evaluations[1] if len(evaluations) > 1 else None
        currentness = evaluate_currentness(
            evaluation=current,
            snapshot=_latest_account_attempt(),
            active_policy=get_active_policy(),
            require_evaluation=True,
        )
        brief = build_change_brief(current, previous)
        brief["currentness"] = currentness
        if current is not None and not currentness["is_current"]:
            brief.update(
                {
                    "state": "stale_evaluation",
                    "changes": [],
                    "source_alert": {
                        "state": "stale_evaluation",
                        "message": "저장된 평가가 현재 Toss 스냅샷·활성 정책과 일치하지 않습니다.",
                    },
                }
            )
        return {
            "ok": True,
            "data": brief,
            "error": None,
        }

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
