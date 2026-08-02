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
from storage.account_observation_store import (
    latest_complete,
    latest_verified_complete,
    list_snapshots,
)
from storage.database import initialize_database
from storage.evaluation_store import list_evaluation_runs, latest_evaluation_run
from storage.performance_store import get_performance_run, latest_performance_run
from storage.policy_store import get_active_policy


SUPPORTED_INSPECTION_ENGINE_VERSION = "phase5-v2"


def _contract_supported(evaluation: dict[str, Any] | None) -> bool:
    """Return whether an evaluation uses the approved v2 result contract.

    API evaluation payloads are persisted wrappers, so only their top-level
    engine version is authoritative.  No nested result is adapted into a v2
    wrapper at read time.
    """
    if not isinstance(evaluation, dict):
        return False
    return evaluation.get("engine_version") == SUPPORTED_INSPECTION_ENGINE_VERSION


def _evaluation_currentness(
    evaluation: dict[str, Any] | None,
    current_snapshot: dict[str, Any] | None,
    active_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare one immutable evaluation with the current source contracts."""
    evaluation_snapshot_id = (
        evaluation.get("snapshot_id") if evaluation is not None else None
    )
    current_snapshot_id = (
        current_snapshot.get("id") if current_snapshot is not None else None
    )
    evaluation_policy_version_id = (
        evaluation.get("policy_version_id") if evaluation is not None else None
    )
    active_policy_version_id = (
        active_policy.get("id") if active_policy is not None else None
    )
    reasons: list[str] = []
    if evaluation is None:
        reasons.append("evaluation_missing")
    if current_snapshot is None:
        reasons.append("current_snapshot_missing")
    if active_policy is None:
        reasons.append("active_policy_missing")
    if (
        evaluation is not None
        and current_snapshot is not None
        and evaluation_snapshot_id != current_snapshot_id
    ):
        reasons.append("snapshot_mismatch")
    if (
        evaluation is not None
        and active_policy is not None
        and evaluation_policy_version_id != active_policy_version_id
    ):
        reasons.append("policy_version_mismatch")
    return {
        "is_current": not reasons,
        "reasons": reasons,
        "evaluation_snapshot_id": evaluation_snapshot_id,
        "current_snapshot_id": current_snapshot_id,
        "evaluation_policy_version_id": evaluation_policy_version_id,
        "active_policy_version_id": active_policy_version_id,
    }


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


_ACCOUNT_SUMMARY_KEYS = (
    "total_value_krw",
    "invested_value_krw",
    "cash_value_krw",
    "cash_weight_gross",
    "investment_principal_krw",
    "account_profit_krw",
    "account_return",
)


def _evaluation_view(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose the current account contract for evaluations persisted by older code."""
    if evaluation is None:
        return None
    visible = dict(evaluation)
    result = dict(evaluation.get("result") or {})
    account = dict(result.get("account") or {})
    performance_run_id = evaluation.get("performance_run_id")
    if performance_run_id is not None:
        performance_run = get_performance_run(int(performance_run_id))
        points = (
            performance_run.get("points", [])
            if isinstance(performance_run, dict)
            else []
        )
        evaluable_points = [
            point
            for point in points
            if isinstance(point, dict) and point.get("evaluation_state") == "evaluable"
        ]
        latest_point = max(
            evaluable_points,
            key=lambda point: (
                str(point.get("point_at") or ""),
                int(point.get("id") or 0),
            ),
            default=None,
        )
        if latest_point is not None:
            account.update(
                {
                    "total_value_krw": latest_point.get("total_value_krw"),
                    "invested_value_krw": latest_point.get("invested_value_krw"),
                    "cash_value_krw": latest_point.get("cash_value_krw"),
                    "investment_principal_krw": latest_point.get(
                        "investment_principal_krw"
                    ),
                    "account_profit_krw": latest_point.get("account_gain_krw"),
                    "account_return": latest_point.get("simple_return"),
                }
            )
    result["account"] = {
        key: account[key] for key in _ACCOUNT_SUMMARY_KEYS if key in account
    }
    visible["result"] = result
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
        evaluation = _evaluation_view(latest_evaluation_run())
        currentness = _evaluation_currentness(
            evaluation,
            latest_complete(),
            get_active_policy(),
        )
        return {
            "ok": True,
            "data": {
                "evaluation": evaluation,
                "contract_supported": _contract_supported(evaluation),
                "currentness": currentness,
            },
            "error": None,
        }

    @app.get("/api/change-brief", response_model=ApiEnvelope)
    async def change_brief() -> dict[str, Any]:
        evaluations = list_evaluation_runs(limit=2)
        current = evaluations[0] if evaluations else latest_evaluation_run()
        previous = evaluations[1] if len(evaluations) > 1 else None
        currentness = _evaluation_currentness(
            current,
            latest_complete(),
            get_active_policy(),
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
