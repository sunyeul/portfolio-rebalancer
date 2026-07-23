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
from storage.market_store import latest_policy_candidate, list_candles
from storage.performance_store import get_performance_run, latest_performance_run
from storage.policy_store import get_active_policy, get_policy_version
from services.market_context import evaluate_market_context


SUPPORTED_INSPECTION_ENGINE_VERSION = "phase5-v2"


def _contract_supported(evaluation: dict[str, Any] | None) -> bool:
    """Return whether an evaluation uses the approved v2 result contract.

    Persisted runs keep the engine version on the wrapper while preview-style
    payloads may expose the result directly.  Both forms are checked strictly;
    no historical payload is adapted into the v2 action vocabulary.
    """
    if not isinstance(evaluation, dict):
        return False
    if evaluation.get("engine_version") == SUPPORTED_INSPECTION_ENGINE_VERSION:
        return True
    result = evaluation.get("result")
    return isinstance(result, dict) and result.get(
        "engine_version"
    ) == SUPPORTED_INSPECTION_ENGINE_VERSION


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
    async def performance(run_id: int | None = None) -> dict[str, Any]:
        run = (
            get_performance_run(run_id)
            if run_id is not None
            else latest_performance_run()
        )
        return {"ok": True, "data": {"run": run}, "error": None}

    @app.get("/api/policy", response_model=ApiEnvelope)
    async def policy(version_id: int | None = None) -> dict[str, Any]:
        selected = (
            get_policy_version(version_id)
            if version_id is not None
            else get_active_policy()
        )
        return {"ok": True, "data": {"policy": selected}, "error": None}

    @app.get("/api/profiles", response_model=ApiEnvelope)
    async def profiles() -> dict[str, Any]:
        return {"ok": True, "data": {"profiles": list_profiles()}, "error": None}

    @app.get("/api/inspection", response_model=ApiEnvelope)
    async def inspection() -> dict[str, Any]:
        evaluation = _evaluation_view(latest_evaluation_run())
        return {
            "ok": True,
            "data": {
                "evaluation": evaluation,
                "contract_supported": _contract_supported(evaluation),
            },
            "error": None,
        }

    @app.get("/api/review-queue", response_model=ApiEnvelope)
    async def review_queue() -> dict[str, Any]:
        evaluation = latest_evaluation_run()
        contract_supported = _contract_supported(evaluation)
        result = (evaluation or {}).get("result", {})
        data: dict[str, Any] = {
            "items": result.get("review_queue", []),
            "run_id": evaluation["id"] if evaluation else None,
            "contract_supported": contract_supported,
        }
        if contract_supported:
            data["adjustment_suggestions"] = result.get(
                "adjustment_suggestions", []
            )
        return {
            "ok": True,
            "data": data,
            "error": None,
        }

    @app.get("/api/market-context", response_model=ApiEnvelope)
    async def market_context() -> dict[str, Any]:
        try:
            active = get_active_policy()
            if active is None:
                return _error("활성 정책을 찾을 수 없습니다.")
            cash_policy = active["policy"].get("cash_reserve", {})
            candles = list_candles(
                source_kind="market_indicator", market_country="KR", symbol="KOSPI"
            )
            context = evaluate_market_context(
                candles,
                current_target=float(cash_policy.get("target", 0.15)),
                last_change_at=active.get("created_at"),
            )
            return {
                "ok": True,
                "data": {
                    "benchmark": "KR/KOSPI",
                    "policy_version_id": active["id"],
                    "context": context,
                    "latest_candidate": latest_policy_candidate(
                        active["account_alias"], active["id"]
                    ),
                    "activation": "human approval required; active policy unchanged",
                },
                "error": None,
            }
        except Exception as exc:
            return _error(str(exc))

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
