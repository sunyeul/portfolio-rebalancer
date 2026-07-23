"""Agent-oriented Toss Securities inspection CLI."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from integrations.toss.auth import TossAuthorizedReader, TossTokenProvider
from integrations.toss.config import TossApiConfig
from integrations.toss.observation import TossObservationService
from integrations.toss.market import TossMarketDataService
from integrations.toss.transport import TossTransport
from services.account_projection import build_account_projection
from services.inspection_service import preview_inspection, run_inspection
from services.market_context import evaluate_market_context
from storage.account_observation_store import (
    get_snapshot as get_account_snapshot,
    list_snapshots as list_account_snapshots,
)
from storage.database import initialize_database
from storage.evaluation_store import get_evaluation_run, latest_evaluation_run
from storage.instrument_profile_store import upsert_profile
from storage.policy_store import (
    activate_policy,
    get_active_policy,
    list_observed_identities,
    policy_template,
)
from storage.market_store import (
    insert_candles,
    insert_policy_candidate,
    latest_policy_candidate,
    list_candles,
)
from storage.performance_store import (
    append_cash_flow_decision,
    create_baseline,
    get_performance_run,
    latest_performance_run,
    preview_baseline,
    refresh_performance,
)


app = typer.Typer(
    help="IPS Pilot Toss Securities inspection CLI.",
    no_args_is_help=True,
)
performance_app = typer.Typer(help="Inspect Toss account performance history.")
profiles_app = typer.Typer(help="Manage Toss instrument IPS profiles.")
policy_app = typer.Typer(help="Inspect and activate versioned Toss IPS policies.")
inspection_app = typer.Typer(help="Run and inspect deterministic Toss evaluations.")
market_app = typer.Typer(help="Inspect official Toss market context.")
app.add_typer(performance_app, name="performance")
app.add_typer(profiles_app, name="profiles")
app.add_typer(policy_app, name="policy")
app.add_typer(inspection_app, name="inspection")
app.add_typer(market_app, name="market")


SUPPORTED_INSPECTION_ENGINE_VERSION = "phase5-v2"


def _contract_supported(evaluation: dict[str, Any] | None) -> bool:
    """Gate consumers on the approved v2 evaluation contract only."""
    if not isinstance(evaluation, dict):
        return False
    return evaluation.get("engine_version") == SUPPORTED_INSPECTION_ENGINE_VERSION


class CliError(Exception):
    """Raised for user-facing CLI errors that should become JSON."""

    def __init__(self, stage: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.hint = hint


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _emit_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))


def _exit_with_command_error(command: str, exc: Exception) -> None:
    if isinstance(exc, CliError):
        error = {"stage": exc.stage, "message": exc.message, "hint": exc.hint}
    else:
        error = {"stage": "unexpected", "message": str(exc), "hint": None}
    _emit_json({"ok": False, "command": command, "error": error})
    raise typer.Exit(code=1)


def _build_toss_service() -> tuple[TossObservationService, TossTransport]:
    config = TossApiConfig.from_env()
    transport = TossTransport(config)
    tokens = TossTokenProvider(config, transport)
    reader = TossAuthorizedReader(config, transport, tokens)
    return TossObservationService(config, reader), transport


def _build_toss_market_service() -> tuple[TossMarketDataService, TossTransport]:
    config = TossApiConfig.from_env()
    transport = TossTransport(config)
    tokens = TossTokenProvider(config, transport)
    reader = TossAuthorizedReader(config, transport, tokens)
    return TossMarketDataService(reader), transport


@app.command("toss-health")
def toss_health() -> None:
    """Check Toss config, OAuth, account discovery, and account match as JSON."""
    transport: TossTransport | None = None
    try:
        service, transport = _build_toss_service()
        _emit_json({"ok": True, "command": "toss-health", **service.health()})
    except Exception as exc:
        _exit_with_command_error("toss-health", exc)
    finally:
        if transport is not None:
            transport.close()


@app.command("toss-sync")
def toss_sync(
    from_date: Annotated[str | None, typer.Option("--from")] = None,
    to_date: Annotated[str | None, typer.Option("--to")] = None,
    max_order_pages: Annotated[int, typer.Option("--max-order-pages")] = 100,
) -> None:
    """Read Toss observations and persist one immutable snapshot as JSON."""
    transport: TossTransport | None = None
    try:
        initialize_database()
        service, transport = _build_toss_service()
        snapshot = service.sync(
            from_date=from_date,
            to_date=to_date,
            max_order_pages=max_order_pages,
        )
        _emit_json(
            {
                "ok": True,
                "command": "toss-sync",
                "snapshot_id": snapshot["id"],
                "state": snapshot["state"],
                "snapshot": snapshot,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("toss-sync", exc)
    finally:
        if transport is not None:
            transport.close()


@app.command("toss-snapshots")
def toss_snapshots(
    latest: Annotated[bool, typer.Option("--latest")] = False,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Inspect locally persisted Toss snapshots without contacting Toss."""
    try:
        initialize_database()
        if latest and snapshot_id is not None:
            raise CliError(
                "input",
                "--latest와 --snapshot-id는 함께 사용할 수 없습니다.",
            )
        if latest:
            snapshots = list_account_snapshots(limit=100)
            payload: Any = snapshots[0] if snapshots else None
        elif snapshot_id is not None:
            payload = get_account_snapshot(snapshot_id)
            if payload is None:
                raise CliError(
                    "persistence", f"snapshot_id={snapshot_id}를 찾을 수 없습니다."
                )
        else:
            payload = list_account_snapshots(limit)
        _emit_json(
            {
                "ok": True,
                "command": "toss-snapshots",
                "latest": latest,
                "snapshot_id": snapshot_id,
                "snapshots": payload,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("toss-snapshots", exc)


@app.command("account-view")
def account_view(
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Project the latest or explicit complete Toss account snapshot."""
    try:
        initialize_database()
        projection = build_account_projection(snapshot_id=snapshot_id)
        _emit_json(
            {
                "ok": True,
                "command": "account-view",
                "projection": projection,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("account-view", exc)


@profiles_app.command("set")
def profiles_set(
    symbol: Annotated[str, typer.Option("--symbol")],
    market_country: Annotated[str, typer.Option("--market-country")],
    layer: Annotated[str, typer.Option("--layer")],
    thesis_status: Annotated[str, typer.Option("--thesis-status")],
    note: Annotated[str, typer.Option("--note")] = "",
    overlap_status: Annotated[str, typer.Option("--overlap-status")] = "unknown",
    management_burden_status: Annotated[
        str, typer.Option("--management-burden-status")
    ] = "unknown",
    holdability_status: Annotated[str, typer.Option("--holdability-status")] = "unknown",
    etf_substitution_status: Annotated[
        str, typer.Option("--etf-substitution-status")
    ] = "unknown",
    review_factors_note: Annotated[
        str, typer.Option("--review-factors-note")
    ] = "",
) -> None:
    """Set IPS metadata for a previously observed Toss instrument."""
    try:
        initialize_database()
        profile = upsert_profile(
            symbol=symbol,
            market_country=market_country,
            layer=layer,
            thesis_status=thesis_status,
            thesis_note=note,
            overlap_status=overlap_status,
            management_burden_status=management_burden_status,
            holdability_status=holdability_status,
            etf_substitution_status=etf_substitution_status,
            review_factors_note=review_factors_note,
        )
        _emit_json(
            {
                "ok": True,
                "command": "profiles set",
                "profile": profile,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("profiles set", exc)


@profiles_app.command("list")
def profiles_list(
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Show current Toss holdings and their IPS classification coverage."""
    try:
        initialize_database()
        projection = build_account_projection(snapshot_id=snapshot_id)
        profiles = {
            (item["market_country"], item["symbol"]): item
            for item in projection["positions"]
            if item["layer"] is not None
        }
        _emit_json(
            {
                "ok": True,
                "command": "profiles list",
                "snapshot_id": projection["snapshot_id"],
                "profiles": [profiles[key] for key in sorted(profiles)],
                "unclassified": projection["unclassified"],
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("profiles list", exc)


@policy_app.command("show")
def policy_show(
    active: Annotated[bool, typer.Option("--active")] = False,
) -> None:
    """Show the active immutable Toss IPS policy."""
    try:
        initialize_database()
        if not active:
            raise CliError("input", "--active가 필요합니다.")
        _emit_json(
            {
                "ok": True,
                "command": "policy show",
                "policy": get_active_policy(),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("policy show", exc)


@policy_app.command("template")
def policy_template_command(
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Build an IPS target template from observed Toss identities."""
    try:
        initialize_database()
        _emit_json(
            {
                "ok": True,
                "command": "policy template",
                "template": policy_template(snapshot_id),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("policy template", exc)


@policy_app.command("validate")
def policy_validate_command(
    file: Annotated[Path, typer.Option("--file")],
) -> None:
    """Validate an app-owned IPS policy file without changing state."""
    try:
        initialize_database()
        from services.policy_validation import policy_metadata, validate_policy

        if not file.is_file():
            raise CliError("input", f"정책 파일을 찾을 수 없습니다: {file}")
        payload = json.loads(file.read_text(encoding="utf-8"))
        normalized = validate_policy(payload, list_observed_identities())
        _emit_json(
            {
                "ok": True,
                "command": "policy validate",
                "policy": normalized,
                "metadata": policy_metadata(normalized),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("policy validate", exc)


@policy_app.command("activate")
def policy_activate_command(
    file: Annotated[Path, typer.Option("--file")],
    expected_current_version: Annotated[
        int, typer.Option("--expected-current-version")
    ],
) -> None:
    """Atomically activate one validated IPS policy version."""
    try:
        initialize_database()
        if not file.is_file():
            raise CliError("input", f"정책 파일을 찾을 수 없습니다: {file}")
        payload = json.loads(file.read_text(encoding="utf-8"))
        policy = (
            payload.get("policy", payload) if isinstance(payload, dict) else payload
        )
        activated = activate_policy(policy, expected_current_version)
        _emit_json(
            {
                "ok": True,
                "command": "policy activate",
                "policy": activated,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("policy activate", exc)


@inspection_app.command("preview")
def inspection_preview_command(
    policy_file: Annotated[Path, typer.Option("--policy-file")],
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Evaluate a proposed Toss IPS policy without changing local state."""
    try:
        if not policy_file.is_file():
            raise CliError("input", f"정책 파일을 찾을 수 없습니다: {policy_file}")
        initialize_database()
        from services.policy_validation import policy_metadata, validate_policy

        payload = json.loads(policy_file.read_text(encoding="utf-8"))
        policy = payload.get("policy", payload) if isinstance(payload, dict) else payload
        normalized = validate_policy(policy, list_observed_identities())
        preview = preview_inspection(normalized, snapshot_id=snapshot_id)
        _emit_json(
            {
                "ok": True,
                "command": "inspection preview",
                "persisted": False,
                "snapshot_id": preview["snapshot_id"],
                "policy": normalized,
                "metadata": policy_metadata(normalized),
                "evaluation": preview["evaluation"],
                "preview": preview,
                "contract_supported": _contract_supported(preview.get("evaluation")),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("inspection preview", exc)


@inspection_app.command("run")
def inspection_run_command(
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Run or reuse one deterministic Toss inspection evaluation."""
    try:
        evaluation = run_inspection(snapshot_id=snapshot_id)
        _emit_json(
            {
                "ok": True,
                "command": "inspection run",
                "run_id": evaluation["id"],
                "snapshot_id": evaluation["snapshot_id"],
                "state": evaluation["state"],
                "evaluation": evaluation,
                "contract_supported": _contract_supported(evaluation),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("inspection run", exc)


@inspection_app.command("show")
def inspection_show_command(
    run_id: Annotated[int | None, typer.Option("--run-id")] = None,
    latest: Annotated[bool, typer.Option("--latest")] = False,
) -> None:
    """Show a persisted Toss inspection evaluation."""
    try:
        initialize_database()
        if (run_id is None) == (not latest):
            raise CliError("input", "--run-id 또는 --latest 중 하나를 지정해야 합니다.")
        evaluation = (
            latest_evaluation_run() if latest else get_evaluation_run(int(run_id))
        )
        if evaluation is None:
            raise CliError("persistence", "평가 실행을 찾을 수 없습니다.")
        _emit_json(
            {
                "ok": True,
                "command": "inspection show",
                "run_id": evaluation["id"],
                "snapshot_id": evaluation["snapshot_id"],
                "performance_run_id": evaluation["performance_run_id"],
                "policy_version_id": evaluation["policy_version_id"],
                "evaluation": evaluation,
                "contract_supported": _contract_supported(evaluation),
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("inspection show", exc)


@market_app.command("sync")
def market_sync_command(
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    max_pages: Annotated[int, typer.Option("--max-pages")] = 5,
) -> None:
    """Read official Toss daily candles and persist immutable observations."""
    transport: TossTransport | None = None
    try:
        initialize_database()
        service, transport = _build_toss_market_service()
        identities = list_observed_identities()
        requested = [
            item.strip() for item in (symbols or "").split(",") if item.strip()
        ]
        stock_symbols = requested or [
            f"{country}/{symbol}" for country, symbol in identities
        ]
        stored: list[dict[str, Any]] = []
        for identity in stock_symbols:
            if "/" in identity:
                market_country, symbol = identity.split("/", 1)
            else:
                market_country, symbol = "", identity
            candles = service.collect_history(
                symbol=symbol,
                market_country=market_country,
                source_kind="stock",
                max_pages=max_pages,
            )
            stored.extend(insert_candles(item.as_dict() for item in candles))
        benchmark = service.collect_history(
            symbol="KOSPI",
            source_kind="market_indicator",
            market_country="KR",
            max_pages=max_pages,
        )
        stored.extend(insert_candles(item.as_dict() for item in benchmark))
        active = get_active_policy()
        context = None
        candidate = None
        if active is not None:
            context = evaluate_market_context(
                list_candles(
                    source_kind="market_indicator",
                    market_country="KR",
                    symbol="KOSPI",
                ),
                current_target=float(
                    active["policy"].get("cash_reserve", {}).get("target", 0.15)
                ),
                last_change_at=active.get("created_at"),
            )
            if context.get("candidate_state") == "candidate":
                candidate = insert_policy_candidate(
                    {
                        "account_alias": active["account_alias"],
                        "base_policy_version_id": active["id"],
                        "candidate_json": {"benchmark": "KR/KOSPI", "context": context},
                    }
                )
        _emit_json(
            {
                "ok": True,
                "command": "market sync",
                "symbols": stock_symbols + ["KR/KOSPI"],
                "candle_count": len(stored),
                "context": context,
                "candidate": candidate,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("market sync", exc)
    finally:
        if transport is not None:
            transport.close()


@market_app.command("context")
def market_context_command(
    benchmark: Annotated[str, typer.Option("--benchmark")] = "KOSPI",
) -> None:
    """Evaluate stored Toss market context without activating a policy."""
    try:
        initialize_database()
        active = get_active_policy()
        if active is None:
            raise CliError("persistence", "활성 정책을 찾을 수 없습니다.")
        policy = active["policy"]
        current_target = float(policy.get("cash_reserve", {}).get("target", 0.15))
        candles = list_candles(
            source_kind="market_indicator", market_country="KR", symbol=benchmark
        )
        context = evaluate_market_context(
            candles,
            current_target=current_target,
            last_change_at=active.get("created_at"),
        )
        candidate = None
        if context.get("candidate_state") == "candidate":
            candidate = insert_policy_candidate(
                {
                    "account_alias": active["account_alias"],
                    "base_policy_version_id": active["id"],
                    "candidate_json": {
                        "benchmark": f"KR/{benchmark.upper()}",
                        "context": context,
                    },
                }
            )
        _emit_json(
            {
                "ok": True,
                "command": "market context",
                "benchmark": f"KR/{benchmark.upper()}",
                "policy_version_id": active["id"],
                "context": context,
                "candidate": candidate,
                "latest_candidate": latest_policy_candidate(
                    active["account_alias"], active["id"]
                ),
                "activation": "human approval required; active policy unchanged",
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("market context", exc)


@app.command("web")
def web_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Serve the read-only Toss dashboard API and built frontend on loopback."""
    try:
        if host not in {"127.0.0.1", "localhost"}:
            raise CliError("input", "web은 loopback 주소만 허용합니다.")
        import uvicorn

        _emit_json(
            {
                "ok": True,
                "command": "web",
                "host": host,
                "port": port,
                "error": None,
            }
        )
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_config=None,
            access_log=False,
        )
    except Exception as exc:
        _exit_with_command_error("web", exc)


@performance_app.command("baseline-preview")
def performance_baseline_preview(
    snapshot_id: Annotated[int, typer.Option("--snapshot-id")],
) -> None:
    """Preview one complete snapshot as a possible performance baseline."""
    try:
        initialize_database()
        payload = preview_baseline(snapshot_id)
        if payload is None:
            raise CliError(
                "persistence", f"snapshot_id={snapshot_id}를 찾을 수 없습니다."
            )
        _emit_json(
            {
                "ok": True,
                "command": "performance baseline-preview",
                "snapshot": payload,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance baseline-preview", exc)


@performance_app.command("baseline-confirm")
def performance_baseline_confirm(
    snapshot_id: Annotated[int, typer.Option("--snapshot-id")],
    expected_principal_krw: Annotated[float, typer.Option("--expected-principal-krw")],
) -> None:
    """Confirm one immutable account performance tracking baseline."""
    try:
        initialize_database()
        baseline = create_baseline(snapshot_id, expected_principal_krw)
        _emit_json(
            {
                "ok": True,
                "command": "performance baseline-confirm",
                "baseline": baseline,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance baseline-confirm", exc)


@performance_app.command("refresh")
def performance_refresh_command() -> None:
    """Build and persist one immutable local performance projection."""
    try:
        initialize_database()
        run = refresh_performance()
        _emit_json(
            {
                "ok": True,
                "command": "performance refresh",
                "run_id": run["id"],
                "state": run["state"],
                "run": run,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance refresh", exc)


@performance_app.command("candidates")
def performance_candidates(
    run_id: Annotated[int | None, typer.Option("--run-id")] = None,
) -> None:
    """List cash-flow candidates attached to a performance run."""
    try:
        initialize_database()
        run = (
            get_performance_run(run_id)
            if run_id is not None
            else latest_performance_run()
        )
        if run is None:
            raise CliError("persistence", "성과 실행을 찾을 수 없습니다.")
        _emit_json(
            {
                "ok": True,
                "command": "performance candidates",
                "run_id": run["id"],
                "candidates": run["candidates"],
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance candidates", exc)


@performance_app.command("decide-flow")
def performance_decide_flow(
    candidate_id: Annotated[int, typer.Option("--candidate-id")],
    classification: Annotated[str, typer.Option("--classification")],
    amount_native: Annotated[float | None, typer.Option("--amount-native")] = None,
    effective_at: Annotated[str | None, typer.Option("--effective-at")] = None,
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """Append a user decision for one cash-flow candidate."""
    try:
        initialize_database()
        decision = append_cash_flow_decision(
            candidate_id,
            classification=classification,
            confirmed_amount_native=amount_native,
            effective_at=effective_at,
            note=note,
        )
        _emit_json(
            {
                "ok": True,
                "command": "performance decide-flow",
                "decision": decision,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance decide-flow", exc)


@performance_app.command("history")
def performance_history(
    latest: Annotated[bool, typer.Option("--latest")] = False,
    run_id: Annotated[int | None, typer.Option("--run-id")] = None,
) -> None:
    """Inspect one local account performance run."""
    try:
        initialize_database()
        if latest and run_id is not None:
            raise CliError("input", "--latest와 --run-id는 함께 사용할 수 없습니다.")
        run = (
            latest_performance_run()
            if latest or run_id is None
            else get_performance_run(run_id)
        )
        if run is None:
            raise CliError("persistence", "성과 실행을 찾을 수 없습니다.")
        _emit_json(
            {
                "ok": True,
                "command": "performance history",
                "latest": latest,
                "run_id": run["id"],
                "run": run,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("performance history", exc)


if __name__ == "__main__":
    sys.exit(app())
