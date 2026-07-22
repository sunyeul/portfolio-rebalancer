"""Agent-oriented Typer CLI for IPS Pilot v2."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from services.analysis_service import DEFAULT_RF, AnalysisError, run_analysis
from services.evaluation_engine import run_evaluation
from services.evaluation_period import (
    EvaluationPeriodError,
    resolve_evaluation_period,
)
from services.evaluation_units import DEFAULT_LAYER_BENCHMARKS
from services.portfolio_service import (
    PortfolioInputError,
    normalize_and_validate_assets,
    parse_csv_to_assets,
    parse_text_to_assets_service,
)
from storage.database import db_path, initialize_database
from storage.portfolio_store import (
    StorageError,
    create_snapshot,
    get_current_state,
    get_snapshot,
    list_portfolios,
    list_snapshots,
    save_current_state,
)
from api.v1.serialization import safe_mapping
from integrations.toss.auth import TossAuthorizedReader, TossTokenProvider
from integrations.toss.config import TossApiConfig
from integrations.toss.observation import TossObservationService
from integrations.toss.transport import TossTransport
from storage.account_observation_store import (
    get_snapshot as get_account_snapshot,
    latest_complete as latest_account_snapshot,
    list_snapshots as list_account_snapshots,
)
from storage.performance_store import (
    append_cash_flow_decision,
    create_baseline,
    get_performance_run,
    latest_performance_run,
    preview_baseline,
    refresh_performance,
)


LAYER_BENCHMARK_LAYERS = ("core", "satellite", "experiment")
LAYER_BENCHMARK_HELP = "Layer benchmark override as layer=BENCHMARK. Repeat for core, satellite, or experiment."

app = typer.Typer(
    help="IPS Pilot CLI for Evaluation Framework v2.",
    no_args_is_help=True,
)
portfolios_app = typer.Typer(help="Inspect saved portfolios.")
snapshots_app = typer.Typer(help="Inspect saved portfolio snapshots.")
performance_app = typer.Typer(help="Inspect account performance history.")
app.add_typer(portfolios_app, name="portfolios")
app.add_typer(snapshots_app, name="snapshots")
app.add_typer(performance_app, name="performance")


class CliError(Exception):
    """Raised for user-facing CLI errors that should become JSON."""

    def __init__(self, stage: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.hint = hint


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return safe_mapping({"value": value})["value"]


def _emit_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))


def _empty_v2_payload(
    command: str, error: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "ok": error is None,
        "command": command,
        "input": None,
        "evaluation_period": None,
        "layer_evaluations": [],
        "asset_evaluations": [],
        "review_queue": [],
        "journal_draft": [],
        "warnings": [],
        "guardrails": {
            "not_investment_advice": True,
            "no_immediate_order_instruction": True,
        },
        "error": error,
    }


def _exit_with_error(command: str, exc: Exception) -> None:
    if isinstance(exc, CliError):
        error = {"stage": exc.stage, "message": exc.message, "hint": exc.hint}
    else:
        error = {
            "stage": "unexpected",
            "message": str(exc),
            "hint": "명령 옵션과 입력 데이터를 확인한 뒤 다시 시도하세요.",
        }
    _emit_json(_empty_v2_payload(command, error))
    raise typer.Exit(code=1)


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


def _selected_sources(*values: Any) -> int:
    return sum(value is not None for value in values)


def _parse_layer_benchmarks(values: list[str] | None) -> dict[str, str]:
    layer_benchmarks = DEFAULT_LAYER_BENCHMARKS.copy()
    seen: set[str] = set()
    for raw_value in values or []:
        if "=" not in raw_value:
            raise CliError(
                "input",
                "--layer-benchmark은 layer=BENCHMARK 형식이어야 합니다.",
                "예: --layer-benchmark core=SPY:80,QQQ:20",
            )
        layer, benchmark = raw_value.split("=", 1)
        layer = layer.strip().lower()
        benchmark = benchmark.strip().upper()
        if layer not in LAYER_BENCHMARK_LAYERS:
            raise CliError(
                "input",
                f"지원하지 않는 layer benchmark 계층입니다: {layer}",
                "core, satellite, experiment 중 하나를 사용하세요.",
            )
        if layer in seen:
            raise CliError(
                "input",
                f"--layer-benchmark {layer}=... 옵션이 중복되었습니다.",
                "계층별 벤치마크는 계층마다 한 번만 지정하세요.",
            )
        if not benchmark:
            raise CliError(
                "input",
                f"{layer} 계층의 벤치마크가 비어 있습니다.",
                "예: --layer-benchmark core=SPY:80,QQQ:20",
            )
        layer_benchmarks[layer] = benchmark
        seen.add(layer)
    return layer_benchmarks


def _load_asset_df(
    *,
    file_path: Path | None,
    text: str | None,
    portfolio_id: int | None,
    snapshot_id: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], int | None]:
    selected_count = _selected_sources(file_path, text, portfolio_id, snapshot_id)
    if selected_count != 1:
        raise CliError(
            "input",
            "입력은 --file, --text, --portfolio-id, --snapshot-id 중 정확히 하나만 지정해야 합니다.",
        )

    try:
        if file_path is not None:
            separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(file_path, sep=separator)
            assets, warnings = parse_csv_to_assets(df)
            asset_df, validation_warnings = normalize_and_validate_assets(assets)
            return (
                asset_df,
                warnings + validation_warnings,
                {"source": "file", "file": str(file_path)},
                None,
            )

        if text is not None:
            assets, warnings = parse_text_to_assets_service(text)
            asset_df, validation_warnings = normalize_and_validate_assets(assets)
            return asset_df, warnings + validation_warnings, {"source": "text"}, None

        initialize_database()
        if portfolio_id is not None:
            state = get_current_state(portfolio_id)
            if state is None:
                raise CliError(
                    "input",
                    f"portfolio_id={portfolio_id}의 current-state를 찾을 수 없습니다.",
                )
            return (
                pd.DataFrame(state["session_state"].get("asset_df") or []),
                [],
                {"source": "portfolio_current_state", "portfolio_id": portfolio_id},
                portfolio_id,
            )

        snapshot = get_snapshot(snapshot_id or 0)
        if snapshot is None:
            raise CliError("input", f"snapshot_id={snapshot_id}를 찾을 수 없습니다.")
        source_portfolio_id = int(snapshot["summary"]["portfolio_id"])
        return (
            pd.DataFrame(snapshot["session_state"].get("asset_df") or []),
            [],
            {
                "source": "snapshot",
                "snapshot_id": snapshot_id,
                "portfolio_id": source_portfolio_id,
            },
            source_portfolio_id,
        )
    except PortfolioInputError as exc:
        raise CliError("input", str(exc)) from exc
    except FileNotFoundError as exc:
        raise CliError("input", f"파일을 찾을 수 없습니다: {file_path}") from exc
    except OSError as exc:
        raise CliError("input", f"파일을 읽을 수 없습니다: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise CliError("input", f"CSV/TSV 파싱 실패: {exc}") from exc


def _save_run(
    *,
    portfolio_id: int,
    snapshot_name: str | None,
    note: str,
    session_data: dict[str, Any],
) -> dict[str, Any]:
    name = (
        snapshot_name or f"CLI evaluation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    snapshot = create_snapshot(portfolio_id, name, note, session_data)
    current_state = save_current_state(portfolio_id, session_data)
    return {
        "saved": True,
        "portfolio_id": portfolio_id,
        "snapshot_id": snapshot["id"],
        "snapshot_name": snapshot["name"],
        "current_state_updated_at": current_state["updated_at"],
    }


def _run_v2(
    *,
    command: str,
    file_path: Path | None,
    text: str | None,
    portfolio_id: int | None,
    snapshot_id: int | None,
    period: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    layer_benchmark: list[str] | None,
) -> tuple[dict[str, Any], pd.DataFrame, Any, dict[str, Any], int | None]:
    asset_df, warnings, input_meta, source_portfolio_id = _load_asset_df(
        file_path=file_path,
        text=text,
        portfolio_id=portfolio_id,
        snapshot_id=snapshot_id,
    )
    try:
        evaluation_period = resolve_evaluation_period(
            period=period,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
    except EvaluationPeriodError as exc:
        raise CliError("input", str(exc)) from exc

    layer_benchmarks = _parse_layer_benchmarks(layer_benchmark)
    bench_ticker = layer_benchmarks["core"]
    try:
        analysis = run_analysis(
            asset_df,
            evaluation_period,
            DEFAULT_RF,
            bench_ticker,
            extra_benchmarks=list(layer_benchmarks.values()),
        )
    except AnalysisError as exc:
        raise CliError("analysis", str(exc)) from exc

    result = run_evaluation(
        analysis=analysis,
        evaluation_period=evaluation_period,
        bench=bench_ticker,
        layer_benchmarks=layer_benchmarks,
    )
    payload = result.to_payload()
    payload.update(
        {
            "ok": True,
            "command": command,
            "input": {
                **input_meta,
                "period": evaluation_period.label,
                "start_date": evaluation_period.start_date.isoformat(),
                "end_date": evaluation_period.end_date.isoformat(),
                "bench": bench_ticker,
                "layer_benchmarks": layer_benchmarks,
                "database_path": str(db_path()),
            },
            "warnings": warnings + payload["warnings"],
            "error": None,
        }
    )
    return payload, asset_df, analysis, input_meta, source_portfolio_id


def _session_data(
    asset_df: pd.DataFrame, analysis, evaluation_payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "asset_df": asset_df.to_dict(orient="records"),
        "prices": analysis.prices.reset_index().to_dict(orient="records"),
        "returns": analysis.returns.reset_index().to_dict(orient="records"),
        "returns_smooth": analysis.returns_smooth.reset_index().to_dict(
            orient="records"
        ),
        "weights_no_bench": analysis.weights_no_bench.to_dict(),
        "metrics_df": analysis.metrics_df.reset_index().to_dict(orient="records"),
        "portfolio_metrics": analysis.portfolio_metrics,
        "benchmark_metrics": analysis.benchmark_metrics,
        "missing_tickers": analysis.missing_tickers,
        "analysis_settings": {
            "period": evaluation_payload["input"]["period"],
            "rf": DEFAULT_RF,
            "bench": evaluation_payload["input"]["bench"],
        },
        "evaluation_v2": {
            key: evaluation_payload[key]
            for key in [
                "evaluation_period",
                "layer_evaluations",
                "asset_evaluations",
                "review_queue",
                "journal_draft",
                "warnings",
                "guardrails",
            ]
        },
        "evaluation_settings": {
            "period": evaluation_payload["input"]["period"],
            "start_date": evaluation_payload["input"]["start_date"],
            "end_date": evaluation_payload["input"]["end_date"],
            "bench": evaluation_payload["input"]["bench"],
            "layer_benchmarks": evaluation_payload["input"]["layer_benchmarks"],
        },
    }


@app.command()
def evaluate(
    file_path: Annotated[Path | None, typer.Option("--file")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    portfolio_id: Annotated[int | None, typer.Option("--portfolio-id")] = None,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    period: Annotated[str, typer.Option("--period")] = "3M",
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
    as_of_date: Annotated[str | None, typer.Option("--as-of-date")] = None,
    layer_benchmark: Annotated[
        list[str] | None, typer.Option("--layer-benchmark", help=LAYER_BENCHMARK_HELP)
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    save: Annotated[bool, typer.Option("--save")] = False,
    save_to_portfolio_id: Annotated[
        int | None, typer.Option("--save-to-portfolio-id")
    ] = None,
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """Run Evaluation Framework v2 and emit one JSON object."""
    try:
        payload, asset_df, analysis, _input_meta, source_portfolio_id = _run_v2(
            command="evaluate",
            file_path=file_path,
            text=text,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
            period=period,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
            layer_benchmark=layer_benchmark,
        )

        artifacts: dict[str, str] = {}
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for key, filename in {
                "layer_evaluations": "layer_evaluations.csv",
                "asset_evaluations": "asset_evaluations.csv",
                "review_queue": "review_queue.csv",
            }.items():
                path = output_dir / filename
                pd.DataFrame(payload[key]).to_csv(path, index=False)
                artifacts[f"{key}_csv"] = str(path)

        target_portfolio_id = save_to_portfolio_id
        if target_portfolio_id is None and save:
            target_portfolio_id = source_portfolio_id
        if save and target_portfolio_id is None:
            raise CliError(
                "persistence",
                "--save는 DB 입력 또는 --save-to-portfolio-id와 함께 사용해야 합니다.",
            )
        saved = (
            _save_run(
                portfolio_id=target_portfolio_id,
                snapshot_name=snapshot_name,
                note=note,
                session_data=_session_data(asset_df, analysis, payload),
            )
            if target_portfolio_id is not None
            else {"saved": False}
        )
        payload["artifacts"] = artifacts
        payload["saved"] = saved
        _emit_json(payload)
    except Exception as exc:
        _exit_with_error("evaluate", exc)


@app.command("agent-brief")
def agent_brief(
    file_path: Annotated[Path | None, typer.Option("--file")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    portfolio_id: Annotated[int | None, typer.Option("--portfolio-id")] = None,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    period: Annotated[str, typer.Option("--period")] = "3M",
    as_of_date: Annotated[str | None, typer.Option("--as-of-date")] = None,
    layer_benchmark: Annotated[
        list[str] | None, typer.Option("--layer-benchmark", help=LAYER_BENCHMARK_HELP)
    ] = None,
) -> None:
    """Emit a compact v2 IPS brief for agents."""
    try:
        payload, *_ = _run_v2(
            command="agent-brief",
            file_path=file_path,
            text=text,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
            period=period,
            start_date=None,
            end_date=None,
            as_of_date=as_of_date,
            layer_benchmark=layer_benchmark,
        )
        _emit_json(
            {
                "ok": True,
                "command": "agent-brief",
                "input": payload["input"],
                "evaluation_period": payload["evaluation_period"],
                "status_summary": _status_summary(payload),
                "review_queue": payload["review_queue"],
                "guardrails": payload["guardrails"],
                "warnings": payload["warnings"],
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_error("agent-brief", exc)


def _status_summary(payload: dict[str, Any]) -> dict[str, int]:
    counts = {"OK": 0, "Watch": 0, "Review": 0, "Action": 0}
    for record in payload["layer_evaluations"] + payload["asset_evaluations"]:
        status = record.get("output", {}).get("status")
        if status in counts:
            counts[status] += 1
    return counts


@app.command("review-queue")
def review_queue(
    file_path: Annotated[Path | None, typer.Option("--file")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    portfolio_id: Annotated[int | None, typer.Option("--portfolio-id")] = None,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    period: Annotated[str, typer.Option("--period")] = "3M",
    as_of_date: Annotated[str | None, typer.Option("--as-of-date")] = None,
    layer_benchmark: Annotated[
        list[str] | None, typer.Option("--layer-benchmark", help=LAYER_BENCHMARK_HELP)
    ] = None,
) -> None:
    """Emit v2 review queue only."""
    try:
        payload, *_ = _run_v2(
            command="review-queue",
            file_path=file_path,
            text=text,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
            period=period,
            start_date=None,
            end_date=None,
            as_of_date=as_of_date,
            layer_benchmark=layer_benchmark,
        )
        _emit_json(
            {
                "ok": True,
                "command": "review-queue",
                "input": payload["input"],
                "evaluation_period": payload["evaluation_period"],
                "review_queue": payload["review_queue"],
                "guardrails": payload["guardrails"],
                "warnings": payload["warnings"],
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_error("review-queue", exc)


@app.command("risk")
def risk(
    file_path: Annotated[Path | None, typer.Option("--file")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    portfolio_id: Annotated[int | None, typer.Option("--portfolio-id")] = None,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    period: Annotated[str, typer.Option("--period")] = "3M",
    as_of_date: Annotated[str | None, typer.Option("--as-of-date")] = None,
    layer_benchmark: Annotated[
        list[str] | None, typer.Option("--layer-benchmark", help=LAYER_BENCHMARK_HELP)
    ] = None,
) -> None:
    """Emit v2 units with risk-related triggers."""
    try:
        payload, *_ = _run_v2(
            command="risk",
            file_path=file_path,
            text=text,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
            period=period,
            start_date=None,
            end_date=None,
            as_of_date=as_of_date,
            layer_benchmark=layer_benchmark,
        )
        risk_items = [
            item
            for item in payload["review_queue"]
            if any(
                "risk" in code or "mdd" in code or "volatility" in code
                for code in item.get("triggered_by", [])
            )
        ]
        _emit_json(
            {
                "ok": True,
                "command": "risk",
                "input": payload["input"],
                "evaluation_period": payload["evaluation_period"],
                "risk_review": risk_items,
                "guardrails": payload["guardrails"],
                "warnings": payload["warnings"],
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_error("risk", exc)


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
    """Read Toss account observations and persist one immutable snapshot as JSON."""
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
            payload: Any = latest_account_snapshot()
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


@portfolios_app.command("list")
def list_saved_portfolios() -> None:
    """List saved portfolios as JSON."""
    try:
        initialize_database()
        _emit_json(
            {
                "ok": True,
                "command": "portfolios list",
                "database_path": str(db_path()),
                "portfolios": list_portfolios(),
                "error": None,
            }
        )
    except StorageError as exc:
        _exit_with_error("portfolios list", CliError("persistence", str(exc)))
    except Exception as exc:
        _exit_with_error("portfolios list", exc)


@snapshots_app.command("list")
def list_saved_snapshots(
    portfolio_id: Annotated[int, typer.Option("--portfolio-id")],
) -> None:
    """List snapshots for a saved portfolio as JSON."""
    try:
        initialize_database()
        _emit_json(
            {
                "ok": True,
                "command": "snapshots list",
                "database_path": str(db_path()),
                "portfolio_id": portfolio_id,
                "snapshots": list_snapshots(portfolio_id),
                "error": None,
            }
        )
    except StorageError as exc:
        _exit_with_error("snapshots list", CliError("persistence", str(exc)))
    except Exception as exc:
        _exit_with_error("snapshots list", exc)


if __name__ == "__main__":
    sys.exit(app())
