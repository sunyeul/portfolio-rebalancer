"""Agent-oriented Typer CLI for IPS Pilot v2."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from services.analysis_service import DEFAULT_BENCH, DEFAULT_RF, AnalysisError, run_analysis
from services.evaluation_engine import run_evaluation
from services.evaluation_period import (
    EvaluationPeriodError,
    analysis_period_value,
    resolve_evaluation_period,
)
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


app = typer.Typer(
    help="IPS Pilot CLI for Evaluation Framework v2.",
    no_args_is_help=True,
)
portfolios_app = typer.Typer(help="Inspect saved portfolios.")
snapshots_app = typer.Typer(help="Inspect saved portfolio snapshots.")
app.add_typer(portfolios_app, name="portfolios")
app.add_typer(snapshots_app, name="snapshots")


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


def _empty_v2_payload(command: str, error: dict[str, Any] | None = None) -> dict[str, Any]:
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


def _selected_sources(*values: Any) -> int:
    return sum(value is not None for value in values)


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
            return asset_df, warnings + validation_warnings, {"source": "file", "file": str(file_path)}, None

        if text is not None:
            assets, warnings = parse_text_to_assets_service(text)
            asset_df, validation_warnings = normalize_and_validate_assets(assets)
            return asset_df, warnings + validation_warnings, {"source": "text"}, None

        initialize_database()
        if portfolio_id is not None:
            state = get_current_state(portfolio_id)
            if state is None:
                raise CliError("input", f"portfolio_id={portfolio_id}의 current-state를 찾을 수 없습니다.")
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
            {"source": "snapshot", "snapshot_id": snapshot_id, "portfolio_id": source_portfolio_id},
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
    name = snapshot_name or f"CLI evaluation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
    rf: float,
    bench: str,
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
        )
    except EvaluationPeriodError as exc:
        raise CliError("input", str(exc)) from exc

    bench_ticker = bench.upper()
    try:
        analysis = run_analysis(
            asset_df,
            analysis_period_value(evaluation_period),
            rf,
            bench_ticker,
        )
    except AnalysisError as exc:
        raise CliError("analysis", str(exc)) from exc

    result = run_evaluation(
        analysis=analysis,
        evaluation_period=evaluation_period,
        rf=rf,
        bench=bench_ticker,
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
                "rf": rf,
                "bench": bench_ticker,
                "database_path": str(db_path()),
            },
            "warnings": warnings + payload["warnings"],
            "error": None,
        }
    )
    return payload, asset_df, analysis, input_meta, source_portfolio_id


def _session_data(asset_df: pd.DataFrame, analysis, evaluation_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_df": asset_df.to_dict(orient="records"),
        "prices": analysis.prices.reset_index().to_dict(orient="records"),
        "returns": analysis.returns.reset_index().to_dict(orient="records"),
        "returns_smooth": analysis.returns_smooth.reset_index().to_dict(orient="records"),
        "weights_no_bench": analysis.weights_no_bench.to_dict(),
        "metrics_df": analysis.metrics_df.reset_index().to_dict(orient="records"),
        "portfolio_metrics": analysis.portfolio_metrics,
        "benchmark_metrics": analysis.benchmark_metrics,
        "missing_tickers": analysis.missing_tickers,
        "analysis_settings": {
            "period": evaluation_payload["input"]["period"],
            "rf": evaluation_payload["input"]["rf"],
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
            "rf": evaluation_payload["input"]["rf"],
            "bench": evaluation_payload["input"]["bench"],
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
    rf: Annotated[float, typer.Option("--rf")] = DEFAULT_RF,
    bench: Annotated[str, typer.Option("--bench")] = DEFAULT_BENCH,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    save: Annotated[bool, typer.Option("--save")] = False,
    save_to_portfolio_id: Annotated[int | None, typer.Option("--save-to-portfolio-id")] = None,
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
            rf=rf,
            bench=bench,
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
            raise CliError("persistence", "--save는 DB 입력 또는 --save-to-portfolio-id와 함께 사용해야 합니다.")
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
    rf: Annotated[float, typer.Option("--rf")] = DEFAULT_RF,
    bench: Annotated[str, typer.Option("--bench")] = DEFAULT_BENCH,
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
            rf=rf,
            bench=bench,
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
    rf: Annotated[float, typer.Option("--rf")] = DEFAULT_RF,
    bench: Annotated[str, typer.Option("--bench")] = DEFAULT_BENCH,
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
            rf=rf,
            bench=bench,
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
    rf: Annotated[float, typer.Option("--rf")] = DEFAULT_RF,
    bench: Annotated[str, typer.Option("--bench")] = DEFAULT_BENCH,
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
            rf=rf,
            bench=bench,
        )
        risk_items = [
            item
            for item in payload["review_queue"]
            if any("risk" in code or "mdd" in code or "volatility" in code for code in item.get("triggered_by", []))
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
