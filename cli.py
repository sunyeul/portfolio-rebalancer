"""Agent-oriented Typer CLI for the portfolio rebalancer."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from api.v1.serialization import (
    GROUP_SUMMARY_COLUMNS,
    METRICS_COLUMNS,
    PROPOSAL_COLUMNS,
    RC_VIOLATION_COLUMNS,
    dataframe_records,
    safe_mapping,
)
from services.analysis_service import AnalysisError, run_analysis
from services.evaluation_service import EvaluationError, run_evaluation
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
from utils.metrics import annualize_cov


app = typer.Typer(
    help="Portfolio rebalancer CLI optimized for agent-readable JSON output.",
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


def _exit_with_error(command: str, exc: Exception) -> None:
    if isinstance(exc, CliError):
        stage = exc.stage
        message = exc.message
        hint = exc.hint
    else:
        stage = "unexpected"
        message = str(exc)
        hint = "명령 옵션과 입력 데이터를 확인한 뒤 다시 실행하세요."
    _emit_json(
        {
            "ok": False,
            "command": command,
            "input": None,
            "warnings": [],
            "analysis": None,
            "evaluation": None,
            "agent_summary": None,
            "artifacts": {},
            "saved": None,
            "error": {
                "stage": stage,
                "message": message,
                "hint": hint,
            },
        }
    )
    raise typer.Exit(code=1)


def _parse_period(period: str) -> int | str:
    normalized = period.strip()
    if normalized.isdigit():
        return int(normalized)
    if normalized.upper() == "YTD":
        return "YTD"
    if normalized.lower() == "max":
        return "Max"
    raise CliError("input", "period는 개월 수, YTD, Max 중 하나여야 합니다.")


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
            return (
                asset_df,
                warnings + validation_warnings,
                {"source": "file", "file": str(file_path)},
                None,
            )

        if text is not None:
            assets, warnings = parse_text_to_assets_service(text)
            asset_df, validation_warnings = normalize_and_validate_assets(assets)
            return (
                asset_df,
                warnings + validation_warnings,
                {"source": "text"},
                None,
            )

        initialize_database()
        if portfolio_id is not None:
            state = get_current_state(portfolio_id)
            if state is None:
                raise CliError(
                    "input",
                    f"portfolio_id={portfolio_id}의 current-state를 찾을 수 없습니다.",
                    "웹앱에서 포트폴리오를 저장하거나 --snapshot-id를 사용하세요.",
                )
            asset_rows = state["session_state"].get("asset_df") or []
            return (
                pd.DataFrame(asset_rows),
                [],
                {"source": "portfolio_current_state", "portfolio_id": portfolio_id},
                portfolio_id,
            )

        snapshot = get_snapshot(snapshot_id or 0)
        if snapshot is None:
            raise CliError(
                "input",
                f"snapshot_id={snapshot_id}를 찾을 수 없습니다.",
                "portfolios list와 snapshots list로 사용 가능한 ID를 확인하세요.",
            )
        asset_rows = snapshot["session_state"].get("asset_df") or []
        source_portfolio_id = int(snapshot["summary"]["portfolio_id"])
        return (
            pd.DataFrame(asset_rows),
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


def _cov_matrix(returns_smooth: pd.DataFrame, metrics_df: pd.DataFrame):
    common = returns_smooth.columns.intersection(metrics_df.index)
    if len(common) == 0:
        return None
    return annualize_cov(returns_smooth[common])


def _records_to_csv(
    records: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
) -> str | None:
    if not records:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def _agent_summary(
    proposal: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    missing_tickers: list[str],
) -> dict[str, Any]:
    recommended = [row for row in proposal if row.get("should_execute") is True]
    hold = [row for row in proposal if row.get("should_execute") is not True]
    data_quality = [
        row
        for row in proposal
        if row.get("data_quality_low")
        or (row.get("missing_ratio") is not None and row.get("missing_ratio") > 0.2)
        or (
            row.get("observation_count") is not None
            and row.get("observation_count") < 60
        )
    ]
    top_risk = sorted(
        metrics,
        key=lambda row: row.get("risk_contribution") or 0,
        reverse=True,
    )[:5]
    if missing_tickers or data_quality:
        health = "data_insufficient"
    elif recommended:
        health = "needs_review"
    else:
        health = "ok"
    return {
        "portfolio_health": health,
        "rebalance_needed": bool(recommended),
        "recommended_actions": recommended,
        "hold_actions": hold,
        "data_quality_warnings": {
            "missing_tickers": missing_tickers,
            "low_quality_rows": data_quality,
        },
        "top_risk_contributors": top_risk,
    }


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


@app.command()
def evaluate(
    file_path: Annotated[
        Path | None,
        typer.Option("--file"),
    ] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    portfolio_id: Annotated[int | None, typer.Option("--portfolio-id")] = None,
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
    period: Annotated[str, typer.Option("--period")] = "12",
    rf: Annotated[float, typer.Option("--rf")] = 0.0,
    bench: Annotated[str, typer.Option("--bench")] = "SPY",
    rc_threshold: Annotated[float, typer.Option("--rc-threshold")] = 1.5,
    e_threshold: Annotated[float, typer.Option("--e-threshold")] = 0.5,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    save: Annotated[bool, typer.Option("--save")] = False,
    save_to_portfolio_id: Annotated[int | None, typer.Option("--save-to-portfolio-id")] = None,
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """Run analysis and evaluation, emitting a single JSON object on stdout."""
    try:
        parsed_period = _parse_period(period)
        asset_df, warnings, input_meta, source_portfolio_id = _load_asset_df(
            file_path=file_path,
            text=text,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
        )
        bench_ticker = bench.upper()

        try:
            analysis = run_analysis(asset_df, parsed_period, rf, bench_ticker)
        except AnalysisError as exc:
            raise CliError("analysis", str(exc)) from exc

        try:
            evaluation = run_evaluation(
                analysis.metrics_df,
                None,
                rc_threshold,
                e_threshold,
                cov_matrix=_cov_matrix(analysis.returns_smooth, analysis.metrics_df),
            )
        except EvaluationError as exc:
            raise CliError("evaluation", str(exc)) from exc

        metrics = dataframe_records(
            analysis.metrics_df, METRICS_COLUMNS, include_index=True
        )
        proposal = dataframe_records(evaluation.proposal_df, PROPOSAL_COLUMNS)
        ips_actions = dataframe_records(evaluation.ips_action_df)
        group_summary = dataframe_records(
            evaluation.group_summary_df, GROUP_SUMMARY_COLUMNS
        )
        rc_violations = dataframe_records(
            evaluation.rc_violations, RC_VIOLATION_COLUMNS
        )

        artifacts: dict[str, str] = {}
        if output_dir is not None:
            outputs = {
                "metrics_csv": _records_to_csv(metrics, output_dir, "metrics.csv"),
                "proposal_csv": _records_to_csv(proposal, output_dir, "proposal.csv"),
                "ips_actions_csv": _records_to_csv(
                    ips_actions, output_dir, "ips_actions.csv"
                ),
                "group_summary_csv": _records_to_csv(
                    group_summary, output_dir, "group_summary.csv"
                ),
                "rc_violations_csv": _records_to_csv(
                    rc_violations, output_dir, "rc_violations.csv"
                ),
            }
            artifacts = {key: value for key, value in outputs.items() if value}

        session_data = {
            "asset_df": asset_df.to_dict(orient="records"),
            "metrics_df": analysis.metrics_df.reset_index().to_dict(orient="records"),
            "portfolio_metrics": analysis.portfolio_metrics,
            "benchmark_metrics": analysis.benchmark_metrics,
            "missing_tickers": analysis.missing_tickers,
            "returns_smooth": analysis.returns_smooth.reset_index().to_dict(
                orient="records"
            ),
            "analysis_settings": {
                "period": parsed_period,
                "rf": rf,
                "bench": bench_ticker,
            },
            "proposal_df": evaluation.proposal_df.to_dict(orient="records"),
            "ips_action_df": evaluation.ips_action_df.to_dict(orient="records"),
            "group_summary_df": evaluation.group_summary_df.to_dict(orient="records"),
            "rc_violations": evaluation.rc_violations.to_dict(orient="records"),
            "evaluation_settings": {
                "rc_over_thresh_pct": rc_threshold,
                "e_thresh": e_threshold,
                "target_weights": None,
            },
            "ips_config_snapshot": evaluation.ips_config_snapshot,
        }

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
                session_data=session_data,
            )
            if target_portfolio_id is not None
            else {"saved": False}
        )

        _emit_json(
            {
                "ok": True,
                "command": "evaluate",
                "input": {
                    **input_meta,
                    "period": parsed_period,
                    "rf": rf,
                    "bench": bench_ticker,
                    "database_path": str(db_path()),
                },
                "warnings": warnings,
                "analysis": {
                    "metrics": metrics,
                    "portfolio_metrics": safe_mapping(analysis.portfolio_metrics),
                    "benchmark_metrics": safe_mapping(analysis.benchmark_metrics),
                    "missing_tickers": analysis.missing_tickers,
                },
                "evaluation": {
                    "proposal": proposal,
                    "ips_actions": ips_actions,
                    "group_summary": group_summary,
                    "rc_violations": rc_violations,
                    "ips_config_snapshot": evaluation.ips_config_snapshot,
                },
                "agent_summary": {
                    **_agent_summary(proposal, metrics, analysis.missing_tickers),
                    "save_status": saved,
                },
                "artifacts": artifacts,
                "saved": saved,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_error("evaluate", exc)


@portfolios_app.command("list")
def list_saved_portfolios() -> None:
    """List saved portfolios as JSON."""
    try:
        initialize_database()
        portfolios = list_portfolios()
        _emit_json(
            {
                "ok": True,
                "command": "portfolios list",
                "database_path": str(db_path()),
                "portfolios": portfolios,
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
        snapshots = list_snapshots(portfolio_id)
        _emit_json(
            {
                "ok": True,
                "command": "snapshots list",
                "database_path": str(db_path()),
                "portfolio_id": portfolio_id,
                "snapshots": snapshots,
                "error": None,
            }
        )
    except StorageError as exc:
        _exit_with_error("snapshots list", CliError("persistence", str(exc)))
    except Exception as exc:
        _exit_with_error("snapshots list", exc)


if __name__ == "__main__":
    sys.exit(app())
