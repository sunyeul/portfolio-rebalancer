import json
import tomllib
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from cli import app
from services.analysis_service import AnalysisResult


runner = CliRunner()


def _payload(result):
    return json.loads(result.stdout)


def test_canonical_cli_script_is_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["ips-pilot"] == "cli:app"
    assert "portfolio-rebalancer" not in pyproject["project"]["scripts"]


def _fake_analysis(asset_df, period, rf, bench):
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO"],
            "CAGR": [0.1],
            "변동성": [0.15],
            "샤프": [0.7],
            "최대낙폭": [-0.1],
            "IR": [0.2],
            "베타": [1.0],
            "알파": [0.01],
            "위험기여도": [0.1],
            "수익기여도": [0.08],
            "가중치": [1.0],
            "E": [0.8],
            "return_total": [0.1],
            "layer": ["core"],
            "category": ["core_market"],
            "dca_enabled": [True],
            "thesis_status": ["intact"],
        }
    ).set_index("ticker")
    returns = pd.DataFrame({"VOO": [0.01, -0.01, 0.02]})
    return AnalysisResult(
        prices=pd.DataFrame({"VOO": [100.0, 101.0, 103.0]}),
        returns=returns,
        returns_smooth=returns,
        weights_no_bench=pd.Series({"VOO": 1.0}),
        metrics_df=metrics_df,
        port_nav=pd.Series([1.0, 1.03]),
        bench_nav=pd.Series([1.0, 1.02]),
        portfolio_metrics={"cagr": 0.1, "volatility": 0.2, "sharpe": 0.5},
        benchmark_metrics={"cagr": 0.08, "volatility": 0.15, "sharpe": 0.4},
        missing_tickers=[],
    )


def test_evaluate_outputs_v2_envelope(monkeypatch):
    monkeypatch.setattr("cli.run_analysis", _fake_analysis)

    result = runner.invoke(app, ["evaluate", "--text", "VOO 100", "--period", "YTD"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert set(payload) >= {
        "ok",
        "command",
        "input",
        "evaluation_period",
        "layer_evaluations",
        "asset_evaluations",
        "review_queue",
        "journal_draft",
        "warnings",
        "guardrails",
        "error",
    }
    assert payload["ok"] is True
    assert payload["command"] == "evaluate"
    assert payload["evaluation_period"]["label"] == "YTD"
    assert payload["layer_evaluations"]
    assert payload["asset_evaluations"]
    assert payload["guardrails"]["no_immediate_order_instruction"] is True


def test_review_queue_command_uses_v2(monkeypatch):
    monkeypatch.setattr("cli.run_analysis", _fake_analysis)

    result = runner.invoke(app, ["review-queue", "--text", "VOO 100"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["command"] == "review-queue"
    assert "review_queue" in payload


def test_cli_help_lists_v2_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "evaluate",
        "agent-brief",
        "review-queue",
        "risk",
        "portfolios",
        "snapshots",
    ]:
        assert command in result.stdout


def test_evaluate_rejects_one_sided_custom_dates():
    result = runner.invoke(
        app,
        ["evaluate", "--text", "VOO 100", "--start-date", "2026-01-01"],
    )

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "input"
