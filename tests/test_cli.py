import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli import app
from services.analysis_service import AnalysisResult
from services.evaluation_service import EvaluationResult
from storage.database import initialize_database
from storage.portfolio_store import (
    create_portfolio,
    get_snapshot,
    save_current_state,
)


runner = CliRunner()


def _payload(result):
    return json.loads(result.stdout)


def test_cli_script_aliases_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["name"] == "ips-pilot"
    assert pyproject["project"]["scripts"]["ips-pilot"] == "cli:app"
    assert pyproject["project"]["scripts"]["portfolio-rebalancer"] == "cli:app"


def _fake_analysis(asset_df, period, rf, bench):
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ"],
            "CAGR": [0.1, 0.2],
            "변동성": [0.15, 0.22],
            "샤프": [0.7, 0.9],
            "최대낙폭": [-0.1, -0.2],
            "IR": [0.2, 0.4],
            "베타": [1.0, 1.2],
            "알파": [0.01, 0.02],
            "data_start": ["2026-06-01", "2026-06-01"],
            "data_end": ["2026-06-05", "2026-06-05"],
            "observation_count": [80, 80],
            "missing_ratio": [0.0, 0.0],
            "위험기여도": [0.4, 0.6],
            "수익기여도": [0.04, 0.12],
            "가중치": [0.4, 0.6],
            "E": [0.8, 0.3],
            "return_total": [0.1, -0.05],
            "group": ["core", "satellite_ai_infra"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
        }
    ).set_index("ticker")
    returns = pd.DataFrame(
        {"VOO": [0.01, 0.02, -0.01], "QQQ": [0.02, -0.01, 0.03]}
    )
    return AnalysisResult(
        prices=pd.DataFrame({"VOO": [100, 101], "QQQ": [200, 202]}),
        returns=returns,
        returns_smooth=returns,
        weights_no_bench=pd.Series({"VOO": 0.4, "QQQ": 0.6}),
        metrics_df=metrics_df,
        port_nav=pd.Series([1.0, 1.1]),
        bench_nav=None,
        portfolio_metrics={"cagr": 0.15, "volatility": 0.18, "sharpe": 0.8},
        benchmark_metrics=None,
        missing_tickers=[],
    )


def _fake_evaluation(*args, **kwargs):
    proposal_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ"],
            "현재%": [40.0, 60.0],
            "목표%": [50.0, 50.0],
            "갭%": [10.0, -10.0],
            "E": [0.8, 0.3],
            "RC_Gap%": [0.0, 10.0],
            "RC_Over%": [0.0, 10.0],
            "RC_Target%": [40.0, 50.0],
            "return_total%": [10.0, -5.0],
            "group": ["core", "satellite_ai_infra"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
            "risk_over": [False, True],
            "efficiency_warning": [False, True],
            "IPS적합도": [82.0, 48.0],
            "IPS등급": ["high", "low"],
            "히스테리시스제외": [False, False],
            "최소거래미만": [False, False],
            "수치후보": [True, True],
            "참고조정%": [10.0, -10.0],
            "실행": [True, False],
            "제안조정%": [10.0, 0.0],
            "판단사유": ["목표 대비 부족", "투자 논리 점검"],
        }
    )
    return EvaluationResult(
        proposal_df=proposal_df,
        ips_action_df=pd.DataFrame(
            {"ticker": ["VOO", "QQQ"], "ips_action": ["increase_dca", "review_before_action"]}
        ),
        group_summary_df=pd.DataFrame(
            {"group": ["core"], "weight": [0.4], "risk_contribution": [0.4]}
        ),
        sell_list=pd.DataFrame(),
        buy_list=pd.DataFrame(),
        fine_tune_list=pd.DataFrame(),
        rc_violations=pd.DataFrame({"ticker": ["QQQ"], "현재RC%": [60.0]}),
        ips_config_snapshot={"rules": {}},
        playbook={
            "code": "market_correction",
            "label": "시장 조정 대응",
            "confidence": "high",
            "reasons": ["코어 비중이 IPS 목표보다 낮습니다."],
            "steps": ["정기매수 조정 가능성을 먼저 봅니다."],
            "manual_context": kwargs.get("decision_context", "regular_review"),
            "is_manual_override": False,
        },
    )


def test_evaluate_text_outputs_agent_readable_json(monkeypatch):
    captured_kwargs = {}

    def fake_evaluation_with_capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _fake_evaluation(*args, **kwargs)

    monkeypatch.setattr("cli.run_analysis", _fake_analysis)
    monkeypatch.setattr("cli.run_evaluation", fake_evaluation_with_capture)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--text",
            "VOO 40\nQQQ 60",
            "--decision-context",
            "market_correction",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["input"]["source"] == "text"
    assert payload["input"]["decision_context"] == "market_correction"
    assert captured_kwargs["decision_context"] == "market_correction"
    assert payload["agent_summary"]["rebalance_needed"] is True
    assert payload["agent_summary"]["recommended_actions"][0]["ticker"] == "VOO"
    assert payload["error"] is None


def test_agent_brief_text_outputs_ips_schema_without_rebalance_legacy_keys(monkeypatch):
    captured_kwargs = {}

    def fake_evaluation_with_capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _fake_evaluation(*args, **kwargs)

    monkeypatch.setattr("cli.run_analysis", _fake_analysis)
    monkeypatch.setattr("cli.run_evaluation", fake_evaluation_with_capture)

    result = runner.invoke(
        app,
        [
            "agent-brief",
            "--text",
            "VOO 40\nQQQ 60",
            "--decision-context",
            "market_correction",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["command"] == "agent-brief"
    assert payload["input"]["source"] == "text"
    assert payload["input"]["decision_context"] == "market_correction"
    assert captured_kwargs["decision_context"] == "market_correction"
    assert payload["ips_status"]["status"] == "review_required"
    assert payload["ips_status"]["status_code"] == "review_required"
    assert payload["ips_status"]["status_label"] == "검토 필요"
    assert payload["dca_plan"]["increase"][0]["ticker"] == "VOO"
    assert payload["review_queue"]["thesis_review"][0]["ticker"] == "QQQ"
    assert payload["risk_flags"][0]["type"] == "risk_contribution_limit"
    assert payload["risk_flags"][0]["type_code"] == "risk_contribution_limit"
    assert payload["risk_flags"][0]["type_label"] == "위험기여도 한도"
    assert payload["risk_flags"][0]["severity_code"] == "warning"
    assert payload["risk_flags"][0]["severity_label"] == "주의"
    assert payload["playbook"]["code"] == "market_correction"
    assert payload["guardrails"]["not_investment_advice"] is True
    assert payload["guardrails"]["no_immediate_order_instruction"] is True
    assert "rebalance_needed" not in payload
    assert "recommended_actions" not in payload
    assert "agent_summary" not in payload
    assert payload["error"] is None


def test_agent_brief_failure_uses_same_ips_envelope():
    result = runner.invoke(app, ["agent-brief"])

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["command"] == "agent-brief"
    assert payload["ips_status"] is None
    assert payload["dca_plan"] == {
        "increase": [],
        "reduce_or_pause": [],
        "hold": [],
    }
    assert payload["review_queue"] == {
        "thesis_review": [],
        "risk_review": [],
        "sell_review": [],
        "blocked": [],
    }
    assert payload["risk_flags"] == []
    assert payload["guardrails"]["not_investment_advice"] is True
    assert payload["error"]["stage"] == "input"
    assert "정확히 하나" in payload["error"]["message"]


@pytest.mark.parametrize(
    ("command", "expected_key"),
    [
        ("diagnose", "ips_status"),
        ("dca-plan", "dca_plan"),
        ("review-queue", "review_queue"),
        ("risk", "risk_flags"),
    ],
)
def test_purpose_built_agent_commands_emit_scoped_json(
    monkeypatch,
    command,
    expected_key,
):
    monkeypatch.setattr("cli.run_analysis", _fake_analysis)
    monkeypatch.setattr("cli.run_evaluation", _fake_evaluation)

    result = runner.invoke(app, [command, "--text", "VOO 40\nQQQ 60"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["command"] == command
    assert expected_key in payload
    assert payload["guardrails"]["no_immediate_order_instruction"] is True
    assert "rebalance_needed" not in payload
    assert "recommended_actions" not in payload
    assert "agent_summary" not in payload


def test_evaluate_can_include_counterfactual_and_backtest(monkeypatch):
    monkeypatch.setattr("cli.run_analysis", _fake_analysis)
    monkeypatch.setattr("cli.run_evaluation", _fake_evaluation)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--text",
            "VOO 40\nQQQ 60",
            "--decision-context",
            "market_correction",
            "--counterfactual-scenario",
            "pause_satellite_new_buys",
            "--backtest-strategy",
            "current_ips",
            "--backtest-strategy",
            "core_first_dca",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["input"]["counterfactual_scenario"] == "pause_satellite_new_buys"
    assert payload["input"]["backtest_strategies"] == ["current_ips", "core_first_dca"]
    assert payload["simulation"]["counterfactual"]["interpretation"]
    assert len(payload["simulation"]["backtest"]["strategy_summaries"]) == 2
    assert {
        row["strategy_label"] for row in payload["simulation"]["backtest"]["strategy_summaries"]
    } == {"현재 IPS 유지", "코어 부족분 우선"}


def test_evaluate_rejects_missing_or_multiple_inputs_as_json():
    result = runner.invoke(app, ["evaluate"])

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "input"
    assert "정확히 하나" in payload["error"]["message"]


def test_evaluate_missing_file_error_is_json():
    result = runner.invoke(app, ["evaluate", "--file", "missing.csv"])

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "input"
    assert "파일을 찾을 수 없습니다" in payload["error"]["message"]


def test_codex_cli_usage_lives_in_repo_skill_not_docs():
    readme = Path("README.md").read_text()
    removed_doc = Path("docs/codex-cli-usage.md")
    skill = Path(".agents/skills/ips-pilot-cli-review/SKILL.md")

    assert "docs/codex-cli-usage.md" not in readme
    assert not removed_doc.exists()
    assert ".agents/skills/ips-pilot-cli-review/SKILL.md" in readme
    assert skill.exists()

    text = skill.read_text()
    assert "agent-brief" in text
    assert "not investment advice" in text
    assert "no immediate buy/sell" in text


def test_portfolio_current_state_can_be_evaluated_and_saved(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()
    portfolio = create_portfolio("Agent account")
    save_current_state(
        portfolio["id"],
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 40.0,
                    "return_total": None,
                    "group": "core",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                    "weight": 0.4,
                },
                {
                    "ticker": "QQQ",
                    "allocation": 60.0,
                    "return_total": None,
                    "group": "satellite_ai_infra",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                    "weight": 0.6,
                },
            ]
        },
    )
    monkeypatch.setattr("cli.run_analysis", _fake_analysis)
    monkeypatch.setattr("cli.run_evaluation", _fake_evaluation)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--portfolio-id",
            str(portfolio["id"]),
            "--save",
            "--snapshot-name",
            "CLI saved run",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["saved"]["saved"] is True
    snapshot = get_snapshot(payload["saved"]["snapshot_id"])
    assert snapshot is not None
    assert snapshot["summary"]["name"] == "CLI saved run"
    assert snapshot["summary"]["has_evaluation"] is True


def test_list_commands_share_database(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()
    portfolio = create_portfolio("Listed account")

    portfolios_result = runner.invoke(app, ["portfolios", "list"])
    snapshots_result = runner.invoke(
        app,
        ["snapshots", "list", "--portfolio-id", str(portfolio["id"])],
    )

    assert portfolios_result.exit_code == 0
    assert _payload(portfolios_result)["portfolios"][0]["name"] == "Listed account"
    assert snapshots_result.exit_code == 0
    assert _payload(snapshots_result)["snapshots"] == []
