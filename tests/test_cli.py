import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from cli import app
from services.dynamic_allocation import DEFAULT_ALLOCATION_REVIEW


runner = CliRunner()


def _payload(result):
    return json.loads(result.stdout)


def test_canonical_cli_script_is_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["ips-pilot"] == "cli:app"
    assert "portfolio-rebalancer" not in pyproject["project"]["scripts"]


def test_help_exposes_only_toss_product_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "toss-health",
        "toss-sync",
        "toss-snapshots",
        "performance",
        "account-view",
        "policy",
        "inspection",
        "market",
        "web",
    ):
        assert command in result.stdout
    for removed in ("evaluate", "agent-brief", "review-queue", "risk", "portfolios"):
        assert f"│ {removed} " not in result.stdout
    assert "│ snapshots " not in result.stdout


def test_account_view_emits_one_projection_json(monkeypatch):
    monkeypatch.setattr(
        "cli.initialize_database",
        lambda: None,
    )
    monkeypatch.setattr(
        "cli.build_account_projection",
        lambda snapshot_id=None, layer_map=None: {"snapshot_id": snapshot_id or 7},
    )

    result = runner.invoke(app, ["account-view", "--snapshot-id", "7"])

    assert result.exit_code == 0
    assert _payload(result) == {
        "ok": True,
        "command": "account-view",
        "projection": {"snapshot_id": 7},
        "error": None,
    }


def test_account_view_returns_machine_readable_error(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.build_account_projection",
        lambda snapshot_id=None, layer_map=None: (_ for _ in ()).throw(
            ValueError("complete Toss snapshot not found")
        ),
    )

    result = runner.invoke(app, ["account-view"])

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["command"] == "account-view"
    assert payload["error"]["message"] == "complete Toss snapshot not found"


def test_inspection_preview_emits_one_non_persisting_json_object(monkeypatch, tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"risk_review": {"lookback_sessions": 252}}))
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.list_observed_identities", lambda: [])
    monkeypatch.setattr(
        "services.policy_validation.validate_policy",
        lambda payload, identities: payload,
    )
    monkeypatch.setattr(
        "cli.preview_inspection",
        lambda policy, snapshot_id=None: {
            "persisted": False,
            "policy_version_id": None,
            "snapshot_id": snapshot_id,
            "evaluation": {"state": "not_evaluable"},
        },
    )

    result = runner.invoke(
        app,
        [
            "inspection",
            "preview",
            "--policy-file",
            str(policy_path),
            "--snapshot-id",
            "7",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["command"] == "inspection preview"
    assert payload["persisted"] is False
    assert payload["snapshot_id"] == 7
    assert payload["contract_supported"] is False


def test_inspection_preview_contract_gate_accepts_v2_result(monkeypatch, tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"risk_review": {"lookback_sessions": 252}}))
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.list_observed_identities", lambda: [])
    monkeypatch.setattr(
        "services.policy_validation.validate_policy",
        lambda payload, identities: payload,
    )
    monkeypatch.setattr(
        "cli.preview_inspection",
        lambda policy, snapshot_id=None: {
            "persisted": False,
            "snapshot_id": snapshot_id,
            "evaluation": {
                "engine_version": "phase5-v2",
                "state": "complete",
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "inspection",
            "preview",
            "--policy-file",
            str(policy_path),
            "--snapshot-id",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert _payload(result)["contract_supported"] is True


def test_inspection_run_and_show_expose_v2_contract_gate(monkeypatch):
    evaluation = {
        "id": 11,
        "snapshot_id": 7,
        "state": "complete",
        "engine_version": "phase5-v2",
        "performance_run_id": None,
        "policy_version_id": 3,
        "result": {"engine_version": "phase5-v2", "state": "complete"},
    }
    monkeypatch.setattr("cli.run_inspection", lambda snapshot_id=None: evaluation)
    run_result = runner.invoke(app, ["inspection", "run", "--snapshot-id", "7"])
    assert run_result.exit_code == 0
    assert _payload(run_result)["contract_supported"] is True

    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_evaluation_run", lambda run_id: evaluation)
    show_result = runner.invoke(app, ["inspection", "show", "--run-id", "11"])
    assert show_result.exit_code == 0
    assert _payload(show_result)["contract_supported"] is True


def test_inspection_show_marks_v1_contract_unsupported(monkeypatch):
    evaluation = {
        "id": 12,
        "snapshot_id": 7,
        "state": "complete",
        "engine_version": "phase5-v1",
        "performance_run_id": None,
        "policy_version_id": 3,
        "result": {
            "engine_version": "phase5-v2",
            "state": "complete",
            "review_queue": [],
        },
    }
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_evaluation_run", lambda run_id: evaluation)

    result = runner.invoke(app, ["inspection", "show", "--run-id", "12"])

    assert result.exit_code == 0
    assert _payload(result)["contract_supported"] is False


def test_inspection_preview_rejects_missing_policy_file():
    result = runner.invoke(
        app,
        ["inspection", "preview", "--policy-file", "/tmp/missing-phase5-policy.json"],
    )

    assert result.exit_code == 1
    assert _payload(result)["error"]["stage"] == "input"


def test_policy_show_requires_active_flag(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    result = runner.invoke(app, ["policy", "show"])
    assert result.exit_code == 1
    assert _payload(result)["error"]["stage"] == "input"


def test_web_rejects_non_loopback_binding():
    result = runner.invoke(app, ["web", "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert _payload(result)["error"]["stage"] == "input"


def test_market_context_persists_composite_candidate_without_activation(monkeypatch):
    candidate_calls = []
    captured = {}
    active = {
        "id": 8,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-01T00:00:00+00:00",
        "policy": {
            "cash_reserve": {"target": 0.05},
            "allocation_review": DEFAULT_ALLOCATION_REVIEW,
        },
    }
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_active_policy", lambda: active)
    monkeypatch.setattr("cli.list_candles", lambda **kwargs: [kwargs])

    def evaluate(series, **kwargs):
        captured["series"] = series
        captured.update(kwargs)
        return {
            "status": "Review",
            "candidate_state": "candidate",
            "regime": "risk_on",
            "verification_task": "verify",
        }

    monkeypatch.setattr("cli.evaluate_dynamic_allocation", evaluate)

    def persist(candidate):
        candidate_calls.append(candidate)
        return {"id": 3, **candidate}

    monkeypatch.setattr("cli.insert_policy_candidate", persist)
    monkeypatch.setattr("cli.latest_policy_candidate", lambda *args: {"id": 3})

    result = runner.invoke(app, ["market", "context"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["candidate"]["id"] == 3
    assert candidate_calls[0]["base_policy_version_id"] == 8
    assert set(captured["series"]) == {
        "US/SPY",
        "US/QQQ",
        "KR/KOSPI",
        "KR/KOSDAQ",
    }
    assert captured["active_policy"] is active["policy"]
    assert captured["last_change_at"] == active["created_at"]
    assert payload["activation"] == "human approval required; active policy unchanged"


def test_market_sync_deduplicates_required_stocks_and_collects_both_indicators(
    monkeypatch,
):
    calls = []

    class Service:
        def collect_history(self, **kwargs):
            calls.append(kwargs)
            return []

    class Transport:
        def close(self):
            return None

    active = {
        "id": 8,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-01T00:00:00+00:00",
        "policy": {"allocation_review": DEFAULT_ALLOCATION_REVIEW},
    }
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_active_policy", lambda: active)
    monkeypatch.setattr("cli.list_observed_identities", lambda: [("US", "SPY")])
    monkeypatch.setattr(
        "cli._build_toss_market_service", lambda: (Service(), Transport())
    )
    monkeypatch.setattr("cli.insert_candles", lambda candles: list(candles))
    monkeypatch.setattr("cli.list_candles", lambda **_: [])
    monkeypatch.setattr(
        "cli.evaluate_dynamic_allocation",
        lambda *args, **kwargs: {
            "status": "Watch",
            "candidate_state": "observe",
        },
    )

    result = runner.invoke(
        app,
        ["market", "sync", "--symbols", "US/SPY,US/QQQ"],
    )

    assert result.exit_code == 0
    stock_calls = [call for call in calls if call["source_kind"] == "stock"]
    indicator_calls = [
        call for call in calls if call["source_kind"] == "market_indicator"
    ]
    assert [(call["market_country"], call["symbol"]) for call in stock_calls] == [
        ("US", "SPY"),
        ("US", "QQQ"),
    ]
    assert [call["symbol"] for call in indicator_calls] == ["KOSPI", "KOSDAQ"]
    assert _payload(result)["symbols"] == [
        "US/SPY",
        "US/QQQ",
        "KR/KOSPI",
        "KR/KOSDAQ",
    ]


def test_market_sync_research_only_collects_policy_universe_without_policy_side_effects(
    monkeypatch,
):
    calls = []

    class Service:
        def collect_history(self, **kwargs):
            calls.append(kwargs)
            return []

    class Transport:
        def close(self):
            return None

    active = {
        "id": 8,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-01T00:00:00+00:00",
        "policy": {
            "instruments": [
                {"market_country": "US", "symbol": "SPY"},
                {"market_country": "US", "symbol": "GLD"},
            ],
            "allocation_review": DEFAULT_ALLOCATION_REVIEW,
        },
    }
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_active_policy", lambda: active)
    monkeypatch.setattr("cli.list_observed_identities", lambda: [("US", "SPY")])
    monkeypatch.setattr(
        "cli._build_toss_market_service", lambda: (Service(), Transport())
    )
    monkeypatch.setattr("cli.insert_candles", lambda candles: list(candles))
    monkeypatch.setattr("cli.list_candles", lambda **_: [])
    monkeypatch.setattr(
        "cli.evaluate_dynamic_allocation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("research-only must not evaluate allocation")
        ),
    )
    monkeypatch.setattr(
        "cli.insert_policy_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("research-only must not persist policy candidates")
        ),
    )

    result = runner.invoke(
        app,
        [
            "market",
            "sync",
            "--research-only",
            "--target-points",
            "756",
            "--max-pages",
            "4",
        ],
    )

    assert result.exit_code == 0
    stock_calls = [call for call in calls if call["source_kind"] == "stock"]
    indicator_calls = [
        call for call in calls if call["source_kind"] == "market_indicator"
    ]
    assert [(call["market_country"], call["symbol"]) for call in stock_calls] == [
        ("US", "SPY"),
        ("US", "GLD"),
        ("US", "QQQ"),
    ]
    assert [call["symbol"] for call in indicator_calls] == ["KOSPI", "KOSDAQ"]
    assert all(call["target_points"] == 756 for call in calls)
    assert _payload(result) == {
        "ok": True,
        "command": "market sync",
        "symbols": [
            "US/SPY",
            "US/GLD",
            "US/QQQ",
            "KR/KOSPI",
            "KR/KOSDAQ",
        ],
        "candle_count": 0,
        "context": None,
        "candidate": None,
        "research_only": True,
        "target_points": 756,
        "error": None,
    }


def test_market_sync_rejects_invalid_target_points_before_database_access(monkeypatch):
    monkeypatch.setattr(
        "cli.initialize_database",
        lambda: (_ for _ in ()).throw(
            AssertionError("invalid input must not initialize the database")
        ),
    )

    result = runner.invoke(app, ["market", "sync", "--target-points", "0"])

    assert result.exit_code == 1
    assert _payload(result) == {
        "ok": False,
        "command": "market sync",
        "error": {
            "stage": "input",
            "message": "--target-points는 1 이상이어야 합니다.",
            "hint": None,
        },
    }


def test_policy_validate_emits_one_json_object(monkeypatch, tmp_path):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.list_observed_identities", lambda: [("US", "AAA")])
    policy = {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "trailing_12_month_twr",
            "minimum_history_days": 365,
        },
        "risk_review": {
            "lookback_sessions": 252,
            "minimum_history_points": 200,
            "max_data_age_days": 7,
            "max_gap_days": 7,
            "account_drawdown_review": -0.15,
            "instrument_drawdown_review": {
                "core": -0.25,
                "satellite": -0.20,
                "experiment": -0.15,
            },
        },
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 0.8, "target": 1.0, "maximum": 1.0},
            "satellite": {"minimum": 0.0, "target": 0.0, "maximum": 0.0},
            "experiment": {"minimum": 0.0, "target": 0.0, "maximum": 0.0},
        },
        "instruments": [
            {
                "market_country": "US",
                "symbol": "AAA",
                "layer": "core",
                "minimum": 0.8,
                "target": 1.0,
                "maximum": 1.0,
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    result = runner.invoke(app, ["policy", "validate", "--file", str(path)])
    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["policy"]["performance"]["annual_return_target"] == 0.1
