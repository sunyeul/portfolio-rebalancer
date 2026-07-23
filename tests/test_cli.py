import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from cli import app


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
        "profiles",
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
        lambda snapshot_id=None: {"snapshot_id": snapshot_id or 7},
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
        lambda snapshot_id=None: (_ for _ in ()).throw(
            ValueError("complete Toss snapshot not found")
        ),
    )

    result = runner.invoke(app, ["account-view"])

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["command"] == "account-view"
    assert payload["error"]["message"] == "complete Toss snapshot not found"


def test_profiles_set_only_emits_profile_metadata(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.upsert_profile",
        lambda **kwargs: {
            "account_alias": "toss-brokerage",
            "market_country": kwargs["market_country"].upper(),
            "symbol": kwargs["symbol"].upper(),
            "layer": kwargs["layer"],
            "thesis_status": kwargs["thesis_status"],
            "thesis_note": kwargs["thesis_note"],
        },
    )

    result = runner.invoke(
        app,
        [
            "profiles",
            "set",
            "--symbol",
            "AAPL",
            "--market-country",
            "US",
            "--layer",
            "core",
            "--thesis-status",
            "valid",
            "--note",
            "Long-term core",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["command"] == "profiles set"
    assert payload["profile"]["symbol"] == "AAPL"
    assert "quantity" not in payload["profile"]
    assert "market_value_krw" not in payload["profile"]


def test_policy_show_requires_active_flag(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    result = runner.invoke(app, ["policy", "show"])
    assert result.exit_code == 1
    assert _payload(result)["error"]["stage"] == "input"


def test_web_rejects_non_loopback_binding():
    result = runner.invoke(app, ["web", "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert _payload(result)["error"]["stage"] == "input"


def test_market_context_persists_candidate_without_activation(monkeypatch):
    candidate_calls = []
    active = {
        "id": 8,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-01T00:00:00+00:00",
        "policy": {"cash_reserve": {"target": 0.15}},
    }
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.get_active_policy", lambda: active)
    monkeypatch.setattr("cli.list_candles", lambda **_: [{"close_price": 1}])
    monkeypatch.setattr(
        "cli.evaluate_market_context",
        lambda candles, **kwargs: {
            "status": "Review",
            "candidate_state": "candidate",
            "current_target": 0.15,
            "proposed_target": 0.2,
            "history_points": 200,
            "verification_task": "verify",
        },
    )

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
    assert payload["activation"] == "human approval required; active policy unchanged"


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
