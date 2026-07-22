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
