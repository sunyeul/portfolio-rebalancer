from fastapi.testclient import TestClient
from typer.testing import CliRunner

from api.app import create_app
from cli import app


def test_api_exposes_only_the_reduced_read_surface(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    client = TestClient(create_app())

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/inspection").status_code == 200
    assert client.get("/api/account").status_code == 404
    assert client.get("/api/market-context").status_code == 404
    assert client.get("/api/policy-preflight").status_code == 404


def test_removed_cli_commands_keep_json_parse_errors():
    result = CliRunner().invoke(app, ["market", "context"])

    assert result.exit_code != 0
    assert '"ok": false' in result.output
