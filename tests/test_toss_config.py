import tomllib
from pathlib import Path

import pytest

from integrations.toss.config import TossApiConfig, TossConfigError
from integrations.toss.redaction import REDACTED, redact_headers, redact_known_values


def test_toss_config_loads_required_environment_without_secret_repr(monkeypatch):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-visible-only-to-server")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "super-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "42")

    config = TossApiConfig.from_env()

    assert config.client_id == "client-visible-only-to-server"
    assert config.client_secret == "super-secret"
    assert config.account_seq == 42
    assert config.base_url == "https://openapi.tossinvest.com"
    assert "client-visible-only-to-server" not in repr(config)
    assert "super-secret" not in repr(config)
    assert "42" not in repr(config)


@pytest.mark.parametrize(
    "missing_name",
    [
        "TOSS_OPEN_API_CLIENT_ID",
        "TOSS_OPEN_API_CLIENT_SECRET",
        "TOSS_OPEN_API_ACCOUNT_SEQ",
    ],
)
def test_toss_config_reports_missing_variable_name_only(monkeypatch, missing_name):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-id")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "7")
    monkeypatch.delenv(missing_name)

    with pytest.raises(TossConfigError) as exc_info:
        TossApiConfig.from_env()

    assert missing_name in str(exc_info.value)
    assert "client-id" not in str(exc_info.value)
    assert "client-secret" not in str(exc_info.value)


def test_toss_config_rejects_non_positive_account_sequence(monkeypatch):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-id")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "0")

    with pytest.raises(TossConfigError, match="positive integer"):
        TossApiConfig.from_env()


def test_sensitive_headers_and_known_values_are_redacted():
    headers = redact_headers(
        {
            "Authorization": "Bearer access-token",
            "X-Tossinvest-Account": "42",
            "Accept": "application/json",
        }
    )

    assert headers == {
        "Authorization": REDACTED,
        "X-Tossinvest-Account": REDACTED,
        "Accept": "application/json",
    }
    assert (
        redact_known_values(
            "client-id client-secret access-token account-1234",
            ["client-id", "client-secret", "access-token", "account-1234"],
        )
        == f"{REDACTED} {REDACTED} {REDACTED} {REDACTED}"
    )


def test_toss_runtime_dependencies_and_package_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert "httpx>=0.28.0" in pyproject["project"]["dependencies"]
    assert "httpx>=0.28.0" not in pyproject["dependency-groups"]["dev"]
    assert (
        "integrations*"
        in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    )
