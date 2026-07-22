"""Environment-backed Toss Open API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_TOSS_BASE_URL = "https://openapi.tossinvest.com"


class TossConfigError(RuntimeError):
    """Raised when required Toss configuration is absent or invalid."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise TossConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class TossApiConfig:
    """Server-only Toss credentials and endpoint configuration."""

    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    account_seq: int = field(repr=False)
    base_url: str = DEFAULT_TOSS_BASE_URL

    @classmethod
    def from_env(cls) -> "TossApiConfig":
        account_seq_text = _required_env("TOSS_OPEN_API_ACCOUNT_SEQ")
        try:
            account_seq = int(account_seq_text)
        except ValueError as exc:
            raise TossConfigError(
                "TOSS_OPEN_API_ACCOUNT_SEQ must be a positive integer."
            ) from exc
        if account_seq <= 0:
            raise TossConfigError(
                "TOSS_OPEN_API_ACCOUNT_SEQ must be a positive integer."
            )
        return cls(
            client_id=_required_env("TOSS_OPEN_API_CLIENT_ID"),
            client_secret=_required_env("TOSS_OPEN_API_CLIENT_SECRET"),
            account_seq=account_seq,
        )
