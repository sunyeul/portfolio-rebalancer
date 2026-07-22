"""HTTP transport that makes Toss order mutation unreachable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from integrations.toss.config import TossApiConfig


TOKEN_PATH = "/oauth2/token"
ALLOWED_GET_PATHS = frozenset(
    {
        "/api/v1/accounts",
        "/api/v1/holdings",
        "/api/v1/buying-power",
        "/api/v1/exchange-rate",
        "/api/v1/orders",
    }
)


class TossRequestBlocked(RuntimeError):
    """Raised before a non-read-only request can reach the network."""


class TossTransportError(RuntimeError):
    """Sanitized Toss transport failure."""


def _normalized_path(path: str) -> str:
    if not path.startswith("/") or "?" in path or "#" in path:
        raise TossRequestBlocked("Toss request blocked by read-only policy.")
    return path.rstrip("/") or "/"


def _assert_allowed(method: str, path: str) -> tuple[str, str]:
    normalized_method = method.upper()
    normalized_path = _normalized_path(path)
    if normalized_method == "GET" and normalized_path in ALLOWED_GET_PATHS:
        return normalized_method, normalized_path
    if normalized_method == "POST" and normalized_path == TOKEN_PATH:
        return normalized_method, normalized_path
    raise TossRequestBlocked("Toss request blocked by read-only policy.")


class TossTransport:
    """Send only explicitly allowlisted Toss observation/auth requests."""

    def __init__(self, config: TossApiConfig, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(10.0),
        )
        self._owns_client = client is None

    def request_json(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str | int] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        allowed_method, allowed_path = _assert_allowed(method, path)
        try:
            response = self._client.request(
                allowed_method,
                allowed_path,
                headers=dict(headers or {}),
                params=dict(params or {}),
                data=dict(data or {}),
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise TossTransportError(
                f"Toss API request failed: status={exc.response.status_code} path={allowed_path}"
            ) from None
        except (httpx.HTTPError, ValueError, TypeError):
            raise TossTransportError(
                f"Toss API request failed: status=unavailable path={allowed_path}"
            ) from None
        if not isinstance(payload, dict):
            raise TossTransportError(
                f"Toss API request failed: status=invalid-json path={allowed_path}"
            )
        return payload

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
