"""In-memory OAuth token handling for read-only Toss observations."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from integrations.toss.config import TossApiConfig
from integrations.toss.transport import TossTransport, TossTransportError


TOKEN_REFRESH_SKEW_SECONDS = 60.0


@dataclass(frozen=True)
class _CachedToken:
    value: str = field(repr=False)
    refresh_at: float


class TossTokenProvider:
    """Issue and cache an OAuth token in process memory only."""

    def __init__(
        self,
        config: TossApiConfig,
        transport: TossTransport,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._transport = transport
        self._clock = clock
        self._cached: _CachedToken | None = None

    def access_token(self) -> str:
        now = self._clock()
        if self._cached is not None and now < self._cached.refresh_at:
            return self._cached.value

        payload = self._transport.request_json(
            "POST",
            "/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            },
        )
        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        try:
            lifetime = float(expires_in)
        except (TypeError, ValueError):
            lifetime = 0.0
        if not isinstance(token, str) or not token or lifetime <= 0:
            raise TossTransportError("Toss API request failed: invalid token response")

        refresh_at = now + max(1.0, lifetime - TOKEN_REFRESH_SKEW_SECONDS)
        self._cached = _CachedToken(value=token, refresh_at=refresh_at)
        return token

    def invalidate(self) -> None:
        self._cached = None


class TossAuthorizedReader:
    """Add OAuth and account headers to allowlisted Toss GET observations."""

    def __init__(
        self,
        config: TossApiConfig,
        transport: TossTransport,
        tokens: TossTokenProvider,
    ):
        self._config = config
        self._transport = transport
        self._tokens = tokens

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> dict[str, Any]:
        return self._transport.request_json(
            "GET",
            path,
            headers={
                "Authorization": f"Bearer {self._tokens.access_token()}",
                "X-Tossinvest-Account": str(self._config.account_seq),
            },
            params=params,
        )
