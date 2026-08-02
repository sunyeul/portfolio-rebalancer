"""Read-only Toss market-data normalization for Phase 4 context review."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from integrations.toss.auth import TossAuthorizedReader


class MarketObservationError(RuntimeError):
    """Raised when an official Toss market response cannot be normalized."""


@dataclass(frozen=True)
class NormalizedCandle:
    source_kind: str
    market_country: str
    symbol: str
    interval: str
    candle_at: str
    currency: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    adjusted: bool
    adjusted_supported: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MarketObservationError(f"invalid {field}") from exc
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise MarketObservationError(f"invalid {field}")
    result = float(parsed)
    if not math.isfinite(result):
        raise MarketObservationError(f"invalid {field}")
    return result


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketObservationError("invalid candle timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MarketObservationError("invalid candle timestamp") from exc
    return value


class TossMarketDataService:
    """Collect official Toss daily candles with bounded backward pagination."""

    def __init__(self, reader: TossAuthorizedReader):
        self._reader = reader

    def read_page(
        self,
        *,
        symbol: str,
        market_country: str = "",
        interval: str = "1d",
        count: int = 200,
        before: str | None = None,
        adjusted: bool = True,
    ) -> tuple[list[NormalizedCandle], str | None]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise MarketObservationError("symbol is required")
        if interval != "1d":
            raise MarketObservationError("only daily candles are supported")
        bounded_count = max(1, min(int(count), 200))
        path = "/api/v1/candles"
        params: dict[str, str | int] = {
            "interval": interval,
            "count": bounded_count,
        }
        params["symbol"] = normalized_symbol
        params["adjusted"] = "true" if adjusted else "false"
        if before is not None:
            params["before"] = before
        payload = self._reader.get_json(
            path, params=params, include_account_header=False
        )
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise MarketObservationError("missing candle result")
        raw_candles = result.get("candles")
        if not isinstance(raw_candles, list):
            raise MarketObservationError("missing candle list")
        normalized: list[NormalizedCandle] = []
        for raw in raw_candles:
            if not isinstance(raw, Mapping):
                raise MarketObservationError("invalid candle item")
            open_price = _number(raw.get("openPrice"), "openPrice")
            high_price = _number(raw.get("highPrice"), "highPrice")
            low_price = _number(raw.get("lowPrice"), "lowPrice")
            close_price = _number(raw.get("closePrice"), "closePrice")
            if low_price > min(open_price, close_price) or high_price < max(
                open_price, close_price
            ):
                raise MarketObservationError("candle OHLC invariant failed")
            raw_currency = raw.get("currency")
            if not isinstance(raw_currency, str):
                raise MarketObservationError("missing candle currency")
            normalized.append(
                NormalizedCandle(
                    source_kind="stock",
                    market_country=market_country.upper(),
                    symbol=normalized_symbol,
                    interval=interval,
                    candle_at=_timestamp(raw.get("timestamp")),
                    currency=str(raw_currency or "").upper(),
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=_number(raw.get("volume"), "volume", nonnegative=True),
                    adjusted=adjusted,
                    adjusted_supported=True,
                )
            )
        next_before = result.get("nextBefore")
        if next_before is not None and not isinstance(next_before, str):
            raise MarketObservationError("invalid nextBefore")
        return normalized, next_before

    def collect_history(
        self,
        *,
        symbol: str,
        market_country: str = "",
        count: int = 200,
        max_pages: int = 5,
        target_points: int | None = None,
        adjusted: bool = True,
    ) -> list[NormalizedCandle]:
        if max_pages < 1:
            raise MarketObservationError("max_pages must be positive")
        if target_points is not None and target_points < 1:
            raise MarketObservationError("target_points must be positive")
        candles: dict[str, NormalizedCandle] = {}
        before: str | None = None
        seen_cursors: set[str | None] = {None}
        for _ in range(max_pages):
            page, next_before = self.read_page(
                symbol=symbol,
                market_country=market_country,
                count=count,
                before=before,
                adjusted=adjusted,
            )
            for candle in page:
                candles[candle.candle_at] = candle
            if target_points is not None and len(candles) >= target_points:
                break
            if not next_before:
                break
            if next_before in seen_cursors:
                raise MarketObservationError("repeated candle cursor")
            seen_cursors.add(next_before)
            before = next_before
        else:
            raise MarketObservationError("candle history exceeds max_pages")
        ordered = sorted(candles.values(), key=lambda candle: candle.candle_at)
        return ordered[-target_points:] if target_points is not None else ordered
