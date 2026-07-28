"""Read-only export of persisted Toss inputs for Qlib research validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, time
from functools import cache
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Literal
from zoneinfo import ZoneInfo

from research.qlib_validation.contracts import Candle, SeriesSpec, SourceSnapshot
from services.dynamic_allocation import DynamicAllocationError, allocation_benchmarks


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


class SourceError(RuntimeError):
    """Raised when a reproducible research input cannot be read safely."""


@cache
def _availability() -> dict[str, tuple[ZoneInfo, time]]:
    rules = json.loads(PROTOCOL_PATH.read_text())["availability_rules"]
    return {
        market: (
            ZoneInfo(value["timezone"]),
            time.fromisoformat(value["conservative_close"]),
        )
        for market, value in rules.items()
    }


@contextmanager
def open_readonly(path: Path) -> Iterator[sqlite3.Connection]:
    """Open an existing database in SQLite's immutable read-only mode."""
    database = Path(path)
    if not database.is_file():
        raise SourceError(f"database not found: {database}")
    conn = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        yield conn
    finally:
        conn.close()


def _require_aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SourceError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _available_at(candle_at: datetime, market_country: str) -> datetime:
    """Return the conservative UTC availability time for a market session."""
    _require_aware(candle_at, label="candle timestamp")
    market = market_country.upper()
    rule = _availability().get(market)
    if rule is None:
        raise SourceError(f"unsupported market availability rule: {market_country}")
    timezone, conservative_close = rule
    session_date = candle_at.astimezone(timezone).date()
    return datetime.combine(session_date, conservative_close, timezone).astimezone(UTC)


def _active_policy(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, account_alias, version, policy_json, policy_hash, created_at
        FROM ips_policy_versions
        WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL
        ORDER BY version DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SourceError("active toss-brokerage policy not found")
    try:
        policy = json.loads(row["policy_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise SourceError("active policy JSON is invalid") from error
    if not isinstance(policy, dict):
        raise SourceError("active policy JSON must be an object")
    return {
        "id": int(row["id"]),
        "account_alias": str(row["account_alias"]),
        "version": int(row["version"]),
        "policy": policy,
        "policy_hash": str(row["policy_hash"]),
        "created_at": str(row["created_at"]),
    }


def _text(value: Any, *, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise SourceError(f"{label} is required")
    return result


def _weight(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SourceError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise SourceError(f"{label} must be finite")
    return result


def _series_spec(
    item: Mapping[str, Any], *, role: Literal["benchmark", "policy_instrument"]
) -> SeriesSpec:
    market_country = _text(item.get("market_country"), label="market_country").upper()
    if market_country not in _availability():
        raise SourceError(f"unsupported market availability rule: {market_country}")
    symbol = _text(item.get("symbol"), label="symbol").upper()
    key = (
        _text(item.get("key"), label="benchmark key")
        if role == "benchmark"
        else f"{market_country}/{symbol}"
    )
    source_kind = (
        _text(item.get("source_kind"), label="source_kind")
        if role == "benchmark"
        else "stock"
    )
    return SeriesSpec(
        key=key,
        source_kind=source_kind,
        market_country=market_country,
        symbol=symbol,
        weight=_weight(
            item.get("weight" if role == "benchmark" else "target"), label="weight"
        ),
        role=role,
    )


def _specs(
    policy: Mapping[str, Any],
) -> tuple[tuple[SeriesSpec, ...], tuple[SeriesSpec, ...]]:
    try:
        benchmarks = allocation_benchmarks(policy)
    except DynamicAllocationError as error:
        raise SourceError(
            f"dynamic allocation policy configuration is invalid: {error}"
        ) from error
    benchmark_specs = tuple(
        sorted(
            (_series_spec(item, role="benchmark") for item in benchmarks),
            key=lambda item: item.key,
        )
    )

    instruments = policy.get("instruments")
    if not isinstance(instruments, list):
        raise SourceError("active policy instruments are required")
    policy_specs = tuple(
        sorted(
            (
                _series_spec(item, role="policy_instrument")
                for item in instruments
                if isinstance(item, Mapping)
            ),
            key=lambda item: item.key,
        )
    )
    if len(policy_specs) != len(instruments):
        raise SourceError("active policy instruments must be objects")
    if any(item.weight < 0 for item in policy_specs):
        raise SourceError("policy instrument targets must not be negative")
    total = sum(item.weight for item in policy_specs)
    if total <= 0:
        raise SourceError("active policy requires a positive-total policy instrument")
    policy_specs = tuple(
        SeriesSpec(
            key=item.key,
            source_kind=item.source_kind,
            market_country=item.market_country,
            symbol=item.symbol,
            weight=item.weight / total,
            role=item.role,
        )
        for item in policy_specs
    )
    return benchmark_specs, policy_specs


def _candle_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SourceError("stored candle timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SourceError(f"stored candle timestamp is invalid: {value}") from error
    return _require_aware(parsed, label="stored candle timestamp")


def _load_series(
    conn: sqlite3.Connection, spec: SeriesSpec, *, as_of: datetime
) -> tuple[Candle, ...]:
    adjusted_filter = (
        " AND adjusted = 1 AND adjusted_supported = 1"
        if spec.source_kind == "stock"
        else ""
    )
    rows = conn.execute(
        """
        SELECT source_kind, market_country, symbol, candle_at, currency,
               open_price, high_price, low_price, close_price, volume,
               adjusted, adjusted_supported
        FROM toss_market_candles
        WHERE source_kind = ? AND market_country = ? AND symbol = ? AND interval = '1d'
        """
        + adjusted_filter
        + " ORDER BY datetime(candle_at), id",
        (spec.source_kind, spec.market_country, spec.symbol),
    ).fetchall()
    result: list[Candle] = []
    timezone = _availability()[spec.market_country][0]
    for row in rows:
        candle_at = _candle_at(row["candle_at"])
        available_at = _available_at(candle_at, spec.market_country)
        if available_at > as_of:
            continue
        result.append(
            Candle(
                key=spec.key,
                source_kind=str(row["source_kind"]),
                market_country=str(row["market_country"]),
                symbol=str(row["symbol"]),
                session_date=candle_at.astimezone(timezone).date(),
                candle_at=candle_at,
                available_at=available_at,
                currency=str(row["currency"]),
                open_price=float(row["open_price"]),
                high_price=float(row["high_price"]),
                low_price=float(row["low_price"]),
                close_price=float(row["close_price"]),
                volume=float(row["volume"]),
                adjusted=bool(row["adjusted"]),
                adjusted_supported=bool(row["adjusted_supported"]),
                factor=None,
            )
        )
    return tuple(result)


def load_snapshot(path: Path, *, as_of: datetime) -> SourceSnapshot:
    """Load one point-in-time, immutable research input snapshot."""
    as_of_utc = _require_aware(as_of, label="as_of")
    with open_readonly(path) as conn:
        policy_record = _active_policy(conn)
        benchmark_specs, policy_specs = _specs(policy_record["policy"])
        unique_specs: dict[str, SeriesSpec] = {}
        for spec in (*benchmark_specs, *policy_specs):
            existing = unique_specs.get(spec.key)
            if existing is not None and (
                existing.source_kind,
                existing.market_country,
                existing.symbol,
            ) != (spec.source_kind, spec.market_country, spec.symbol):
                raise SourceError(
                    f"conflicting series specifications for key: {spec.key}"
                )
            unique_specs.setdefault(spec.key, spec)
        candles = tuple(
            candle
            for spec in sorted(unique_specs.values(), key=lambda item: item.key)
            for candle in _load_series(conn, spec, as_of=as_of_utc)
        )
    return SourceSnapshot(
        policy_record=policy_record,
        benchmark_specs=benchmark_specs,
        policy_specs=policy_specs,
        candles=candles,
    )
