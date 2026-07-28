"""Read-only export of persisted Toss inputs for Qlib research validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
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
from storage.policy_store import policy_hash


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
REQUIRED_BENCHMARKS = {
    "KR/KOSDAQ": ("market_indicator", "KR", "KOSDAQ"),
    "KR/KOSPI": ("market_indicator", "KR", "KOSPI"),
    "US/QQQ": ("stock", "US", "QQQ"),
    "US/SPY": ("stock", "US", "SPY"),
}
FACTOR_LOOKBACK_SESSIONS = 20
UNIVERSE_MODES = frozenset({"active-policy", "current-holdings"})


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
        conn.execute("BEGIN")
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
    stored_hash = str(row["policy_hash"])
    if policy_hash(policy) != stored_hash:
        raise SourceError("active policy hash mismatch")
    return {
        "id": int(row["id"]),
        "account_alias": str(row["account_alias"]),
        "version": int(row["version"]),
        "policy": policy,
        "policy_hash": stored_hash,
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
    if not benchmark_specs:
        raise SourceError("active policy requires benchmark specifications")
    if len({item.key for item in benchmark_specs}) != len(benchmark_specs):
        raise SourceError("active policy benchmarks contain duplicate keys")
    if len(
        {
            (item.source_kind, item.market_country, item.symbol)
            for item in benchmark_specs
        }
    ) != len(benchmark_specs):
        raise SourceError("active policy benchmarks contain duplicate identities")
    if any(item.weight < 0 for item in benchmark_specs):
        raise SourceError("benchmark weights must not be negative")
    benchmark_total = sum(item.weight for item in benchmark_specs)
    if not math.isclose(benchmark_total, 1.0, abs_tol=1e-9):
        raise SourceError("benchmark weights must sum to one")
    actual_benchmarks = {
        item.key: (item.source_kind, item.market_country, item.symbol)
        for item in benchmark_specs
    }
    if actual_benchmarks != REQUIRED_BENCHMARKS:
        raise SourceError("active policy must contain exactly the required identities")

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
    if len({item.key for item in policy_specs}) != len(policy_specs):
        raise SourceError("active policy instruments contain duplicate identities")
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


def _current_holdings_universe(
    conn: sqlite3.Connection,
    policy_specs: tuple[SeriesSpec, ...],
    *,
    as_of: datetime,
) -> tuple[tuple[SeriesSpec, ...], dict[str, Any]]:
    """Select active-policy instruments held in the current complete Toss snapshot."""
    snapshot = conn.execute(
        """
        SELECT id, synced_at, state
        FROM broker_account_snapshots
        WHERE account_alias = 'toss-brokerage'
          AND datetime(synced_at) <= datetime(?)
        ORDER BY synced_at DESC, id DESC
        LIMIT 1
        """,
        (as_of.isoformat(),),
    ).fetchone()
    if snapshot is None:
        raise SourceError("Toss snapshot is unavailable at the research as_of time")
    if str(snapshot["state"]) != "complete":
        raise SourceError("latest Toss snapshot at research as_of is not complete")
    rows = conn.execute(
        """
        SELECT market_country, symbol
        FROM broker_holdings
        WHERE snapshot_id = ? AND quantity > 0
        ORDER BY market_country, symbol, id
        """,
        (int(snapshot["id"]),),
    ).fetchall()
    holding_identities = {
        (
            _text(row["market_country"], label="holding market_country").upper(),
            _text(row["symbol"], label="holding symbol").upper(),
        )
        for row in rows
    }
    if not holding_identities:
        raise SourceError("current complete Toss snapshot has no positive holdings")
    selected = tuple(
        item
        for item in policy_specs
        if (item.market_country, item.symbol) in holding_identities
    )
    if not selected:
        raise SourceError(
            "current Toss holdings do not match active policy instruments"
        )
    selected_keys = [item.key for item in selected]
    return selected, {
        "mode": "current-holdings",
        "account_snapshot_id": int(snapshot["id"]),
        "account_snapshot_synced_at": str(snapshot["synced_at"]),
        "selected_policy_instruments": selected_keys,
        "excluded_policy_instruments": [
            item.key for item in policy_specs if item.key not in set(selected_keys)
        ],
    }


def _candle_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SourceError("stored candle timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SourceError(f"stored candle timestamp is invalid: {value}") from error
    return _require_aware(parsed, label="stored candle timestamp")


def _maximum_gap_days(policy: Mapping[str, Any]) -> int:
    config = policy.get("allocation_review")
    value = config.get("max_gap_days") if isinstance(config, Mapping) else None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SourceError(
            "allocation_review.max_gap_days must be a positive integer"
        ) from error
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise SourceError("allocation_review.max_gap_days must be a positive integer")
    return int(number)


def _validate_series(
    spec: SeriesSpec, candles: tuple[Candle, ...], *, maximum_gap_days: int
) -> None:
    seen_sessions = set()
    previous: Candle | None = None
    for candle in candles:
        prices = (
            candle.open_price,
            candle.high_price,
            candle.low_price,
            candle.close_price,
        )
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise SourceError(f"invalid positive OHLC for series: {spec.key}")
        if candle.low_price > min(
            candle.open_price, candle.close_price
        ) or candle.high_price < max(candle.open_price, candle.close_price):
            raise SourceError(f"invalid OHLC relationship for series: {spec.key}")
        if not math.isfinite(candle.volume) or candle.volume < 0:
            raise SourceError(f"invalid volume for series: {spec.key}")
        if candle.session_date in seen_sessions:
            raise SourceError(f"duplicate session for series: {spec.key}")
        seen_sessions.add(candle.session_date)
        if previous is not None:
            gap_days = (candle.session_date - previous.session_date).days
            if gap_days > maximum_gap_days:
                raise SourceError(
                    f"history gap exceeds policy limit for series: {spec.key}"
                )
        previous = candle


def _load_series(
    conn: sqlite3.Connection, spec: SeriesSpec, *, as_of: datetime
) -> tuple[Candle, ...]:
    adjusted_filter = (
        " AND current.adjusted = 1 AND current.adjusted_supported = 1"
        if spec.source_kind == "stock"
        else ""
    )
    rows = conn.execute(
        """
        SELECT current.source_kind, current.market_country, current.symbol,
               current.candle_at, current.currency, current.open_price,
               current.high_price, current.low_price, current.close_price,
               current.volume, current.adjusted, current.adjusted_supported
        FROM toss_market_candles AS current
        WHERE current.source_kind = ? AND current.market_country = ?
          AND current.symbol = ? AND current.interval = '1d'
          AND current.id = (
              SELECT MAX(revision.id)
              FROM toss_market_candles AS revision
              WHERE revision.source_kind = current.source_kind
                AND revision.market_country = current.market_country
                AND revision.symbol = current.symbol
                AND revision.interval = current.interval
                AND revision.candle_at = current.candle_at
                AND revision.adjusted = current.adjusted
          )
        """
        + adjusted_filter
        + " ORDER BY datetime(current.candle_at), current.id",
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


def _with_trailing_factors(candles: tuple[Candle, ...]) -> tuple[Candle, ...]:
    """Attach a factor using only each row's prior observed closes."""
    if len(candles) <= FACTOR_LOOKBACK_SESSIONS:
        return ()
    result: list[Candle] = []
    for index in range(FACTOR_LOOKBACK_SESSIONS, len(candles)):
        baseline = candles[index - FACTOR_LOOKBACK_SESSIONS]
        current = candles[index]
        result.append(
            replace(
                current,
                factor=current.close_price / baseline.close_price - 1.0,
            )
        )
    return tuple(result)


def load_snapshot(
    path: Path,
    *,
    as_of: datetime,
    universe: Literal["active-policy", "current-holdings"] = "active-policy",
) -> SourceSnapshot:
    """Load one point-in-time, immutable research input snapshot."""
    as_of_utc = _require_aware(as_of, label="as_of")
    if universe not in UNIVERSE_MODES:
        raise SourceError(f"unsupported research universe: {universe}")
    with open_readonly(path) as conn:
        policy_record = _active_policy(conn)
        benchmark_specs, policy_specs = _specs(policy_record["policy"])
        research_universe: dict[str, Any] = {"mode": "active-policy"}
        if universe == "current-holdings":
            policy_specs, research_universe = _current_holdings_universe(
                conn, policy_specs, as_of=as_of_utc
            )
        maximum_gap_days = _maximum_gap_days(policy_record["policy"])
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
        loaded: list[Candle] = []
        for spec in sorted(unique_specs.values(), key=lambda item: item.key):
            series = _load_series(conn, spec, as_of=as_of_utc)
            _validate_series(spec, series, maximum_gap_days=maximum_gap_days)
            loaded.extend(_with_trailing_factors(series))
        candles = tuple(loaded)
    return SourceSnapshot(
        policy_record=policy_record,
        benchmark_specs=benchmark_specs,
        policy_specs=policy_specs,
        candles=candles,
        research_universe=research_universe,
    )
