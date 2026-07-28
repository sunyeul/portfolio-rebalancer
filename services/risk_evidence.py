"""Pure profit/loss and drawdown evidence for Toss IPS inspection."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_return(pnl: float | None, cost_basis: float | None) -> float | None:
    if pnl is None or cost_basis is None or cost_basis <= 0:
        return None
    result = pnl / cost_basis
    return result if math.isfinite(result) else None


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unavailable_drawdown(state: str, history_points: int = 0) -> dict[str, Any]:
    return {
        "state": state,
        "current": None,
        "maximum": None,
        "history_points": history_points,
    }


def _market_descriptor(
    state: str,
    candles: Sequence[Mapping[str, Any]],
    history_points: int = 0,
) -> dict[str, Any]:
    return {
        "state": state,
        "history_points": history_points,
        "raw_points": len(candles),
        "source_fingerprint": _fingerprint(
            [
                str(candle.get("source_fingerprint") or _fingerprint(dict(candle)))
                for candle in candles
            ]
        ),
    }


def _curve_drawdown(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timestamped = [(_timestamp(point.get("point_at")), point) for point in points]
    if any(timestamp is None for timestamp, _ in timestamped):
        return _unavailable_drawdown("invalid_point_timestamp")
    ordered = sorted(
        ((timestamp, point) for timestamp, point in timestamped if timestamp),
        key=lambda item: item[0],
    )
    if len(ordered) < 2:
        return _unavailable_drawdown("insufficient_history", len(ordered))
    if any(point.get("evaluation_state") != "evaluable" for _, point in ordered):
        return _unavailable_drawdown("boundary_unresolved")

    curve = 1.0
    peak = 1.0
    drawdowns: list[float] = []
    for _, point in ordered:
        interval = _finite(point.get("interval_twr"))
        if interval is None or interval <= -1:
            return _unavailable_drawdown("invalid_interval", len(ordered))
        curve *= 1.0 + interval
        peak = max(peak, curve)
        drawdowns.append(curve / peak - 1.0)
    return {
        "state": "complete",
        "current": drawdowns[-1],
        "maximum": min(drawdowns),
        "history_points": len(ordered),
        "first_point_at": ordered[0][0].isoformat(),
        "latest_point_at": ordered[-1][0].isoformat(),
    }


def _position_totals(
    projection: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    if projection is None:
        return None, None
    positions = list(projection.get("positions", []))
    costs = [_finite(position.get("cost_krw")) for position in positions]
    pnls = [_finite(position.get("profit_loss_krw")) for position in positions]
    cost = (
        sum(costs) if positions and all(value is not None for value in costs) else None
    )
    pnl = sum(pnls) if positions and all(value is not None for value in pnls) else None
    return cost, pnl


def build_account_profit_loss(
    snapshot_id: int,
    performance_run: Mapping[str, Any] | None,
    projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate account facts without assigning an IPS status."""
    points = list((performance_run or {}).get("points", []))
    latest = points[-1] if points else {}
    fallback_cost, fallback_pnl = _position_totals(projection)
    cost_basis = _finite(latest.get("current_cost_basis_krw")) or fallback_cost
    unrealized_pnl = _finite(latest.get("unrealized_pnl_krw"))
    if unrealized_pnl is None:
        unrealized_pnl = fallback_pnl
    run_state = (performance_run or {}).get("state")
    execution_count = int((performance_run or {}).get("execution_count") or 0)
    data_quality = (performance_run or {}).get("data_quality") or {}
    realized_supported = bool(
        run_state == "complete"
        and execution_count > 0
        and not data_quality.get("issues")
        and latest.get("tracked_realized_pnl_krw") is not None
        and latest.get("actual_realized_pnl_krw") is not None
    )
    return {
        "snapshot_id": snapshot_id,
        "performance_run_id": (performance_run or {}).get("id"),
        "state": "source_unavailable"
        if performance_run is None
        else ("complete" if run_state == "complete" else "performance_unavailable"),
        "cost_basis_krw": cost_basis,
        "unrealized_pnl_krw": unrealized_pnl,
        "unrealized_return": _safe_return(unrealized_pnl, cost_basis),
        "tracked_realized_pnl_krw": (
            _finite(latest.get("tracked_realized_pnl_krw"))
            if realized_supported
            else None
        ),
        "actual_realized_pnl_krw": (
            _finite(latest.get("actual_realized_pnl_krw"))
            if realized_supported
            else None
        ),
        "realized_pnl_supported": realized_supported,
        "drawdown": _curve_drawdown(points)
        if performance_run
        else _unavailable_drawdown("performance_unavailable"),
    }


def _candle_drawdown(
    candles: Sequence[Mapping[str, Any]],
    *,
    through_at: Any,
    risk_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    through = _timestamp(through_at)
    if through is None:
        return _unavailable_drawdown("invalid_snapshot_timestamp"), _market_descriptor(
            "invalid_snapshot_timestamp", candles
        )
    eligible: list[tuple[datetime, float, Mapping[str, Any]]] = []
    seen: set[datetime] = set()
    for candle in candles:
        timestamp = _timestamp(candle.get("candle_at"))
        if timestamp is None:
            return _unavailable_drawdown("invalid_candle"), _market_descriptor(
                "invalid_candle", candles
            )
        if timestamp > through:
            continue
        if timestamp in seen:
            return _unavailable_drawdown("duplicate_timestamp"), _market_descriptor(
                "duplicate_timestamp", candles
            )
        seen.add(timestamp)
        if not bool(candle.get("adjusted")) or not bool(
            candle.get("adjusted_supported")
        ):
            return _unavailable_drawdown("unadjusted"), _market_descriptor(
                "unadjusted", candles
            )
        close = _finite(candle.get("close_price"))
        if close is None or close <= 0:
            return _unavailable_drawdown("invalid_candle"), _market_descriptor(
                "invalid_candle", candles
            )
        eligible.append((timestamp, close, candle))
    eligible.sort(key=lambda item: item[0])
    lookback = int(risk_policy["lookback_sessions"])
    selected = eligible[-lookback:]
    minimum = int(risk_policy["minimum_history_points"])
    if len(selected) < minimum:
        return _unavailable_drawdown(
            "insufficient_history", len(selected)
        ), _market_descriptor("insufficient_history", candles, len(selected))
    latest_timestamp = selected[-1][0]
    age_days = (through - latest_timestamp).total_seconds() / 86400
    if age_days > int(risk_policy["max_data_age_days"]):
        return _unavailable_drawdown("stale", len(selected)), _market_descriptor(
            "stale", candles, len(selected)
        )
    max_gap_days = int(risk_policy["max_gap_days"])
    for previous, current in zip(selected, selected[1:]):
        gap_days = (current[0] - previous[0]).total_seconds() / 86400
        if gap_days > max_gap_days:
            return _unavailable_drawdown("gap", len(selected)), _market_descriptor(
                "gap", candles, len(selected)
            )

    curve = selected[0][1]
    peak = curve
    drawdowns: list[float] = [0.0]
    for _, close, _ in selected[1:]:
        peak = max(peak, close)
        drawdowns.append(close / peak - 1.0)
    source_fingerprints = [
        str(candle.get("source_fingerprint") or _fingerprint(dict(candle)))
        for _, _, candle in selected
    ]
    descriptor = {
        "state": "complete",
        "first_candle_at": selected[0][0].isoformat(),
        "latest_candle_at": selected[-1][0].isoformat(),
        "history_points": len(selected),
        "source_fingerprint": _fingerprint(source_fingerprints),
    }
    return (
        {
            "state": "complete",
            "current": drawdowns[-1],
            "maximum": min(drawdowns),
            "lookback_sessions": lookback,
            "history_points": len(selected),
            **descriptor,
        },
        descriptor,
    )


def build_risk_evidence(
    projection: Mapping[str, Any] | None,
    performance_run: Mapping[str, Any] | None,
    candles_by_identity: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    risk_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return source-linked facts only; never assign IPS status."""
    snapshot_id = int(projection.get("snapshot_id", 0)) if projection else 0
    account = build_account_profit_loss(snapshot_id, performance_run, projection)
    instruments: dict[str, Any] = {}
    market_evidence: dict[str, Any] = {}
    if projection is not None:
        for position in projection.get("positions", []):
            country = str(position.get("market_country", "")).upper()
            symbol = str(position.get("symbol", "")).upper()
            identity = f"{country}/{symbol}"
            market_value = _finite(position.get("market_value_krw"))
            cost_basis = _finite(position.get("cost_krw"))
            unrealized_pnl = _finite(position.get("profit_loss_krw"))
            drawdown, descriptor = _candle_drawdown(
                candles_by_identity.get((country, symbol), []),
                through_at=projection.get("synced_at"),
                risk_policy=risk_policy,
            )
            instruments[identity] = {
                "snapshot_id": snapshot_id,
                "market_value_krw": market_value,
                "cost_basis_krw": cost_basis,
                "unrealized_pnl_krw": unrealized_pnl,
                "unrealized_return": _safe_return(unrealized_pnl, cost_basis),
                "drawdown": drawdown,
            }
            market_evidence[identity] = descriptor
    return {
        "schema_version": 1,
        "account_profit_loss": account,
        "instruments": instruments,
        "market_evidence": market_evidence,
        "market_evidence_fingerprint": _fingerprint(market_evidence),
    }
