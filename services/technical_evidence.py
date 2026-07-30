"""Pure technical facts from validated adjusted Toss candles."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


def _unavailable(reason: str, history_points: int) -> dict[str, Any]:
    return {
        "state": "unavailable",
        "reason": reason,
        "history_points": history_points,
    }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _midpoint(highs: Sequence[float], lows: Sequence[float]) -> float:
    return (max(highs) + min(lows)) / 2.0


def build_technical_evidence(
    candles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return standard Ichimoku and Bollinger facts without policy judgment."""
    if len(candles) < 78:
        return _unavailable("technical_history_insufficient", len(candles))
    highs = [_number(item.get("high_price")) for item in candles]
    lows = [_number(item.get("low_price")) for item in candles]
    closes = [_number(item.get("close_price")) for item in candles]
    if any(value is None for value in highs + lows + closes):
        return _unavailable("technical_data_invalid", len(candles))
    high_values = [float(value) for value in highs if value is not None]
    low_values = [float(value) for value in lows if value is not None]
    close_values = [float(value) for value in closes if value is not None]
    if any(
        low > close or close > high or low > high
        for low, close, high in zip(low_values, close_values, high_values)
    ):
        return _unavailable("technical_data_invalid", len(candles))

    conversion = _midpoint(high_values[-9:], low_values[-9:])
    base = _midpoint(high_values[-26:], low_values[-26:])
    displaced_conversion = _midpoint(high_values[-35:-26], low_values[-35:-26])
    displaced_base = _midpoint(high_values[-52:-26], low_values[-52:-26])
    span_a = (displaced_conversion + displaced_base) / 2.0
    span_b = _midpoint(high_values[-78:-26], low_values[-78:-26])
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    latest = close_values[-1]
    price_position = (
        "above"
        if latest > cloud_top
        else "below"
        if latest < cloud_bottom
        else "inside"
    )
    line_alignment = (
        "positive"
        if conversion > base
        else "negative"
        if conversion < base
        else "mixed"
    )
    direction = (
        1
        if price_position == "above" and line_alignment == "positive"
        else -1
        if price_position == "below" and line_alignment == "negative"
        else 0
    )

    window = close_values[-20:]
    middle = mean(window)
    deviation = pstdev(window)
    upper = middle + 2.0 * deviation
    lower = middle - 2.0 * deviation
    width = upper - lower
    percent_b = (latest - lower) / width if width else 0.5
    extension = "above" if percent_b > 1 else "below" if percent_b < 0 else "inside"
    return {
        "state": "complete",
        "reason": "technical_evidence_complete",
        "history_points": len(candles),
        "ichimoku": {
            "conversion": conversion,
            "base": base,
            "span_a": span_a,
            "span_b": span_b,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "price_position": price_position,
            "line_alignment": line_alignment,
            "direction": direction,
        },
        "bollinger": {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": width / middle,
            "percent_b": percent_b,
            "extension": extension,
        },
    }
