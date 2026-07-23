"""Pure Toss-market context signals for candidate cash-policy review."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MarketContextThresholds:
    minimum_history_points: int = 200
    max_data_age_days: int = 7
    max_gap_days: int = 7
    drawdown_review: float = -0.15
    volatility_review: float = 0.30
    positive_trend_floor: float = 0.05
    cooling_period_days: int = 30
    cash_floor: float = 0.10
    cash_target: float = 0.15
    cash_ceiling: float = 0.20


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _close(candle: Mapping[str, Any]) -> float | None:
    try:
        value = float(candle.get("close_price", candle.get("closePrice")))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _drawdown(closes: Sequence[float]) -> float:
    peak = max(closes)
    return closes[-1] / peak - 1.0 if peak else 0.0


def _volatility(closes: Sequence[float]) -> float | None:
    if len(closes) < 21:
        return None
    returns = [
        math.log(closes[index] / closes[index - 1]) for index in range(1, len(closes))
    ]
    window = returns[-20:]
    return pstdev(window) * math.sqrt(252) if window else None


def evaluate_market_context(
    candles: Sequence[Mapping[str, Any]],
    *,
    current_target: float = 0.15,
    last_change_at: str | None = None,
    now: datetime | None = None,
    thresholds: MarketContextThresholds = MarketContextThresholds(),
) -> dict[str, Any]:
    points: list[tuple[datetime, float]] = []
    invalid_points = 0
    for candle in candles:
        close = _close(candle)
        timestamp = _timestamp(candle.get("candle_at", candle.get("timestamp")))
        if close is None or timestamp is None:
            invalid_points += 1
            continue
        points.append((timestamp, close))
    ordered = sorted(points, key=lambda item: item[0])
    timestamps = [timestamp for timestamp, _ in ordered]
    valid_closes = [close for _, close in ordered]
    duplicate_timestamps = len(set(timestamps)) != len(timestamps)
    if invalid_points or duplicate_timestamps:
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "market_data_invalid",
            "history_points": len(valid_closes),
            "invalid_points": invalid_points,
            "duplicate_timestamps": duplicate_timestamps,
            "thresholds": asdict(thresholds),
            "signals": [],
            "current_target": current_target,
            "proposed_target": None,
            "verification_task": "Toss 시장 데이터의 봉 시각과 종가 무결성을 확인합니다.",
        }
    if len(valid_closes) < thresholds.minimum_history_points:
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "market_history_insufficient",
            "history_points": len(valid_closes),
            "thresholds": asdict(thresholds),
            "signals": [],
            "current_target": current_target,
            "proposed_target": None,
            "verification_task": "Toss 일봉 이력과 시장 지표 지원 여부를 확인합니다.",
        }

    raw_now = now or datetime.now(timezone.utc)
    now_value = raw_now.replace(tzinfo=raw_now.tzinfo or timezone.utc).astimezone(
        timezone.utc
    )
    latest_at = timestamps[-1]
    data_age_days = (now_value - latest_at).total_seconds() / 86400
    if data_age_days < -1:
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "market_data_future_timestamp",
            "history_points": len(valid_closes),
            "latest_candle_at": latest_at.isoformat(),
            "data_age_days": data_age_days,
            "thresholds": asdict(thresholds),
            "signals": [],
            "current_target": current_target,
            "proposed_target": None,
            "verification_task": "시장 봉 시각과 로컬 현재 시각의 기준을 확인합니다.",
        }
    if data_age_days > thresholds.max_data_age_days:
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "market_data_stale",
            "history_points": len(valid_closes),
            "latest_candle_at": latest_at.isoformat(),
            "data_age_days": data_age_days,
            "thresholds": asdict(thresholds),
            "signals": [],
            "current_target": current_target,
            "proposed_target": None,
            "verification_task": "시장 데이터 마지막 수집 시각과 최신 영업일을 확인합니다.",
        }
    gaps = [
        (later - earlier).total_seconds() / 86400
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    largest_gap_days = max(gaps, default=0.0)
    if largest_gap_days > thresholds.max_gap_days:
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "market_history_gap",
            "history_points": len(valid_closes),
            "latest_candle_at": latest_at.isoformat(),
            "data_age_days": data_age_days,
            "largest_gap_days": largest_gap_days,
            "thresholds": asdict(thresholds),
            "signals": [],
            "current_target": current_target,
            "proposed_target": None,
            "verification_task": "시장 이력의 누락 구간과 Toss 페이지네이션 완결 여부를 확인합니다.",
        }

    latest = valid_closes[-1]
    drawdown = _drawdown(valid_closes)
    volatility = _volatility(valid_closes)
    medium_mean = mean(valid_closes[-60:])
    long_mean = mean(valid_closes[-200:])
    medium_trend = latest / medium_mean - 1.0 if medium_mean else None
    long_trend = latest / long_mean - 1.0 if long_mean else None
    signals = [
        {
            "name": "benchmark_drawdown",
            "value": drawdown,
            "threshold": thresholds.drawdown_review,
            "confirmed": drawdown <= thresholds.drawdown_review,
            "meaning": "벤치마크가 최근 고점 대비 승인된 검토 폭을 벗어났는지 확인합니다.",
            "verification_task": "하락 원인과 IPS 현금 범위 검토 필요성을 확인합니다.",
        },
        {
            "name": "realized_volatility",
            "value": volatility,
            "threshold": thresholds.volatility_review,
            "confirmed": volatility is not None
            and volatility >= thresholds.volatility_review,
            "meaning": "최근 실현 변동성이 검토 기준을 넘었는지 확인합니다.",
            "verification_task": "변동성 수치의 출처 시각과 시장 범위를 확인합니다.",
        },
        {
            "name": "medium_long_trend",
            "value": {"medium": medium_trend, "long": long_trend},
            "threshold": 0.0,
            "confirmed": bool(
                medium_trend is not None
                and long_trend is not None
                and medium_trend < 0
                and long_trend < 0
            ),
            "meaning": "중기·장기 기준선 아래에 동시에 있는지 확인합니다.",
            "verification_task": "추세 신호가 일시 현상인지 정책 검토 신호인지 확인합니다.",
        },
    ]
    confirmed = sum(1 for signal in signals if signal["confirmed"])
    positive_trend = bool(
        medium_trend is not None
        and long_trend is not None
        and medium_trend >= thresholds.positive_trend_floor
        and long_trend >= thresholds.positive_trend_floor
        and drawdown > thresholds.drawdown_review / 3
    )
    if confirmed >= 2:
        proposed_target = thresholds.cash_ceiling
    elif positive_trend:
        proposed_target = thresholds.cash_floor
    else:
        proposed_target = thresholds.cash_target
    proposed_target = min(
        thresholds.cash_ceiling, max(thresholds.cash_floor, proposed_target)
    )
    current_target = min(
        thresholds.cash_ceiling, max(thresholds.cash_floor, current_target)
    )
    last_change = _timestamp(last_change_at)
    cooling = bool(
        last_change is not None
        and (now_value.astimezone(timezone.utc) - last_change).days
        < thresholds.cooling_period_days
    )
    eligible = confirmed >= 2 and proposed_target != current_target and not cooling
    if (
        confirmed == 0
        and positive_trend
        and proposed_target != current_target
        and not cooling
    ):
        eligible = True
    return {
        "status": "Review"
        if confirmed >= 2
        else "Watch"
        if proposed_target != current_target
        else "OK",
        "candidate_state": "candidate" if eligible else "observe",
        "reason": "multiple_confirming_signals"
        if confirmed >= 2
        else "positive_trend_or_no_change",
        "history_points": len(valid_closes),
        "latest_close": latest,
        "drawdown": drawdown,
        "realized_volatility": volatility,
        "medium_trend": medium_trend,
        "long_trend": long_trend,
        "confirmed_signal_count": confirmed,
        "cooling": cooling,
        "thresholds": asdict(thresholds),
        "signals": signals,
        "current_target": current_target,
        "proposed_target": proposed_target,
        "verification_task": "후보 현금 목표를 활성화하기 전에 임계값·히스테리시스와 시장 데이터 신선도를 승인합니다.",
    }
