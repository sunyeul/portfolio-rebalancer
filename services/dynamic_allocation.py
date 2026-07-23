"""Pure market-regime evidence and policy-target construction.

The module produces inspection candidates only. It does not activate policies or
translate target gaps into orders.
"""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from services.policy_validation import LAYERS, validate_policy


REGIME_TARGETS: dict[str, dict[str, Any]] = {
    "risk_on": {
        "cash_target": 0.03,
        "layers": {"core": 0.52, "satellite": 0.44, "experiment": 0.04},
    },
    "neutral": {
        "cash_target": 0.05,
        "layers": {"core": 0.60, "satellite": 0.38, "experiment": 0.02},
    },
    "risk_off": {
        "cash_target": 0.10,
        "layers": {"core": 0.70, "satellite": 0.29, "experiment": 0.01},
    },
}

DEFAULT_ALLOCATION_REVIEW: dict[str, Any] = {
    "strategy": "us_kr_three_regime_v1",
    "cooldown_days": 30,
    "minimum_history_points": 200,
    "max_data_age_days": 7,
    "max_gap_days": 7,
    "drawdown_review": -0.15,
    "volatility_review": 0.30,
    "risk_on_trend": 0.50,
    "risk_off_trend": -0.50,
    "risk_on_max_risk_weight": 0.30,
    "risk_off_risk_weight": 0.50,
    "benchmarks": [
        {
            "key": "US/SPY",
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "SPY",
            "weight": 0.30,
        },
        {
            "key": "US/QQQ",
            "source_kind": "stock",
            "market_country": "US",
            "symbol": "QQQ",
            "weight": 0.30,
        },
        {
            "key": "KR/KOSPI",
            "source_kind": "market_indicator",
            "market_country": "KR",
            "symbol": "KOSPI",
            "weight": 0.25,
        },
        {
            "key": "KR/KOSDAQ",
            "source_kind": "market_indicator",
            "market_country": "KR",
            "symbol": "KOSDAQ",
            "weight": 0.15,
        },
    ],
    "regimes": deepcopy(REGIME_TARGETS),
}

APPROVED_CASH_RANGE = {"minimum": 0.03, "target": 0.05, "maximum": 0.10}
APPROVED_LAYER_RANGES = {
    "core": {"minimum": 0.50, "target": 0.60, "maximum": 0.70},
    "satellite": {"minimum": 0.28, "target": 0.38, "maximum": 0.48},
    "experiment": {"minimum": 0.00, "target": 0.02, "maximum": 0.04},
}


class DynamicAllocationError(ValueError):
    """Raised when a complete review policy cannot be constructed."""


def _identity(item: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("market_country", "")).strip().upper(),
        str(item.get("symbol", "")).strip().upper(),
    )


def _ordered(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: _identity(item))


def _proportional_values(
    items: Sequence[dict[str, Any]],
    *,
    total: float,
    weights: Mapping[tuple[str, str], float],
    layer: str,
) -> dict[tuple[str, str], float]:
    ordered = _ordered(items)
    if not ordered:
        raise DynamicAllocationError(f"{layer} source layer is empty")
    denominator = sum(float(weights.get(_identity(item), 0.0)) for item in ordered)
    if not math.isfinite(denominator) or denominator <= 0:
        raise DynamicAllocationError(f"{layer} source layer target must be positive")
    result: dict[tuple[str, str], float] = {}
    subtotal = 0.0
    for item in ordered[:-1]:
        identity = _identity(item)
        value = total * float(weights.get(identity, 0.0)) / denominator
        result[identity] = value
        subtotal += value
    result[_identity(ordered[-1])] = total - subtotal
    return result


def _assign_layer_total(
    items: Sequence[dict[str, Any]],
    *,
    field: str,
    total: float,
    weights: Mapping[tuple[str, str], float],
    layer: str,
) -> None:
    values = _proportional_values(items, total=total, weights=weights, layer=layer)
    for item in items:
        item[field] = values[_identity(item)]


def build_neutral_policy(active_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return the approved neutral baseline derived from an active policy."""
    policy = deepcopy(dict(active_policy))
    raw_instruments = policy.get("instruments")
    if not isinstance(raw_instruments, list):
        raise DynamicAllocationError("active policy instruments are required")
    instruments = [deepcopy(item) for item in raw_instruments if isinstance(item, dict)]
    gld_matches = [item for item in instruments if _identity(item) == ("US", "GLD")]
    if len(gld_matches) != 1:
        raise DynamicAllocationError("exactly one US/GLD instrument is required")
    gld = gld_matches[0]
    original_weights = {
        _identity(item): float(item.get("target", 0.0)) for item in instruments
    }
    gld["layer"] = "core"
    core_others = [
        item
        for item in instruments
        if item is not gld and str(item.get("layer", "")).lower() == "core"
    ]
    satellite = [
        item
        for item in instruments
        if str(item.get("layer", "")).lower() == "satellite"
    ]
    experiment = [
        item
        for item in instruments
        if str(item.get("layer", "")).lower() == "experiment"
    ]
    _assign_layer_total(
        core_others,
        field="target",
        total=0.50,
        weights=original_weights,
        layer="core",
    )
    gld["target"] = 0.10
    _assign_layer_total(
        satellite,
        field="target",
        total=0.38,
        weights=original_weights,
        layer="satellite",
    )
    _assign_layer_total(
        experiment,
        field="target",
        total=0.02,
        weights=original_weights,
        layer="experiment",
    )
    policy["cash_reserve"] = deepcopy(APPROVED_CASH_RANGE)
    policy["layers"] = deepcopy(APPROVED_LAYER_RANGES)
    for layer in LAYERS:
        items = [item for item in instruments if item.get("layer") == layer]
        neutral_weights = {_identity(item): float(item["target"]) for item in items}
        _assign_layer_total(
            items,
            field="minimum",
            total=APPROVED_LAYER_RANGES[layer]["minimum"],
            weights=neutral_weights,
            layer=layer,
        )
        _assign_layer_total(
            items,
            field="maximum",
            total=APPROVED_LAYER_RANGES[layer]["maximum"],
            weights=neutral_weights,
            layer=layer,
        )
    policy["instruments"] = _ordered(instruments)
    policy["allocation_review"] = deepcopy(DEFAULT_ALLOCATION_REVIEW)
    return policy


def scale_policy_to_regime(policy: Mapping[str, Any], regime: str) -> dict[str, Any]:
    """Scale instrument targets within each layer to one configured regime."""
    result = deepcopy(dict(policy))
    config = result.get("allocation_review")
    if not isinstance(config, dict):
        raise DynamicAllocationError(
            "allocation_review policy configuration is required"
        )
    regimes = config.get("regimes")
    if not isinstance(regimes, dict) or regime not in regimes:
        raise DynamicAllocationError(f"unknown allocation regime: {regime}")
    preset = regimes[regime]
    result["cash_reserve"]["target"] = float(preset["cash_target"])
    instruments = result.get("instruments")
    if not isinstance(instruments, list):
        raise DynamicAllocationError("policy instruments are required")
    for layer in LAYERS:
        target = float(preset["layers"][layer])
        result["layers"][layer]["target"] = target
        items = [item for item in instruments if item.get("layer") == layer]
        weights = {_identity(item): float(item.get("target", 0.0)) for item in items}
        _assign_layer_total(
            items,
            field="target",
            total=target,
            weights=weights,
            layer=layer,
        )
    result["instruments"] = _ordered(instruments)
    return result


def target_summary(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the gross-cash and invested-layer targets."""
    return {
        "cash_target": float(policy["cash_reserve"]["target"]),
        "layers": {layer: float(policy["layers"][layer]["target"]) for layer in LAYERS},
    }


def allocation_benchmarks(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the validated benchmark specifications for an active policy."""
    config = policy.get("allocation_review")
    if not isinstance(config, dict) or not isinstance(config.get("benchmarks"), list):
        raise DynamicAllocationError(
            "allocation_review benchmark configuration is required"
        )
    return deepcopy(config["benchmarks"])


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


def _quality_failure(
    reason: str,
    *,
    history_points: int,
    verification_task: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "history_points": history_points,
        **details,
        "verification_task": verification_task,
    }


def _series_evidence(
    candles: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    points: list[tuple[datetime, float]] = []
    invalid_points = 0
    for candle in candles:
        timestamp = _timestamp(candle.get("candle_at", candle.get("timestamp")))
        close = _close(candle)
        if timestamp is None or close is None:
            invalid_points += 1
            continue
        points.append((timestamp, close))
    ordered = sorted(points, key=lambda item: item[0])
    timestamps = [item[0] for item in ordered]
    closes = [item[1] for item in ordered]
    duplicate_timestamps = len(timestamps) != len(set(timestamps))
    if invalid_points or duplicate_timestamps:
        return _quality_failure(
            "market_data_invalid",
            history_points=len(closes),
            invalid_points=invalid_points,
            duplicate_timestamps=duplicate_timestamps,
            verification_task="Toss 시장 데이터의 봉 시각과 종가 무결성을 확인합니다.",
        )
    minimum_history = int(config["minimum_history_points"])
    if len(closes) < minimum_history:
        return _quality_failure(
            "market_history_insufficient",
            history_points=len(closes),
            verification_task="Toss 일봉 이력과 필수 시장 지표 지원 여부를 확인합니다.",
        )
    latest_at = timestamps[-1]
    data_age_days = (now - latest_at).total_seconds() / 86400
    if data_age_days < -1:
        return _quality_failure(
            "market_data_future_timestamp",
            history_points=len(closes),
            latest_candle_at=latest_at.isoformat(),
            data_age_days=data_age_days,
            verification_task="시장 봉 시각과 현재 시각의 기준을 확인합니다.",
        )
    if data_age_days > float(config["max_data_age_days"]):
        return _quality_failure(
            "market_data_stale",
            history_points=len(closes),
            latest_candle_at=latest_at.isoformat(),
            data_age_days=data_age_days,
            verification_task="시장 데이터의 마지막 수집 시각과 최신 영업일을 확인합니다.",
        )
    gaps = [
        (later - earlier).total_seconds() / 86400
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    largest_gap_days = max(gaps, default=0.0)
    if largest_gap_days > float(config["max_gap_days"]):
        return _quality_failure(
            "market_history_gap",
            history_points=len(closes),
            latest_candle_at=latest_at.isoformat(),
            data_age_days=data_age_days,
            largest_gap_days=largest_gap_days,
            verification_task="시장 이력의 누락 구간과 Toss 페이지네이션을 확인합니다.",
        )
    lookback = closes[-252:]
    latest = lookback[-1]
    drawdown = latest / max(lookback) - 1.0
    log_returns = [
        math.log(lookback[index] / lookback[index - 1])
        for index in range(1, len(lookback))
    ]
    volatility = pstdev(log_returns[-20:]) * math.sqrt(252)
    medium_mean = mean(lookback[-60:])
    long_mean = mean(lookback[-200:])
    medium_trend = latest / medium_mean - 1.0
    long_trend = latest / long_mean - 1.0
    trend_direction = (
        1
        if medium_trend > 0 and long_trend > 0
        else -1
        if medium_trend < 0 and long_trend < 0
        else 0
    )
    severe_risk = bool(
        drawdown <= float(config["drawdown_review"])
        or volatility >= float(config["volatility_review"])
    )
    return {
        "valid": True,
        "reason": "market_data_valid",
        "history_points": len(closes),
        "latest_candle_at": latest_at.isoformat(),
        "data_age_days": data_age_days,
        "largest_gap_days": largest_gap_days,
        "latest_close": latest,
        "drawdown": drawdown,
        "realized_volatility": volatility,
        "medium_trend": medium_trend,
        "long_trend": long_trend,
        "trend_direction": trend_direction,
        "severe_risk": severe_risk,
        "verification_task": "시장 지표의 추세·낙폭·변동성 근거를 함께 검토합니다.",
    }


def _same_targets(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return math.isclose(
        float(left["cash_target"]),
        float(right["cash_target"]),
        abs_tol=1e-9,
    ) and all(
        math.isclose(
            float(left["layers"][layer]),
            float(right["layers"][layer]),
            abs_tol=1e-9,
        )
        for layer in LAYERS
    )


def evaluate_dynamic_allocation(
    series_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    active_policy: Mapping[str, Any],
    last_change_at: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate required market series and return a human-review policy candidate."""
    config = active_policy.get("allocation_review")
    if not isinstance(config, dict):
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": "allocation_review_missing",
            "regime": None,
            "benchmarks": {},
            "failed_benchmarks": [],
            "current_targets": target_summary(active_policy),
            "proposed_targets": None,
            "proposed_policy": None,
            "verification_task": "활성 정책의 동적 목표 비중 검토 설정을 확인합니다.",
        }
    raw_now = now or datetime.now(timezone.utc)
    now_value = raw_now.replace(tzinfo=raw_now.tzinfo or timezone.utc).astimezone(
        timezone.utc
    )
    evidence: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for benchmark in config["benchmarks"]:
        key = str(benchmark["key"])
        item = _series_evidence(
            series_by_key.get(key, []), now=now_value, config=config
        )
        item["weight"] = float(benchmark["weight"])
        evidence[key] = item
        if not item["valid"]:
            failed.append(key)
    current_targets = target_summary(active_policy)
    if failed:
        first = evidence[failed[0]]
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "reason": first["reason"],
            "regime": None,
            "weighted_trend": None,
            "severe_risk_weight": None,
            "benchmarks": evidence,
            "failed_benchmarks": failed,
            "current_targets": current_targets,
            "proposed_targets": None,
            "proposed_policy": None,
            "cooling": False,
            "verification_task": first["verification_task"],
        }
    weighted_trend = sum(
        float(item["weight"]) * int(item["trend_direction"])
        for item in evidence.values()
    )
    severe_risk_weight = sum(
        float(item["weight"]) for item in evidence.values() if item["severe_risk"]
    )
    if weighted_trend <= float(config["risk_off_trend"]) or severe_risk_weight >= float(
        config["risk_off_risk_weight"]
    ):
        regime = "risk_off"
    elif weighted_trend >= float(
        config["risk_on_trend"]
    ) and severe_risk_weight < float(config["risk_on_max_risk_weight"]):
        regime = "risk_on"
    else:
        regime = "neutral"
    proposed_policy = scale_policy_to_regime(active_policy, regime)
    identities = [_identity(item) for item in proposed_policy["instruments"]]
    proposed_policy = validate_policy(proposed_policy, identities)
    proposed_targets = target_summary(proposed_policy)
    targets_changed = not _same_targets(current_targets, proposed_targets)
    last_change = _timestamp(last_change_at)
    cooling = bool(
        last_change is not None
        and 0
        <= (now_value - last_change).total_seconds()
        < int(config["cooldown_days"]) * 86400
    )
    if not targets_changed:
        status = "OK"
        candidate_state = "observe"
        reason = "regime_targets_already_active"
    elif cooling:
        status = "Watch"
        candidate_state = "observe"
        reason = "allocation_review_cooling_period"
    else:
        status = "Review"
        candidate_state = "candidate"
        reason = "regime_target_change"
    return {
        "status": status,
        "candidate_state": candidate_state,
        "reason": reason,
        "regime": regime,
        "weighted_trend": weighted_trend,
        "severe_risk_weight": severe_risk_weight,
        "benchmarks": evidence,
        "failed_benchmarks": [],
        "current_targets": current_targets,
        "proposed_targets": proposed_targets,
        "proposed_policy": proposed_policy,
        "cooling": cooling,
        "cooldown_days": int(config["cooldown_days"]),
        "verification_task": "목표 비중 후보의 시장 근거와 정책 범위를 승인 전에 확인합니다.",
    }
