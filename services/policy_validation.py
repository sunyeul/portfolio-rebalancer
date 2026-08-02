"""Validation and activation rules for app-owned Toss IPS policy intent."""

from __future__ import annotations

import math
from typing import Any, Iterable

from storage.policy_store import canonical_policy_json, policy_hash


LAYERS = ("core", "satellite", "experiment")
SUM_TOLERANCE = 1e-9


class PolicyValidationError(ValueError):
    """Raised when a policy cannot be safely evaluated or activated."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _number(value: Any, path: str, errors: list[str]) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{path} must be a finite number")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be a finite number")
        return None
    if not math.isfinite(number) or not 0 <= number <= 1:
        errors.append(f"{path} must be between 0 and 1")
        return None
    return number


def _range(value: Any, path: str, errors: list[str]) -> dict[str, float] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    result: dict[str, float] = {}
    for key in ("minimum", "target", "maximum"):
        number = _number(value.get(key), f"{path}.{key}", errors)
        if number is not None:
            result[key] = number
    if len(result) == 3 and not (
        result["minimum"] <= result["target"] <= result["maximum"]
    ):
        errors.append(f"{path} must satisfy minimum <= target <= maximum")
    return result if len(result) == 3 else None


def _positive_integer(value: Any, path: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{path} must be a positive integer")
        return None
    return value


def _negative_rate(value: Any, path: str, errors: list[str]) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    if not math.isfinite(number) or not -1 < number < 0:
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    return number


def _signed_unit(value: Any, path: str, errors: list[str]) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{path} must be a finite number between -1 and 1")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be a finite number between -1 and 1")
        return None
    if not math.isfinite(number) or not -1 <= number <= 1:
        errors.append(f"{path} must be a finite number between -1 and 1")
        return None
    return number


def _allocation_benchmarks(value: Any, errors: list[str]) -> list[dict[str, Any]]:
    path = "allocation_review.benchmarks"
    if not isinstance(value, list):
        errors.append(f"{path} must be an array")
        return []
    required = {
        "US/SPY": ("stock", "US", "SPY"),
        "US/QQQ": ("stock", "US", "QQQ"),
        "KR/KOSPI": ("market_indicator", "KR", "KOSPI"),
        "KR/KOSDAQ": ("market_indicator", "KR", "KOSDAQ"),
    }
    result: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_identities: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{item_path} must be an object")
            continue
        key = str(raw.get("key", "")).strip().upper()
        source_kind = str(raw.get("source_kind", "")).strip().lower()
        market_country = str(raw.get("market_country", "")).strip().upper()
        symbol = str(raw.get("symbol", "")).strip().upper()
        weight = _number(raw.get("weight"), f"{item_path}.weight", errors)
        identity = (source_kind, market_country, symbol)
        if not key:
            errors.append(f"{item_path}.key is required")
        if source_kind not in {"stock", "market_indicator"}:
            errors.append(f"{item_path}.source_kind is invalid")
        if not market_country or not symbol:
            errors.append(f"{item_path} requires market_country and symbol")
        if key in seen_keys or identity in seen_identities:
            errors.append(f"{path} contains a duplicate key or identity")
        seen_keys.add(key)
        seen_identities.add(identity)
        if weight is not None:
            result.append(
                {
                    "key": key,
                    "source_kind": source_kind,
                    "market_country": market_country,
                    "symbol": symbol,
                    "weight": weight,
                }
            )
    if set(seen_keys) != set(required):
        errors.append(
            f"{path} must contain exactly US/SPY, US/QQQ, KR/KOSPI, KR/KOSDAQ"
        )
    for item in result:
        expected = required.get(item["key"])
        actual = (item["source_kind"], item["market_country"], item["symbol"])
        if expected is not None and actual != expected:
            errors.append(f"{path} identity does not match {item['key']}")
    if not math.isclose(
        sum(float(item["weight"]) for item in result),
        1.0,
        abs_tol=SUM_TOLERANCE,
    ):
        errors.append(f"{path} weights must sum to 1")
    return result


def _allocation_regimes(
    value: Any,
    cash: dict[str, float] | None,
    layers: dict[str, dict[str, float]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    path = "allocation_review.regimes"
    regimes = ("risk_on", "neutral", "risk_off")
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    if set(value) != set(regimes):
        errors.append(f"{path} must contain exactly risk_on, neutral, risk_off")
    result: dict[str, dict[str, Any]] = {}
    for regime in regimes:
        raw = value.get(regime)
        item_path = f"{path}.{regime}"
        if not isinstance(raw, dict):
            errors.append(f"{item_path} must be an object")
            continue
        cash_target = _number(
            raw.get("cash_target"), f"{item_path}.cash_target", errors
        )
        raw_layer_targets = raw.get("layers")
        if not isinstance(raw_layer_targets, dict):
            errors.append(f"{item_path}.layers must be an object")
            continue
        if set(raw_layer_targets) != set(LAYERS):
            errors.append(
                f"{item_path}.layers must contain exactly core, satellite, experiment"
            )
        layer_targets: dict[str, float] = {}
        for layer in LAYERS:
            target = _number(
                raw_layer_targets.get(layer),
                f"{item_path}.layers.{layer}",
                errors,
            )
            if target is not None:
                layer_targets[layer] = target
        if len(layer_targets) == len(LAYERS) and not math.isclose(
            sum(layer_targets.values()), 1.0, abs_tol=SUM_TOLERANCE
        ):
            errors.append(f"{item_path} layer targets must sum to 1")
        if (
            cash_target is not None
            and cash is not None
            and not (cash["minimum"] <= cash_target <= cash["maximum"])
        ):
            errors.append(f"{item_path} cash target must be within cash_reserve range")
        for layer, target in layer_targets.items():
            layer_range = layers.get(layer)
            if layer_range is not None and not (
                layer_range["minimum"] <= target <= layer_range["maximum"]
            ):
                errors.append(
                    f"{item_path}.layers.{layer} must be within layers.{layer} range"
                )
        if cash_target is not None and len(layer_targets) == len(LAYERS):
            result[regime] = {
                "cash_target": cash_target,
                "layers": layer_targets,
            }
    return result


def _allocation_review(
    value: Any,
    cash: dict[str, float] | None,
    layers: dict[str, dict[str, float]],
    errors: list[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append("allocation_review must be an object")
        return None
    before = len(errors)
    strategy = str(value.get("strategy", "")).strip()
    if strategy != "us_kr_three_regime_v1":
        errors.append("allocation_review.strategy must be us_kr_three_regime_v1")
    cooldown_days = _positive_integer(
        value.get("cooldown_days"), "allocation_review.cooldown_days", errors
    )
    minimum_history_points = _positive_integer(
        value.get("minimum_history_points"),
        "allocation_review.minimum_history_points",
        errors,
    )
    max_data_age_days = _positive_integer(
        value.get("max_data_age_days"),
        "allocation_review.max_data_age_days",
        errors,
    )
    max_gap_days = _positive_integer(
        value.get("max_gap_days"), "allocation_review.max_gap_days", errors
    )
    drawdown_review = _negative_rate(
        value.get("drawdown_review"), "allocation_review.drawdown_review", errors
    )
    volatility_review = _number(
        value.get("volatility_review"),
        "allocation_review.volatility_review",
        errors,
    )
    risk_on_trend = _signed_unit(
        value.get("risk_on_trend"), "allocation_review.risk_on_trend", errors
    )
    risk_off_trend = _signed_unit(
        value.get("risk_off_trend"), "allocation_review.risk_off_trend", errors
    )
    risk_on_max_risk_weight = _number(
        value.get("risk_on_max_risk_weight"),
        "allocation_review.risk_on_max_risk_weight",
        errors,
    )
    risk_off_risk_weight = _number(
        value.get("risk_off_risk_weight"),
        "allocation_review.risk_off_risk_weight",
        errors,
    )
    benchmarks = _allocation_benchmarks(value.get("benchmarks"), errors)
    regimes = _allocation_regimes(value.get("regimes"), cash, layers, errors)
    if (
        risk_off_trend is not None
        and risk_on_trend is not None
        and risk_off_trend >= risk_on_trend
    ):
        errors.append("allocation_review risk_off_trend must be below risk_on_trend")
    if (
        risk_on_max_risk_weight is not None
        and risk_off_risk_weight is not None
        and risk_on_max_risk_weight > risk_off_risk_weight
    ):
        errors.append(
            "allocation_review risk_on_max_risk_weight must not exceed risk_off_risk_weight"
        )
    if len(errors) != before:
        return None
    return {
        "strategy": strategy,
        "cooldown_days": cooldown_days,
        "minimum_history_points": minimum_history_points,
        "max_data_age_days": max_data_age_days,
        "max_gap_days": max_gap_days,
        "drawdown_review": drawdown_review,
        "volatility_review": volatility_review,
        "risk_on_trend": risk_on_trend,
        "risk_off_trend": risk_off_trend,
        "risk_on_max_risk_weight": risk_on_max_risk_weight,
        "risk_off_risk_weight": risk_off_risk_weight,
        "benchmarks": benchmarks,
        "regimes": regimes,
    }


def _identity(
    item: dict[str, Any], index: int, errors: list[str]
) -> tuple[str, str] | None:
    symbol = str(item.get("symbol", "")).strip().upper()
    country = str(item.get("market_country", "")).strip().upper()
    if not symbol or not country:
        errors.append(f"instruments[{index}] requires market_country and symbol")
        return None
    return country, symbol


def validate_policy(
    policy: dict[str, Any],
    observed_identities: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Return a normalized policy or raise with every violated invariant."""
    errors: list[str] = []
    if not isinstance(policy, dict):
        raise PolicyValidationError(["policy must be an object"])

    cash = _range(policy.get("cash_reserve"), "cash_reserve", errors)
    performance = policy.get("performance")
    if not isinstance(performance, dict):
        errors.append("performance must be an object")
        performance = {}
    annual_target = _number(
        performance.get("annual_return_target"),
        "performance.annual_return_target",
        errors,
    )
    minimum_days = performance.get("minimum_history_days")
    if isinstance(minimum_days, bool) or not isinstance(minimum_days, int):
        errors.append("performance.minimum_history_days must be an integer")
        minimum_days = None
    elif minimum_days != 365:
        errors.append("performance.minimum_history_days must be exactly 365")
    measurement = str(performance.get("measurement", "")).strip()
    if measurement not in {"ytd_twr", "trailing_12_month_twr"}:
        errors.append(
            "performance.measurement must be ytd_twr or trailing_12_month_twr"
        )

    risk_review: dict[str, Any] = {}
    raw_risk_review = policy.get("risk_review")
    if not isinstance(raw_risk_review, dict):
        errors.append("risk_review must be an object")
        raw_risk_review = {}
    lookback_sessions = _positive_integer(
        raw_risk_review.get("lookback_sessions"),
        "risk_review.lookback_sessions",
        errors,
    )
    minimum_history_points = _positive_integer(
        raw_risk_review.get("minimum_history_points"),
        "risk_review.minimum_history_points",
        errors,
    )
    max_data_age_days = _positive_integer(
        raw_risk_review.get("max_data_age_days"),
        "risk_review.max_data_age_days",
        errors,
    )
    max_gap_days = _positive_integer(
        raw_risk_review.get("max_gap_days"),
        "risk_review.max_gap_days",
        errors,
    )
    if (
        lookback_sessions is not None
        and minimum_history_points is not None
        and minimum_history_points > lookback_sessions
    ):
        errors.append(
            "risk_review.minimum_history_points must be <= risk_review.lookback_sessions"
        )
    account_drawdown_review = _negative_rate(
        raw_risk_review.get("account_drawdown_review"),
        "risk_review.account_drawdown_review",
        errors,
    )
    raw_instrument_drawdown = raw_risk_review.get("instrument_drawdown_review")
    instrument_drawdown_review: dict[str, float] = {}
    if not isinstance(raw_instrument_drawdown, dict):
        errors.append("risk_review.instrument_drawdown_review must be an object")
        raw_instrument_drawdown = {}
    if set(raw_instrument_drawdown) != set(LAYERS):
        errors.append(
            "risk_review.instrument_drawdown_review must contain exactly core, satellite, experiment"
        )
    for layer in LAYERS:
        parsed = _negative_rate(
            raw_instrument_drawdown.get(layer),
            f"risk_review.instrument_drawdown_review.{layer}",
            errors,
        )
        if parsed is not None:
            instrument_drawdown_review[layer] = parsed
    if (
        lookback_sessions is not None
        and minimum_history_points is not None
        and max_data_age_days is not None
        and max_gap_days is not None
        and account_drawdown_review is not None
        and len(instrument_drawdown_review) == len(LAYERS)
    ):
        risk_review = {
            "lookback_sessions": lookback_sessions,
            "minimum_history_points": minimum_history_points,
            "max_data_age_days": max_data_age_days,
            "max_gap_days": max_gap_days,
            "account_drawdown_review": account_drawdown_review,
            "instrument_drawdown_review": instrument_drawdown_review,
        }

    cadence = policy.get("cadence")
    if not isinstance(cadence, dict):
        errors.append("cadence must be an object")
        cadence = {}
    observation = str(cadence.get("observation", "")).strip().lower()
    inspection = str(cadence.get("inspection", "")).strip().lower()
    if observation != "weekly":
        errors.append("cadence.observation must be weekly")
    if inspection != "monthly":
        errors.append("cadence.inspection must be monthly")

    layers: dict[str, dict[str, float]] = {}
    raw_layers = policy.get("layers")
    if not isinstance(raw_layers, dict):
        errors.append("layers must be an object")
        raw_layers = {}
    for layer in LAYERS:
        parsed = _range(raw_layers.get(layer), f"layers.{layer}", errors)
        if parsed is not None:
            layers[layer] = parsed
    if len(layers) == len(LAYERS):
        total = sum(item["target"] for item in layers.values())
        if not math.isclose(total, 1.0, abs_tol=SUM_TOLERANCE):
            errors.append("layers target values must sum to 1")

    observed = {
        (str(country).strip().upper(), str(symbol).strip().upper())
        for country, symbol in observed_identities
    }
    instruments: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    grouped = {
        layer: {"minimum": 0.0, "target": 0.0, "maximum": 0.0} for layer in LAYERS
    }
    raw_instruments = policy.get("instruments", [])
    if not isinstance(raw_instruments, list):
        errors.append("instruments must be an array")
        raw_instruments = []
    for index, raw in enumerate(raw_instruments):
        if not isinstance(raw, dict):
            errors.append(f"instruments[{index}] must be an object")
            continue
        identity = _identity(raw, index, errors)
        layer = str(raw.get("layer", "")).strip().lower()
        if layer not in LAYERS:
            errors.append(f"instruments[{index}].layer must be a valid layer")
        parsed = _range(raw, f"instruments[{index}]", errors)
        if identity is None or parsed is None or layer not in LAYERS:
            continue
        if identity in seen:
            errors.append(f"duplicate instrument identity: {identity[0]}/{identity[1]}")
        seen.add(identity)
        if identity not in observed:
            errors.append(
                f"instrument not observed by Toss: {identity[0]}/{identity[1]}"
            )
        if layer in layers:
            layer_target = layers[layer]
            for bound in ("minimum", "target", "maximum"):
                if parsed[bound] > layer_target["maximum"] + SUM_TOLERANCE:
                    errors.append(
                        f"instrument {identity[0]}/{identity[1]} exceeds {layer} maximum"
                    )
            if parsed["target"] > layer_target["target"] + SUM_TOLERANCE:
                errors.append(
                    f"instrument {identity[0]}/{identity[1]} target exceeds {layer} target"
                )
        for bound in grouped.get(layer, {}):
            grouped[layer][bound] += parsed[bound]
        instruments.append(
            {
                "market_country": identity[0],
                "symbol": identity[1],
                "layer": layer,
                **parsed,
            }
        )

    for layer, totals in grouped.items():
        if layer not in layers:
            continue
        target = layers[layer]["target"]
        if not math.isclose(totals["target"], target, abs_tol=SUM_TOLERANCE):
            errors.append(f"{layer} instrument targets must equal layer target")
        if totals["minimum"] > target + SUM_TOLERANCE:
            errors.append(f"{layer} instrument minimums exceed layer target")
        if totals["maximum"] < target - SUM_TOLERANCE:
            errors.append(f"{layer} instrument maximums do not cover layer target")

    if errors:
        raise PolicyValidationError(errors)
    normalized = {
        "cash_reserve": cash,
        "performance": {
            "annual_return_target": annual_target,
            "measurement": measurement,
            "minimum_history_days": minimum_days,
        },
        "risk_review": risk_review,
        "cadence": {"observation": observation, "inspection": inspection},
        "layers": layers,
        "instruments": sorted(
            instruments, key=lambda item: (item["market_country"], item["symbol"])
        ),
    }
    return normalized


def policy_metadata(policy: dict[str, Any]) -> dict[str, str]:
    """Return canonical persistence metadata for a validated policy."""
    return {
        "policy_json": canonical_policy_json(policy),
        "policy_hash": policy_hash(policy),
    }
