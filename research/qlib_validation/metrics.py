from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
import random
from statistics import mean, pstdev
from typing import Any

from research.qlib_validation.contracts import ReplayPoint, SourceSnapshot


REQUIRED_EFFECTS = frozenset(
    {
        ("benchmarks", 20),
        ("benchmarks", 60),
        ("policy_instruments", 20),
        ("policy_instruments", 60),
    }
)
ALLOWED_BASKETS = frozenset({"benchmarks", "policy_instruments"})


@dataclass(frozen=True)
class SignalRule:
    horizons: tuple[int, ...]
    required_effects: frozenset[tuple[str, int]]
    supported_upper_ci_below: float
    not_supported_point_at_or_above: float


def parse_signal_rule(value: Any, horizons: Sequence[Any]) -> SignalRule:
    if not isinstance(value, Mapping):
        raise ValueError("signal_rule must be an object")
    if value.get("primary_metric") != "max_drawdown":
        raise ValueError("signal_rule.primary_metric must be max_drawdown")
    if value.get("comparison") != "risk_off_minus_other":
        raise ValueError("signal_rule.comparison must be risk_off_minus_other")
    raw_baskets = value.get("required_baskets")
    if not isinstance(raw_baskets, list) or not raw_baskets:
        raise ValueError("signal_rule.required_baskets must be a non-empty array")
    baskets = tuple(str(item) for item in raw_baskets)
    if len(set(baskets)) != len(baskets) or not set(baskets) <= ALLOWED_BASKETS:
        raise ValueError("signal_rule.required_baskets contains invalid values")
    parsed_horizons: list[int] = []
    for item in horizons:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError("signal_rule horizons must be positive integers")
        parsed_horizons.append(item)
    if not parsed_horizons or len(set(parsed_horizons)) != len(parsed_horizons):
        raise ValueError("signal_rule horizons must be non-empty and unique")
    try:
        supported = float(value.get("supported_upper_ci_below"))
        not_supported = float(value.get("not_supported_point_at_or_above"))
    except (TypeError, ValueError) as error:
        raise ValueError("signal_rule thresholds must be finite") from error
    if not isfinite(supported) or not isfinite(not_supported):
        raise ValueError("signal_rule thresholds must be finite")
    if supported > not_supported:
        raise ValueError("signal_rule thresholds are ambiguous")
    parsed = tuple(parsed_horizons)
    return SignalRule(
        horizons=parsed,
        required_effects=frozenset(
            (basket, horizon) for basket in baskets for horizon in parsed
        ),
        supported_upper_ci_below=supported,
        not_supported_point_at_or_above=not_supported,
    )


def downside_metrics(closes: list[float]) -> dict[str, float | int | None]:
    if not closes:
        raise ValueError("at least one close is required")
    returns = [
        current / previous - 1.0 for previous, current in zip(closes, closes[1:])
    ]
    peak = closes[0]
    max_drawdown = 0.0
    trough_index = 0
    peak_before_trough = closes[0]
    for index, close in enumerate(closes):
        if close > peak:
            peak = close
        drawdown = close / peak - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            trough_index = index
            peak_before_trough = peak
    recovery = next(
        (
            index - trough_index
            for index in range(trough_index + 1, len(closes))
            if closes[index] >= peak_before_trough
        ),
        None,
    )
    return {
        "annualized_volatility": pstdev(returns) * sqrt(252)
        if len(returns) > 1
        else 0.0,
        "max_drawdown": max_drawdown,
        "worst_daily_return": min(returns) if returns else 0.0,
        "recovery_sessions": recovery,
    }


def block_bootstrap_effect(
    observations: list[tuple[str, float]],
    *,
    block_months: int = 3,
    samples: int = 10000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> dict[str, float]:
    if block_months < 1 or samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap protocol")

    def difference(items: list[tuple[str, float]]) -> float | None:
        risk_off = [value for regime, value in items if regime == "risk_off"]
        other = [value for regime, value in items if regime in {"neutral", "risk_on"}]
        return mean(risk_off) - mean(other) if risk_off and other else None

    estimate = difference(observations)
    if estimate is None:
        raise ValueError("both risk_off and comparison observations are required")
    rng = random.Random(seed)
    generated: list[float] = []
    attempts = 0
    while len(generated) < samples and attempts < samples * 20:
        attempts += 1
        sample: list[tuple[str, float]] = []
        while len(sample) < len(observations):
            start = rng.randrange(len(observations))
            sample.extend(
                observations[(start + offset) % len(observations)]
                for offset in range(block_months)
            )
        value = difference(sample[: len(observations)])
        if value is not None:
            generated.append(value)
    if len(generated) < samples:
        raise ValueError("bootstrap could not preserve both comparison groups")
    generated.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_low": generated[int(samples * tail)],
        "ci_high": generated[min(samples - 1, int(samples * (1.0 - tail)))],
    }


def _next_closes(
    snapshot: SourceSnapshot,
    key: str,
    decision_timestamp,
    horizon: int,
) -> list[float] | None:
    series = sorted(snapshot.candles_for(key), key=lambda item: item.available_at)
    eligible = [
        index
        for index, item in enumerate(series)
        if item.available_at <= decision_timestamp
    ]
    if not eligible:
        return None
    start = eligible[-1]
    window = series[start : start + horizon + 1]
    return [item.close_price for item in window] if len(window) == horizon + 1 else None


def _risk_off_episodes(points: list[ReplayPoint]) -> int:
    count = 0
    previous = None
    for point in points:
        if point.regime == "risk_off" and previous != "risk_off":
            count += 1
        previous = point.regime
    return count


def build_forward_observations(
    snapshot: SourceSnapshot,
    replay_points: list[ReplayPoint],
    horizons: tuple[int, ...] = (20, 60),
    *,
    block_months: int = 3,
    samples: int = 10000,
    seed: int = 20260728,
    confidence: float = 0.95,
    required_effects: frozenset[tuple[str, int]] = REQUIRED_EFFECTS,
) -> dict[str, Any]:
    series_rows: list[dict[str, Any]] = []
    basket_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    available_baskets = {
        "benchmarks": snapshot.benchmark_specs,
        "policy_instruments": snapshot.policy_specs,
    }
    required_basket_names = {basket for basket, _ in required_effects}
    if not required_basket_names or not required_basket_names <= set(available_baskets):
        raise ValueError("required_effects contains invalid baskets")
    baskets = {
        name: specs
        for name, specs in available_baskets.items()
        if name in required_basket_names
    }
    reference_specs = (
        snapshot.benchmark_specs
        if "benchmarks" in required_basket_names
        else tuple(spec for specs in baskets.values() for spec in specs)
    )
    partial_reason = (
        "benchmark_forward_history_short"
        if "benchmarks" in required_basket_names
        else "required_series_forward_history_short"
    )
    for point in replay_points:
        if point.regime not in {"risk_on", "neutral", "risk_off"}:
            continue
        for horizon in horizons:
            reference_missing = [
                spec.key
                for spec in reference_specs
                if _next_closes(snapshot, spec.key, point.decision_timestamp, horizon)
                is None
            ]
            if reference_missing:
                right_censored = len(reference_missing) == len(reference_specs)
                missing.extend(
                    {
                        "month": point.month,
                        "key": key,
                        "horizon": horizon,
                        "reason": (
                            "right_censored" if right_censored else partial_reason
                        ),
                        "blocking": not right_censored,
                    }
                    for key in reference_missing
                )
                continue
            for basket, specs in baskets.items():
                measured: list[tuple[float, dict[str, float | int | None]]] = []
                for spec in specs:
                    closes = _next_closes(
                        snapshot, spec.key, point.decision_timestamp, horizon
                    )
                    if closes is None:
                        missing.append(
                            {
                                "month": point.month,
                                "key": spec.key,
                                "horizon": horizon,
                                "reason": "forward_history_short",
                                "blocking": basket == "policy_instruments",
                            }
                        )
                        continue
                    values = downside_metrics(closes)
                    measured.append((spec.weight, values))
                    series_rows.append(
                        {
                            "scope": "series",
                            "month": point.month,
                            "regime": point.regime,
                            "basket": basket,
                            "key": spec.key,
                            "horizon": horizon,
                            **values,
                        }
                    )
                if len(measured) != len(specs):
                    continue
                weight_total = sum(weight for weight, _ in measured)
                basket_rows.append(
                    {
                        "scope": "basket",
                        "month": point.month,
                        "regime": point.regime,
                        "basket": basket,
                        "horizon": horizon,
                        "max_drawdown": sum(
                            weight * float(value["max_drawdown"])
                            for weight, value in measured
                        )
                        / weight_total,
                        "annualized_volatility": sum(
                            weight * float(value["annualized_volatility"])
                            for weight, value in measured
                        )
                        / weight_total,
                        "worst_daily_return": sum(
                            weight * float(value["worst_daily_return"])
                            for weight, value in measured
                        )
                        / weight_total,
                    }
                )
    effects: dict[tuple[str, int], dict[str, float]] = {}
    for basket, horizon in required_effects:
        observations = [
            (str(row["regime"]), float(row["max_drawdown"]))
            for row in basket_rows
            if row["basket"] == basket and row["horizon"] == horizon
        ]
        try:
            effects[(basket, horizon)] = block_bootstrap_effect(
                observations,
                block_months=block_months,
                samples=samples,
                seed=seed,
                confidence=confidence,
            )
        except ValueError:
            continue
    serializable = {
        f"{basket}:{horizon}": value
        for (basket, horizon), value in sorted(effects.items())
    }
    return {
        "rows": [*series_rows, *basket_rows],
        "missing": missing,
        "analysis": {
            "effects": effects,
            "effects_serializable": serializable,
            "risk_off_episodes": _risk_off_episodes(replay_points),
        },
    }


def signal_verdict(
    effects: dict[tuple[str, int], dict[str, float]],
    *,
    risk_off_episodes: int,
    complete_coverage: bool,
    reproducible: bool = True,
    source_fresh: bool = True,
    replay_complete: bool = True,
    minimum_risk_off_episodes: int = 3,
    required_effects: frozenset[tuple[str, int]] = REQUIRED_EFFECTS,
    supported_upper_ci_below: float = 0.0,
    not_supported_point_at_or_above: float = 0.0,
) -> dict[str, Any]:
    if not complete_coverage:
        return {
            "verdict": "inconclusive",
            "reason": "policy_instrument_coverage_incomplete",
        }
    if not reproducible:
        return {"verdict": "inconclusive", "reason": "relevant_source_dirty"}
    if not source_fresh:
        return {"verdict": "inconclusive", "reason": "source_stale"}
    if not replay_complete:
        return {"verdict": "inconclusive", "reason": "replay_incomplete"}
    if risk_off_episodes < minimum_risk_off_episodes:
        return {"verdict": "inconclusive", "reason": "risk_off_episodes_below_three"}
    if set(effects) != required_effects:
        return {"verdict": "inconclusive", "reason": "required_effects_missing"}
    if any(
        item["estimate"] >= not_supported_point_at_or_above for item in effects.values()
    ):
        return {"verdict": "not_supported", "reason": "max_drawdown_direction_failed"}
    if all(item["ci_high"] < supported_upper_ci_below for item in effects.values()):
        return {"verdict": "supported", "reason": "downside_signal_confirmed"}
    return {"verdict": "inconclusive", "reason": "confidence_interval_crosses_zero"}
