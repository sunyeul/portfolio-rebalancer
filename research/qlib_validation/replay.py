from collections.abc import Callable
from typing import Any

from services.dynamic_allocation import (
    build_neutral_policy,
    evaluate_dynamic_allocation,
)
from research.qlib_validation.contracts import Candle, ReplayPoint, SourceSnapshot


Evaluator = Callable[..., dict[str, Any]]


def _by_month(candles: tuple[Candle, ...]) -> dict[str, list[Candle]]:
    grouped: dict[str, list[Candle]] = {}
    for candle in candles:
        grouped.setdefault(candle.session_date.strftime("%Y-%m"), []).append(candle)
    return grouped


def replay_regimes(
    snapshot: SourceSnapshot,
    *,
    evaluator: Evaluator = evaluate_dynamic_allocation,
    minimum_history: int = 200,
) -> list[ReplayPoint]:
    policy = snapshot.policy_record["policy"]
    neutral = build_neutral_policy(policy)
    all_specs = {
        spec.key: spec for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    monthly = {
        spec.key: _by_month(snapshot.candles_for(spec.key))
        for spec in all_specs.values()
    }
    if not monthly:
        return []
    common_months = sorted(
        set.intersection(*(set(value) for value in monthly.values()))
    )
    points: list[ReplayPoint] = []
    for month in common_months:
        selected = {
            spec.key: max(monthly[spec.key][month], key=lambda item: item.session_date)
            for spec in snapshot.benchmark_specs
        }
        if not selected:
            continue
        decision_timestamp = max(item.available_at for item in selected.values())
        histories = {
            key: [
                item
                for item in snapshot.candles_for(key)
                if item.available_at <= decision_timestamp
            ]
            for key in all_specs
        }
        if any(len(items) < minimum_history for items in histories.values()):
            continue
        series_by_key: dict[str, list[dict[str, Any]]] = {}
        cutoffs: dict[str, str] = {}
        for key, cutoff in selected.items():
            cutoffs[key] = cutoff.session_date.isoformat()
            series_by_key[key] = [
                {**item.evaluator_row(), "available_at": item.available_at.isoformat()}
                for item in sorted(histories[key], key=lambda value: value.available_at)
                if item.session_date <= cutoff.session_date
            ]
        result = evaluator(
            series_by_key,
            active_policy=neutral,
            last_change_at=None,
            now=decision_timestamp,
        )
        points.append(
            ReplayPoint(
                month=month,
                decision_timestamp=decision_timestamp,
                regime=result.get("regime"),
                reason=str(result.get("reason", "unknown")),
                cutoffs=cutoffs,
            )
        )
    return points
