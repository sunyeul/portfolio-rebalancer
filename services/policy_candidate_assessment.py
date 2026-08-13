"""Read-only allocation-policy candidate research, separate from IPS results."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from services.account_projection import (
    AccountProjectionError,
    layer_map_from_policy,
    project_complete_snapshot,
)
from storage.account_observation_store import list_snapshots
from storage.database import connect_readonly
from storage.policy_store import get_active_policy, policy_hash


EPSILON = 1e-6


class CandidateScenarioError(ValueError):
    """Raised when a read-only candidate scenario is malformed."""


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise CandidateScenarioError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateScenarioError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise CandidateScenarioError(f"{field} is outside the allowed range")
    return number


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateScenarioError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise CandidateScenarioError(f"{field} must be between {minimum} and {maximum}")
    return value


def normalize_candidate_scenario(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact, explicit scenario contract for candidate research."""
    allowed = {
        "monthly_contribution_krw",
        "horizon_months",
        "review_interval_days",
        "transaction_cost_bps",
    }
    if not isinstance(raw, Mapping):
        raise CandidateScenarioError("scenario must be an object")
    unknown = sorted(set(raw) - allowed)
    missing = sorted(allowed - set(raw))
    if unknown:
        raise CandidateScenarioError(f"unknown scenario fields: {', '.join(unknown)}")
    if missing:
        raise CandidateScenarioError(f"missing scenario fields: {', '.join(missing)}")
    monthly = _finite(raw["monthly_contribution_krw"], "monthly_contribution_krw")
    if monthly <= 0:
        raise CandidateScenarioError("monthly_contribution_krw must be greater than 0")
    cost = _finite(raw["transaction_cost_bps"], "transaction_cost_bps", minimum=0)
    if cost >= 10_000:
        raise CandidateScenarioError("transaction_cost_bps must be less than 10000")
    return {
        "monthly_contribution_krw": monthly,
        "horizon_months": _integer(
            raw["horizon_months"], "horizon_months", minimum=1, maximum=120
        ),
        "review_interval_days": _integer(
            raw["review_interval_days"],
            "review_interval_days",
            minimum=1,
            maximum=365,
        ),
        "transaction_cost_bps": cost,
    }


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _range_result(current: float | None, bounds: Mapping[str, Any]) -> dict[str, Any]:
    minimum, target, maximum = (
        float(bounds["minimum"]),
        float(bounds["target"]),
        float(bounds["maximum"]),
    )
    if current is None or not math.isfinite(current):
        state = "not_evaluable"
        drift = None
    elif current < minimum - EPSILON:
        state, drift = "below", current - target
    elif current > maximum + EPSILON:
        state, drift = "above", current - target
    else:
        state, drift = "within", current - target
    return {
        "current": current,
        "minimum": minimum,
        "target": target,
        "maximum": maximum,
        "target_drift": drift,
        "band_state": state,
    }


def _assess_policy(
    snapshot: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        projection = project_complete_snapshot(snapshot, layer_map_from_policy(policy))
    except AccountProjectionError as exc:
        return {"state": "not_evaluable", "reason": str(exc)}
    if projection["unclassified"]:
        return {
            "state": "not_evaluable",
            "reason": "policy_coverage_incomplete",
            "unclassified": projection["unclassified"],
        }
    invested = float(projection["invested_value_krw"])
    cash = _range_result(float(projection["cash_weight_gross"]), policy["cash_reserve"])
    layers = [
        {
            "identity": layer,
            **_range_result(
                float(projection["layer_weights_invested"][layer])
                if projection["invested_weights_evaluable"]
                else None,
                policy["layers"][layer],
            ),
        }
        for layer in ("core", "satellite", "experiment")
    ]
    values = {
        (str(position["market_country"]), str(position["symbol"])): float(
            position["market_value_krw"]
        )
        for position in projection["positions"]
    }
    instruments = []
    for item in policy["instruments"]:
        identity = (str(item["market_country"]), str(item["symbol"]))
        current = values.get(identity, 0.0) / invested if invested > 0 else None
        instruments.append(
            {
                "identity": f"{identity[0]}/{identity[1]}",
                "layer": item["layer"],
                **_range_result(current, item),
            }
        )
    return {
        "state": "evaluable" if invested > 0 else "not_evaluable",
        "reason": None if invested > 0 else "invested_denominator_unavailable",
        "cash": cash,
        "layers": layers,
        "instruments": instruments,
        "projection": projection,
    }


def _recovery(
    assessment: Mapping[str, Any],
    policy: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Find a no-sale, flat-price, continuous-capital band-feasibility month."""
    if assessment.get("state") != "evaluable":
        return {
            "state": "not_evaluable",
            "reason": assessment.get("reason"),
            "earliest_month": None,
            "cumulative_gross_contribution_krw": None,
            "cumulative_net_contribution_krw": None,
            "binding_constraints": [],
        }
    projection = assessment["projection"]
    invested = float(projection["invested_value_krw"])
    values = {
        (str(position["market_country"]), str(position["symbol"])): float(
            position["market_value_krw"]
        )
        for position in projection["positions"]
    }
    layer_values = {
        layer: sum(
            value
            for identity, value in values.items()
            if layer_map_from_policy(policy).get(identity) == layer
        )
        for layer in ("core", "satellite", "experiment")
    }
    net_monthly = float(scenario["monthly_contribution_krw"]) * (
        1.0 - float(scenario["transaction_cost_bps"]) / 10_000.0
    )
    last_reasons: list[str] = []
    for month in range(int(scenario["horizon_months"]) + 1):
        added = month * net_monthly
        denominator = invested + added
        lower_total = upper_total = 0.0
        reasons: list[str] = []
        for layer in ("core", "satellite", "experiment"):
            instrument_lower = instrument_upper = 0.0
            for item in (
                item for item in policy["instruments"] if item["layer"] == layer
            ):
                identity = (str(item["market_country"]), str(item["symbol"]))
                value = values.get(identity, 0.0)
                lower = max(0.0, float(item["minimum"]) * denominator - value)
                upper = float(item["maximum"]) * denominator - value
                if upper < -EPSILON:
                    reasons.append(f"instrument_maximum:{identity[0]}/{identity[1]}")
                instrument_lower += lower
                instrument_upper += upper
            bounds = policy["layers"][layer]
            lower = max(
                0.0,
                float(bounds["minimum"]) * denominator - layer_values[layer],
                instrument_lower,
            )
            upper = min(
                float(bounds["maximum"]) * denominator - layer_values[layer],
                instrument_upper,
            )
            if upper + EPSILON < lower:
                reasons.append(f"layer_range:{layer}")
            lower_total += lower
            upper_total += upper
        if lower_total - EPSILON <= added <= upper_total + EPSILON and not reasons:
            gross = month * float(scenario["monthly_contribution_krw"])
            return {
                "state": "within_band" if month == 0 else "feasible",
                "reason": None,
                "earliest_month": month,
                "cumulative_gross_contribution_krw": gross,
                "cumulative_net_contribution_krw": added,
                "binding_constraints": [],
            }
        last_reasons = sorted(set(reasons)) or ["contribution_capacity"]
    return {
        "state": "not_feasible_within_horizon",
        "reason": None,
        "earliest_month": None,
        "cumulative_gross_contribution_krw": None,
        "cumulative_net_contribution_krw": None,
        "binding_constraints": last_reasons,
    }


def _diff(active: Any, candidate: Any, path: str = "") -> list[dict[str, Any]]:
    if path == "cadence":
        return []
    if isinstance(active, Mapping) and isinstance(candidate, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(active) | set(candidate)):
            child = f"{path}.{key}" if path else str(key)
            result.extend(_diff(active.get(key), candidate.get(key), child))
        return result
    if isinstance(active, list) and isinstance(candidate, list):
        result: list[dict[str, Any]] = []
        for index in range(max(len(active), len(candidate))):
            child = f"{path}[{index}]"
            result.extend(
                _diff(
                    active[index] if index < len(active) else None,
                    candidate[index] if index < len(candidate) else None,
                    child,
                )
            )
        return result
    return (
        []
        if active == candidate
        else [{"path": path, "active": active, "candidate": candidate}]
    )


def _has_breach(assessment: Mapping[str, Any]) -> bool:
    if assessment.get("state") != "evaluable":
        return False
    units = [assessment["cash"], *assessment["layers"], *assessment["instruments"]]
    return any(item["band_state"] != "within" for item in units)


def _cadence(entries: list[dict[str, Any]], interval_days: int) -> dict[str, Any]:
    if not entries:
        return {
            "state": "insufficient_history",
            "selected_snapshot_ids": [],
            "candidate_breach_count": 0,
        }
    first, last = (
        _timestamp(entries[0]["synced_at"]),
        _timestamp(entries[-1]["synced_at"]),
    )
    selected = [entries[0]]
    boundary = first + timedelta(days=interval_days)
    while boundary <= last:
        next_entry = next(
            (entry for entry in entries if _timestamp(entry["synced_at"]) >= boundary),
            None,
        )
        if (
            next_entry is not None
            and next_entry["snapshot_id"] != selected[-1]["snapshot_id"]
        ):
            selected.append(next_entry)
        boundary += timedelta(days=interval_days)
    state = "complete" if len(selected) >= 2 else "insufficient_history"
    return {
        "state": state,
        "selected_snapshot_ids": [entry["snapshot_id"] for entry in selected],
        "candidate_breach_count": sum(
            _has_breach(entry["candidate"]) for entry in selected
        ),
    }


def _analysis_fingerprint(
    active: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    scenario: Mapping[str, Any],
    snapshots: list[Mapping[str, Any]],
) -> str:
    payload = {
        "contract": "candidate-preview-v1",
        "active_policy_hash": active["policy_hash"] if active is not None else None,
        "candidate_policy_hash": policy_hash(dict(candidate)),
        "scenario": scenario,
        "sources": [snapshot["source_fingerprint"] for snapshot in snapshots],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode()).hexdigest()


def assess_candidate_history(
    candidate_policy: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    snapshot_limit: int = 100,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Return deterministic candidate research without mutating local IPS state."""
    normalized_scenario = normalize_candidate_scenario(scenario)
    bounded_limit = max(1, min(int(snapshot_limit), 100))
    conn = connect_readonly()
    try:
        active = get_active_policy(account_alias, conn=conn)
        attempts = sorted(
            list_snapshots(bounded_limit, account_alias, conn=conn),
            key=lambda item: (str(item["synced_at"]), int(item["id"])),
        )
    finally:
        conn.close()
    assumptions = {
        "flat_prices": True,
        "fractional_investment": True,
        "no_sales": True,
        "tax_treatment": "not_applicable_without_sales",
        "transaction_cost_treatment": "deducted_from_monthly_contribution",
    }
    excluded = [
        {
            "snapshot_id": item["id"],
            "state": item["state"],
            "reason": "snapshot_not_complete",
        }
        for item in attempts
        if item["state"] != "complete"
    ]
    complete_snapshots = [item for item in attempts if item["state"] == "complete"]
    if active is None:
        return {
            "state": "not_evaluable",
            "reason": "active_policy_missing",
            "persisted": False,
            "may_change_primary_evaluation": False,
            "account_alias": account_alias,
            "active_policy": {"id": None, "hash": None},
            "candidate_policy": {"hash": policy_hash(dict(candidate_policy))},
            "policy_differences": [],
            "scenario": normalized_scenario,
            "assumptions": assumptions,
            "analysis_period": {
                "start_at": None,
                "end_at": None,
                "used_snapshot_ids": [],
                "excluded_snapshots": excluded,
            },
            "cadence": _cadence([], normalized_scenario["review_interval_days"]),
            "snapshots": [],
            "analysis_fingerprint": _analysis_fingerprint(
                None, candidate_policy, normalized_scenario, complete_snapshots
            ),
        }
    entries: list[dict[str, Any]] = []
    for snapshot in complete_snapshots:
        active_assessment = _assess_policy(snapshot, active["policy"])
        candidate_assessment = _assess_policy(snapshot, candidate_policy)
        entries.append(
            {
                "snapshot_id": snapshot["id"],
                "synced_at": snapshot["synced_at"],
                "source_fingerprint": snapshot["source_fingerprint"],
                "active": {
                    **{
                        key: value
                        for key, value in active_assessment.items()
                        if key != "projection"
                    },
                    "recovery": _recovery(
                        active_assessment, active["policy"], normalized_scenario
                    ),
                },
                "candidate": {
                    **{
                        key: value
                        for key, value in candidate_assessment.items()
                        if key != "projection"
                    },
                    "recovery": _recovery(
                        candidate_assessment, candidate_policy, normalized_scenario
                    ),
                },
            }
        )
    cadence = _cadence(entries, normalized_scenario["review_interval_days"])
    state = "complete"
    if not entries:
        state = "not_evaluable"
    elif (
        excluded
        or cadence["state"] == "insufficient_history"
        or any(
            entry["active"]["state"] != "evaluable"
            or entry["candidate"]["state"] != "evaluable"
            for entry in entries
        )
    ):
        state = "partial"
    return {
        "state": state,
        "reason": None if entries else "complete_snapshot_unavailable",
        "persisted": False,
        "may_change_primary_evaluation": False,
        "account_alias": account_alias,
        "active_policy": {"id": active["id"], "hash": active["policy_hash"]},
        "candidate_policy": {"hash": policy_hash(dict(candidate_policy))},
        "policy_differences": _diff(active["policy"], candidate_policy),
        "scenario": normalized_scenario,
        "assumptions": assumptions,
        "analysis_period": {
            "start_at": entries[0]["synced_at"] if entries else None,
            "end_at": entries[-1]["synced_at"] if entries else None,
            "used_snapshot_ids": [entry["snapshot_id"] for entry in entries],
            "excluded_snapshots": excluded,
        },
        "cadence": cadence,
        "snapshots": entries,
        "analysis_fingerprint": _analysis_fingerprint(
            active, candidate_policy, normalized_scenario, complete_snapshots
        ),
    }


def unavailable_policy_candidate_assessment(
    active_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expose the reduced-core automatic-candidate boundary without inventing evidence."""
    return {
        "state": "not_implemented",
        "reason": "market_candidate_evidence_unavailable",
        "active_policy_version_id": active_policy.get("id")
        if active_policy is not None
        else None,
        "candidate": None,
        "feasibility": None,
        "may_change_primary_evaluation": False,
        "verification_task": (
            "완전한 Toss 시장·기술 근거와 후보 정책의 실행 가능성 검증을 제공하기 전에는 "
            "정책 재검토 후보를 만들지 않습니다."
        ),
    }
