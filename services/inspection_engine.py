"""Deterministic, inspection-only IPS evaluation for Toss account projections."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from services.policy_validation import LAYERS


ENGINE_VERSION = "phase5-v1"
SEVERITY = {"OK": 0, "Watch": 1, "Review": 2, "Action": 3}


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


def _range_gap(current: float | None, target: float | None) -> float | None:
    return None if current is None or target is None else current - target


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _within(current: float | None, bounds: Mapping[str, Any] | None) -> bool:
    if current is None or not bounds:
        return False
    return float(bounds["minimum"]) <= current <= float(bounds["maximum"])


def _cumulative_twr(points: list[Mapping[str, Any]]) -> float | None:
    factors: list[float] = []
    for point in points:
        if point.get("evaluation_state") != "evaluable":
            continue
        value = point.get("interval_twr")
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= -1:
            return None
        factors.append(1 + number)
    if not factors:
        return None
    result = 1.0
    for factor in factors:
        result *= factor
    return result - 1.0


def _history_days(points: list[Mapping[str, Any]]) -> int:
    timestamps = [
        _timestamp(point.get("point_at"))
        for point in points
        if point.get("evaluation_state") == "evaluable"
    ]
    valid = [value for value in timestamps if value is not None]
    if len(valid) < 2:
        return 0
    return max(0, (max(valid) - min(valid)).days)


def _trailing_twr(points: list[Mapping[str, Any]], minimum_days: int) -> float | None:
    """Compound supported intervals after the supported point at the cutoff."""
    supported = sorted(
        (
            (_timestamp(point.get("point_at")), point)
            for point in points
            if point.get("evaluation_state") == "evaluable"
        ),
        key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc),
    )
    supported = [(timestamp, point) for timestamp, point in supported if timestamp]
    if len(supported) < 2:
        return None
    cutoff = supported[-1][0] - timedelta(days=minimum_days)
    baseline_index = max(
        (
            index
            for index, (timestamp, _) in enumerate(supported)
            if timestamp <= cutoff
        ),
        default=-1,
    )
    if baseline_index < 0:
        return None
    window_days = (supported[-1][0] - supported[baseline_index][0]).days
    if window_days > minimum_days + 7:
        return None
    factors: list[float] = []
    for _, point in supported[baseline_index + 1 :]:
        value = point.get("interval_twr")
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= -1:
            return None
        factors.append(1 + number)
    if not factors:
        return None
    result = 1.0
    for factor in factors:
        result *= factor
    return result - 1.0


def _ytd_twr(points: list[Mapping[str, Any]]) -> float | None:
    """Compound supported intervals after the last point on or before Jan 1."""
    supported = sorted(
        (
            (_timestamp(point.get("point_at")), point)
            for point in points
            if point.get("evaluation_state") == "evaluable"
        ),
        key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc),
    )
    supported = [(timestamp, point) for timestamp, point in supported if timestamp]
    if not supported:
        return None
    year_start = datetime(
        supported[-1][0].year, 1, 1, tzinfo=timezone.utc
    )
    baseline_index = max(
        (
            index
            for index, (timestamp, _) in enumerate(supported)
            if timestamp <= year_start
        ),
        default=-1,
    )
    if baseline_index < 0:
        return None
    result = 1.0
    for _, point in supported[baseline_index + 1 :]:
        value = point.get("interval_twr")
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= -1:
            return None
        result *= 1 + number
    return result - 1.0


def _unit(
    *,
    kind: str,
    identity: str,
    current: float | None,
    target: float | None,
    minimum: float | None,
    maximum: float | None,
    denominator: str,
    triggers: list[str],
    meaning: str,
    next_step: str,
    status: str = "OK",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = {
        "kind": kind,
        "identity": identity,
        "status": status,
        "current": current,
        "target": target,
        "minimum": minimum,
        "maximum": maximum,
        "gap": _range_gap(current, target),
        "denominator": denominator,
        "triggers": sorted(set(triggers)),
        "meaning": meaning,
        "next_step": next_step,
        "verification_task": next_step,
    }
    if extra:
        values.update(extra)
    return values


def _raise_status(current: str, candidate: str) -> str:
    return candidate if SEVERITY[candidate] > SEVERITY[current] else current


def evaluate_inspection(
    projection: Mapping[str, Any] | None,
    performance_run: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    risk_evidence: Mapping[str, Any] | None = None,
    evidence_refs: Mapping[str, Any] | None = None,
    *,
    source_error: str | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable input set without side effects or broker actions."""
    profiles = profiles or {}
    risk_evidence = risk_evidence or {}
    evidence_refs = evidence_refs or {}
    policy_dict = dict(policy)
    performance_ok = (
        performance_run is not None and performance_run.get("state") == "complete"
    )
    points = list((performance_run or {}).get("points", []))
    reconciliation = (projection or {}).get("reconciliation") or {}
    reconciled = (reconciliation.get("holdings") or {}).get(
        "all_within_tolerance"
    ) is True
    source_ok = projection is not None and source_error is None and reconciled
    result: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "state": "complete" if source_ok and performance_ok else "not_evaluable",
        "non_evaluable_reason": None,
        "source": {
            "snapshot_id": projection.get("snapshot_id") if projection else None,
            "account_alias": projection.get("account_alias")
            if projection
            else "toss-brokerage",
            "state": "complete" if source_ok else "failed",
            "source_fingerprint": None,
            "reconciled": reconciled,
            "synced_at": projection.get("synced_at") if projection else None,
        },
        "account": {},
        "account_profit_loss": risk_evidence.get("account_profit_loss", {}),
        "performance": {},
        "cash": None,
        "layers": [],
        "instruments": [],
        "review_queue": [],
        "evidence_refs": dict(evidence_refs),
    }
    if not source_ok:
        result["non_evaluable_reason"] = source_error or "source_not_evaluable"
    elif not performance_ok:
        result["non_evaluable_reason"] = "performance_not_evaluable"
    if projection is not None:
        result["source"]["source_fingerprint"] = projection.get("source_fingerprint")
        latest_point = (
            points[-1]
            if points and points[-1].get("evaluation_state") == "evaluable"
            else {}
        )
        result["account"] = {
            "total_value_krw": projection.get("total_value_krw"),
            "invested_value_krw": projection.get("invested_value_krw"),
            "cash_value_krw": projection.get("cash_value_krw"),
            "cash_weight_gross": projection.get("cash_weight_gross"),
            "investment_principal_krw": latest_point.get(
                "investment_principal_krw"
            ),
            "account_profit_krw": latest_point.get("account_gain_krw"),
            "account_return": latest_point.get("simple_return"),
        }

    performance_policy = policy_dict.get("performance") or {
        "annual_return_target": 0.10,
        "measurement": "ytd_twr",
        "minimum_history_days": 365,
    }
    measurement = str(performance_policy.get("measurement") or "ytd_twr")
    minimum_days = int(performance_policy.get("minimum_history_days", 365))
    history_days = _history_days(points)
    cumulative = _cumulative_twr(points) if performance_ok else None
    ytd = _ytd_twr(points) if performance_ok else None
    trailing = (
        _trailing_twr(points, minimum_days)
        if performance_ok and history_days >= minimum_days
        else None
    )
    annual = ytd if measurement == "ytd_twr" else trailing
    annual_target = performance_policy.get("annual_return_target")
    performance_status = "OK"
    performance_triggers: list[str] = []
    if not performance_ok:
        performance_status = "Review"
        performance_triggers.append("performance_not_evaluable")
    elif measurement == "ytd_twr" and ytd is None:
        performance_status = "Watch"
        performance_triggers.append("ytd_return_history_insufficient")
    elif measurement == "trailing_12_month_twr" and history_days < minimum_days:
        performance_status = "Watch"
        performance_triggers.append("annual_return_history_insufficient")
    elif annual is None:
        performance_status = "Watch"
        performance_triggers.append(
            "ytd_return_points_insufficient"
            if measurement == "ytd_twr"
            else "annual_return_points_insufficient"
        )
    elif annual is not None and annual < float(annual_target):
        performance_status = "Watch"
        performance_triggers.append("annual_return_below_target")
    result["performance"] = {
        "status": performance_status,
        "cumulative_twr": cumulative,
        "ytd_twr": ytd,
        "trailing_12m_twr": trailing,
        "annual_twr": annual,
        "annual_target": annual_target,
        "history_days": history_days,
        "minimum_history_days": minimum_days,
        "measurement": measurement,
        "triggers": performance_triggers,
        "meaning": (
            "올해 누적 계좌 성과와 보유 평가손익을 분리해 관찰합니다."
            if measurement == "ytd_twr"
            else "최근 1년 계좌 성과와 보유 평가손익을 분리해 관찰합니다."
        ),
        "verification_task": "성과 이력과 현금흐름 분류를 확인합니다.",
    }

    risk_policy = policy_dict.get("risk_review") or {}
    account_evidence = result["account_profit_loss"]
    if account_evidence and source_ok and performance_ok:
        account_status = "OK"
        account_triggers: list[str] = []
        account_drawdown = account_evidence.get("drawdown") or {}
        if account_drawdown.get("state") != "complete":
            account_status = "Watch"
            account_triggers.append("account_drawdown_evidence_unavailable")
        elif (
            account_drawdown.get("current") is not None
            and risk_policy.get("account_drawdown_review") is not None
            and float(account_drawdown["current"])
            <= float(risk_policy["account_drawdown_review"])
        ):
            account_status = "Review"
            account_triggers.append("account_drawdown_review_threshold")
        result["account_profit_loss"] = {
            **account_evidence,
            "kind": "account_risk",
            "identity": "account_profit_loss",
            "status": account_status,
            "triggers": account_triggers,
            "meaning": "계좌 TWR 곡선의 현재 drawdown과 손익 근거를 확인합니다.",
            "next_step": "계좌 성과 이력과 현금흐름 분류를 검토합니다.",
            "verification_task": "계좌 성과 이력과 현금흐름 분류를 검토합니다.",
        }

    if not source_ok or not performance_ok:
        queue: list[dict[str, Any]] = []
        if not source_ok:
            queue.append(
                {
                    "priority": 0,
                    "kind": "source",
                    "identity": "account",
                    "status": "Review",
                    "triggers": [result["non_evaluable_reason"]],
                    "meaning": "Toss 원천 자료가 완전하고 최신인지 확인해야 합니다.",
                    "next_step": "최근 동기화 결과와 조정 상태를 확인합니다.",
                    "verification_task": "최근 동기화 결과와 조정 상태를 확인합니다.",
                }
            )
        if not performance_ok:
            queue.append(
                {
                    "priority": 6,
                    "kind": "performance",
                    "identity": "account_return",
                    "status": "Review",
                    "triggers": ["performance_not_evaluable"],
                    "meaning": "계좌 성과 이력이 평가 가능한지 확인해야 합니다.",
                    "next_step": "성과 이력과 현금흐름 분류를 확인합니다.",
                    "verification_task": "성과 이력과 현금흐름 분류를 확인합니다.",
                }
            )
        result["review_queue"] = queue
        return result

    configured = {
        (
            str(item.get("market_country", "")).upper(),
            str(item.get("symbol", "")).upper(),
        ): item
        for item in policy_dict.get("instruments", [])
        if isinstance(item, Mapping)
    }
    positions = list(projection.get("positions", []))

    def _target_ready(target: Mapping[str, Any] | None) -> bool:
        return bool(
            target
            and all(
                isinstance(target.get(bound), (int, float))
                and not isinstance(target.get(bound), bool)
                and math.isfinite(float(target[bound]))
                for bound in ("minimum", "target", "maximum")
            )
        )

    coverage_complete = all(
        position.get("layer") is not None
        and (str(position["market_country"]).upper(), str(position["symbol"]).upper())
        in configured
        and _target_ready(
            configured.get(
                (
                    str(position["market_country"]).upper(),
                    str(position["symbol"]).upper(),
                )
            )
        )
        and (
            str(position["market_country"]).upper(),
            str(position["symbol"]).upper(),
        )
        in profiles
        for position in positions
    )
    if not coverage_complete:
        result["state"] = "not_evaluable"
        result["non_evaluable_reason"] = "policy_coverage_incomplete"

    cash_policy = policy_dict.get("cash_reserve", {})
    cash_current = float(projection["cash_weight_gross"])
    cash_status = "OK" if _within(cash_current, cash_policy) else "Review"
    cash_triggers = [] if cash_status == "OK" else ["cash_reserve_out_of_range"]
    result["cash"] = _unit(
        kind="cash",
        identity="cash_reserve",
        current=cash_current,
        target=float(cash_policy["target"]),
        minimum=float(cash_policy["minimum"]),
        maximum=float(cash_policy["maximum"]),
        denominator="gross_account_value",
        triggers=cash_triggers,
        meaning="예수금 비중이 설정한 총계좌 기준 보유 범위 안에 있는지 확인합니다.",
        next_step="현금 범위와 향후 정기매수 정책을 검토합니다.",
        status=cash_status,
        extra={"cash_value_krw": projection["cash_value_krw"]},
    )

    layer_policy = policy_dict.get("layers", {})
    layer_weights = projection.get("layer_weights_invested", {})
    for layer in LAYERS:
        bounds = layer_policy.get(layer)
        current = layer_weights.get(layer) if coverage_complete else None
        status = "OK" if _within(current, bounds) else "Review"
        triggers = [] if status == "OK" else ["layer_out_of_range"]
        if not coverage_complete:
            triggers = ["profile_or_target_missing"]
        result["layers"].append(
            _unit(
                kind="layer",
                identity=layer,
                current=float(current) if current is not None else None,
                target=float(bounds["target"])
                if bounds and coverage_complete
                else None,
                minimum=float(bounds["minimum"])
                if bounds and coverage_complete
                else None,
                maximum=float(bounds["maximum"])
                if bounds and coverage_complete
                else None,
                denominator="invested_account_value",
                triggers=triggers,
                meaning="투자금 기준 레이어 비중이 승인된 범위에 맞는지 확인합니다.",
                next_step="레이어 목표와 향후 정기매수 정책을 검토합니다.",
                status=status,
            )
        )

    for position in positions:
        key = (str(position["market_country"]).upper(), str(position["symbol"]).upper())
        target = configured.get(key)
        profile = profiles.get(key)
        evidence = (risk_evidence.get("instruments") or {}).get(
            f"{key[0]}/{key[1]}", {}
        )
        triggers: list[str] = []
        status = "OK"
        ready = profile is not None and _target_ready(target)
        current = position.get("invested_weight") if ready else None
        thesis = str(profile.get("thesis_status", "unknown")) if profile else "unknown"
        layer = str(position.get("layer") or "").lower()
        if profile is None or target is None or position.get("layer") is None:
            status = "Review"
            triggers.append("profile_or_target_missing")
        else:
            if position.get("layer") != target.get("layer"):
                status = "Review"
                triggers.append("layer_identity_mismatch")
            if not _target_ready(target):
                status = _raise_status(status, "Review")
                triggers.append("profile_or_target_missing")
            elif not _within(current, target):
                status = _raise_status(status, "Review")
                triggers.append("instrument_out_of_range")

            drawdown = evidence.get("drawdown") or {}
            if drawdown:
                if drawdown.get("state") != "complete":
                    status = _raise_status(status, "Watch")
                    triggers.append("instrument_drawdown_evidence_unavailable")
                else:
                    threshold = (risk_policy.get("instrument_drawdown_review") or {}).get(
                        layer
                    )
                    if (
                        threshold is not None
                        and drawdown.get("current") is not None
                        and float(drawdown["current"]) <= float(threshold)
                    ):
                        candidate = "Watch" if layer == "core" else "Review"
                        status = _raise_status(status, candidate)
                        triggers.append(
                            "core_drawdown_review_threshold"
                            if layer == "core"
                            else "strict_layer_drawdown_review_threshold"
                        )

            pnl = _finite_number(
                evidence.get("unrealized_pnl_krw", position.get("profit_loss_krw"))
            )
            hard_maximum_breach = (
                current is not None and float(current) > float(target["maximum"])
            )
            if pnl is not None and pnl > 0 and hard_maximum_breach:
                status = _raise_status(status, "Review")
                triggers.append("gain_with_instrument_overweight")
            if pnl is not None and pnl < 0 and (
                thesis in {"watch", "broken"}
                or profile.get("holdability_status", "unknown") == "review"
            ):
                status = _raise_status(status, "Review")
                triggers.append("loss_with_thesis_or_holdability_concern")

            if thesis == "watch":
                status = _raise_status(status, "Watch")
                triggers.append("thesis_watch")
            elif thesis == "broken":
                status = _raise_status(status, "Review")
                triggers.append("thesis_broken")

            factor_fields = (
                ("overlap_status", "overlap"),
                ("management_burden_status", "management_burden"),
                ("holdability_status", "holdability"),
                ("etf_substitution_status", "etf_substitution"),
            )
            strict_layer = layer in {"satellite", "experiment"}
            for field, trigger_name in factor_fields:
                factor_status = str(profile.get(field, "unknown")).lower()
                if factor_status == "review":
                    status = _raise_status(status, "Review")
                    triggers.append(f"{trigger_name}_review")
                elif strict_layer and factor_status == "unknown":
                    status = _raise_status(status, "Review")
                    triggers.append(f"{trigger_name}_unknown")

            if thesis == "broken" and hard_maximum_breach:
                status = _raise_status(status, "Action")
                triggers.append("broken_thesis_and_hard_maximum_breach")
        next_step = (
            "예외 개입 가능성을 점검하고 조건이 확인되지 않으면 보류합니다."
            if status == "Action"
            else "프로필, 투자 논지, 겹침과 향후 정기매수 정책을 검토합니다."
        )
        result["instruments"].append(
            _unit(
                kind="instrument",
                identity=f"{key[0]}/{key[1]}",
                current=float(current) if current is not None else None,
                target=float(target["target"]) if ready else None,
                minimum=float(target["minimum"]) if ready else None,
                maximum=float(target["maximum"]) if ready else None,
                denominator="invested_account_value",
                triggers=triggers,
                meaning="종목의 현재 비중과 IPS 프로필·목표 범위를 함께 확인합니다.",
                next_step=next_step,
                status=status,
                extra={
                    "market_country": key[0],
                    "symbol": key[1],
                    "layer": position.get("layer"),
                    "thesis_status": profile.get("thesis_status") if profile else None,
                    "overlap_status": profile.get("overlap_status", "unknown") if profile else "unknown",
                    "management_burden_status": profile.get("management_burden_status", "unknown") if profile else "unknown",
                    "holdability_status": profile.get("holdability_status", "unknown") if profile else "unknown",
                    "etf_substitution_status": profile.get("etf_substitution_status", "unknown") if profile else "unknown",
                    "review_factors_note": profile.get("review_factors_note", "") if profile else "",
                    "unrealized_pnl_krw": position.get("profit_loss_krw"),
                    "evidence": evidence,
                },
            )
        )

    queue: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = [result["cash"], *result["layers"], *result["instruments"]]
    if result["account_profit_loss"].get("status"):
        units.insert(0, result["account_profit_loss"])
    for item in units:
        if item["status"] != "OK":
            priority = {
                "account_risk": 5,
                "cash": 2,
                "layer": 4,
                "instrument": 5,
            }.get(item["kind"], 9)
            if any("thesis" in trigger for trigger in item["triggers"]):
                priority = 3
            if any("drawdown_evidence_unavailable" in trigger for trigger in item["triggers"]):
                priority = 9
            queue.append(
                {
                    "priority": priority,
                    "kind": item["kind"],
                    "identity": item["identity"],
                    "status": item["status"],
                    "triggers": item["triggers"],
                    "meaning": item["meaning"],
                    "next_step": item["next_step"],
                    "verification_task": item["verification_task"],
                    "evidence_refs": {
                        **dict(evidence_refs),
                        **(
                            {
                                "market_source_fingerprint": item["evidence"]
                                .get("drawdown", {})
                                .get("source_fingerprint")
                            }
                            if item["kind"] == "instrument"
                            else {}
                        ),
                    },
                }
            )
    if performance_status != "OK":
        queue.append(
            {
                "priority": 6,
                "kind": "performance",
                "identity": "account_return",
                "status": performance_status,
                "triggers": performance_triggers,
                "meaning": result["performance"]["meaning"],
                "next_step": "성과 이력과 현금흐름 분류를 확인합니다.",
                "verification_task": result["performance"]["verification_task"],
                "evidence_refs": dict(evidence_refs),
            }
        )
    result["review_queue"] = sorted(
        queue,
        key=lambda item: (
            -SEVERITY[item["status"]],
            item["priority"],
            item["kind"],
            item["identity"],
        ),
    )
    return result


def evaluation_fingerprint(
    *,
    source_fingerprint: str,
    performance_fingerprint: str | None,
    policy_hash: str,
    profile_hash: str,
    market_evidence_fingerprint: str = "",
    engine_version: str = ENGINE_VERSION,
) -> str:
    payload = "|".join(
        (
            source_fingerprint,
            performance_fingerprint or "",
            policy_hash,
            profile_hash,
            market_evidence_fingerprint,
            engine_version,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_snapshot(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    values = [dict(value) for _, value in sorted(profiles.items())]
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return values, hashlib.sha256(encoded.encode("utf-8")).hexdigest()
