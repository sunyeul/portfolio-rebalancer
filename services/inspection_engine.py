"""Deterministic, inspection-only IPS evaluation for Toss account projections."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from services.action_contract import (
    attach_decision,
    priority_label,
    priority_rank,
    queue_class,
    suggestion,
)
from services.policy_validation import LAYERS


ENGINE_VERSION = "phase5-v2"
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
    year_start = datetime(supported[-1][0].year, 1, 1, tzinfo=timezone.utc)
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


def _red_team(item: Mapping[str, Any], *, blocking: bool) -> dict[str, str]:
    """Return a deterministic counterargument without changing IPS judgment."""
    kind = str(item.get("kind") or "")
    if blocking:
        counterargument = (
            "현재 Toss 원천과 정책 커버리지가 평가 가능한 상태인지 확인하기 전에는 "
            "비중 판단을 확정할 수 없습니다."
        )
    elif kind in {"cash", "layer", "instrument"}:
        counterargument = "비중 범위 이탈만으로 거래나 예외 개입을 확정할 수 없습니다."
    elif kind in {"performance", "account_risk"}:
        counterargument = (
            "수익률·손익·drawdown만으로 비중 조정이나 예외 개입을 확정할 수 없습니다."
        )
    else:
        counterargument = "현재 검사 신호만으로 결론을 확정할 수 없습니다."
    return {
        "counterargument": counterargument,
        "evidence_needed": str(
            item.get("verification_task") or "근거와 확인 과제를 검토합니다."
        ),
    }


def evaluate_inspection(
    projection: Mapping[str, Any] | None,
    performance_run: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    layer_map: Mapping[tuple[str, str], str] | None = None,
    risk_evidence: Mapping[str, Any] | None = None,
    evidence_refs: Mapping[str, Any] | None = None,
    *,
    source_error: str | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable input set without side effects or broker actions."""
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

    result: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "allocation_state": "not_evaluable",
        "allocation_reason": None,
        "source": {
            "snapshot_id": projection.get("snapshot_id") if projection else None,
            "account_alias": projection.get("account_alias")
            if projection
            else "toss-brokerage",
            "state": "complete" if source_ok else "failed",
            "source_fingerprint": projection.get("source_fingerprint")
            if projection
            else None,
            "reconciled": reconciled,
            "synced_at": projection.get("synced_at") if projection else None,
        },
        "account": {},
        "account_profit_loss": risk_evidence.get("account_profit_loss", {}),
        "performance": {},
        "cash": None,
        "layers": [],
        "instruments": [],
        "adjustment_suggestions": [],
        "review_queue": [],
        "evidence_refs": dict(evidence_refs),
    }

    if projection is not None:
        latest_point = (
            points[-1]
            if performance_ok
            and points
            and points[-1].get("evaluation_state") == "evaluable"
            else {}
        )
        result["account"] = {
            "total_value_krw": projection.get("total_value_krw"),
            "invested_value_krw": projection.get("invested_value_krw"),
            "cash_value_krw": projection.get("cash_value_krw"),
            "cash_weight_gross": projection.get("cash_weight_gross"),
            "investment_principal_krw": latest_point.get("investment_principal_krw"),
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
        performance_status = "Watch"
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
            "verification_task": "계좌 성과 이력과 현금흐름 분류를 검토합니다.",
        }

    # Required allocation evidence is evaluated independently from optional performance.
    positions = list(projection.get("positions", [])) if projection else []
    allocation_reason: str | None = None
    if source_error is not None or projection is None:
        allocation_reason = "source_not_current_evaluable"
    elif not reconciled:
        allocation_reason = "holdings_reconciliation_failed"
    else:
        total = _finite_number(projection.get("total_value_krw"))
        cash_weight = _finite_number(projection.get("cash_weight_gross"))
        if total is None or total <= 0 or cash_weight is None:
            allocation_reason = "gross_account_denominator_invalid"
        elif not projection.get(
            "invested_weights_evaluable",
            bool(projection.get("layer_weights_invested"))
            or any(
                position.get("invested_weight") is not None for position in positions
            ),
        ):
            # A zero-investment account still has a valid gross cash denominator.
            allocation_reason = "invested_denominator_unavailable"

    configured = {
        (
            str(item.get("market_country", "")).upper(),
            str(item.get("symbol", "")).upper(),
        ): item
        for item in policy_dict.get("instruments", [])
        if isinstance(item, Mapping)
    }
    position_map = {
        (
            str(position.get("market_country", "")).upper(),
            str(position.get("symbol", "")).upper(),
        ): position
        for position in positions
    }
    invested_denominator_evaluable = bool(
        projection
        and projection.get(
            "invested_weights_evaluable",
            bool(projection.get("layer_weights_invested"))
            or any(
                position.get("invested_weight") is not None for position in positions
            ),
        )
    )
    holding_coverage_complete = bool(source_ok) and all(
        position.get("layer") is not None
        and (
            str(position.get("market_country", "")).upper(),
            str(position.get("symbol", "")).upper(),
        )
        in configured
        and _target_ready(
            configured.get(
                (
                    str(position.get("market_country", "")).upper(),
                    str(position.get("symbol", "")).upper(),
                )
            )
        )
        for position in positions
    )
    configured_coverage_complete = all(
        str(target.get("layer", "")).lower() in LAYERS and _target_ready(target)
        for key, target in configured.items()
    )
    if invested_denominator_evaluable:
        holding_coverage_complete = (
            holding_coverage_complete and configured_coverage_complete
        )
    if allocation_reason is None and not holding_coverage_complete:
        allocation_reason = "policy_coverage_incomplete"

    # Cash is the only valid unit for an all-cash account.
    cash_ready = allocation_reason in {None, "invested_denominator_unavailable"}
    cash_allocation_available = True
    cash_policy = policy_dict.get("cash_reserve") or {}
    if cash_ready:
        cash_current = _finite_number(projection.get("cash_weight_gross"))
        cash_bounds_ready = _target_ready(cash_policy)
        if cash_current is None or not cash_bounds_ready:
            allocation_reason = "gross_account_denominator_invalid"
            cash_ready = False
        else:
            cash_status = "OK" if _within(cash_current, cash_policy) else "Review"
            cash_triggers = [] if cash_status == "OK" else ["cash_reserve_out_of_range"]
            result["cash"] = attach_decision(
                _unit(
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
                    extra={"cash_value_krw": projection.get("cash_value_krw")},
                ),
            )
            cash_allocation_available = cash_current >= float(cash_policy["minimum"])

    invested_available = (
        source_ok
        and allocation_reason is None
        and invested_denominator_evaluable
        and holding_coverage_complete
    )
    if invested_available:
        layer_policy = policy_dict.get("layers") or {}
        layer_weights = projection.get("layer_weights_invested") or {}
        for layer in LAYERS:
            bounds = layer_policy.get(layer)
            current = _finite_number(layer_weights.get(layer))
            bounds_ready = _target_ready(bounds)
            status = "OK" if bounds_ready and _within(current, bounds) else "Review"
            triggers = [] if status == "OK" else ["layer_out_of_range"]
            eligible = (
                cash_allocation_available
                and status == "Review"
                and current is not None
                and bounds_ready
                and current < float(bounds["minimum"])
            )
            result["layers"].append(
                attach_decision(
                    _unit(
                        kind="layer",
                        identity=layer,
                        current=current,
                        target=float(bounds["target"]) if bounds_ready else None,
                        minimum=float(bounds["minimum"]) if bounds_ready else None,
                        maximum=float(bounds["maximum"]) if bounds_ready else None,
                        denominator="invested_account_value",
                        triggers=triggers,
                        meaning="투자금 기준 레이어 비중이 승인된 범위에 맞는지 확인합니다.",
                        next_step="레이어 목표와 향후 정기매수 정책을 검토합니다.",
                        status=status,
                    ),
                    eligible_for_increase=eligible,
                )
            )

        universe = sorted(set(position_map) | set(configured))
        for key in universe:
            position = position_map.get(key)
            target = configured.get(key)
            evidence = (risk_evidence.get("instruments") or {}).get(
                f"{key[0]}/{key[1]}", {}
            )
            target_ready = _target_ready(target)
            is_absent = position is None
            layer = str(
                (target or {}).get("layer") or (position or {}).get("layer") or ""
            ).lower()
            triggers: list[str] = []
            status = "OK"
            current = (
                0.0
                if is_absent and target_ready
                else (
                    position.get("invested_weight")
                    if position and target_ready
                    else None
                )
            )
            ready = target_ready and layer in LAYERS
            if not ready:
                status = "Review"
                triggers.append("policy_target_or_layer_missing")
            else:
                if position is not None and position.get("layer") != target.get(
                    "layer"
                ):
                    status = "Review"
                    triggers.append("layer_identity_mismatch")
                if not _within(current, target):
                    status = _raise_status(status, "Review")
                    triggers.append("instrument_out_of_range")
                drawdown = evidence.get("drawdown") or {}
                if drawdown:
                    if drawdown.get("state") != "complete":
                        status = _raise_status(status, "Watch")
                        triggers.append("instrument_drawdown_evidence_unavailable")
                    else:
                        threshold = (
                            risk_policy.get("instrument_drawdown_review") or {}
                        ).get(layer)
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
                    evidence.get(
                        "unrealized_pnl_krw",
                        position.get("profit_loss_krw") if position else None,
                    )
                )
                hard_maximum_breach = current is not None and float(current) > float(
                    target["maximum"]
                )
                if pnl is not None and pnl > 0 and hard_maximum_breach:
                    status = _raise_status(status, "Review")
                    triggers.append("gain_with_instrument_overweight")
            eligible = (
                cash_allocation_available
                and ready
                and current is not None
                and float(current) < float(target["minimum"])
            )
            next_step = "레이어·종목 목표와 향후 정기매수 정책을 검토합니다."
            result["instruments"].append(
                attach_decision(
                    _unit(
                        kind="instrument",
                        identity=f"{key[0]}/{key[1]}",
                        current=float(current) if current is not None else None,
                        target=float(target["target"]) if target_ready else None,
                        minimum=float(target["minimum"]) if target_ready else None,
                        maximum=float(target["maximum"]) if target_ready else None,
                        denominator="invested_account_value",
                        triggers=triggers,
                        meaning="종목의 현재 비중과 IPS 레이어·목표 범위를 함께 확인합니다.",
                        next_step=next_step,
                        status=status,
                        extra={
                            "market_country": key[0],
                            "symbol": key[1],
                            "layer": target.get("layer")
                            if target
                            else (position.get("layer") if position else None),
                            "unrealized_pnl_krw": position.get("profit_loss_krw")
                            if position
                            else None,
                            "evidence": evidence,
                        },
                    ),
                    eligible_for_increase=eligible,
                )
            )

    if allocation_reason is None:
        result["allocation_state"] = "complete"
    elif (
        allocation_reason == "invested_denominator_unavailable"
        and result["cash"] is not None
    ):
        result["allocation_state"] = "partial"
    else:
        result["allocation_state"] = "not_evaluable"
    result["allocation_reason"] = allocation_reason

    queue: list[dict[str, Any]] = []

    def _queue_item(
        item: Mapping[str, Any],
        *,
        blocking: bool = False,
        suggestion_override: dict[str, str] | None = None,
        priority_override: str | None = None,
    ) -> None:
        if item.get("status") == "OK" and not blocking:
            return
        priority = (
            priority_override if priority_override is not None else item.get("priority")
        )
        selected_suggestion = (
            None
            if blocking
            else suggestion_override
            or item.get("suggestion")
            or suggestion("hold_and_observe")
        )
        queue.append(
            {
                "priority": None if blocking else priority,
                "priority_label": "평가 차단" if blocking else priority_label(priority),
                "queue_class": queue_class(
                    priority,
                    blocking=blocking,
                    suggestion_code=(selected_suggestion or {}).get("code"),
                ),
                "kind": item.get("kind"),
                "identity": item.get("identity"),
                "status": item.get("status", "Review"),
                "triggers": list(item.get("triggers") or []),
                "suggestion": selected_suggestion,
                "meaning": item.get("meaning"),
                "verification_task": item.get("verification_task"),
                "red_team": _red_team(item, blocking=blocking),
                "evidence_refs": {
                    **dict(evidence_refs),
                    **(
                        {
                            "market_source_fingerprint": (item.get("evidence") or {})
                            .get("drawdown", {})
                            .get("source_fingerprint")
                        }
                        if item.get("kind") == "instrument"
                        else {}
                    ),
                },
            }
        )

    if result["allocation_state"] == "not_evaluable":
        reason = result["allocation_reason"] or "source_not_current_evaluable"
        _queue_item(
            {
                "kind": "allocation",
                "identity": "account",
                "status": "Review",
                "triggers": [reason],
                "meaning": "현재 Toss 원천 자료로 비중 조정 판단을 평가할 수 없습니다.",
                "verification_task": "최근 동기화·조정 상태와 정책 종목·레이어 커버리지를 확인합니다.",
            },
            blocking=True,
        )

    units: list[dict[str, Any]] = []
    if result["cash"] is not None:
        units.append(result["cash"])
    units.extend(result["layers"])
    units.extend(result["instruments"])
    if result["allocation_state"] != "not_evaluable":
        for item in units:
            _queue_item(item)

    if result["account_profit_loss"].get("status"):
        _queue_item(
            attach_decision(
                {
                    **result["account_profit_loss"],
                    "kind": "account_risk",
                    "identity": "account_profit_loss",
                    "current": None,
                    "minimum": None,
                    "maximum": None,
                    "triggers": result["account_profit_loss"].get("triggers", []),
                }
            )
        )
    if performance_status != "OK":
        _queue_item(
            {
                "kind": "performance",
                "identity": "account_return",
                "status": performance_status,
                "priority": "P4",
                "suggestion": suggestion("hold_and_observe"),
                "triggers": performance_triggers,
                "meaning": result["performance"]["meaning"],
                "verification_task": result["performance"]["verification_task"],
            },
            priority_override="P4",
        )

    result["review_queue"] = sorted(
        queue,
        key=lambda item: (
            0 if item["queue_class"] == "blocking" else 1,
            priority_rank(item["priority"]),
            -SEVERITY.get(item["status"], 0),
            str(item.get("kind", "")),
            str(item.get("identity", "")),
        ),
    )
    result["adjustment_suggestions"] = [
        {
            key: item[key]
            for key in (
                "priority",
                "priority_label",
                "kind",
                "identity",
                "status",
                "current",
                "minimum",
                "target",
                "maximum",
                "gap",
                "denominator",
                "suggestion",
                "meaning",
                "verification_task",
                "triggers",
            )
            if key in item
        }
        for item in sorted(
            units,
            key=lambda value: (
                priority_rank(value.get("priority")),
                value.get("kind", ""),
                value.get("identity", ""),
            ),
        )
        if item.get("status") != "OK"
        and item.get("priority") in {"P1", "P2", "P3"}
        and (item.get("suggestion") or {}).get("code") != "hold_and_observe"
        and result["allocation_state"] in {"complete", "partial"}
    ]
    return result


def evaluation_fingerprint(
    *,
    source_fingerprint: str,
    performance_fingerprint: str | None,
    policy_hash: str,
    market_evidence_fingerprint: str = "",
    engine_version: str = ENGINE_VERSION,
) -> str:
    payload = "|".join(
        (
            source_fingerprint,
            performance_fingerprint or "",
            policy_hash,
            market_evidence_fingerprint,
            engine_version,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
