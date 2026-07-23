"""Pure calculations for Phase 2 account-value and performance history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, MutableMapping, Sequence


ENGINE_VERSION = "phase2-v1"
DECIMAL_TOLERANCE = Decimal("0.00000001")
ONE_BASIS_POINT = Decimal("0.0001")
MIN_MATERIALITY_KRW = Decimal("10000")


class PerformanceCalculationError(ValueError):
    """Raised when account history cannot be calculated safely."""


@dataclass(frozen=True)
class TrackingBaseline:
    id: int
    account_alias: str
    baseline_snapshot_id: int
    tracking_started_at: str
    initial_principal_krw: Decimal
    baseline_fx_rate: Decimal | None


@dataclass(frozen=True)
class CashFlowCandidate:
    baseline_id: int
    from_snapshot_id: int
    to_snapshot_id: int
    currency: str
    observed_delta_native: Decimal
    explained_trade_delta_native: Decimal
    residual_native: Decimal
    residual_krw: Decimal | None
    materiality_threshold_krw: Decimal
    bridge_basis: str
    candidate_fingerprint: str
    id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "baseline_id": self.baseline_id,
            "from_snapshot_id": self.from_snapshot_id,
            "to_snapshot_id": self.to_snapshot_id,
            "currency": self.currency,
            "observed_delta_native": float(self.observed_delta_native),
            "explained_trade_delta_native": float(self.explained_trade_delta_native),
            "residual_native": float(self.residual_native),
            "residual_krw": (
                float(self.residual_krw) if self.residual_krw is not None else None
            ),
            "materiality_threshold_krw": float(self.materiality_threshold_krw),
            "bridge_basis": self.bridge_basis,
            "candidate_fingerprint": self.candidate_fingerprint,
        }


@dataclass(frozen=True)
class PerformanceProjection:
    baseline_id: int
    through_snapshot_id: int
    input_fingerprint: str
    engine_version: str
    state: str
    data_quality: Mapping[str, Any]
    points: tuple[Mapping[str, Any], ...]
    executions: tuple[Mapping[str, Any], ...]
    candidates: tuple[CashFlowCandidate, ...]


def _decimal(value: Any, field: str, *, default: Decimal | None = None) -> Decimal:
    if value is None and default is not None:
        return default
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PerformanceCalculationError(f"invalid {field}") from exc
    if not parsed.is_finite():
        raise PerformanceCalculationError(f"invalid {field}")
    return parsed


def _optional_decimal(value: Any, field: str) -> Decimal | None:
    return None if value is None else _decimal(value, field)


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_time(
    order: Mapping[str, Any], *, settlement: bool = False
) -> datetime | None:
    if settlement:
        value = order.get("settlement_date")
    else:
        value = order.get("filled_at") or order.get("ordered_at")
    return _timestamp(value)


def _snapshot_time(snapshot: Mapping[str, Any]) -> datetime:
    parsed = _timestamp(snapshot.get("synced_at"))
    if parsed is None:
        raise PerformanceCalculationError("snapshot synced_at is invalid")
    return parsed


def _money(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _baseline_from_row(row: Mapping[str, Any]) -> TrackingBaseline:
    return TrackingBaseline(
        id=int(row["id"]),
        account_alias=str(row["account_alias"]),
        baseline_snapshot_id=int(row["baseline_snapshot_id"]),
        tracking_started_at=str(row["tracking_started_at"]),
        initial_principal_krw=_decimal(
            row["initial_principal_krw"], "initial_principal_krw"
        ),
        baseline_fx_rate=_optional_decimal(
            row.get("baseline_fx_rate"), "baseline_fx_rate"
        ),
    )


def validate_baseline_snapshot(
    snapshot: Mapping[str, Any], expected_principal_krw: Decimal
) -> TrackingBaseline:
    if snapshot.get("state") != "complete" or not snapshot.get(
        "is_current_evaluable", False
    ):
        raise PerformanceCalculationError(
            "baseline requires a current complete snapshot"
        )
    total = _decimal(snapshot.get("total_value_krw"), "snapshot.total_value_krw")
    expected = _decimal(expected_principal_krw, "expected_principal_krw")
    if abs(total - expected) > Decimal("0.01"):
        raise PerformanceCalculationError("expected principal does not match snapshot")
    fx = snapshot.get("fx_rate") or {}
    return TrackingBaseline(
        id=0,
        account_alias=str(snapshot.get("account_alias") or ""),
        baseline_snapshot_id=int(snapshot["id"]),
        tracking_started_at=str(snapshot["synced_at"]),
        initial_principal_krw=total,
        baseline_fx_rate=_optional_decimal(fx.get("rate"), "baseline_fx_rate"),
    )


def materiality_threshold_krw(previous_total_krw: Decimal) -> Decimal:
    return max(MIN_MATERIALITY_KRW, previous_total_krw * ONE_BASIS_POINT)


def canonical_executions(
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    by_id: dict[str, Mapping[str, Any]] = {}
    comparable = (
        "symbol",
        "currency",
        "side",
        "status",
        "filled_quantity",
        "average_filled_price_native",
        "filled_amount_native",
        "commission_native",
        "tax_native",
        "filled_at",
        "settlement_date",
    )
    for snapshot in snapshots:
        for order in snapshot.get("orders", []):
            quantity = _decimal(order.get("filled_quantity", 0), "filled_quantity")
            if quantity <= 0:
                continue
            order_id = str(order.get("order_id") or "")
            if not order_id:
                raise PerformanceCalculationError("order_id is missing")
            existing = by_id.get(order_id)
            if existing is not None:
                if any(existing.get(key) != order.get(key) for key in comparable):
                    raise PerformanceCalculationError(
                        f"conflicting execution: order_id={order_id}"
                    )
                continue
            by_id[order_id] = dict(order)
    return tuple(
        sorted(
            by_id.values(),
            key=lambda order: (
                _event_time(order) or datetime.max.replace(tzinfo=timezone.utc),
                str(order["order_id"]),
            ),
        )
    )


def _cash_delta(order: Mapping[str, Any]) -> Decimal:
    amount = _optional_decimal(
        order.get("filled_amount_native"), "filled_amount_native"
    )
    if amount is None:
        quantity = _decimal(order.get("filled_quantity"), "filled_quantity")
        price = _decimal(
            order.get("average_filled_price_native"), "average_filled_price_native"
        )
        amount = quantity * price
    commission = _decimal(order.get("commission_native", 0), "commission_native")
    tax = _decimal(order.get("tax_native", 0), "tax_native")
    side = str(order.get("side") or "").upper()
    if side == "BUY":
        return -(amount + commission + tax)
    if side == "SELL":
        return amount - commission - tax
    raise PerformanceCalculationError(f"unsupported execution side: {side}")


def _orders_in_interval(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    executions: Sequence[Mapping[str, Any]],
    *,
    settlement: bool,
) -> list[Mapping[str, Any]]:
    start = _snapshot_time(previous)
    end = _snapshot_time(current)
    selected = []
    for order in executions:
        event = _event_time(order, settlement=settlement)
        if event is not None and start < event <= end:
            selected.append(order)
    return selected


def _cash_map(snapshot: Mapping[str, Any]) -> dict[str, Decimal]:
    return {
        str(item["currency"]): _decimal(
            item.get("buying_power_native"),
            f"{item.get('currency')}.buying_power_native",
        )
        for item in snapshot.get("cash", [])
    }


def _fx_rate(snapshot: Mapping[str, Any]) -> Decimal | None:
    return _optional_decimal((snapshot.get("fx_rate") or {}).get("rate"), "fx.rate")


def _candidate_fingerprint(
    baseline_id: int,
    previous_id: int,
    current_id: int,
    currency: str,
    residual: Decimal,
    bridge_basis: str,
) -> str:
    value = f"{baseline_id}|{previous_id}|{current_id}|{currency}|{residual.normalize()}|{bridge_basis}"
    return hashlib.sha256(value.encode()).hexdigest()


def _bridge(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    currency: str,
    executions: Sequence[Mapping[str, Any]],
    *,
    settlement: bool,
) -> tuple[Decimal, Decimal] | None:
    previous_cash = _cash_map(previous).get(currency)
    current_cash = _cash_map(current).get(currency)
    if previous_cash is None or current_cash is None:
        return None
    orders = [
        order
        for order in _orders_in_interval(
            previous, current, executions, settlement=settlement
        )
        if str(order.get("currency")) == currency
    ]
    explained = sum((_cash_delta(order) for order in orders), Decimal("0"))
    return current_cash - previous_cash, explained


def detect_cash_candidates(
    baseline_id: int,
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    executions: Sequence[Mapping[str, Any]],
) -> tuple[CashFlowCandidate, ...]:
    previous_total = _decimal(
        previous.get("total_value_krw"), "previous.total_value_krw"
    )
    threshold_krw = materiality_threshold_krw(previous_total)
    currencies = set(_cash_map(previous)) | set(_cash_map(current))
    candidates = []
    current_fx = _fx_rate(current)
    for currency in sorted(currencies):
        filled_bridge = _bridge(
            previous, current, currency, executions, settlement=False
        )
        settlement_bridge = _bridge(
            previous, current, currency, executions, settlement=True
        )
        bridges = {
            "filled_at": filled_bridge,
            "settlement_date": settlement_bridge,
        }
        residuals = {
            basis: observed - explained
            for basis, value in bridges.items()
            if value is not None
            for observed, explained in (value,)
        }
        if not residuals:
            continue

        def residual_krw(value: Decimal) -> Decimal | None:
            if currency == "KRW":
                return value
            return value * current_fx if current_fx is not None else None

        if any(
            residual_krw(value) is not None
            and abs(residual_krw(value)) <= threshold_krw
            for value in residuals.values()
        ):
            continue
        basis, residual = min(residuals.items(), key=lambda item: abs(item[1]))
        observed, explained = bridges[basis] or (Decimal("0"), Decimal("0"))
        candidates.append(
            CashFlowCandidate(
                baseline_id=baseline_id,
                from_snapshot_id=int(previous["id"]),
                to_snapshot_id=int(current["id"]),
                currency=currency,
                observed_delta_native=observed,
                explained_trade_delta_native=explained,
                residual_native=residual,
                residual_krw=residual_krw(residual),
                materiality_threshold_krw=threshold_krw,
                bridge_basis=basis,
                candidate_fingerprint=_candidate_fingerprint(
                    baseline_id,
                    int(previous["id"]),
                    int(current["id"]),
                    currency,
                    residual,
                    basis,
                ),
            )
        )
    return tuple(candidates)


def _holding_map(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], Decimal]:
    return {
        (str(item["symbol"]), str(item["currency"])): _decimal(
            item.get("quantity"), f"{item.get('symbol')}.quantity"
        )
        for item in snapshot.get("holdings", [])
    }


def detect_quantity_issues(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    executions: Sequence[Mapping[str, Any]],
) -> list[str]:
    observed = _holding_map(current)
    previous_map = _holding_map(previous)
    keys = set(observed) | set(previous_map)
    expected: dict[tuple[str, str], Decimal] = {key: Decimal("0") for key in keys}
    for order in _orders_in_interval(previous, current, executions, settlement=False):
        key = (str(order.get("symbol")), str(order.get("currency")))
        quantity = _decimal(order.get("filled_quantity"), "filled_quantity")
        side = str(order.get("side") or "").upper()
        expected[key] = expected.get(key, Decimal("0")) + (
            quantity if side == "BUY" else -quantity if side == "SELL" else Decimal("0")
        )
    issues = []
    for key in sorted(keys):
        actual_delta = observed.get(key, Decimal("0")) - previous_map.get(
            key, Decimal("0")
        )
        if abs(actual_delta - expected.get(key, Decimal("0"))) > DECIMAL_TOLERANCE:
            issues.append(
                f"quantity mismatch {key[0]} {key[1]}: observed={actual_delta} expected={expected.get(key, Decimal('0'))}"
            )
    return issues


def _seed_state(
    snapshot: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Decimal]]:
    state = {}
    for item in snapshot.get("holdings", []):
        key = (str(item["symbol"]), str(item["currency"]))
        quantity = _decimal(item.get("quantity"), f"{key}.quantity")
        state[key] = {
            "quantity": quantity,
            "actual_total_basis": quantity
            * _decimal(
                item.get("average_purchase_price"), f"{key}.average_purchase_price"
            ),
            "tracking_total_basis": _decimal(
                item.get("market_value_native"), f"{key}.market_value_native"
            ),
        }
    return state


def apply_execution(
    state: MutableMapping[tuple[str, str], dict[str, Decimal]],
    order: Mapping[str, Any],
) -> dict[str, Any]:
    key = (str(order.get("symbol")), str(order.get("currency")))
    quantity = _decimal(order.get("filled_quantity"), "filled_quantity")
    amount = _optional_decimal(
        order.get("filled_amount_native"), "filled_amount_native"
    )
    if amount is None:
        amount = quantity * _decimal(
            order.get("average_filled_price_native"), "average_filled_price_native"
        )
    commission = _decimal(order.get("commission_native", 0), "commission_native")
    tax = _decimal(order.get("tax_native", 0), "tax_native")
    side = str(order.get("side") or "").upper()
    position = state.setdefault(
        key,
        {
            "quantity": Decimal("0"),
            "actual_total_basis": Decimal("0"),
            "tracking_total_basis": Decimal("0"),
        },
    )
    actual_before = (
        position["actual_total_basis"] / position["quantity"]
        if position["quantity"]
        else Decimal("0")
    )
    tracking_before = (
        position["tracking_total_basis"] / position["quantity"]
        if position["quantity"]
        else Decimal("0")
    )
    if side == "BUY":
        total_cost = amount + commission + tax
        position["quantity"] += quantity
        position["actual_total_basis"] += total_cost
        position["tracking_total_basis"] += total_cost
        actual_realized = Decimal("0")
        tracking_realized = Decimal("0")
    elif side == "SELL":
        if position["quantity"] + DECIMAL_TOLERANCE < quantity:
            raise PerformanceCalculationError(
                f"sell exceeds tracked quantity: {key[0]}"
            )
        proceeds = amount - commission - tax
        actual_realized = proceeds - actual_before * quantity
        tracking_realized = proceeds - tracking_before * quantity
        position["quantity"] -= quantity
        position["actual_total_basis"] -= actual_before * quantity
        position["tracking_total_basis"] -= tracking_before * quantity
    else:
        raise PerformanceCalculationError(f"unsupported execution side: {side}")
    return {
        "order_id": str(order["order_id"]),
        "symbol": key[0],
        "currency": key[1],
        "side": side,
        "filled_at": order.get("filled_at"),
        "settlement_date": order.get("settlement_date"),
        "filled_quantity_native": float(quantity),
        "filled_amount_native": float(amount),
        "commission_native": float(commission),
        "tax_native": float(tax),
        "actual_basis_before_native": _float(actual_before),
        "tracking_basis_before_native": _float(tracking_before),
        "actual_realized_pnl_native": _float(actual_realized),
        "tracking_realized_pnl_native": _float(tracking_realized),
        "realized_pnl_krw": None,
        "krw_conversion_snapshot_id": None,
    }


def _current_cost_basis(snapshot: Mapping[str, Any]) -> Decimal | None:
    values = [
        _optional_decimal(item.get("cost_krw"), "holding.cost_krw")
        for item in snapshot.get("holdings", [])
    ]
    if any(value is None for value in values):
        return None
    return sum((value for value in values if value is not None), Decimal("0"))


def _unrealized(snapshot: Mapping[str, Any]) -> Decimal | None:
    values = [
        _optional_decimal(item.get("profit_loss_krw"), "holding.profit_loss_krw")
        for item in snapshot.get("holdings", [])
    ]
    if any(value is None for value in values):
        return None
    return sum((value for value in values if value is not None), Decimal("0"))


def _fx_remeasurement(
    baseline_snapshot: Mapping[str, Any], current_snapshot: Mapping[str, Any]
) -> Decimal | None:
    baseline_rate = _fx_rate(baseline_snapshot)
    current_rate = _fx_rate(current_snapshot)
    if baseline_rate is None or current_rate is None:
        return None
    foreign_value = sum(
        (
            _decimal(item.get("market_value_native"), "holding.market_value_native")
            for item in current_snapshot.get("holdings", [])
            if str(item.get("currency")) == "USD"
        ),
        Decimal("0"),
    ) + _cash_map(current_snapshot).get("USD", Decimal("0"))
    return foreign_value * (current_rate - baseline_rate)


def fingerprint_inputs(
    baseline: TrackingBaseline,
    snapshots: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, Mapping[str, Any]],
) -> str:
    payload = {
        "baseline": {
            "id": baseline.id,
            "snapshot_id": baseline.baseline_snapshot_id,
            "principal": str(baseline.initial_principal_krw),
        },
        "snapshots": [
            {
                "id": snapshot.get("id"),
                "fingerprint": snapshot.get("source_fingerprint"),
            }
            for snapshot in snapshots
        ],
        "decisions": decisions,
        "engine": ENGINE_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_projection(
    baseline: TrackingBaseline | Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, Mapping[str, Any]],
    candidate_ids: Mapping[str, int] | None = None,
) -> PerformanceProjection:
    if not snapshots:
        raise PerformanceCalculationError("no complete snapshots")
    baseline_model = (
        baseline
        if isinstance(baseline, TrackingBaseline)
        else _baseline_from_row(baseline)
    )
    ordered = sorted(
        snapshots, key=lambda item: (_snapshot_time(item), int(item["id"]))
    )
    baseline_index = next(
        (
            index
            for index, item in enumerate(ordered)
            if int(item["id"]) == baseline_model.baseline_snapshot_id
        ),
        None,
    )
    if baseline_index is None:
        raise PerformanceCalculationError(
            "baseline snapshot is not in complete history"
        )
    ordered = ordered[baseline_index:]
    executions = canonical_executions(ordered)
    baseline_snapshot = ordered[0]
    baseline_time = _snapshot_time(baseline_snapshot)
    state = _seed_state(baseline_snapshot)
    points: list[Mapping[str, Any]] = []
    ledger: list[Mapping[str, Any]] = []
    candidates: list[CashFlowCandidate] = []
    processed: set[str] = set()
    cumulative_flow = Decimal("0")
    investment_principal = baseline_model.initial_principal_krw
    segment_id = 0
    segment_twr = Decimal("0")
    previous = None
    all_issues: list[str] = []
    for snapshot in ordered:
        point_time = _snapshot_time(snapshot)
        interval_issues: list[str] = []
        interval_candidates: list[CashFlowCandidate] = []
        if previous is not None:
            interval_executions = [
                order
                for order in executions
                if order["order_id"] not in processed
                and (_event_time(order) is not None)
                and _snapshot_time(previous) < _event_time(order) <= point_time
            ]
            interval_candidates = list(
                detect_cash_candidates(
                    baseline_model.id,
                    previous,
                    snapshot,
                    executions,
                )
            )
            if candidate_ids:
                interval_candidates = [
                    replace(
                        candidate,
                        id=candidate_ids.get(candidate.candidate_fingerprint),
                    )
                    for candidate in interval_candidates
                ]
            candidates.extend(interval_candidates)
            interval_issues.extend(
                detect_quantity_issues(previous, snapshot, executions)
            )
            for candidate in interval_candidates:
                decision = (
                    decisions.get(candidate.id) if candidate.id is not None else None
                )
                if decision is None:
                    interval_issues.append(
                        f"unresolved cash flow candidate: {candidate.candidate_fingerprint}"
                    )
                else:
                    classification = str(decision.get("classification"))
                    if classification in {"external_deposit", "external_withdrawal"}:
                        amount = _optional_decimal(
                            decision.get("confirmed_amount_krw"),
                            "decision.confirmed_amount_krw",
                        )
                        if amount is None:
                            amount = candidate.residual_krw or Decimal("0")
                        signed = (
                            amount if classification == "external_deposit" else -amount
                        )
                        cumulative_flow += signed
                        investment_principal += signed
                        interval_issues.append(
                            "confirmed external flow requires TWR boundary"
                        )
            if interval_issues:
                segment_id += 1
                segment_twr = Decimal("0")
        for order in interval_executions if previous is not None else []:
            if _event_time(order) is None or _event_time(order) <= baseline_time:
                continue
            try:
                row = apply_execution(state, order)
            except PerformanceCalculationError as exc:
                interval_issues.append(str(exc))
                continue
            current_rate = _fx_rate(snapshot)
            if row["currency"] == "KRW":
                row["realized_pnl_krw"] = (
                    row["tracking_realized_pnl_native"]
                    if row["tracking_realized_pnl_native"] is not None
                    else None
                )
                row["actual_realized_pnl_krw"] = row["actual_realized_pnl_native"]
            elif (
                current_rate is not None
                and row["tracking_realized_pnl_native"] is not None
            ):
                row["realized_pnl_krw"] = row["tracking_realized_pnl_native"] * float(
                    current_rate
                )
                row["actual_realized_pnl_krw"] = row[
                    "actual_realized_pnl_native"
                ] * float(current_rate)
            row["source_snapshot_id"] = int(snapshot["id"])
            row["krw_conversion_snapshot_id"] = int(snapshot["id"])
            ledger.append(row)
            processed.add(str(order["order_id"]))
        total = _optional_decimal(snapshot.get("total_value_krw"), "total_value_krw")
        invested = _optional_decimal(
            snapshot.get("invested_value_krw"), "invested_value_krw"
        )
        cash = _optional_decimal(snapshot.get("cash_value_krw"), "cash_value_krw")
        if total is None or invested is None or cash is None:
            interval_issues.append("missing required account value")
        gain = (
            total - baseline_model.initial_principal_krw - cumulative_flow
            if total is not None
            else None
        )
        simple_return = (
            gain / investment_principal
            if gain is not None and investment_principal > 0
            else None
        )
        interval_twr = None
        if previous is None and total is not None:
            interval_twr = Decimal("0")
        elif previous is not None and not interval_issues and total is not None:
            previous_total = _decimal(
                previous.get("total_value_krw"), "previous.total_value_krw"
            )
            if previous_total > 0:
                interval_twr = total / previous_total - Decimal("1")
                segment_twr = (Decimal("1") + segment_twr) * (
                    Decimal("1") + interval_twr
                ) - Decimal("1")
        if interval_issues:
            interval_twr = None
            all_issues.extend(interval_issues)
        points.append(
            {
                "snapshot_id": int(snapshot["id"]),
                "previous_snapshot_id": int(previous["id"])
                if previous is not None
                else None,
                "point_at": snapshot["synced_at"],
                "evaluation_state": "non_evaluable" if interval_issues else "evaluable",
                "evaluation_reason": "; ".join(dict.fromkeys(interval_issues)) or None,
                "total_value_krw": _money(total),
                "invested_value_krw": _money(invested),
                "cash_value_krw": _money(cash),
                "current_cost_basis_krw": _money(_current_cost_basis(snapshot)),
                "unrealized_pnl_krw": _money(_unrealized(snapshot)),
                "investment_principal_krw": _money(investment_principal),
                "cumulative_external_flow_krw": _money(cumulative_flow),
                "account_gain_krw": _money(gain),
                "simple_return": _money(simple_return),
                "interval_twr": _money(interval_twr),
                "segment_id": segment_id,
                "segment_twr": _money(segment_twr)
                if interval_twr is not None
                else None,
                "tracked_realized_pnl_krw": _money(
                    sum(
                        (
                            _decimal(
                                row.get("realized_pnl_krw"),
                                "realized_pnl_krw",
                                default=Decimal("0"),
                            )
                            for row in ledger
                        ),
                        Decimal("0"),
                    )
                ),
                "actual_realized_pnl_krw": _money(
                    sum(
                        (
                            _decimal(
                                row.get("actual_realized_pnl_krw"),
                                "actual_realized_pnl_krw",
                                default=Decimal("0"),
                            )
                            for row in ledger
                        ),
                        Decimal("0"),
                    )
                ),
                "fx_remeasurement_krw": _money(
                    _fx_remeasurement(baseline_snapshot, snapshot)
                ),
            }
        )
        previous = snapshot
    if not points:
        raise PerformanceCalculationError("no performance points")
    state_value = "complete" if not all_issues else "partial"
    return PerformanceProjection(
        baseline_id=baseline_model.id,
        through_snapshot_id=int(ordered[-1]["id"]),
        input_fingerprint=fingerprint_inputs(baseline_model, ordered, decisions),
        engine_version=ENGINE_VERSION,
        state=state_value,
        data_quality={"issues": list(dict.fromkeys(all_issues))},
        points=tuple(points),
        executions=tuple(ledger),
        candidates=tuple(candidates),
    )
