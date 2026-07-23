"""Pure projection of a complete Toss account snapshot for IPS inspection."""

from __future__ import annotations

import math
from typing import Any

from storage.account_observation_store import get_snapshot, latest_complete
from storage.instrument_profile_store import profile_map


TOLERANCE_KRW = 1.0
LAYERS = ("core", "satellite", "experiment")


class AccountProjectionError(ValueError):
    """Raised when an account snapshot cannot be projected safely."""


def _finite(value: Any, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise AccountProjectionError(f"missing {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AccountProjectionError(f"invalid {field}") from exc
    if not math.isfinite(number):
        raise AccountProjectionError(f"invalid {field}")
    return number


def _project_complete_snapshot(
    snapshot: dict[str, Any],
    profiles: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    total = _finite(snapshot["total_value_krw"], "total_value_krw")
    invested = _finite(snapshot["invested_value_krw"], "invested_value_krw")
    cash = _finite(snapshot["cash_value_krw"], "cash_value_krw")
    if total <= 0:
        raise AccountProjectionError("total_value_krw must be positive")
    if invested < 0 or cash < 0:
        raise AccountProjectionError("invested and cash values must be nonnegative")
    if not math.isclose(total, invested + cash, abs_tol=TOLERANCE_KRW):
        raise AccountProjectionError("account totals do not reconcile")
    reconciliation = snapshot.get("reconciliation") or {}
    holdings_reconciliation = reconciliation.get("holdings") or {}
    if holdings_reconciliation.get("all_within_tolerance") is not True:
        raise AccountProjectionError("snapshot reconciliation failed")
    if any(
        isinstance(value, dict) and value.get("within_tolerance") is False
        for value in holdings_reconciliation.values()
    ):
        raise AccountProjectionError("snapshot reconciliation failed")

    holdings = sorted(
        snapshot["holdings"],
        key=lambda item: (
            str(item.get("market_country", "")).upper(),
            str(item.get("symbol", "")).upper(),
        ),
    )
    positions: list[dict[str, Any]] = []
    unclassified: list[dict[str, str]] = []
    layer_values = {layer: 0.0 for layer in LAYERS}
    classified_value = 0.0
    holding_total = 0.0

    for holding in holdings:
        symbol = str(holding.get("symbol", "")).strip().upper()
        market_country = str(holding.get("market_country", "")).strip().upper()
        if not symbol or not market_country:
            raise AccountProjectionError("holding identity is missing")
        market_value = _finite(
            holding.get("market_value_krw"), f"{symbol} market_value_krw"
        )
        if market_value < 0:
            raise AccountProjectionError(
                f"{symbol} market_value_krw must be nonnegative"
            )
        holding_total += market_value
        profile = profiles.get((market_country, symbol))
        position: dict[str, Any] = {
            "symbol": symbol,
            "name": holding.get("name"),
            "market_country": market_country,
            "currency": holding.get("currency"),
            "quantity": holding.get("quantity"),
            "market_value_krw": market_value,
            "cost_krw": holding.get("cost_krw"),
            "profit_loss_krw": holding.get("profit_loss_krw"),
            "gross_weight": market_value / total,
            "invested_weight": market_value / invested if invested > 0 else None,
            "layer": profile["layer"] if profile else None,
            "thesis_status": profile["thesis_status"] if profile else None,
            "thesis_note": profile["thesis_note"] if profile else None,
        }
        positions.append(position)
        if profile is None:
            unclassified.append({"market_country": market_country, "symbol": symbol})
        else:
            classified_value += market_value
            layer_values[profile["layer"]] += market_value

    if not math.isclose(holding_total, invested, abs_tol=TOLERANCE_KRW):
        raise AccountProjectionError("holding totals do not reconcile")
    if invested == 0 and any(value > TOLERANCE_KRW for value in (holding_total,)):
        raise AccountProjectionError("zero invested value has positive holdings")

    invested_evaluable = invested > 0
    if not invested_evaluable:
        layer_weights: dict[str, float] = {layer: 0.0 for layer in LAYERS}
        coverage: float | None = None
    else:
        layer_weights = {
            layer: value / invested for layer, value in layer_values.items()
        }
        coverage = classified_value / invested

    return {
        "snapshot_id": int(snapshot["id"]),
        "account_alias": snapshot["account_alias"],
        "source_fingerprint": snapshot["source_fingerprint"],
        "synced_at": snapshot["synced_at"],
        "source_timestamps": snapshot["source_timestamps"],
        "total_value_krw": total,
        "invested_value_krw": invested,
        "cash_value_krw": cash,
        "cash_weight_gross": cash / total,
        "invested_weights_evaluable": invested_evaluable,
        "classification_coverage_invested": coverage,
        "positions": positions,
        "layer_weights_invested": layer_weights,
        "unclassified": unclassified,
        "data_quality": snapshot["data_quality"],
        "reconciliation": snapshot["reconciliation"],
    }


def build_account_projection(
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Project one explicit or latest complete Toss snapshot."""
    snapshot = (
        get_snapshot(snapshot_id)
        if snapshot_id is not None
        else latest_complete(account_alias)
    )
    if snapshot is None:
        raise AccountProjectionError("complete Toss snapshot not found")
    if snapshot["account_alias"] != account_alias:
        raise AccountProjectionError("snapshot account alias mismatch")
    if snapshot["state"] != "complete":
        raise AccountProjectionError(
            f"snapshot {snapshot['id']} is not complete: {snapshot['state']}"
        )
    if not snapshot.get("is_current_evaluable", False):
        raise AccountProjectionError(
            f"snapshot {snapshot['id']} is not current evaluable"
        )
    return _project_complete_snapshot(snapshot, profile_map(account_alias))
