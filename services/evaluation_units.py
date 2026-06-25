"""Evaluation unit builders for Evaluation Framework v2."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from core.asset import DEFAULT_LAYER, LAYER_TYPES
from core.evaluation import EvaluationPeriod, EvaluationUnit


DEFAULT_LAYER_LIMITS = {
    "core": {
        "allowed_mdd": -0.35,
        "allowed_volatility": 0.30,
        "max_weight": 0.90,
        "min_efficiency": -0.50,
        "benchmark": "SPY:80,QQQ:20",
        "check_frequency": "monthly",
        "manual_intervention_allowed": False,
    },
    "satellite": {
        "allowed_mdd": -0.45,
        "allowed_volatility": 0.55,
        "max_weight": 0.30,
        "min_efficiency": 0.00,
        "benchmark": "QQQ",
        "check_frequency": "monthly",
        "manual_intervention_allowed": True,
    },
    "experiment": {
        "allowed_mdd": -0.25,
        "allowed_volatility": 0.70,
        "max_weight": 0.05,
        "min_efficiency": 0.20,
        "benchmark": "QQQ",
        "check_frequency": "weekly",
        "manual_intervention_allowed": True,
    },
}

DEFAULT_LAYER_BENCHMARKS = {
    layer: str(limits["benchmark"]) for layer, limits in DEFAULT_LAYER_LIMITS.items()
}


class EvaluationUnitSet(NamedTuple):
    """Generated layer and asset units plus target weights."""

    layer_units: list[EvaluationUnit]
    asset_units: list[EvaluationUnit]
    layer_targets: dict[str, float]
    asset_targets: dict[str, float]


def normalize_layer_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure metrics data has a v2 layer column."""
    result = df.copy()

    if "layer" not in result.columns:
        result["layer"] = None

    for idx, row in result.iterrows():
        layer = row.get("layer") or DEFAULT_LAYER
        layer = str(layer).strip().lower()
        if layer not in LAYER_TYPES or layer not in DEFAULT_LAYER_LIMITS:
            layer = DEFAULT_LAYER
        result.at[idx, "layer"] = layer
    return result


def layer_targets_from_ips_config(ips_config: dict | None) -> dict[str, float]:
    """Return v2 layer targets from runtime IPS config."""
    config = ips_config or {}
    target_cfg = config.get("target_allocation", {})
    targets = {"core": 0.80, "satellite": 0.20, "experiment": 0.0}
    for layer in targets:
        values = target_cfg.get(layer)
        if isinstance(values, dict):
            targets[layer] = float(values.get("target", targets[layer]))
    return targets


def _layer_unit(
    layer: str,
    target_weight: float,
    evaluation_period: EvaluationPeriod,
    benchmark: str,
    layer_benchmarks: dict[str, str] | None = None,
) -> EvaluationUnit:
    limits = DEFAULT_LAYER_LIMITS[layer]
    return EvaluationUnit(
        level="layer",
        name=layer,
        benchmark=_benchmark_for_layer(layer, benchmark, layer_benchmarks),
        target_weight=target_weight,
        allowed_mdd=limits["allowed_mdd"],
        allowed_volatility=limits["allowed_volatility"],
        max_weight=limits["max_weight"],
        min_efficiency=limits["min_efficiency"],
        check_frequency=limits["check_frequency"],
        manual_intervention_allowed=limits["manual_intervention_allowed"],
        evaluation_period=evaluation_period,
    )


def _benchmark_for_layer(
    layer: str,
    fallback_benchmark: str,
    layer_benchmarks: dict[str, str] | None = None,
) -> str:
    if layer_benchmarks is not None:
        configured = str(layer_benchmarks.get(layer, "") or "").strip().upper()
        if configured:
            return configured
        return DEFAULT_LAYER_LIMITS[layer]["benchmark"]
    fallback = str(fallback_benchmark or "").strip().upper()
    return DEFAULT_LAYER_LIMITS[layer]["benchmark"] if fallback == "" else fallback


def build_evaluation_units(
    metrics_df: pd.DataFrame,
    ips_config: dict | None,
    evaluation_period: EvaluationPeriod,
    benchmark: str,
    layer_benchmarks: dict[str, str] | None = None,
) -> EvaluationUnitSet:
    """Build v2 layer and asset units from metrics data."""
    metrics = normalize_layer_metadata(metrics_df)
    if "가중치" in metrics.columns:
        weights = pd.to_numeric(metrics["가중치"], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(0.0, index=metrics.index)

    layer_targets = layer_targets_from_ips_config(ips_config)
    layer_units = [
        _layer_unit(layer, target, evaluation_period, benchmark, layer_benchmarks)
        for layer, target in layer_targets.items()
        if target > 0 or bool((metrics["layer"] == layer).any()) or layer == "experiment"
    ]

    layer_weight = metrics.groupby("layer")["가중치"].sum() if "가중치" in metrics.columns else pd.Series(dtype=float)
    asset_targets: dict[str, float] = {}
    asset_units: list[EvaluationUnit] = []
    for ticker, row in metrics.iterrows():
        layer = str(row["layer"])
        current_layer_weight = float(layer_weight.get(layer, 0.0))
        layer_target = float(layer_targets.get(layer, 0.0))
        current_weight = float(weights.get(ticker, 0.0))
        target_weight = (
            layer_target * current_weight / current_layer_weight
            if current_layer_weight > 0
            else None
        )
        if target_weight is not None:
            asset_targets[str(ticker)] = target_weight
        limits = DEFAULT_LAYER_LIMITS.get(layer, DEFAULT_LAYER_LIMITS["core"])
        asset_units.append(
            EvaluationUnit(
                level="asset",
                name=str(ticker),
                parent_layer=layer,
                benchmark=_benchmark_for_layer(layer, benchmark, layer_benchmarks),
                target_weight=target_weight,
                allowed_mdd=limits["allowed_mdd"],
                allowed_volatility=limits["allowed_volatility"],
                max_weight=limits["max_weight"],
                min_efficiency=limits["min_efficiency"],
                thesis=str(row.get("thesis", "") or "") or None,
                check_frequency=limits["check_frequency"],
                manual_intervention_allowed=limits["manual_intervention_allowed"],
                evaluation_period=evaluation_period,
            )
        )

    return EvaluationUnitSet(
        layer_units=layer_units,
        asset_units=asset_units,
        layer_targets=layer_targets,
        asset_targets=asset_targets,
    )
