from datetime import date

import pandas as pd
import pytest

from core.asset import Asset
from core.evaluation import EvaluationPeriod
from services.evaluation_units import build_evaluation_units, normalize_layer_category


def _period():
    return EvaluationPeriod(label="3M", start_date=date(2026, 3, 23), end_date=date(2026, 6, 23))


def test_asset_fills_current_layer_and_category_defaults():
    asset = Asset(ticker="UFO", allocation=3, layer="satellite")

    assert asset.layer == "satellite"
    assert asset.category == "satellite_ai_infra"


def test_normalize_layer_category_uses_current_fields():
    metrics = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "layer": ["core", "satellite"],
            "category": ["core_market", "satellite_nextgen"],
            "가중치": [0.8, 0.2],
        }
    ).set_index("ticker")

    normalized = normalize_layer_category(metrics)

    assert normalized.loc["VOO", "layer"] == "core"
    assert normalized.loc["UFO", "layer"] == "satellite"
    assert normalized.loc["UFO", "category"] == "satellite_nextgen"


def test_build_evaluation_units_aggregates_layers_and_assets():
    metrics = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "layer": ["core", "satellite"],
            "category": ["core_market", "satellite_nextgen"],
            "가중치": [0.8, 0.2],
        }
    ).set_index("ticker")
    ips_config = {
        "target_allocation": {
            "core": {"target": 0.8},
            "satellite": {"target": 0.2},
        }
    }

    unit_set = build_evaluation_units(metrics, ips_config, _period(), "SPY")

    assert {unit.name for unit in unit_set.layer_units} >= {"core", "satellite"}
    asset = next(unit for unit in unit_set.asset_units if unit.name == "UFO")
    assert asset.parent_layer == "satellite"
    assert asset.target_weight == pytest.approx(0.2)


def test_build_evaluation_units_applies_layer_benchmarks_to_layers_and_assets():
    metrics = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO", "QLD"],
            "layer": ["core", "satellite", "experiment"],
            "category": ["core_market", "satellite_nextgen", "experiment_leverage"],
            "가중치": [0.7, 0.25, 0.05],
        }
    ).set_index("ticker")
    layer_benchmarks = {
        "core": "SPY:80,QQQ:20",
        "satellite": "QQQ",
        "experiment": "CASH",
    }

    unit_set = build_evaluation_units(
        metrics,
        None,
        _period(),
        "",
        layer_benchmarks=layer_benchmarks,
    )

    layer_units = {unit.name: unit for unit in unit_set.layer_units}
    asset_units = {unit.name: unit for unit in unit_set.asset_units}
    assert layer_units["core"].benchmark == "SPY:80,QQQ:20"
    assert layer_units["satellite"].benchmark == "QQQ"
    assert layer_units["experiment"].benchmark == "CASH"
    assert asset_units["VOO"].benchmark == "SPY:80,QQQ:20"
    assert asset_units["UFO"].benchmark == "QQQ"
    assert asset_units["QLD"].benchmark == "CASH"


def test_build_evaluation_units_defaults_satellite_and_experiment_to_qqq():
    metrics = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO", "QLD"],
            "layer": ["core", "satellite", "experiment"],
            "category": ["core_market", "satellite_nextgen", "experiment_leverage"],
            "가중치": [0.7, 0.25, 0.05],
        }
    ).set_index("ticker")

    unit_set = build_evaluation_units(
        metrics,
        None,
        _period(),
        "",
        layer_benchmarks={},
    )

    layer_units = {unit.name: unit for unit in unit_set.layer_units}
    asset_units = {unit.name: unit for unit in unit_set.asset_units}
    assert layer_units["core"].benchmark == "SPY:80,QQQ:20"
    assert layer_units["satellite"].benchmark == "QQQ"
    assert layer_units["experiment"].benchmark == "QQQ"
    assert asset_units["VOO"].benchmark == "SPY:80,QQQ:20"
    assert asset_units["UFO"].benchmark == "QQQ"
    assert asset_units["QLD"].benchmark == "QQQ"
