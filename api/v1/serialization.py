"""JSON serialization helpers for dataframe-backed service results."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


METRICS_COLUMNS = {
    "ticker": "ticker",
    "CAGR": "cagr",
    "변동성": "volatility",
    "샤프": "sharpe",
    "최대낙폭": "max_drawdown",
    "IR": "information_ratio",
    "베타": "beta",
    "알파": "alpha",
    "위험기여도": "risk_contribution",
    "수익기여도": "return_contribution",
    "가중치": "weight",
    "E": "efficiency_score",
    "DCA강도점수": "dca_intensity_score",
    "return_total": "return_total",
    "group": "group",
    "role": "role",
    "dca_enabled": "dca_enabled",
    "thesis_status": "thesis_status",
}

PROPOSAL_COLUMNS = {
    "ticker": "ticker",
    "현재%": "current_weight_pct",
    "목표%": "target_weight_pct",
    "갭%": "gap_pct",
    "E": "efficiency_score",
    "DCA강도점수": "dca_intensity_score",
    "RC_Over%": "rc_over_pct",
    "RC_Target%": "rc_target_pct",
    "return_total%": "return_total_pct",
    "group": "group",
    "role": "role",
    "dca_enabled": "dca_enabled",
    "thesis_status": "thesis_status",
    "risk_over": "risk_over",
    "efficiency_good": "efficiency_good",
    "히스테리시스제외": "within_hysteresis",
    "최소거래미만": "below_min_trade",
    "실행": "should_execute",
    "조정갭%": "adjusted_gap_pct",
}

RC_VIOLATION_COLUMNS = {
    "ticker": "ticker",
    "현재RC%": "current_rc_pct",
    "RC상한%": "rc_cap_pct",
    "상태": "status",
}

GROUP_SUMMARY_COLUMNS = {
    "group_type": "group_type",
    "group": "group",
    "weight": "weight",
    "risk_contribution": "risk_contribution",
    "avg_efficiency": "avg_efficiency",
    "avg_dca_score": "avg_dca_score",
}


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe Python values."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dataframe_records(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    include_index: bool = False,
    index_name: str = "ticker",
) -> list[dict[str, Any]]:
    """Return dataframe rows with stable API keys and JSON-safe values."""
    if df is None or df.empty:
        return []

    result = df.copy()
    if include_index or result.index.name:
        result = result.reset_index()
        if result.columns[0] == "index":
            result = result.rename(columns={"index": index_name})

    if column_map:
        result = result.rename(columns=column_map)

    records = result.to_dict(orient="records")
    return [{key: json_safe(value) for key, value in row.items()} for row in records]


def safe_mapping(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-safe copy of a plain metrics mapping."""
    if mapping is None:
        return None
    return {key: json_safe(value) for key, value in mapping.items()}
