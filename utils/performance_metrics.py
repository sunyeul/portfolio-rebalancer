"""Performance metric helpers for Evaluation Framework v2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.metrics import cagr_from_series


def _clean_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def period_return(price_or_nav: pd.Series) -> float | None:
    """Return total period return from a price or NAV series."""
    clean = _clean_series(price_or_nav)
    if len(clean) < 2:
        return None
    first = float(clean.iloc[0])
    last = float(clean.iloc[-1])
    if not math.isfinite(first) or first <= 0 or not math.isfinite(last):
        return None
    return last / first - 1.0


def cagr(price_or_nav: pd.Series) -> float | None:
    """Return CAGR from a price or NAV series, or None when unavailable."""
    clean = _clean_series(price_or_nav)
    if len(clean) < 2:
        return None
    value = float(cagr_from_series(clean))
    if not math.isfinite(value):
        return None
    return value


def benchmark_excess_return(
    unit_return: float | None,
    benchmark_return: float | None,
) -> float | None:
    """Return active performance versus benchmark."""
    if unit_return is None or benchmark_return is None:
        return None
    if np.isnan(unit_return) or np.isnan(benchmark_return):
        return None
    return float(unit_return - benchmark_return)
