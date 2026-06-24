"""Risk metric helpers for Evaluation Framework v2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.metrics import daily_to_annual_vol, max_drawdown, risk_contributions


def maximum_drawdown(price_or_nav: pd.Series) -> float | None:
    """Return maximum drawdown as a negative decimal."""
    clean = pd.to_numeric(price_or_nav, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    value = float(max_drawdown(clean))
    if not math.isfinite(value):
        return None
    return value


def annualized_volatility(daily_returns: pd.Series) -> float | None:
    """Return annualized volatility from daily returns."""
    clean = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    value = float(daily_to_annual_vol(clean))
    if not math.isfinite(value):
        return None
    return value


def concentration(weights: pd.Series) -> float | None:
    """Return Herfindahl concentration for a set of weights."""
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(clean.sum())
    if total <= 0:
        return None
    normalized = clean / total
    value = float((normalized**2).sum())
    if not math.isfinite(value):
        return None
    return value


def risk_contribution(weights: pd.Series, cov_matrix: pd.DataFrame | None) -> pd.Series:
    """Return risk contributions when covariance is available."""
    if cov_matrix is None:
        return pd.Series(np.nan, index=weights.index)
    return risk_contributions(weights, cov_matrix)
