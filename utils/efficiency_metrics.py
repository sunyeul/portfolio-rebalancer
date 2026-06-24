"""Efficiency metric helpers for Evaluation Framework v2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.metrics import sharpe_ratio


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def return_mdd_ratio(period_return: float | None, mdd: float | None) -> float | None:
    """Return period return divided by absolute MDD."""
    if not _finite(period_return) or not _finite(mdd) or float(mdd) == 0:
        return None
    return float(period_return) / abs(float(mdd))


def cagr_mdd_ratio(cagr: float | None, mdd: float | None) -> float | None:
    """Return CAGR divided by absolute MDD."""
    if not _finite(cagr) or not _finite(mdd) or float(mdd) == 0:
        return None
    return float(cagr) / abs(float(mdd))


def sharpe(cagr: float | None, volatility: float | None, rf: float = 0.0) -> float | None:
    """Return Sharpe ratio, preserving None for unavailable inputs."""
    if not _finite(cagr) or not _finite(volatility):
        return None
    value = float(sharpe_ratio(float(cagr), float(volatility), rf))
    return value if math.isfinite(value) else None


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float | None:
    """Return annualized Sortino ratio from daily returns."""
    clean = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    downside = clean[clean < 0]
    if downside.empty:
        return None
    downside_dev = float(downside.std() * np.sqrt(252))
    if not math.isfinite(downside_dev) or downside_dev == 0:
        return None
    annual_return = float(clean.mean() * 252)
    return (annual_return - rf) / downside_dev
