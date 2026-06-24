"""Efficiency metric helpers for Evaluation Framework v2."""

from __future__ import annotations

import math


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
