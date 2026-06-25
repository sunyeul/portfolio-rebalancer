"""Evaluation period parsing for Evaluation Framework v2."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from core.evaluation import EvaluationPeriod


VALID_PERIOD_LABELS = {"1M", "3M", "6M", "YTD", "1Y", "Max"}


class EvaluationPeriodError(ValueError):
    """Raised when a v2 evaluation period cannot be resolved."""


def _coerce_date(value: str | date | None, field_name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise EvaluationPeriodError(f"{field_name} must be YYYY-MM-DD") from exc


def resolve_evaluation_period(
    *,
    period: str = "3M",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    as_of_date: str | date | None = None,
    today: date | None = None,
) -> EvaluationPeriod:
    """Resolve preset or custom period into an EvaluationPeriod."""
    resolved_end = _coerce_date(end_date, "end_date")
    resolved_start = _coerce_date(start_date, "start_date")
    resolved_as_of = _coerce_date(as_of_date, "as_of_date")
    if (resolved_start is None) != (resolved_end is None):
        raise EvaluationPeriodError("start_date and end_date must be provided together")
    if resolved_start is not None and resolved_end is not None:
        return EvaluationPeriod(
            label="custom",
            start_date=resolved_start,
            end_date=resolved_end,
        )

    end = resolved_as_of or today or date.today()
    label = str(period).strip()
    normalized = label.upper()
    if normalized == "MAX":
        normalized = "Max"
    if normalized not in VALID_PERIOD_LABELS:
        raise EvaluationPeriodError("period must be one of 1M, 3M, 6M, YTD, 1Y, Max")

    if normalized == "YTD":
        start = date(end.year, 1, 1)
    elif normalized == "Max":
        start = date(end.year - 15, end.month, end.day)
    elif normalized == "1Y":
        start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
    else:
        months = int(normalized[:-1])
        start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()

    return EvaluationPeriod(label=normalized, start_date=start, end_date=end)


def analysis_period_value(evaluation_period: EvaluationPeriod) -> int | str | EvaluationPeriod:
    """Return a value acceptable by run_analysis."""
    if evaluation_period.label == "1M":
        return 1
    if evaluation_period.label == "3M":
        return 3
    if evaluation_period.label == "6M":
        return 6
    if evaluation_period.label == "1Y":
        return 12
    if evaluation_period.label == "YTD":
        return "YTD"
    if evaluation_period.label == "Max":
        return "Max"
    return evaluation_period
