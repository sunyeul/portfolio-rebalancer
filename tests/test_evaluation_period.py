from datetime import date

import pytest

from services.evaluation_period import EvaluationPeriodError, resolve_evaluation_period


def test_resolves_month_and_year_periods_from_today():
    today = date(2026, 6, 23)

    assert resolve_evaluation_period(period="1M", today=today).start_date == date(2026, 5, 23)
    assert resolve_evaluation_period(period="3M", today=today).start_date == date(2026, 3, 23)
    assert resolve_evaluation_period(period="6M", today=today).start_date == date(2025, 12, 23)
    assert resolve_evaluation_period(period="1Y", today=today).start_date == date(2025, 6, 23)


def test_resolves_ytd_and_max():
    today = date(2026, 6, 23)

    assert resolve_evaluation_period(period="YTD", today=today).start_date == date(2026, 1, 1)
    max_period = resolve_evaluation_period(period="Max", today=today)
    assert max_period.label == "Max"
    assert max_period.start_date == date(2011, 6, 23)


def test_custom_dates_override_period():
    period = resolve_evaluation_period(
        period="3M",
        start_date="2026-01-01",
        end_date="2026-06-30",
        today=date(2026, 6, 23),
    )

    assert period.label == "custom"
    assert period.start_date == date(2026, 1, 1)
    assert period.end_date == date(2026, 6, 30)


def test_as_of_date_sets_preset_period_end():
    period = resolve_evaluation_period(period="3M", as_of_date="2026-06-15")

    assert period.label == "3M"
    assert period.start_date == date(2026, 3, 15)
    assert period.end_date == date(2026, 6, 15)


def test_rejects_one_sided_custom_dates():
    with pytest.raises(EvaluationPeriodError):
        resolve_evaluation_period(start_date="2026-01-01")


def test_rejects_invalid_period():
    with pytest.raises(EvaluationPeriodError):
        resolve_evaluation_period(period="2W")
