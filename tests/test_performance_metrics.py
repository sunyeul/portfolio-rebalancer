import pandas as pd
import pytest

from utils.performance_metrics import benchmark_excess_return, cagr, period_return


def test_period_return_from_price_series():
    assert period_return(pd.Series([100.0, 110.0])) == pytest.approx(0.1)


def test_period_return_handles_short_or_zero_series():
    assert period_return(pd.Series([100.0])) is None
    assert period_return(pd.Series([0.0, 100.0])) is None


def test_cagr_returns_none_for_short_series():
    assert cagr(pd.Series([100.0])) is None


def test_cagr_from_nav_series_is_positive():
    result = cagr(pd.Series([1.0] * 252 + [1.1]))
    assert result is not None
    assert result > 0


def test_benchmark_excess_return():
    assert benchmark_excess_return(0.12, 0.05) == pytest.approx(0.07)
    assert benchmark_excess_return(None, 0.05) is None
