import pandas as pd
import pytest

from utils.efficiency_metrics import (
    cagr_mdd_ratio,
    return_mdd_ratio,
    sharpe,
    sortino_ratio,
)


def test_return_mdd_ratio():
    assert return_mdd_ratio(0.10, -0.20) == 0.5
    assert return_mdd_ratio(0.10, 0.0) is None


def test_cagr_mdd_ratio():
    assert cagr_mdd_ratio(0.12, -0.30) == 0.4
    assert cagr_mdd_ratio(None, -0.30) is None


def test_sharpe_null_safe():
    assert sharpe(None, 0.2) is None
    assert sharpe(0.1, 0.0) is None
    assert sharpe(0.1, 0.2, 0.02) == pytest.approx(0.4)


def test_sortino_ratio():
    value = sortino_ratio(pd.Series([0.02, -0.01, 0.03, -0.02]), rf=0.0)
    assert value is not None
    assert value > 0
