import pandas as pd

from utils.risk_metrics import annualized_volatility, concentration, maximum_drawdown


def test_maximum_drawdown():
    assert maximum_drawdown(pd.Series([100.0, 120.0, 90.0, 110.0])) == -0.25


def test_maximum_drawdown_handles_short_series():
    assert maximum_drawdown(pd.Series([100.0])) is None


def test_annualized_volatility():
    value = annualized_volatility(pd.Series([0.01, -0.01, 0.02, -0.02]))
    assert value is not None
    assert value > 0


def test_concentration_uses_hhi():
    assert concentration(pd.Series([0.5, 0.5])) == 0.5
    assert concentration(pd.Series([0.0, 0.0])) is None
