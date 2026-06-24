from utils.efficiency_metrics import cagr_mdd_ratio, return_mdd_ratio


def test_return_mdd_ratio():
    assert return_mdd_ratio(0.10, -0.20) == 0.5
    assert return_mdd_ratio(0.10, 0.0) is None


def test_cagr_mdd_ratio():
    assert cagr_mdd_ratio(0.12, -0.30) == 0.4
    assert cagr_mdd_ratio(None, -0.30) is None
