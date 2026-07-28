from dataclasses import replace
from importlib.util import find_spec

import pytest

from research.qlib_validation.capability import assess_capability, static_roundtrip
from research.qlib_validation.contracts import SourceSnapshot


pytestmark = pytest.mark.skipif(
    find_spec("qlib") is None,
    reason="Qlib research environment only",
)


def test_real_toss_shape_fails_closed_when_factor_is_unavailable(snapshot_factory):
    result = assess_capability(snapshot_factory())
    assert result["data_adapter_suitable"] is False
    assert "factor_unavailable" in result["reasons"]
    assert result["backtest_engine_suitable"] is False


def test_static_loader_round_trip_preserves_verified_fixture(snapshot_factory):
    snapshot = snapshot_factory()
    candles = tuple(replace(item, factor=1.0) for item in snapshot.candles)
    verified = SourceSnapshot(
        snapshot.policy_record, snapshot.benchmark_specs, snapshot.policy_specs, candles
    )
    assert static_roundtrip(verified) == {"rows": len(candles), "matched": True}
