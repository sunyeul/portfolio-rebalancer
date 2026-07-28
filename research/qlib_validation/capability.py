from importlib import metadata
from typing import Any

from research.qlib_validation.contracts import SourceSnapshot


def static_roundtrip(snapshot: SourceSnapshot) -> dict[str, Any]:
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from qlib.data.dataset.loader import StaticDataLoader

    records = [
        {
            "datetime": item.session_date.isoformat(),
            "instrument": item.key.replace("/", "_"),
            "open": item.open_price,
            "high": item.high_price,
            "low": item.low_price,
            "close": item.close_price,
            "volume": item.volume,
            "factor": item.factor,
        }
        for item in snapshot.candles
    ]
    frame = pd.DataFrame.from_records(records)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    expected = frame.set_index(["datetime", "instrument"]).sort_index()
    loaded = StaticDataLoader(expected).load().sort_index()
    assert_frame_equal(loaded, expected, check_dtype=False, check_names=True)
    return {"rows": len(expected), "matched": True}


def assess_capability(snapshot: SourceSnapshot) -> dict[str, Any]:
    reasons: list[str] = []
    if not snapshot.candles or any(item.factor is None for item in snapshot.candles):
        reasons.append("factor_unavailable")
    if {item.market_country for item in snapshot.benchmark_specs} != {"KR", "US"}:
        reasons.append("required_markets_missing")
    if any(item.available_at.tzinfo is None for item in snapshot.candles):
        reasons.append("utc_availability_missing")
    matched = False
    if not reasons:
        matched = static_roundtrip(snapshot)["matched"]
        if not matched:
            reasons.append("static_loader_mismatch")
    return {
        "pyqlib": metadata.version("pyqlib"),
        "data_adapter_suitable": not reasons and matched,
        "backtest_engine_suitable": False,
        "backtest_reason": "stage2_not_evaluated",
        "reasons": reasons,
    }
