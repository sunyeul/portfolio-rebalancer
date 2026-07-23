from datetime import datetime, timedelta, timezone

import pytest

from services.market_context import evaluate_market_context


def _candles(values):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "candle_at": (start + timedelta(days=index)).isoformat(),
            "close_price": value,
        }
        for index, value in enumerate(values)
    ]


def test_market_context_fails_closed_before_long_history():
    result = evaluate_market_context(_candles([100.0] * 40))

    assert result["status"] == "Watch"
    assert result["candidate_state"] == "observe"
    assert result["proposed_target"] is None


def test_multiple_confirming_signals_create_candidate_without_activation():
    values = [100.0] * 170 + [90.0] * 10 + [70.0] * 20
    result = evaluate_market_context(
        _candles(values),
        current_target=0.15,
        now=datetime(2025, 7, 20, tzinfo=timezone.utc),
    )

    assert result["status"] == "Review"
    assert result["candidate_state"] == "candidate"
    assert result["proposed_target"] == pytest.approx(0.20)
    assert result["confirmed_signal_count"] >= 2


def test_cooling_period_blocks_repeated_candidate():
    values = [100.0] * 170 + [90.0] * 10 + [70.0] * 20
    result = evaluate_market_context(
        _candles(values),
        current_target=0.15,
        last_change_at="2025-07-10T00:00:00+00:00",
        now=datetime(2025, 7, 20, tzinfo=timezone.utc),
    )

    assert result["cooling"] is True
    assert result["candidate_state"] == "observe"


def test_stale_history_fails_closed_without_candidate():
    values = [100.0] * 170 + [90.0] * 10 + [70.0] * 20
    result = evaluate_market_context(
        _candles(values),
        now=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert result["status"] == "Watch"
    assert result["reason"] == "market_data_stale"
    assert result["candidate_state"] == "observe"
    assert result["proposed_target"] is None


def test_large_history_gap_fails_closed_without_candidate():
    candles = _candles([100.0] * 200)
    candles[-1]["candle_at"] = "2026-07-23T00:00:00+00:00"
    result = evaluate_market_context(
        candles,
        now=datetime(2026, 7, 23, 1, tzinfo=timezone.utc),
    )

    assert result["status"] == "Watch"
    assert result["reason"] == "market_history_gap"
    assert result["candidate_state"] == "observe"


def test_duplicate_timestamps_fail_closed_without_candidate():
    candles = _candles([100.0] * 200)
    candles[-1]["candle_at"] = candles[-2]["candle_at"]
    result = evaluate_market_context(
        candles,
        now=datetime(2025, 7, 20, tzinfo=timezone.utc),
    )

    assert result["reason"] == "market_data_invalid"
    assert result["duplicate_timestamps"] is True
