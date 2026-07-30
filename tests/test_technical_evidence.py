from statistics import mean, pstdev

import pytest

from services.technical_evidence import build_technical_evidence


def _candles(closes):
    return [
        {
            "high_price": close + 1.0,
            "low_price": close - 1.0,
            "close_price": close,
        }
        for close in closes
    ]


def test_positive_ichimoku_and_bollinger_values_are_standard():
    closes = [100.0 + index for index in range(78)]
    result = build_technical_evidence(_candles(closes))

    latest = closes[-20:]
    middle = mean(latest)
    deviation = pstdev(latest)
    assert result["state"] == "complete"
    assert result["ichimoku"]["direction"] == 1
    assert result["ichimoku"]["cloud_position"] == "above"
    assert result["ichimoku"]["line_alignment"] == "positive"
    assert result["bollinger"]["middle"] == pytest.approx(middle)
    assert result["bollinger"]["upper"] == pytest.approx(middle + 2 * deviation)
    assert result["bollinger"]["lower"] == pytest.approx(middle - 2 * deviation)


def test_negative_ichimoku_direction_is_exposed():
    result = build_technical_evidence(_candles([200.0 - index for index in range(78)]))

    assert result["ichimoku"]["direction"] == -1
    assert result["ichimoku"]["cloud_position"] == "below"
    assert result["ichimoku"]["line_alignment"] == "negative"


def test_flat_bollinger_band_has_stable_percent_b():
    result = build_technical_evidence(_candles([100.0] * 78))

    assert result["bollinger"]["bandwidth"] == 0.0
    assert result["bollinger"]["percent_b"] == 0.5
    assert result["bollinger"]["extension"] == "inside"


@pytest.mark.parametrize(
    ("closes", "extension"),
    [
        ([100.0] * 77 + [200.0], "above"),
        ([100.0] * 77 + [50.0], "below"),
    ],
)
def test_bollinger_extension_is_descriptive(closes, extension):
    result = build_technical_evidence(_candles(closes))

    assert result["bollinger"]["extension"] == extension


@pytest.mark.parametrize(
    ("candles", "reason"),
    [
        (_candles([100.0] * 77), "technical_history_insufficient"),
        (
            [{"high_price": 90.0, "low_price": 100.0, "close_price": 95.0}] * 78,
            "technical_data_invalid",
        ),
    ],
)
def test_invalid_or_short_input_fails_closed(candles, reason):
    result = build_technical_evidence(candles)

    assert result == {
        "state": "unavailable",
        "reason": reason,
        "history_points": len(candles),
    }
