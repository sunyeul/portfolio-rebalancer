# Technical Evidence Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standard Ichimoku trend confirmation and Bollinger extension context to risk-adjusted instrument target reviews without producing trading signals.

**Architecture:** A new pure `services/technical_evidence.py` module calculates indicator facts from validated adjusted candles. `services/dynamic_allocation.py` remains the only judgment owner: severe risk stays first, moving-trend and Ichimoku must agree for direction, and Bollinger extension changes confidence/reasons only. Existing target-range, cooldown, persistence, and activation behavior remains unchanged.

**Tech Stack:** Python 3.12 standard library (`math`, `statistics`), normalized Toss daily candles, pytest, Ruff.

---

### Task 1: Pure Ichimoku and Bollinger evidence

**Files:**
- Create: `services/technical_evidence.py`
- Create: `tests/test_technical_evidence.py`

- [ ] **Step 1: Write failing calculator tests**

Create `tests/test_technical_evidence.py` with chronological OHLC fixtures:

```python
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
    assert result["ichimoku"]["price_position"] == "above"
    assert result["ichimoku"]["line_alignment"] == "positive"
    assert result["bollinger"]["middle"] == pytest.approx(middle)
    assert result["bollinger"]["upper"] == pytest.approx(middle + 2 * deviation)
    assert result["bollinger"]["lower"] == pytest.approx(middle - 2 * deviation)


def test_negative_ichimoku_direction_is_exposed():
    result = build_technical_evidence(_candles([200.0 - index for index in range(78)]))
    assert result["ichimoku"]["direction"] == -1
    assert result["ichimoku"]["price_position"] == "below"
    assert result["ichimoku"]["line_alignment"] == "negative"


def test_flat_bollinger_band_has_stable_percent_b():
    result = build_technical_evidence(_candles([100.0] * 78))
    assert result["bollinger"]["bandwidth"] == 0.0
    assert result["bollinger"]["percent_b"] == 0.5
    assert result["bollinger"]["extension"] == "inside"


@pytest.mark.parametrize(
    ("candles", "reason"),
    [
        (_candles([100.0] * 77), "technical_history_insufficient"),
        ([{"high_price": 90.0, "low_price": 100.0, "close_price": 95.0}] * 78, "technical_data_invalid"),
    ],
)
def test_invalid_or_short_input_fails_closed(candles, reason):
    result = build_technical_evidence(candles)
    assert result == {
        "state": "unavailable",
        "reason": reason,
        "history_points": len(candles),
    }
```

Add one fixture whose final close lies above the upper band and one below the
lower band; assert `extension` is `above` and `below` respectively.

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
rtk uv run pytest tests/test_technical_evidence.py -v
```

Expected: import failure because `services.technical_evidence` does not exist.

- [ ] **Step 3: Implement the calculator with fixed standard parameters**

Create `services/technical_evidence.py`:

```python
"""Pure technical facts from validated adjusted Toss candles."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


def _unavailable(reason: str, history_points: int) -> dict[str, Any]:
    return {
        "state": "unavailable",
        "reason": reason,
        "history_points": history_points,
    }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _midpoint(highs: Sequence[float], lows: Sequence[float]) -> float:
    return (max(highs) + min(lows)) / 2.0


def build_technical_evidence(
    candles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return standard Ichimoku and Bollinger facts without a policy judgment."""
    if len(candles) < 78:
        return _unavailable("technical_history_insufficient", len(candles))
    highs = [_number(item.get("high_price")) for item in candles]
    lows = [_number(item.get("low_price")) for item in candles]
    closes = [_number(item.get("close_price")) for item in candles]
    if any(value is None for value in highs + lows + closes):
        return _unavailable("technical_data_invalid", len(candles))
    high_values = [float(value) for value in highs if value is not None]
    low_values = [float(value) for value in lows if value is not None]
    close_values = [float(value) for value in closes if value is not None]
    if any(
        low > close or close > high or low > high
        for low, close, high in zip(low_values, close_values, high_values)
    ):
        return _unavailable("technical_data_invalid", len(candles))

    conversion = _midpoint(high_values[-9:], low_values[-9:])
    base = _midpoint(high_values[-26:], low_values[-26:])
    displaced_conversion = _midpoint(high_values[-35:-26], low_values[-35:-26])
    displaced_base = _midpoint(high_values[-52:-26], low_values[-52:-26])
    span_a = (displaced_conversion + displaced_base) / 2.0
    span_b = _midpoint(high_values[-78:-26], low_values[-78:-26])
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    latest = close_values[-1]
    price_position = (
        "above" if latest > cloud_top else "below" if latest < cloud_bottom else "inside"
    )
    line_alignment = (
        "positive" if conversion > base else "negative" if conversion < base else "mixed"
    )
    direction = (
        1
        if price_position == "above" and line_alignment == "positive"
        else -1
        if price_position == "below" and line_alignment == "negative"
        else 0
    )

    window = close_values[-20:]
    middle = mean(window)
    deviation = pstdev(window)
    upper = middle + 2.0 * deviation
    lower = middle - 2.0 * deviation
    width = upper - lower
    percent_b = (latest - lower) / width if width else 0.5
    extension = "above" if percent_b > 1 else "below" if percent_b < 0 else "inside"
    return {
        "state": "complete",
        "reason": "technical_evidence_complete",
        "history_points": len(candles),
        "ichimoku": {
            "conversion": conversion,
            "base": base,
            "span_a": span_a,
            "span_b": span_b,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "price_position": price_position,
            "line_alignment": line_alignment,
            "direction": direction,
        },
        "bollinger": {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": width / middle,
            "percent_b": percent_b,
            "extension": extension,
        },
    }
```

- [ ] **Step 4: Run calculator tests**

```bash
rtk uv run pytest tests/test_technical_evidence.py -v
```

Expected: all calculator tests pass.

- [ ] **Step 5: Commit the pure calculator**

```bash
rtk git add services/technical_evidence.py tests/test_technical_evidence.py
rtk git commit -m "feat: add technical evidence calculator"
```

### Task 2: Require Ichimoku confirmation in instrument judgment

**Files:**
- Modify: `services/dynamic_allocation.py:298-620`
- Modify: `tests/test_dynamic_allocation.py:159-430`

- [ ] **Step 1: Add failing integration tests and complete OHLC fixtures**

Update `_candles` so every synthetic instrument row contains standard OHLC:

```python
{
    "candle_at": timestamp,
    "high_price": value + 1.0,
    "low_price": max(0.01, value - 1.0),
    "close_price": value,
    "adjusted": adjusted,
    "adjusted_supported": adjusted,
}
```

Add a helper returning a complete overridden technical result and use
`monkeypatch` to verify composition:

```python
def _technical(direction, extension="inside"):
    return {
        "state": "complete",
        "reason": "technical_evidence_complete",
        "history_points": 220,
        "ichimoku": {"direction": direction},
        "bollinger": {"extension": extension},
    }


def test_moving_trend_and_ichimoku_must_agree(monkeypatch):
    policy = build_neutral_policy(_active_policy())
    series = _instrument_series(policy, [100 + index * 0.1 for index in range(220)])
    monkeypatch.setattr(
        "services.dynamic_allocation.build_technical_evidence",
        lambda _: _technical(-1),
    )
    result = build_instrument_target_reviews(
        series,
        active_policy=policy,
        regime_policy=scale_policy_to_regime(policy, "neutral"),
        now=NOW,
    )
    voo = _review_by_identity(result, "US/VOO")
    assert voo["signal"] == "neutral"
    assert "technical_trend_conflict" in voo["reasons"]


def test_bollinger_extension_changes_confidence_not_direction(monkeypatch):
    policy = build_neutral_policy(_active_policy())
    series = _instrument_series(policy, [100 + index * 0.1 for index in range(220)])
    monkeypatch.setattr(
        "services.dynamic_allocation.build_technical_evidence",
        lambda _: _technical(1, "above"),
    )
    result = build_instrument_target_reviews(
        series,
        active_policy=policy,
        regime_policy=scale_policy_to_regime(policy, "neutral"),
        now=NOW,
    )
    voo = _review_by_identity(result, "US/VOO")
    assert voo["signal"] == "supportive"
    assert voo["confidence"] == "caution"
    assert "bollinger_extension_above" in voo["reasons"]
```

Add one test where `build_technical_evidence` returns unavailable and assert the
overall review state is incomplete with no proposed policy. Keep the existing
severe-volatility test as the precedence regression.

- [ ] **Step 2: Run integration tests and verify they fail**

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -k "ichimoku or bollinger or technical" -v
```

Expected: failures because technical evidence is not composed into judgment.

- [ ] **Step 3: Integrate technical evidence without changing range rules**

Import the calculator:

```python
from services.technical_evidence import build_technical_evidence
```

Change `_instrument_signal` to accept technical evidence:

```python
def _instrument_signal(
    evidence: Mapping[str, Any], technical: Mapping[str, Any]
) -> str:
    if evidence["severe_risk"]:
        return "severe"
    moving = int(evidence["trend_direction"])
    ichimoku = int(technical["ichimoku"]["direction"])
    if moving == ichimoku == 1:
        return "supportive"
    if moving == ichimoku == -1:
        return "adverse"
    return "neutral"
```

After close-series evidence is valid, calculate technical evidence. If it is
unavailable, convert the instrument evidence to an invalid machine-readable
result with the technical reason and verification task. For valid rows:

```python
technical = build_technical_evidence(candles)
signal = _instrument_signal(evidence, technical)
extension = technical["bollinger"]["extension"]
reasons = [f"instrument_signal_{signal}"]
if int(evidence["trend_direction"]) != int(technical["ichimoku"]["direction"]):
    reasons.append("technical_trend_conflict")
if extension != "inside":
    reasons.append(f"bollinger_extension_{extension}")
confidence = "caution" if extension != "inside" else "supported"
```

Expose `technical_evidence` directly on each review and inside the source
evidence. Preserve `unavailable` confidence for invalid data, severe precedence,
range mapping, normalization, cooldown, and policy validation.

- [ ] **Step 4: Run dynamic-allocation tests**

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -v
```

Expected: all tests pass, including severe precedence and no-order-field scans.

- [ ] **Step 5: Commit judgment composition**

```bash
rtk git add services/dynamic_allocation.py tests/test_dynamic_allocation.py
rtk git commit -m "feat: confirm target ranges with technical evidence"
```

### Task 3: Full verification and adversarial review

**Files:**
- Test: `tests/test_technical_evidence.py`
- Test: `tests/test_dynamic_allocation.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_api_contract.py`
- Test: `tests/test_risk_evidence.py`
- Test: `tests/research/test_qlib_replay.py`

- [ ] **Step 1: Run formatter and static checks**

```bash
rtk uv run ruff format --check services/technical_evidence.py services/dynamic_allocation.py tests/test_technical_evidence.py tests/test_dynamic_allocation.py
rtk uv run ruff check services/technical_evidence.py services/dynamic_allocation.py tests/test_technical_evidence.py tests/test_dynamic_allocation.py
```

Expected: all checks pass.

- [ ] **Step 2: Run focused and related regressions**

```bash
rtk uv run pytest tests/test_technical_evidence.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py tests/test_risk_evidence.py tests/research/test_qlib_replay.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the complete repository suite**

```bash
rtk uv run pytest -q
```

Expected: all supported tests pass; optional integration tests may remain
skipped according to existing markers.

- [ ] **Step 4: Review serialized guardrails and the final diff**

Confirm tests reject `buy`, `sell`, `execute`, `quantity`, and order-timing
fields; technical conflicts become neutral; Bollinger extension never changes
direction; unavailable technical evidence prevents a policy candidate; and no
active-policy or snapshot file changed.

```bash
rtk git diff --check main...HEAD
rtk git status --short
```

Do not stage the user's unrelated `.codex`, `AGENTS.md`, ACE-hook, or ACE-test
changes.
