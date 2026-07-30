# Technical evidence confirmation

Status: approved recommended increment (2026-07-31)

## Context

Risk-adjusted instrument target ranges currently use role-specific drawdown,
20-session realized volatility, and 60/200-session close trends. Adding more
price-derived indicators as independent votes would double-count correlated
evidence and create unjustified confidence. This increment gives Ichimoku and
Bollinger Bands separate, narrow roles.

## Decision

- Ichimoku confirms trend direction. An instrument is `supportive` or `adverse`
  only when the existing 60/200-session trend and Ichimoku direction agree.
- Bollinger Bands describe extension and bandwidth. They affect evidence
  reasons and confidence, but never create a directional signal by themselves.
- Existing severe-risk logic remains first: a role-specific drawdown or the
  configured realized-volatility breach still produces `severe` regardless of
  trend indicators.
- Conflicting or mixed trend evidence produces `neutral` rather than a forced
  direction.

## Goals

- Add standard, reproducible Ichimoku and Bollinger evidence from normalized
  Toss adjusted daily candles.
- Reduce false certainty by requiring independent trend confirmation.
- Expose enough raw values and stable labels to explain a target range.
- Preserve policy-candidate, cooldown, and human-approval contracts.

## Non-goals

- No candlestick-pattern engine, RSI, MACD, stochastic oscillator, volume
  profile, correlation optimizer, or third-party TA dependency.
- No indicator voting, learned weights, backtest-selected parameters, or
  instrument-specific tuning.
- No order timing, entry/exit label, price target, stop level, or `Action`
  escalation from technical evidence.
- No change to active policy or saved account state.

## Component boundary

Create `services/technical_evidence.py` as a pure, dependency-free calculator.
It receives adjusted daily candle mappings and returns technical facts only. It
does not know policy roles, statuses, target ranges, holdings, or orders.

`services/dynamic_allocation.py` remains the judgment owner. It combines the
technical facts with existing risk evidence and maps the resulting signal to
the already approved target-range rules.

## Input contract

Technical evidence requires at least 78 chronological adjusted daily candles:

- finite positive `high_price`, `low_price`, and `close_price`;
- `low_price <= close_price <= high_price`;
- unique, already validated timestamps; and
- no unadjusted or adjustment-unsupported stock rows.

The surrounding market-evidence path still enforces the stricter policy minimum
history, freshness, and gap rules. Missing or invalid high/low data makes the
instrument target review incomplete; it does not fall back to close-only
Ichimoku.

## Ichimoku calculation

Use standard fixed parameters: conversion 9, base 26, span B 52, displacement
26.

- current conversion line: midpoint of the highest high and lowest low over the
  latest 9 sessions;
- current base line: the same midpoint over 26 sessions;
- current cloud span A: conversion/base midpoint calculated at the 26-session
  displaced observation;
- current cloud span B: 52-session high/low midpoint calculated at that same
  displaced observation;
- cloud top/bottom: maximum/minimum of spans A and B.

Ichimoku direction is:

- `1` when the latest close is above the cloud and conversion is above base;
- `-1` when the latest close is below the cloud and conversion is below base;
- `0` otherwise.

The result also exposes the lines, cloud bounds, price position
(`above/inside/below`), and line alignment (`positive/mixed/negative`).

## Bollinger calculation

Use the latest 20 closes and two population standard deviations:

- middle: 20-session arithmetic mean;
- upper/lower: middle plus/minus `2 * population standard deviation`;
- bandwidth: `(upper - lower) / middle`;
- percent B: `(latest - lower) / (upper - lower)`; and
- extension: `above` when percent B is greater than 1, `below` when less than 0,
  otherwise `inside`.

For a zero-width band, percent B is `0.5` and extension is `inside`.

Bollinger extension adds `bollinger_extension_above` or
`bollinger_extension_below` to evidence reasons and changes confidence from
`supported` to `caution`. It does not alter `supportive`, `neutral`, `adverse`,
or `severe` by itself.

## Judgment composition

Signal precedence is deterministic:

1. `severe` when existing severe-risk evidence is true;
2. `supportive` when 60/200 trend direction is `1` and Ichimoku direction is
   `1`;
3. `adverse` when both directions are `-1`;
4. `neutral` for every conflict or mixed state.

The existing target-range mapping remains unchanged. Technical evidence changes
the quality of trend confirmation, not the meaning of policy bounds.

## Result contract

Each valid instrument review adds:

```json
{
  "technical_evidence": {
    "state": "complete",
    "ichimoku": {
      "direction": 0,
      "cloud_position": "inside",
      "line_alignment": "mixed"
    },
    "bollinger": {
      "extension": "inside",
      "bandwidth": 0.0,
      "percent_b": 0.5
    }
  },
  "confidence": "supported"
}
```

The full numeric indicator values remain inside `technical_evidence`. Stable
reason codes explain trend disagreement and band extension. `signal` remains
evidence detail, not an IPS status.

## Failure behavior

- Invalid technical price fields return `technical_data_invalid`.
- Fewer than 78 usable points return `technical_history_insufficient`.
- Either failure makes the instrument evidence incomplete, so the overall
  dynamic-allocation result is `Watch`, `candidate_state=observe`, and has no
  proposed policy.
- Benchmark regime calculation remains backward compatible and does not require
  high/low fields; technical evidence applies only to policy instruments.

## Verification

Unit tests cover:

- standard Ichimoku line/cloud values and positive, negative, and mixed states;
- Bollinger middle, bands, bandwidth, percent B, extensions, and zero-width
  behavior;
- invalid OHLC and insufficient history;
- severe-risk precedence over all technical signals;
- supportive/adverse requiring moving-trend and Ichimoku agreement;
- disagreement becoming neutral;
- Bollinger extension changing confidence but not direction; and
- absence of order, quantity, execution, price-target, and timing fields.

Dynamic-allocation, CLI/API, research replay, risk-evidence, and full repository
tests remain the regression gate.

## Adversarial review

- Bollinger extension is not treated as overbought/oversold execution timing.
- Ichimoku and moving averages do not cast separate additive votes.
- A technical conflict cannot be hidden by preserving the old directional
  signal.
- Technical evidence cannot override missing/stale source data or severe risk.
- Fixed standard parameters avoid unvalidated optimization and configuration
  sprawl.
