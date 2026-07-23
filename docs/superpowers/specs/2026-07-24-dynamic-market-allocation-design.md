# Dynamic Market Allocation Review Design

## Goal

Replace the single-KOSPI cash-only review with a deterministic three-regime review that can propose coordinated changes to cash and the Core/Satellite/Experiment targets. The review optimizes the portfolio's risk budget for the observed market regime, while remaining an IPS inspection signal: it never creates orders, never activates a policy automatically, and never converts a target gap into a trade instruction.

The approved neutral allocation is:

- gross-account cash: 5%, with a hard 3% to 10% range;
- invested-account Core: 60%, with a hard 50% to 70% range;
- invested-account Satellite: 38%, with a hard 28% to 48% range;
- invested-account Experiment: 2%, with a hard 0% to 4% range.

GLD is a Core holding. Its neutral invested-account target is 10%.

## Product boundary

Dynamic target review and rebalancing inspection are separate stages:

1. Market evidence proposes the portfolio's cash and layer targets.
2. A human reviews and activates a complete policy version.
3. A later portfolio inspection measures actual weights against that active policy.

The market review can emit only `OK`, `Watch`, or `Review`. `Action` remains reserved for the downstream inspection engine's exceptional-intervention review. No market result contains `buy`, `sell`, order quantity, order price, or executable instructions.

## Regime presets

All cash percentages use gross account value. All layer percentages use invested assets after cash, so layer targets always sum to 100%.

| Regime | Cash | Core | Satellite | Experiment | Interpretation |
|---|---:|---:|---:|---:|---|
| `risk_on` | 3% | 52% | 44% | 4% | Broad strength permits the upper approved risk budget. |
| `neutral` | 5% | 60% | 38% | Default growth-oriented allocation. |
| `risk_off` | 10% | 70% | 29% | Preserve optionality and reduce high-variance exposure. |

These are policy targets, not permanent weights. The hard ranges remain 3–10% cash, 50–70% Core, 28–48% Satellite, and 0–4% Experiment.

## Market evidence

### Required primary benchmarks

The first implementation uses four consistently reproducible Toss-backed daily series:

| Key | Toss source | Weight |
|---|---|---:|
| `US/SPY` | adjusted stock candles | 30% |
| `US/QQQ` | adjusted stock candles | 30% |
| `KR/KOSPI` | market-indicator candles | 25% |
| `KR/KOSDAQ` | market-indicator candles | 15% |

The evaluator reads only normalized candles already stored by IPS Pilot. `market sync` obtains the required benchmarks from Toss before evaluation. There is no yfinance, generic broker, manual-price, or web fallback.

### Per-benchmark metrics

For each series, calculate:

- 60-session and 200-session simple moving-average distance;
- drawdown from the maximum close in the available 252-session lookback;
- annualized volatility from the latest 20 log returns.

Trend direction is `+1` when the latest close is above both moving averages, `-1` when it is below both, and `0` otherwise. A benchmark has severe risk when drawdown is at or below -15% or annualized volatility is at or above 30%.

### Composite classification

Compute:

- `weighted_trend`: the weighted sum of the four trend directions;
- `severe_risk_weight`: the total weight of benchmarks with severe risk.

Classify deterministically:

1. `risk_off` when `weighted_trend <= -0.50` or `severe_risk_weight >= 0.50`;
2. `risk_on` when `weighted_trend >= 0.50` and `severe_risk_weight < 0.30`;
3. `neutral` otherwise.

Portfolio breadth and instrument-level trend data may be reported as supplemental evidence when available, but they do not alter the first version's classification. This keeps optional holdings data from changing the result or masking missing primary evidence.

## Data-quality and timing gates

Every required benchmark must have:

- at least 200 valid daily observations;
- no duplicate timestamps or invalid closes;
- a latest observation no more than seven calendar days old;
- no adjacent observation gap greater than ten calendar days, allowing known extended Korean market holidays while still rejecting longer unexplained omissions.

Any failed required input produces `Watch`, `candidate_state=observe`, a machine-readable reason, and no proposed allocation. The output lists the failed benchmark and a verification task.

The allocation review cadence is monthly with a minimum 30-day cooldown measured from the active policy's creation time. During cooldown, the evaluator reports the observed regime and proposed preset, but keeps `candidate_state=observe`. This prevents repeated policy churn without hiding current evidence.

## Policy construction

### Neutral instrument targets

The approved neutral policy keeps the previous policy's relative weights where practical and applies the user's GLD decision explicitly:

- GLD receives 10% of the Core layer.
- The remaining 50% Core target is allocated proportionally across the existing Core targets (`069500`, `QQQ`, `SCHD`, `SPY`, `VOO`).
- The 38% Satellite target is allocated proportionally across the existing non-GLD Satellite targets.
- The 2% Experiment target preserves the existing SOXL:SPCX ratio of 3:2.

The policy builder uses exact normalized ratios and assigns the final rounding remainder to the last stable-sorted instrument so every layer target sums exactly under policy validation.

### Bounds and regime-specific targets

Instrument minimums and maximums are derived from their neutral within-layer share:

- Core minimums sum to 50% and maximums sum to 70%;
- Satellite minimums sum to 28% and maximums sum to 48%;
- Experiment minimums sum to 0% and maximums sum to 4%.

For a regime proposal, instrument targets are the instrument's neutral within-layer share multiplied by that regime's layer target. Minima and maxima remain unchanged. This preserves the approved composition inside each layer and avoids introducing ticker-level market timing into the regime classifier.

The complete proposed policy retains performance, risk-review, and cadence settings from the active policy. It changes only cash, layer ranges and targets, instrument layers/bounds/targets, and the explicit allocation-review configuration.

## Result contract

The composite evaluator returns one JSON object containing:

- `status`: `OK`, `Watch`, or `Review`;
- `candidate_state`: `observe` or `candidate`;
- `reason` and `verification_task`;
- `regime`, `weighted_trend`, and `severe_risk_weight`;
- per-benchmark quality, metrics, signals, and weights;
- current and proposed cash/layer targets;
- the complete validated `proposed_policy` only when all evidence passes;
- cooldown metadata and the base policy version id.

`OK` means the active targets already match the current preset. `Review` means a different valid preset is eligible for human review. `Watch` means evidence, timing, or another guardrail prevents an eligible proposal.

Policy candidates remain immutable and deduplicated by account, base-policy version, and canonical candidate JSON. Persisting a candidate never changes `ips_policy_versions`.

## CLI flow

`market sync` will:

1. collect requested or held Toss stock candles;
2. ensure `US/SPY`, `US/QQQ`, `KR/KOSPI`, and `KR/KOSDAQ` are collected once;
3. persist normalized candles idempotently;
4. evaluate the composite allocation context against the active policy;
5. persist an immutable candidate only when `candidate_state=candidate`.

`market context` will evaluate the four stored primary benchmarks. The old single-benchmark cash-only option is removed because it conflicts with the approved coordinated allocation model. The command continues to state that active policy is unchanged and human approval is required.

The existing `policy validate`, `inspection preview`, and `policy activate` commands remain the approval path. No convenience command will combine context evaluation with activation.

## Initial policy rollout

Implementation will generate and validate one recommended neutral policy document from the current active policy and latest Toss identities. It will be previewed through the inspection engine before activation. The user's instruction to proceed with the recommended plan authorizes activation of this neutral baseline after those checks pass.

Activation creates the new immutable current version. Prior versions referenced by saved evaluations are retained as audit evidence even though normal product surfaces continue to show only the active version. Candidate evaluation begins from the new policy creation time, so the initial 30-day cooldown applies.

## Error handling

- Market normalization, policy validation, storage, and CLI failures remain machine-readable on stdout.
- One failed required fetch stops composite evaluation; partial data is persisted safely but cannot create a candidate.
- A missing active policy or missing observed identity stops policy construction.
- Floating-point totals are normalized before validation rather than relaxed after validation.
- A policy-version conflict stops activation and leaves the old active policy unchanged.

## Verification

Unit tests cover:

- each regime boundary and mixed-market neutrality;
- drawdown/volatility risk-off overrides;
- insufficient, stale, duplicated, future, and gapped series;
- cooldown and unchanged-target behavior;
- exact policy sums, GLD Core assignment, and deterministic proportional scaling;
- candidate persistence and deduplication;
- multi-benchmark CLI sync/context output and no activation side effect.

Integration verification will run the focused market, policy, CLI, and inspection tests, followed by the full test suite. The generated neutral policy will be validated and previewed against the latest saved Toss snapshot before activation, then the active policy and a fresh inspection will be read back to confirm the result.
