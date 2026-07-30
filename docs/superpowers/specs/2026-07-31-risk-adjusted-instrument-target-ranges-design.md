# Risk-adjusted instrument target ranges

Status: approved direction, pending written-spec review (2026-07-31)

## Context

The current dynamic-allocation path selects a market regime and changes cash and
layer targets, but it preserves each instrument's legacy within-layer share.
That makes an inherited number such as the KODEX 200 target look like a current
market judgment even when the instrument's own evidence was never used.

This design keeps the active policy as an explicit anchor and adds a separate,
review-only target range based on validated instrument price evidence. It does
not turn a target gap, loss, drawdown, or range into an order instruction.

## Goals

- Show the active-policy range, regime-scaled anchor, and risk-adjusted analysis
  range separately for every policy instrument.
- Use an explainable four-signal rule based only on validated Toss market data.
- Prefer ranges over false precision while retaining a deterministic internal
  reference target when a complete policy candidate must be validated.
- Fail closed when instrument evidence is incomplete or the proposed ranges
  cannot cover the selected layer target.
- Preserve the existing `OK`, `Watch`, `Review`, and `Action` vocabulary and the
  human-approval boundary for policy activation.

## Non-goals

- No automatic policy activation, order side, quantity, price, or execution
  field.
- No target adjustment from current weight, allocation gap, cost basis,
  unrealized return, or realized profit/loss.
- No hard-coded VOO recurring-purchase plan and no assumption that Toss exposes
  recurring-purchase configuration. A request-scoped plan such as VOO USD 30
  daily belongs in a separate forward projection, not in the strategic target
  calculation.
- No correlation optimizer, risk-parity engine, forecast model, or new external
  dependency.
- No policy-role change. A severe Core signal creates a role-review flag but
  does not silently reclassify or remove the instrument.

## Inputs and source boundaries

The target review consumes:

1. the active normalized IPS policy;
2. the existing benchmark series used to select the market regime;
3. Toss daily candles for every configured policy instrument; and
4. the evaluation timestamp and active-policy change timestamp.

Account holdings, cost, orders, and executions remain normalized Toss snapshot
facts, but they do not enter this target algorithm. Explicit user constraints
remain response-scoped unless the user separately authorizes a policy change.

## Data flow

1. Validate the required benchmark series and select `risk_on`, `neutral`, or
   `risk_off` with the existing regime rules.
2. Scale cash and layer targets with the existing regime presets.
3. Validate each configured instrument's Toss candle series.
4. Classify each valid instrument as `supportive`, `neutral`, `adverse`, or
   `severe`.
5. Convert the classification into an analysis range anchored to the active
   policy and regime-scaled target.
6. If every instrument is evaluable and every layer is feasible, calculate a
   deterministic normalization reference inside the ranges and validate a
   review-only proposed policy.
7. Persist only the composite policy-candidate context. Activation remains a
   separate explicit command requiring the expected current policy version.

## Instrument evidence contract

Instrument evidence uses the same source-quality posture as existing market and
risk evidence:

- positive, finite closes with unique timestamps;
- at least the policy's minimum history points from a 252-session lookback;
- data no older than the configured maximum age;
- calendar-aware gap validation using the configured maximum gap;
- adjusted and adjustment-supported candles for stock instruments; and
- no future timestamps or malformed points.

Valid evidence exposes current drawdown, 20-session annualized realized
volatility, 60-session trend, 200-session trend, history bounds, and a source
fingerprint. A failed series exposes a stable reason code and verification task,
but no analysis range.

If any configured policy instrument is unavailable, valid instruments may still
show their evidence and provisional ranges, but the overall result is `Watch`,
`candidate_state=observe`, `reason=instrument_evidence_incomplete`, and
`proposed_policy=null`. This prevents an incomplete universe from silently
redistributing the missing instrument's weight.

## Explainable classification

The rules are evaluated in this order:

| Signal | Rule |
| --- | --- |
| `severe` | Role-specific drawdown breaches `risk_review.instrument_drawdown_review`, or realized volatility breaches `allocation_review.volatility_review` |
| `adverse` | Not severe, and both 60-session and 200-session trends are negative |
| `supportive` | Not severe, and both trends are positive |
| `neutral` | Valid evidence with mixed or flat trends |

Unrealized return is deliberately excluded because it depends on the holder's
cost basis. Current weight and allocation gap are excluded because using them
would turn the target into a moving response to the portfolio's present state.

## Target-range mapping

For each instrument, `policy_anchor` is the active policy's
`minimum/target/maximum`, and `regime_anchor` is the target after applying the
selected layer regime. The primary analysis output is mapped as follows:

| Signal | Analysis range |
| --- | --- |
| `supportive` | regime anchor to policy maximum |
| `neutral` | policy minimum to policy maximum |
| `adverse` | policy minimum to regime anchor |
| `severe` | zero to policy minimum |

A `severe` Core instrument also sets `role_review_required=true`. Zero is the
lower analytical bound, not an instruction to remove the holding. The role flag
requires a human to decide whether the Core thesis and policy floor still make
sense before any policy candidate is activated.

## Layer normalization and feasibility

The selected regime continues to own each layer's total target. Instrument
ranges must satisfy both conditions for a complete candidate:

- the sum of lower bounds does not exceed the layer target; and
- the sum of upper bounds covers the layer target.

When feasible, calculate `normalization_reference` values that remain inside
each analysis range, sum exactly to the layer target, and minimize movement from
the regime anchors. Distribution is deterministic and uses stable instrument
identity ordering when headroom ties.

`normalization_reference` exists only to validate and compare a complete policy
candidate. User-facing tables lead with the range and must not present the
reference as a uniquely correct target.

When a layer is infeasible, return `Review`, `candidate_state=observe`,
`reason=instrument_target_ranges_infeasible`, and `proposed_policy=null`, with a
verification task to review the layer target or instrument roles. Do not widen
ranges behind the user's back.

The existing cooldown still prevents repeated candidate churn. A range or
reference change is part of target-change comparison; comparison is no longer
limited to cash and layer totals.

## Result contract

Keep the current market-context fields and add an ordered
`instrument_target_reviews` array. Each item contains:

- `identity` and `layer`;
- `policy_anchor`;
- `regime_anchor`;
- `evidence_state` and source-quality details;
- `signal` and stable reason codes;
- `analysis_range` or `null`;
- `normalization_reference` or `null`;
- `role_review_required`; and
- `verification_task`.

`signal` is evidence detail, not a new IPS status. No client or CLI consumer may
derive a replacement `status`, `priority`, `queue_class`, or `suggestion` from
it. Persisted candidates retain the active policy version as their base and do
not activate themselves.

## Presentation

Tables show these columns in this order:

| Instrument | Layer | Current policy | Regime anchor | Risk-adjusted range | Evidence | Confidence |
| --- | --- | --- | --- | --- | --- | --- |

An unavailable series shows `검증 필요` instead of falling back to the legacy
target. Current holdings and request-scoped recurring-purchase projections may
appear in adjacent columns or sections, but they remain labeled separately from
the target source.

## Error handling

- Missing benchmark evidence preserves the existing market-context `Watch`
  behavior and prevents all target candidates.
- Missing instrument evidence prevents a complete instrument policy candidate
  without discarding valid evidence for other instruments.
- Invalid policy bounds or an infeasible layer fail closed with machine-readable
  reason codes and no traceback on CLI stdout.
- Candidate storage remains idempotent through the existing composite hash.
- Active policy and saved snapshots remain unchanged until a separately
  authorized activation command succeeds.

## Verification

Unit tests cover:

- all four signal classifications and precedence of `severe`;
- role-specific drawdown thresholds and the configured volatility threshold;
- the four range mappings, including severe Core role review;
- missing, stale, future, duplicate, unadjusted, and gapped candle histories;
- feasible normalization, stable tie ordering, and exact layer totals;
- infeasible ranges producing no proposed policy;
- current weight, allocation gap, and unrealized return having no effect on the
  target range;
- cooldown behavior when only instrument ranges change; and
- absence of buy, sell, execute, quantity, and price fields.

Integration and contract tests cover:

- market sync collecting every policy instrument without duplicating benchmark
  symbols;
- CLI and API returning the same ordered review contract;
- policy-candidate persistence without activation;
- backward-compatible benchmark regime behavior; and
- successful validation of a feasible proposed policy.

The minimum verification set is the dynamic-allocation, policy-validation,
market-store, CLI market-context, and API market-context suites. Broader
inspection tests run if a shared evidence helper changes.

## Adversarial review

- Poor cost-basis return cannot lower a target by itself.
- A drawdown breach cannot produce `Action` or an execution suggestion.
- Missing KODEX 200 evidence cannot fall back to its inherited 5.3% as though it
  were a current analysis target.
- A complete but infeasible set of ranges cannot be force-normalized outside
  its published bounds.
- A severe Core result cannot silently change the instrument's role.
- A recurring-purchase plan cannot mutate strategic targets or persistent state.
- The algorithm remains deterministic and dependency-free; correlation and
  optimizer work is deferred until there is a separately validated need.
