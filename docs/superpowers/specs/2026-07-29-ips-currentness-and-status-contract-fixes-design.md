# IPS Currentness and Status Contract Fixes Design

## Goal

Prevent stale persisted inspections and contradictory status metadata from being
presented as current IPS guidance, while preserving immutable evaluation history
and the Phase 5-v2 inspection-only boundary.

## Scope

This change fixes four bounded contract defects:

1. compare the latest persisted evaluation with the current evaluable Toss
   snapshot and active policy before presenting it as current;
2. keep the annual-return target as descriptive context rather than a status
   trigger;
3. keep evidence-only `Review` items in the monthly review without classifying
   `hold_and_observe` as an allocation adjustment; and
4. align repository guidance with the runtime-supported freshness and strict
   layer contracts.

Snapshot age thresholds, target-vector hysteresis, new `Action` production
rules, prospective-instrument policy activation, and market-regime model changes
remain outside this change because they require separate policy decisions.

## Evaluation currentness

Persisted evaluations remain immutable and replayable. The API adds a derived
currentness object that compares:

- `evaluation.snapshot_id` with the sole current evaluable complete snapshot;
- `evaluation.policy_version_id` with the active policy version.

The derived object reports both IDs, an `is_current` boolean, and stable reason
codes for a missing evaluation, missing current snapshot, missing active policy,
snapshot mismatch, or policy-version mismatch. It does not rewrite the stored
evaluation or create a replacement evaluation.

`/api/inspection` continues to expose the historical evaluation and its
`contract_supported` axis, plus currentness. The frontend renders allocation,
performance, and Review Queue contents only when currentness is true. When it is
false, it shows a verification banner with the evaluation and current snapshot
IDs and asks for an explicitly authorized inspection refresh.

`/api/review-queue` returns the same currentness object. When currentness is
false, it returns no current queue items or adjustment suggestions, preventing a
consumer from treating historical directions as current.

`/api/change-brief` also returns currentness. If its newest persisted evaluation
is not current, it returns `stale_evaluation` with no changes, and the frontend
does not request or render the evaluation-derived brief. This closes an
alternate surface added alongside the primary inspection view.

## Performance target semantics

The engine continues to calculate and expose `annual_twr` and `annual_target`.
An annual result below the target no longer changes `performance.status` or adds
`annual_return_below_target`. Missing or invalid performance evidence can still
produce `Watch` because that status describes data quality rather than return
performance.

## Review priority and queue classification

Allocation-specific precedence remains unchanged:

- cash below minimum: `P1`, reduce or pause future regular-purchase pace review;
- cash above maximum: `P2`, increase future regular-purchase pace review;
- overweight allocation: `P2`, normalization review;
- cash-supported underweight allocation: `P3`, future-allocation increase
  review.

After those cases, a remaining `Review` item receives `P2` with
`hold_and_observe`. Its queue class is `observation`, because the suggestion is
verification rather than an allocation adjustment. `Watch` and `OK` items keep
`P4` unless another documented precedence rule applies. Blocking items retain
null priority and suggestion.

Queue classification therefore considers both timing and suggestion: P1-P3 is
an `adjustment` only when the suggestion is not `hold_and_observe`.

## Repository contract alignment

`AGENTS.md` will describe snapshot selection using the fields the schema
actually exposes: newest `synced_at`, then largest `id` for a tie. It will also
define stricter Satellite and Experiment inspection in terms of runtime-owned
role, allocation-range, and drawdown evidence, without requiring retired manual
profile metadata.

## Verification

Tests must demonstrate:

- snapshot or policy mismatch makes API currentness false;
- a current evaluation remains visible and exposes its queue and suggestions;
- a stale evaluation exposes no Change Brief items;
- the frontend does not render a stale evaluation result;
- annual return below target remains descriptive and `OK` when evidence is
  otherwise valid;
- drawdown-based `Review` receives `P2`, `hold_and_observe`, and observation
  queue class;
- existing blocking, cash, overweight, and underweight precedence is unchanged;
- the relevant backend and frontend suites pass without live market data.

## Safety boundaries

The change adds no order side, quantity, price, timing, execution flag, or
automatic policy activation. It does not refresh an inspection automatically,
modify snapshots, or reinterpret a historical evaluation.
