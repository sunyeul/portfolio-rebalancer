---
name: ips-snapshot-evaluate
description: Use when evaluating a saved or freshly synchronized Toss account snapshot against the active IPS policy, including cash, instrument, layer, risk, Review Queue, latest-account, target-gap, or candidate-policy comparisons.
---

# IPS Snapshot Evaluate

Acquire the requested Toss snapshot, run or read the backend-owned inspection,
and explain current cash, instrument, and layer differences. Pair this skill
with `ips-judgment-filter` whenever the output contains allocation or
exceptional-intervention language.

## Select the mode

- **Saved latest:** inspect persisted snapshots without contacting Toss. List
  candidates first and select the newest `synced_at`; break missing or tied
  timestamps with the largest `id`. Always report the selected `snapshot_id`.
- **Specific snapshot:** evaluate the explicit `snapshot_id` for reproducible
  review.
- **API refresh:** only when the user explicitly requests an API refresh or
  synchronization, run `task toss-health`, verify configuration, OAuth,
  account discovery, and account match, then run `task toss-sync`. The sync
  writes one new immutable snapshot.
- **Candidate comparison:** only when the user explicitly provides or selects a
  candidate, preview it separately against the same snapshot. Never replace
  the active-policy result with candidate output.

Treat `toss-snapshots`, `policy show`, `inspection preview`, and
`inspection run` as potentially stateful because they may initialize, migrate,
seed, or persist local data. A request to show a saved result does not authorize
a sync or fresh evaluation. A request to run or refresh an evaluation
authorizes the evaluation run, not an account sync or policy activation.

## Evaluate the snapshot

1. Record the acquisition mode, account alias, snapshot ID, synchronization
   time, source timestamps, source fingerprint, and data-quality state.
2. Stop allocation interpretation when the snapshot is missing, stale,
   partial, failed, unreconciled, or has incomplete policy coverage. Preserve
   backend `allocation_state=not_evaluable`, null priority, and null suggestion.
3. Load the active immutable policy version and hash. Do not recalculate target
   weights during snapshot evaluation.
4. Run or read the repository-owned deterministic inspection. Check the CLI's
   single JSON object, `ok`, `error`, and `contract_supported` before reading
   `evaluation.result`. Do not reimplement status classification in the skill.
5. Calculate or report cash against gross account value. Calculate or report
   instruments and `core`, `satellite`, and `experiment` against invested
   account value.
6. Present instrument rows before layer totals so the layer view is visibly an
   aggregation. Preserve the backend's current, minimum, target, maximum,
   target gap, triggers, status, priority, queue class, suggestion, meaning,
   verification task, and evidence references.
7. Show layer totals and cash as independent views. Do not mix gross cash and
   invested-asset denominators in one unexplained total.
8. Show account performance, profit/loss, drawdown, and market evidence as risk
   context only. Do not turn any one of them into an allocation status or trade
   trigger.
9. When the user requests a shared-exposure row such as SPY+VOO, aggregate it
   only in the presentation layer, retain the raw instrument rows, label the
   result as derived, and do not invent a new backend status.
10. When candidate preview is requested, add active and candidate target/gap
    columns for the same snapshot. Keep status, priority, queue class, and
    suggestion anchored to the active policy; state that the candidate is not
    active.

## Return the report

Return these sections in order:

1. Evaluation basis: mode, `run_id` when present, `snapshot_id`, synchronization
   time, active policy version and hash, candidate fingerprint when present,
   contract support, currentness, reconciliation, and `allocation_state`.
2. Account totals: gross account value, invested value, cash value, and data
   quality.
3. Cash comparison: current, minimum, target, maximum, target gap, range state,
   status, and supported review direction.
4. Instrument comparison: identity, layer, current, minimum, target, maximum,
   target gap, `below`/`within`/`above`, status, priority, and supported review
   direction. Add clearly labeled candidate columns only in comparison mode.
5. Layer aggregation: member coverage, current, minimum, target, maximum,
   target gap, range state, and backend status.
6. Risk context: performance and drawdown evidence with availability and source
   timestamps.
7. Review Queue: raw triggers, plain Korean meaning, backend suggestion, and
   verification task for every non-`OK` item.
8. Status counts and the next data-verification or future regular-purchase
   policy review step.

Use only `OK`, `Watch`, `Review`, and `Action`. Interpret `Action` as inspect a
possible exceptional intervention, never as permission to trade. Use directions
such as maintain, future-allocation direction, reduce or pause future regular
purchases, concentration-normalization review, or exceptional-review.

Never add a buy or sell side, order size, quantity, price, timing, execution
step, automatic recurring-sale instruction, policy mutation, or candidate
activation.

## Fail closed

- If `ok` is false, `error` is non-null, or `contract_supported` is false,
  report the machine-readable failure and stop interpreting the payload.
- If account data is not currently evaluable, return the unavailable fact and
  verification task; do not assert current weights or gaps.
- If the account is all cash, keep the backend `partial` result and evaluate
  only cash.
- If a candidate preview is invalid or not evaluable, preserve the active
  evaluation and label the candidate failure separately.
- If the latest snapshot and latest saved evaluation use different snapshot
  IDs, state the mismatch instead of calling the evaluation current.
