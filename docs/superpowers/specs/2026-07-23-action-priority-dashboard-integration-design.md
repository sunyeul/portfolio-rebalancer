# Action Priority and Dashboard Integration Design

**Created:** 2026-07-23  
**Status:** Proposed for written review

## Purpose

Make IPS Pilot lead with the smallest useful answer to one operating question:

> Given the current cash and allocation state, which future allocation adjustment
> should be reviewed first, and why?

The dashboard remains an inspection surface, not a complete operating ledger or
trading terminal. It does not need to display every fact used by the evaluator.
It shows account context, current allocation against permitted bands, and the
highest-priority adjustment suggestions with only the evidence needed to
understand them.

This design integrates the contract into three places:

1. the project-local `ips-judgment-filter` skill;
2. the backend inspection result and Review Queue ordering;
3. the current read-only dashboard.

## Existing Baseline

The current implementation already provides:

- immutable Toss account and performance evidence;
- cash, layer, and instrument current/target/minimum/maximum values;
- account and instrument profit/loss and drawdown evidence;
- `OK`, `Watch`, `Review`, and `Action` statuses;
- plain-language meaning and verification tasks;
- a read-only Overview, Allocation, Performance, Review Queue, Profiles, and
  Source Health UI.

The main mismatch is semantic. Review Queue entries have an internal integer
`priority`, but the queue is sorted by status severity before that priority.
The UI therefore cannot clearly distinguish:

- whether the evidence is usable;
- how serious the condition is;
- when it should be reviewed;
- which allocation mechanism should be considered.

The Overview also contains more repeated allocation and diagnostic content than
is needed for the result-first operating view.

## Selected Scope

### In scope

- Secondary account facts: investment principal, current account value,
  principal-relative account return, and YTD account TWR against the 10% annual
  objective.
- A compact cash/core/satellite/experiment band summary.
- A backend-owned four-axis decision contract:
  evaluability, status, priority, and adjustment suggestion.
- Priority-first ordering of adjustment suggestions.
- Minimal evidence disclosure on Overview and full evidence in Review Queue.
- Skill guidance and pressure tests that teach the same contract.
- Current API, CLI, and frontend contract tests affected by the new result
  fields.

### Out of scope

- Order sizing, order entry, execution flags, or automatic broker mutation.
- A complete cash-cycle ledger, DCA runway, XIRR, layer TWR, covariance risk
  contribution, or experiment lifecycle workbench.
- Automatic satellite validation or experiment promotion.
- Treating return targets, profit, loss, or drawdown as standalone allocation
  instructions.
- Replacing detailed Performance, Profiles, or Source Health views.

Those facts may remain backend evidence or future detailed features, but they
are not required to produce the current allocation review result.

## Supporting Account Facts

Overview keeps four descriptive facts in a compact secondary strip after the
judgment output:

1. `investment_principal_krw`: confirmed initial account principal plus
   classified external deposits and withdrawals.
2. `total_value_krw`: current invested market value plus managed brokerage cash.
3. `account_return`: `(total_value_krw - investment_principal_krw) /
   investment_principal_krw`, with the KRW profit/loss as supporting text.
4. `ytd_twr`: the account's cash-flow-neutral annual-objective measure, shown
   against the existing `10%` target.

The first three fields describe current account state. YTD TWR evaluates the
annual strategy objective. None of them creates an allocation suggestion by
itself.

Current holdings cost basis and holding unrealized return remain in Performance
and instrument evidence. They are not labeled as account investment principal.

## Four-Axis Decision Contract

Every displayed suggestion separates four meanings that must not be collapsed.

### 1. Evaluability

Evaluability answers whether current evidence may support an allocation
judgment.

`phase5-v2` separates allocation evaluability from optional performance
availability:

- `allocation_state = complete`: a current complete/reconciled Toss snapshot,
  valid policy denominators, and complete holding target/profile coverage are
  available;
- `allocation_state = partial`: cash is evaluable but the invested-value
  denominator is unavailable, as in an all-cash account; cash suggestions may
  be emitted while layer/instrument suggestions are withheld;
- `allocation_state = not_evaluable`: one of those required allocation inputs
  is missing, stale, inconsistent, or uncovered.

The account performance component has its own availability and status. A
missing or partial performance run may make investment principal, account
return, or YTD unavailable, but it does not suppress otherwise valid cash and
allocation suggestions. This deliberately removes the current engine's global
dependency on `performance_run.state == complete`.

The new result contains only `allocation_state` and `allocation_reason`; it does
not duplicate the old result-level `state` and `non_evaluable_reason` fields.
The persisted wrapper uses `state = complete` for `complete` or `partial`
allocation results and `state = not_evaluable` only when no allocation unit can
be judged. Wrapper `non_evaluable_reason` mirrors `allocation_reason` only in
that latter case. An exception before an evaluation can be persisted remains an
API/CLI failure, not a fabricated persisted `failed` decision state.

When performance is missing, wrapper state remains `complete`,
`performance.status` is `Watch`, unsupported principal/account-return/YTD
fields are null, and source-backed allocation suggestions remain available.

When `allocation_state` is not evaluable, the UI shows one blocking source/data
message and no allocation suggestion. Missing optional performance or risk
evidence may remain a data-verification `Watch`, but it receives
`hold_and_observe` and can never justify a more active adjustment suggestion or
`Action`.

### 2. Status

Status answers how serious the inspected condition is, independent of timing.

| Status | Contract |
| --- | --- |
| `OK` | Inside the approved inspection frame; no human verification required. |
| `Watch` | Soft evidence or a descriptive variance worth observing; it may be scheduled for the normal monthly inspection without requiring an allocation decision. |
| `Review` | A human must verify policy fit, thesis, or the proposed adjustment direction. |
| `Action` | The same instrument has a broken thesis and exceeds its own hard maximum; inspect possible exceptional intervention. |

`Action` does not authorize a trade. Cash, layer, group, return, profit/loss,
drawdown, or experiment evidence cannot create `Action` without the exact
same-instrument conjunction above.

### 3. Priority

Priority answers when the item should be handled and determines queue order.
It is not derived from status alone.

| Code | Exact label | Qualifying conditions |
| --- | --- | --- |
| `P1` | `다음 정기매수 전` | Cash below its minimum or an `Action` item. This is a relative checkpoint, not a computed calendar date. |
| `P2` | `이번 월간 점검` | Cash above its maximum; layer/instrument above its maximum; broken thesis; thesis-watch observation; overlap, burden, holdability, or ETF-substitution review. |
| `P3` | `다음 정기매수 배분 반영` | Layer/instrument below its minimum while cash is not below its minimum and the instrument has a valid thesis with the required profile evidence. |
| `P4` | `관찰 유지` | Inside-band target gaps, return-target differences, profit/loss, drawdown-only evidence, or other facts that do not justify a nearer allocation review. |

When one unit qualifies for multiple priorities, the lowest number wins. Queue
ordering is:

1. priority rank (`P1` through `P4`);
2. status severity (`Action`, `Review`, `Watch`, `OK`);
3. stable kind and identity ordering.

New `phase5-v2` payloads emit the semantic priority code and timing label. They
do not expose a meaningless implementation-only integer as the user-facing
priority. Historical payloads remain unchanged.

### 4. Adjustment Suggestion

The suggestion answers which policy mechanism should be reviewed. The backend
emits one code and human label from this closed vocabulary:

| Code | Exact backend label |
| --- | --- |
| `review_increase_regular_purchase_pace` | `향후 정기매수 속도 확대 검토` |
| `review_reduce_or_pause_regular_purchase_pace` | `향후 정기매수 속도 축소·중단 검토` |
| `review_increase_regular_purchase_allocation` | `향후 정기매수 배분 확대 검토` |
| `review_overweight_normalization` | `정기매수 축소·중단 우선 후 초과비중 정상화 검토` |
| `review_thesis_or_constraints` | `논지와 제약 조건 검토` |
| `inspect_exceptional_intervention` | `예외적 개입 가능성 검토` |
| `hold_and_observe` | `관찰 유지` |

The suggestion contains no quantity, order side, price, or execution flag.

## Deterministic Selection Rules

### Cash

| Condition | Status | Priority | Suggestion |
| --- | --- | --- | --- |
| Below minimum | `Review` | `P1` | `review_reduce_or_pause_regular_purchase_pace` |
| Within permitted band | `OK` | `P4` | none; an empty adjustment set means maintain |
| Above maximum | `Review` | `P2` | `review_increase_regular_purchase_pace` |

The suggestion applies to future aggregate regular-purchase policy. It does not
size or allocate an order.

### Layer

| Condition | Status | Priority | Suggestion |
| --- | --- | --- | --- |
| Below minimum and cash not below minimum | `Review` | `P3` | `review_increase_regular_purchase_allocation` |
| Within permitted band | `OK` | `P4` | none; an empty adjustment set means maintain |
| Above maximum | `Review` | `P2` | `review_overweight_normalization` |

Layer results remain separate from instruments. A layer result cannot infer that
one particular ticker should receive or fund an adjustment.

### Instrument

The instrument inspection universe is the union of current Toss holdings and
policy-configured instruments. When invested account value is positive, a
configured/profiled instrument absent from current holdings is evaluated at
`current = 0`; this permits a genuine below-minimum result without inventing a
holding. A current holding missing from policy or profiles makes allocation
coverage non-evaluable. When invested account value is zero, instrument weights
are unavailable rather than fabricated as zero and only the cash unit is
evaluable.

- For `core`, below minimum, a `valid` thesis, and cash not below minimum:
  `Review`, `P3`, review a larger share of future regular purchases. Unknown
  factors do not block core because they are not required for core, but any
  factor explicitly marked `review` blocks P3 and receives a P2 constraint
  review.
- For `satellite` or `experiment`, below minimum is eligible for the same `P3`
  suggestion only when thesis is `valid`; overlap, management burden, and
  holdability are all `clear`; and ETF substitution is `clear` or
  `not_applicable`.
- Below minimum with unclear/damaged thesis or unresolved strict-layer factors:
  `Review`, `P2`, review thesis or constraints; do not add merely because the
  weight is low.
- Above maximum with thesis not broken: `Review`, `P2`,
  `review_overweight_normalization`; reducing or pausing future regular
  purchases is considered before any exceptional intervention.
- Broken thesis without own hard maximum breach: `Review`, `P2`, review thesis
  and constraints.
- Broken thesis plus own hard maximum breach: `Action`, `P1`, inspect possible
  exceptional intervention.
- Profit, loss, price movement, or drawdown alone: at most the existing
  evidence-appropriate `Watch`/`Review`; `P4` hold-and-observe unless another
  allocation or thesis condition supplies a higher priority.
- Missing optional drawdown evidence does not suppress an otherwise valid P3
  underweight suggestion. It remains separate secondary evidence in the full
  Review Queue.

Suggestion selection uses this precedence after all triggers are collected:

1. exact `Action` conjunction;
2. cash below/above its band for the cash unit;
3. layer or instrument above its maximum;
4. layer or instrument below its minimum and eligible for future allocation;
5. thesis or profile constraints without an allocation-band suggestion;
6. hold and observe.

This deliberately lets an objective overweight condition supply the primary
normalization suggestion for non-`Action` units even when secondary review
factors are present. Those factors remain available in `verification_task` and
detailed evidence. A below-minimum unit never receives an increase suggestion
until its layer-specific readiness conditions pass.

### Performance

- YTD target gap, cumulative return, principal-relative return, profit/loss,
  and account drawdown remain descriptive strategy-evaluation facts.
- Unsupported performance evidence may request data verification, but it does
  not produce an allocation suggestion.
- Supported return underperformance is `Watch`, `P4`, `hold_and_observe`.

## Backend Result Shape

The backend returns two distinct lists:

- `adjustment_suggestions`: only evaluable cash/layer/instrument items whose
  backend suggestion is a `P1`, `P2`, or `P3` allocation-policy review;
- `review_queue`: every non-OK inspection or data-verification item, including
  performance and account-risk observations.

Review Queue entries also carry a backend-owned `queue_class`:

- `blocking`: required allocation evidence is unusable; `priority` is null,
  `priority_label` is `평가 차단`, and `suggestion` is null;
- `adjustment`: a P1–P3 allocation-policy suggestion;
- `observation`: a P4 data, risk, or performance observation with
  `hold_and_observe` when a suggestion object is useful.

Queue ordering is blocking first, then P1 through P4, then status severity and
stable kind/identity ordering. `allocation_reason` uses this fixed precedence:

1. Toss source state not current-evaluable;
2. holdings reconciliation failure;
3. gross account denominator invalid;
4. invested denominator unavailable, reported as `partial` rather than
   `not_evaluable` when cash remains valid;
5. policy/profile coverage incomplete.

Policy validation exceptions occur before evaluation and remain
machine-readable API/CLI failures rather than persisted blocking reasons.

The frontend never derives `adjustment_suggestions` by filtering
`review_queue`. When `allocation_state` is not evaluable,
`adjustment_suggestions` is always empty while `review_queue` may retain source
or data-verification evidence. `allocation_reason` is the one blocking reason
shown on Overview.

Each adjustment suggestion uses this conceptual shape:

```json
{
  "priority": "P2",
  "priority_label": "이번 월간 점검",
  "status": "Review",
  "kind": "instrument",
  "identity": "US/NBIS",
  "current": 0.027,
  "minimum": 0.0,
  "target": 0.01,
  "maximum": 0.02,
  "suggestion": {
    "code": "review_overweight_normalization",
    "label": "정기매수 축소·중단 우선 후 초과비중 정상화 검토"
  },
  "meaning": "현재 비중과 승인 범위가 어긋나며 위성 검토 근거가 남아 있습니다.",
  "verification_task": "투자 논지와 중복·ETF 대체 가능성을 확인합니다.",
  "triggers": ["instrument_out_of_range", "overlap_review"]
}
```

Backend status, priority, suggestion selection, explanations, and sorting remain
the single source of truth. React formats values and renders the returned order;
it does not reconstruct decision logic.

Persisted historical evaluation JSON stays immutable. New evaluations use a new
`phase5-v2` engine version and current contract without writing compatibility
aliases into new payloads. The new queue item replaces the duplicated
`next_step` field with the structured `suggestion` while retaining `meaning`
and `verification_task`. API/CLI readers continue to hydrate historical records
as stored.

The current dashboard reads the latest persisted evaluation. Until an explicitly
approved `phase5-v2` evaluation is persisted, the new adjustment section shows
an old-contract notice rather than adapting an integer priority or inventing a
suggestion from historical triggers. `/api/inspection` and
`/api/review-queue` expose `contract_supported: false` beside the unchanged old
evaluation; CLI `inspection show` likewise reports the unsupported current
action contract while preserving the raw historical result. This deliberate
cut avoids a semantic compatibility layer and prevents old evidence from being
presented as a new judgment.

The version gate appears in exact locations:

- `/api/inspection`: `data.contract_supported` beside `data.evaluation`;
- `/api/review-queue`: `data.contract_supported` beside `data.items`;
- CLI `inspection show`, `inspection preview`, and `inspection run`: top-level
  `contract_supported`.

It is true exactly when the returned evaluation wrapper has
`engine_version == "phase5-v2"`.

## Dashboard Information Architecture

### Overview

Overview is reduced to three sections in result-first order.

1. **Adjustment suggestions**
   - The first three entries from backend `adjustment_suggestions` in priority
     order.
   - Each card shows priority and timing, status, unit, current versus band,
     suggestion label, and one plain-language reason.
   - Triggers and evidence references stay inside a disclosure element.
   - If `adjustment_suggestions` is empty, show a neutral empty state: no current
     allocation adjustment requires review.
   - `P4` observation items never appear here; they remain in supporting facts
     or the full Review Queue.

2. **Allocation bands and adjacent evidence**
   - One compact row each for cash, core, satellite, and experiment.
   - Current, minimum–maximum band, target marker, and backend status.
   - The permitted-band relationship is more prominent than target gap.
   - Cash retains the `gross_account_value` denominator and layers retain the
     `invested_account_value` denominator. Visible denominator labels make clear
     that the four rows are not additive parts of one 100% total.

3. **Compact account context**
   - Investment principal.
   - Current account value and KRW profit/loss relative to principal.
   - Principal-relative account return.
   - YTD account TWR and the 10% annual target.

The existing full Review Queue remains the place for every item. Performance,
Allocation, Profiles, and Source Health remain detailed secondary views.

The repeated full Cash Reserve and Layer Review sections are removed from
Overview because their essential state is already present in the compact band
summary. Their detailed information remains available in Allocation and
Performance.

### Failure and missing-data presentation

- A non-evaluable allocation result replaces suggestions with one blocking
  banner and a link/pointer to Source Health.
- Unavailable performance facts render as unavailable inside the compact
  account context without hiding an otherwise evaluable adjustment section.
- Missing numeric evidence renders as unavailable, never as zero.
- Optional secondary endpoint failures keep the main evaluation visible and
  show a conditional warning.
- Internal snapshot, run, and policy IDs remain collapsed diagnostics.

## Skill Integration and Verification

The project-local `ips-judgment-filter` skill gains concise guidance that future
agents must:

- separate evaluability, status, priority, and suggestion;
- use the exact priority timing contract above;
- order work by priority rather than status alone;
- treat account returns as supporting facts, not adjustment triggers;
- prefer future regular-purchase allocation changes before exceptional
  intervention;
- preserve the exact `Action` conjunction.

These priority and suggestion rules are fixed product judgment semantics, not
user-editable numeric IPS thresholds. The implementation updates the durable
cash-account observability roadmap and README alongside the skill. Existing
versioned policy values for cash, layer, instrument, performance, and risk
thresholds remain unchanged unless separately approved.

Because an existing skill is being edited, its change follows skill TDD.

### RED pressure scenarios

Before editing the skill, run independent agents without the proposed new
section against at least these scenarios and record whether they conflate the
four axes:

1. Cash below minimum with no damaged thesis: expected `Review`, `P1`, future
   regular-purchase reduction/pause review; never `Action` or a sell order.
2. YTD below target plus underweight core: performance is `Watch`, `P4`; the
   core allocation may be `Review`, `P3` only if cash and evidence permit.
3. Broken thesis plus the same instrument above its own maximum: expected
   `Action`, `P1`, exceptional-intervention inspection without trade sizing.
4. A snapshot already classified `stale` by the Toss source contract plus a
   large allocation gap: expected non-evaluable result and no allocation
   suggestion. This change does not introduce a second wall-clock freshness
   threshold.
5. Missing performance history plus a complete current snapshot and policy:
   expected unavailable principal/YTD facts but still-evaluable cash and
   allocation suggestions.

### GREEN and REFACTOR

Add only the guidance needed to correct observed baseline failures, rerun the
same scenarios with the edited skill, and tighten wording only if an agent still
collapses status into priority or emits direct trade semantics.

## Implementation Boundaries

- Do not activate a new live IPS policy, persist the first new-engine
  evaluation, mutate profiles, or call live Toss endpoints as part of code
  implementation without a separate operational approval.
- Unit, API, CLI, and frontend fixtures provide deterministic offline evidence.
- Existing unrelated worktree changes and user-owned untracked files are not
  staged with this design or implementation.
- Status, priority, and suggestion rules stay centralized in the backend.

## Verification

- Backend unit tests cover every cash/layer/instrument matrix row, precedence,
  and priority-first stable ordering.
- Guardrail tests prove return, profit/loss, drawdown, missing data, and stale
  data cannot create direct allocation instructions or broaden `Action`.
- API and CLI contract tests prove the semantic priority and suggestion shape.
- Old `phase5-v1` and new `phase5-v2` fixtures prove that every API/CLI surface
  applies the same explicit engine-version gate without a read-time adapter.
- Non-evaluable integration tests prove `adjustment_suggestions` is empty even
  when diagnostic Review Queue entries remain.
- Performance-unavailable integration tests prove supporting account facts may
  be unavailable while source-backed allocation suggestions remain evaluable.
- Frontend tests cover zero versus unavailable returns, the four account facts,
  priority/status separation, priority order, and evidence disclosure.
- Frontend typecheck and production build pass.
- Browser verification checks the reduced Overview on desktop and narrow
  viewports.
- Skill pressure tests demonstrate the RED and GREEN behavior change.

## Acceptance Criteria

1. A user can identify the next review item, its timing, its severity, and the
   suggested allocation mechanism without opening a detail view.
2. Priority order is independent from status and is owned by the backend.
3. Investment principal, account value, principal-relative return, and YTD TWR
   are visible as secondary descriptive context when supported, without
   controlling allocation evaluability.
4. Overview does not duplicate the full operating ledger or detailed evidence
   surfaces.
5. No suggestion sizes or authorizes an order.
6. `Action` retains the exact same-instrument broken-thesis plus own-maximum
   breach condition.
7. The skill, backend, API/CLI, and dashboard use the same contract language.
