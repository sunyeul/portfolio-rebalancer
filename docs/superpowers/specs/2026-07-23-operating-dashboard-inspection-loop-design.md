# Operating Dashboard and Inspection Loop Design

**Date:** 2026-07-23  
**Status:** Draft for written review  
**Product:** IPS Pilot

## Context

IPS Pilot now has a Toss-only foundation: read-only account observations,
immutable snapshots, account-performance history, Toss-keyed instrument
profiles, versioned policy storage, and a deterministic account projection. The
generic portfolio runtime and frontend are gone.

The next product outcome is an operating loop for a portfolio that receives no
recurring external contribution. The user must see principal, account value,
returns, current and target weights, gaps, and explainable review items. The
default behavior is steady regular-purchase policy rather than market timing.
Cash and securities form a controlled cycle, but the app remains an inspection
workbench and never becomes an order recommender.

## Confirmed Decisions

- Assume no recurring external cash contribution.
- Observe account state weekly and run the full IPS inspection monthly.
- Manage brokerage cash as a reserve with `10%` minimum, `15%` target, and
  `20%` maximum of gross account value.
- Manage `core`, `satellite`, and `experiment` as first-class invested layers.
- Give every Toss-observed instrument an explicit layer, thesis state, target
  range, and invested-account target weight.
- Instrument targets within a layer sum to that layer's target; all layer
  targets sum to 100% of invested value.
- Track both cumulative account TWR and annual account return.
- Define the annual objective as `10%` trailing-12-month TWR.
- Do not annualize or compare against the annual objective until 365 days of
  supported performance evidence exist.
- Use only `OK`, `Watch`, `Review`, and `Action`.
- Treat `Action` as inspection of a possible exceptional intervention, never
  permission to trade.

## Product Outcome

The dashboard answers five questions in order:

1. Is the Toss source complete, current, and reconciled?
2. What are the principal, account value, cash, and supported returns?
3. What are the current and target cash, layer, and instrument weights?
4. Which gaps or thesis conditions need human attention?
5. What should be verified or changed in the future regular-purchase policy?

The screen does not answer “what order should be placed?” It contains no order
quantity, proposed transaction value, execution flag, or broker mutation path.

## Scope Decomposition

The design is delivered in two consecutive implementation slices.

### Phase 3A — Target policy and inspection engine

Phase 3A extends policy, computes gaps, classifies inspection states, persists
replayable evaluation runs, and exposes one JSON CLI contract. It is the only
source of status and explanation logic.

### Phase 3B — Read-only operating dashboard

Phase 3B adds a loopback-only authenticated API and Toss-only browser UI that
render Phase 3A results. The browser does not reimplement classification and
does not edit broker facts, profile state, or policy in this slice.

Policy/profile editing and review decisions become browser-write operations only
in Phase 6, after their contracts and security behavior receive a separate
approval.

## Non-goals

- Automatic order creation, modification, cancellation, sizing, or execution.
- A direct buy, sell, profit-taking, or stop-loss recommendation.
- Treating gain, loss, drawdown, price decline, cash deviation, or annual-return
  shortfall as a standalone transaction trigger.
- Market timing or dynamic cash-target changes; official Toss market context is
  Phase 4.
- Full risk, overlap, burden, holdability, and ETF-substitution evaluation;
  exceptional-review enrichment is Phase 5.
- Browser editing of targets, profiles, or decisions in Phase 3B.
- Manual portfolio input, CSV upload, yfinance, generic brokers, or Japan-account
  compatibility.

## Source-of-truth Architecture

```text
Toss read-only account API
  -> normalized immutable broker snapshot
  -> account projection + performance history

versioned IPS policy + current Toss instrument profiles
  -> canonical evaluation input

projection + performance + policy + profile snapshot
  -> deterministic inspection engine
  -> immutable evaluation run
  -> CLI / local API / browser dashboard
```

Toss owns quantity, price, average purchase price, market value, cost, cash,
currency, orders, and executions. The app owns targets, layer classification,
thesis state, cadence, evaluation results, and later human decisions. The UI may
edit neither category in Phase 3B.

## Weight and Return Definitions

### Denominators

- `cash_weight_gross = deployable_cash_krw / total_account_value_krw`
- `layer_weight_invested = layer_market_value_krw / invested_market_value_krw`
- `instrument_weight_invested = instrument_market_value_krw /
  invested_market_value_krw`

Cash uses gross account value. Layers and instruments use invested value. Every
API and UI field carries a denominator label; no consumer may infer it from the
field's position.

### Return measures

- Holding unrealized profit/loss compares current market value with the current
  holding cost basis. It is not account return.
- Cumulative account TWR starts at the user-confirmed tracking baseline and has
  no target gap.
- Annual account return is trailing-12-month TWR and is compared with the `10%`
  objective only when at least 365 days of supported points cover the window.
- Shorter history returns `not_evaluable` with
  `annual_return_history_insufficient`; it is never annualized.
- External flows use the existing explicit cash-flow classification and TWR
  segmentation rules.

## Versioned Policy Contract

The active `ips_policy_versions.policy_json` is extended while retaining its
canonical JSON hash.

```json
{
  "cash_reserve": {
    "minimum": 0.10,
    "target": 0.15,
    "maximum": 0.20
  },
  "performance": {
    "annual_return_target": 0.10,
    "measurement": "trailing_12_month_twr",
    "minimum_history_days": 365
  },
  "cadence": {
    "observation": "weekly",
    "inspection": "monthly"
  },
  "layers": {
    "core": {"minimum": 0.70, "target": 0.80, "maximum": 0.90},
    "satellite": {"minimum": 0.10, "target": 0.20, "maximum": 0.30},
    "experiment": {"minimum": 0.00, "target": 0.00, "maximum": 0.05}
  },
  "instruments": [
    {
      "market_country": "US",
      "symbol": "SPY",
      "layer": "core",
      "minimum": 0.00,
      "target": 0.00,
      "maximum": 0.00
    }
  ]
}
```

The example instrument zeroes are structural examples, not approved portfolio
targets. The first policy remains non-evaluable for layer and instrument gaps
until the user assigns every held instrument and explicitly supplies its
minimum, target, and maximum. The app never invents those values.

Policy activation validates:

- every numeric value is finite and between 0 and 1;
- `minimum <= target <= maximum` for cash, layers, and instruments;
- layer targets sum to exactly 1 within a documented numerical tolerance;
- each instrument target and range fits inside its assigned layer;
- instrument targets grouped by layer equal that layer target;
- for each layer, grouped instrument minima do not exceed the layer target and
  grouped instrument maxima are not below it, so the configured ranges are
  jointly feasible;
- one composite `(market_country, symbol)` identity appears only once;
- every target identity was observed in a Toss snapshot;
- every currently held instrument has a profile and target before an evaluation
  can be fully evaluable.

Activation creates a new immutable policy version and supersedes the prior
version. It never rewrites historical policy rows.

## Evaluation Run Contract

Phase 3A adds immutable evaluation runs with:

- account alias;
- source `snapshot_id` and source fingerprint;
- source `performance_run_id` and input fingerprint;
- active `policy_version_id` and policy hash;
- canonical profile snapshot JSON and hash;
- engine version;
- evaluation state and non-evaluable reason;
- canonical result JSON;
- creation timestamp.

Capturing the profile snapshot prevents a later layer or thesis edit from
changing the meaning of a historical evaluation. Repeating identical canonical
inputs with the same engine version produces the same evaluation fingerprint
and result.

## Evaluation Pipeline

### 1. Source gate

Require a `complete`, current, evaluable, reconciled Toss snapshot and an
evaluable performance run. A failed, partial, stale, or inconsistent source
produces a machine-readable non-evaluable run and no portfolio-unit status.

### 2. Metric projection

Compute account facts, supported returns, gross cash weight, invested layer
weights, invested instrument weights, cost basis, and unrealized profit/loss.

### 3. Target gaps

For each evaluable unit, return current, target, minimum, maximum, signed gap,
denominator, and source identity. A missing profile or target returns `null`
metrics and a verification item, never zero.

### 4. Status classification

Classification lives in one backend module and uses this initial matrix:

| Condition | Status | Meaning |
| --- | --- | --- |
| Source and policy valid; unit within range | `OK` | Inside the approved frame |
| Annual TWR history below 365 days | `Watch` | Continue evidence collection |
| Annual TWR below 10% with sufficient history | `Watch` | Observe performance context; not a trade trigger |
| Thesis marked `watch` | `Watch` | Recheck the holding rationale |
| Cash below 10% or above 20% | `Review` | Review reserve and future regular-purchase policy |
| Layer or instrument outside its min/max range | `Review` | Verify target fit and future purchase posture |
| Missing profile, target, or denominator | `Review` | Complete or verify IPS input |
| Thesis marked `broken` without a hard overweight/risk breach | `Review` | Inspect thesis and holdability |
| Broken thesis plus hard maximum/risk breach | `Action` | Inspect possible exceptional intervention |

Profit, loss, price movement, cash deviation, or return-target gap alone cannot
create `Action`. Phase 3A can use hard maximum breaches; Phase 5 may add approved
risk and concentration hard limits without changing the meaning of `Action`.

When several triggers affect the same unit, severity is deterministic:
`Action > Review > Watch > OK`. All triggers remain visible even when a higher
status wins. Trigger lists, layer rows, instrument rows, and Review Queue items
use stable documented sort keys. Review Queue priority is source/data validity,
`Action`, cash reserve, broken thesis, layer range, instrument range, and return
context.

### 5. Explanation

Every non-OK item contains:

- stable raw trigger labels;
- current, target, range, gap, and denominator when supported;
- plain-language meaning;
- a human verification task;
- an allowed next-step posture;
- source snapshot, performance run, policy version, and engine version.

Allowed next-step posture is limited to:

- maintain future regular purchases;
- inspect increasing future regular purchases;
- inspect reducing or pausing future regular purchases;
- hold and observe;
- review thesis, overlap, burden, holdability, or ETF substitution;
- inspect possible exceptional intervention; if conditions are not confirmed,
  hold.

No field is named `buy`, `sell`, `execute`, or contains an order amount or
quantity.

## Operating Cadence

### Weekly observation

- Run Toss health and sync explicitly.
- Persist a new immutable snapshot when evidence changed.
- Refresh supported performance history.
- Show source freshness, reconciliation, account values, cash, and whether the
  monthly inspection is due.
- Do not create background orders or silently activate policy changes.

### Monthly inspection

- Resolve the newest complete snapshot by timestamp and ID tie-break.
- Resolve the active policy version and current profile snapshot.
- Create or reuse the deterministic evaluation run.
- Present the Review Queue for human inspection.
- Record no portfolio mutation or broker action.

The first implementation has no scheduler. CLI or local UI initiation remains
explicit; cadence metadata drives due-state presentation only.

## Phase 3A CLI Surface

Every command emits exactly one JSON object on stdout and sanitized,
machine-readable failures.

- `policy show --active` returns the active immutable policy and version.
- `policy template [--snapshot-id ID]` returns one policy JSON template with all
  observed current holdings and explicit null profile/target fields.
- `policy validate --file PATH` validates app-owned IPS intent without changing
  persistent state.
- `policy activate --file PATH --expected-current-version N` performs one
  compare-and-swap activation after all invariants pass. It creates one policy
  version; partial per-instrument versions are never persisted.
- `inspection run [--snapshot-id ID]` creates or reuses the deterministic
  evaluation for the explicit or actual latest complete snapshot.
- `inspection show (--run-id ID | --latest)` returns a saved evaluation and
  always states its actual snapshot, performance run, policy, and engine IDs.

Policy JSON is app-owned configuration, not a substitute portfolio or broker
fact. Allowing it as an atomic CLI input does not reintroduce manual holdings,
quantities, prices, values, or market data. An activation file may reference
only identities already observed through Toss.

## Dashboard Information Architecture

The approved desktop layout has three columns:

1. left navigation for overview, performance, allocation, profiles, policy, and
   source health;
2. central account facts, cash band, performance history, and target-gap tables;
3. right Review Queue with trigger, meaning, and verification task.

The first viewport shows:

- source state and actual snapshot ID;
- total account value and tracking principal;
- cumulative TWR;
- trailing-12-month TWR and `10%` objective, or an explicit non-evaluable state;
- cash current/target/range/gap;
- layer current/target/range/gap;
- top instrument current weights, target status, and unrealized profit/loss;
- ordered Review Queue items.

Layer and instrument tables remain separate. Sorting is client-local, uses raw
numeric values, preserves backend order when inactive, and places missing values
last. The browser renders backend classifications and does not infer a status.

## Phase 3B Local Web Boundary

Phase 3B establishes the final Toss-only browser foundation rather than a
throwaway static report.

- The backend transport uses FastAPI and Pydantic over the existing
  framework-independent services.
- The frontend uses React, TypeScript, and Vite with local CSS,
  `lucide-react`, and native SVG for the initial time-series charts; it adds no
  charting framework in Phase 3B.
- Production serves the built frontend from the local FastAPI process; Vite
  proxies only the Toss-only API during development.
- The server binds to loopback by default.
- A per-process secret bootstraps an HttpOnly, SameSite session; Toss
  credentials and account identifiers never enter the browser.
- The API exposes sanitized read contracts for source health, snapshots,
  projection, performance, active policy, profiles, evaluation, and Review
  Queue.
- The frontend is a dedicated Toss-only application and imports no removed
  generic portfolio contract. Its TypeScript response types are generated from
  the backend OpenAPI schema during verification.
- Backend services remain framework-independent; API and UI are consumers.
- Phase 3B has no browser write routes for policy, profiles, decisions, or
  broker operations.

This stack is intentionally the Phase 6 foundation. The code and contracts are
new Toss-only implementations; none of the removed generic API or frontend is
restored as a compatibility layer.

## Failure and Empty States

- Missing Toss configuration: show sanitized configuration failure; no network
  attempt or cached-current promotion.
- Latest sync failed or is partial: show that attempt first; an older complete
  snapshot may be labeled “last verified complete” but never “current”.
- Missing exchange rate or reconciliation failure: block KRW aggregate
  evaluation.
- No tracking baseline: show account value but mark account returns
  non-evaluable.
- Fewer than 365 supported days: show cumulative TWR and annual-return history
  insufficiency; do not annualize.
- Missing profile or target: show the instrument and required verification;
  layer and instrument gap remain null.
- Invalid target totals: reject policy activation and show the exact invariant.
- Unknown broker enum or non-finite number: preserve diagnostic evidence and
  fail closed.
- API failure: retain the last rendered view with a visible stale/error banner;
  disable controls that could create new state.

## Verification Strategy

### Phase 3A engine

- Policy validation tests for bounds, ordering, target sums, duplicate
  identities, unseen instruments, and canonical hashes.
- Return-window tests for exact 365-day coverage, shorter history, external
  flows, gaps, and missing points.
- Denominator tests proving cash uses gross value and layers/instruments use
  invested value.
- Table-driven status tests for every row in the classification matrix.
- Determinism tests across snapshot, performance, policy, profile, and engine
  fingerprints.
- Guardrail tests proving gain, loss, return shortfall, and cash deviation alone
  never create `Action` or order language.
- CLI tests preserving exactly one JSON object on stdout and machine-readable
  failures.

### Phase 3B dashboard

- API contract tests against the Phase 3A result without duplicated status
  logic.
- Authentication, redaction, loopback binding, and browser-secret tests.
- Component tests for loading, empty, failed, partial, stale, unclassified,
  insufficient-history, and invalid-policy states.
- Raw-value sorting tests with missing values last.
- Desktop and narrow viewport visual inspection.
- Frontend typecheck, build, and focused behavior tests.
- Repository scans for forbidden order routes, order-sized fields, direct trade
  language, and removed generic-data paths.

Automated tests use local fixtures and never require live Toss credentials or
market data. Live verification remains explicit and optional.

## Adversarial Review

1. **No external cash plus steady purchasing can exhaust the reserve.** Cash
   below the 10% floor produces a review of reducing or pausing future regular
   purchases. The engine does not manufacture a sale solely to replenish cash.
2. **A 10% objective can encourage risk chasing.** Annual underperformance is
   only `Watch` and never combines with target gaps to create an order or
   `Action`.
3. **“Natural profit-taking” can hide a gain trigger.** Gain is displayed as
   evidence; overweight, concentration, thesis, or burden conditions must carry
   the review independently.
4. **“Stop loss” can hide a loss trigger.** Loss alone never escalates. Broken
   thesis plus a hard maximum/risk breach is required for `Action`.
5. **Gross and invested denominators can produce contradictory gaps.** Field
   names, API contracts, validation, tests, and UI labels carry denominators
   explicitly.
6. **Target totals can be mathematically impossible.** Invalid policies cannot
   activate, and existing valid versions remain unchanged.
7. **Mutable profiles can corrupt historical meaning.** Every evaluation run
   captures canonical profile input and hash.
8. **A failed sync can make an old snapshot look current.** The UI distinguishes
   latest attempt from last verified complete and blocks current evaluation.
9. **Client-side status logic can drift.** Only the backend classifies; the UI
   sorts and renders.
10. **A temporary dashboard can become new legacy.** Phase 3B establishes the
    Toss-only web foundation intended for Phase 6 evolution, not a disposable
    report.
11. **Account data can leak through the local browser.** Loopback binding,
    ephemeral session bootstrap, redaction, and sanitized API schemas are
    required before the dashboard is accepted.
12. **The current account has no classifications or instrument targets.** The
    first complete evaluation intentionally remains `Review`/non-evaluable
    until the user supplies explicit values; no migration guesses them.

No critical adversarial finding remains unresolved in the design. Concrete
instrument target values and ranges are intentionally user-owned policy input,
not invented defaults.

## Acceptance Criteria

- The same complete snapshot, performance run, policy version, profile input,
  and engine version produce the same result.
- Current, target, range, signed gap, denominator, status, trigger, meaning, and
  verification task are visible for every evaluable cash, layer, and instrument
  unit.
- Principal, valuation, holding P/L, cumulative TWR, and annual TWR are not
  conflated.
- Annual TWR remains non-evaluable before 365 days and compares with `10%`
  afterward.
- Weekly observation and monthly inspection due states are visible but do not
  schedule background actions.
- Missing or ambiguous data fails closed.
- No gain, loss, drawdown, cash deviation, or return shortfall alone produces
  direct trade language or `Action`.
- `Action` always means inspect possible exceptional intervention.
- No broker mutation, order size, execution field, or credential reaches the
  CLI, API, or browser.
- Phase 3A can be implemented and verified before Phase 3B consumes its stable
  contract.
