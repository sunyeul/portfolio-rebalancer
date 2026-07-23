# Toss-only Account Inspection Roadmap

**Created:** 2026-07-22
**Revised:** 2026-07-23
**Status:** Current roadmap; Phases 0–5 complete, Phase 6 remaining
**Product:** IPS Pilot

## Product Direction

IPS Pilot is a Toss Securities-specific IPS inspection workbench. Toss Open API
is the only source of account facts and market data. The app observes immutable
account state, tracks principal and performance, manages brokerage cash as an
explicit reserve, and produces evidence-backed human review signals.

The product does not support manual portfolio input, CSV/TSV upload,
user-entered ticker analysis, yfinance, generic broker adapters, or
Japan-account-specific behavior. It does not place, prepare, modify, cancel, or
size orders.

All investment judgments use exactly `OK`, `Watch`, `Review`, and `Action`.
`Action` means inspect a possible exceptional intervention, not permission to
trade. Gains, losses, market declines, or cash deviations are evidence, not
standalone buy/sell triggers.

## User Outcome

The user must operate the portfolio without relying on new external
contributions. IPS Pilot therefore provides a reproducible Toss-only view of:

- total account value;
- invested market value;
- deployable brokerage cash and gross cash weight;
- current holdings cost basis and unrealized profit/loss;
- tracked realized profit/loss only where Toss evidence is sufficient;
- user-confirmed tracking principal and external cash-flow adjustments;
- account value, principal, cash, and supported return history;
- a `10%` annual account-return objective measured by YTD TWR from the
  January 1 anchor, with trailing-12-month TWR retained as a secondary view;
- a normal cash policy of `10% / 15% / 20%` minimum, target, and maximum;
- current, target, permitted range, and gap for cash, layers, and instruments;
- `core`, `satellite`, and `experiment` exposure;
- market context derived only from official Toss market data;
- explainable review items linked to exact snapshots and policy versions.

## Durable Product Decisions

- Broker and account source: Toss Securities only.
- Integration: official Toss Open API through a server-side read-only allowlist.
- Managed cash: deployable cash inside the Toss brokerage account only.
- Base currency: KRW.
- Account history: immutable Toss snapshots; latest complete snapshot is current.
- IPS intent: versioned app-owned policy and Toss-instrument profiles.
- Cash: portfolio reserve, not a fourth investment layer.
- Layer model: `core`, `satellite`, and `experiment` remain first-class.
- Default adjustment posture: review future regular-purchase policy before an
  immediate trade.
- Operating constraint: no recurring external contribution is assumed.
- Cadence: observe account state weekly and run the full IPS inspection monthly.
- Return objective: YTD account TWR target is `10%`; trailing-12-month and
  cumulative TWR remain factual secondary views without separate targets.
- Target hierarchy: cash uses gross account weight; layers and instruments use
  invested weight, and instrument targets sum to their layer target.
- No compatibility mode: missing Toss data fails closed rather than falling back
  to manual input, cached generic data, or yfinance.

## Core Data Model

```text
Toss account API
  -> normalized immutable account snapshot
  -> account performance history
  -> Toss-keyed IPS profiles + versioned policy
  -> deterministic account projection
  -> OK / Watch / Review / Action inspection

Toss market-data API
  -> immutable market observations
  -> market-context evidence
  -> policy candidate requiring human approval
```

### Account facts versus IPS intent

Toss owns quantities, prices, average costs, market values, cash, currency,
orders, and executions. The app never allows users to edit those facts.

The app owns layer classification, thesis status, cash/layer policy versions,
evaluation runs, and human review records. This metadata annotates
Toss-observed instruments and cannot form an independent portfolio.

### Value and weight definitions

`total_account_value_krw = invested_market_value_krw + validated_deployable_cash_krw`

- Gross weight uses total account value and is used for cash policy.
- Invested weight uses invested market value and is used for layer and asset
  inspection.
- Tracking principal is the confirmed baseline adjusted only by classified
  external deposits and withdrawals.
- Holdings cost basis is not lifetime principal.
- Partial, stale, failed, or unreconciled observations cannot produce a current
  evaluation.

## Roadmap

### Phase 0 — Trust foundation — Complete

**Purpose:** Establish safe local persistence and a broker boundary that cannot
mutate the account.

**Delivered**

- Forward-only transactional SQLite migrations.
- Environment-only Toss credentials and systematic redaction.
- OAuth client-credentials flow.
- Allowlisted account reads and explicit blocking of broker mutations.
- Machine-readable CLI failure behavior.

The prior generic-data backup policy is superseded by the approved Phase 2.5
clean cut. Toss evidence remains protected; generic data is intentionally not
archived.

### Phase 1 — Toss account observation — Complete

**Purpose:** Establish a trustworthy, reproducible account snapshot.

**Delivered**

- Account discovery and configured account match.
- Holdings, KRW/USD buying power, exchange rate, and closed-order reads.
- Native-currency normalization and KRW reconciliation.
- Immutable `complete`, `partial`, `stale`, and `failed` snapshots.
- Idempotent fingerprints, pagination checks, and current-complete selection.
- `toss-health`, `toss-sync`, and `toss-snapshots` CLI commands.

### Phase 2 — Account value and performance history — Complete

**Purpose:** Track account value and return without claiming evidence that Toss
cannot substantiate.

**Delivered**

- User-confirmed tracking baseline.
- Principal, account value, invested value, cash, cost basis, and unrealized
  profit/loss points.
- External cash-flow candidates and explicit user classifications.
- Deterministic execution ledger and bounded realized profit/loss.
- Immutable performance runs with source fingerprints.
- Machine-readable performance CLI.

### Phase 2.5 — Toss-only foundation convergence — Complete

**Purpose:** Remove the generic product foundation before building policy and
judgment behavior on top of it.

**Delivered**

- Upgrade to schema v4 while preserving every Toss observation and performance
  row.
- Drop generic portfolio, snapshot, configuration, evaluation, and journal
  tables without migration or archive.
- Add Toss-keyed instrument profiles and immutable IPS policy versions.
- Add a pure account projection with explicit gross and invested denominators.
- Remove manual/CSV/text inputs, generic saved portfolios, yfinance analysis,
  Japan-account logic, FastAPI, session middleware, and the existing frontend.
- Reduce the runtime to the Toss/performance/profile/account-view CLI.
- Remove obsolete dependencies and adjacent generic-data migration backups
  after an explicit integrity gate.

**Exit criteria**

- Existing Toss snapshot 4, baseline 1, and performance run 1 remain readable.
- Generic tables, commands, source modules, frontend files, dependencies, and
  named migration backups are absent.
- New holdings cannot be silently treated as classified or `OK`.
- The latest complete snapshot produces a deterministic account projection.
- No broker mutation or direct trade-language path is introduced.

The roadmap itself is the current source for phase status and durable product
direction; completed implementation plans and phase-specific design records
are intentionally not retained here.

### Phase 3A — Target policy and deterministic inspection engine — Complete

**Purpose:** Turn Toss account evidence and versioned user targets into a
deterministic monthly IPS inspection without creating a trading recommender.

**Goals**

- Extend the immutable policy with a `10%` YTD TWR objective, weekly
  observation/monthly review cadence, and per-instrument target ranges.
- Validate that layer targets total 100% of invested value and instrument
  targets sum to their assigned layer target.
- Evaluate gross cash weight against `10% / 15% / 20%` and invested layer and
  instrument weights against their active policy ranges.
- Separate cumulative return, holding unrealized profit/loss, YTD TWR, and
  trailing-12-month TWR; do not annualize shorter histories.
- Derive only regular-purchase-policy, observe, thesis-review, and exceptional
  human-review language.
- Emit the source snapshot ID, policy version ID, trigger, plain meaning, and
  verification task.
- Persist immutable Toss evaluation runs and expose CLI inspection commands.

**Exit criteria**

- Identical snapshot and policy inputs produce identical results.
- Missing profiles, stale data, or denominator inconsistencies fail closed.
- Cash and invested-layer denominators cannot be confused.
- Annual-return underperformance, profit, loss, or cash deviation alone cannot
  produce `Action`.
- `Action` requires a broken thesis plus a hard overweight or risk breach and
  means inspect possible exceptional intervention.
- No result sizes an order or directly recommends buying or selling.

Phase 3A is the operational Toss-only inspection engine. It is the single
backend source of status and explanation logic for the CLI, API, and dashboard.

### Phase 3B — Read-only Toss operating dashboard — Complete

**Purpose:** Make current state, targets, gaps, performance, and Review Queue
visible without duplicating judgment logic in the browser.

**Goals**

- Add a loopback-only authenticated API over the Phase 3A projection,
  performance, policy, profile, and evaluation contracts.
- Rebuild a Toss-only browser foundation with account overview, performance,
  allocation, instrument profile, policy, Review Queue, and source-health
  surfaces.
- Show account principal, valuation, cash, cumulative TWR, YTD TWR, and
  supported trailing-12-month TWR on one time axis.
- Keep cash, layer, and instrument denominators explicit in every table and
  chart.
- Render backend statuses and explanations without client-side reclassification.
- Keep the first dashboard read-only for broker facts and evaluation results;
  profile and policy edits remain CLI-controlled until the write-enabled
  workbench phase.

**Exit criteria**

- The dashboard reproduces the Phase 3A CLI result for the same snapshot and
  policy version.
- Partial, stale, failed, unclassified, insufficient-history, and invalid-policy
  states have explicit non-evaluable UI states.
- No credential, raw account identifier, order-sized field, or broker mutation
  endpoint reaches the browser.
- Desktop and narrow viewport verification passes.

### Phase 4 — Toss market-context policy review — Complete

**Purpose:** Inspect whether the normal 15% cash target should move within the
approved 10%–20% envelope using only official Toss market data.

**Goals**

- Extend the read-only allowlist with Toss stock candles, current prices, stock
  master data, and market-indicator candles.
- Persist immutable market observations and source timestamps.
- Define benchmark drawdown, realized volatility, and medium/long trend from
  Toss daily candles rather than yfinance.
- Require multiple confirming signals, hysteresis, and a cooling period.
- Produce only a candidate policy version; human approval activates it.
- Treat incomplete history, unsupported symbols, or stale market data as a
  verification task.

Toss officially provides current prices, daily candles, stock metadata, and
market-indicator candles through its Open API:
<https://developers.tossinvest.com/docs>.

**Exit criteria**

- No external market-data source or user-entered benchmark exists.
- Offline official-response fixtures cover KR, US, index, pagination/window,
  missing-history, and rate-limit behavior.
- A market decline alone never becomes an immediate-purchase reason.
- Approved target changes remain within the policy envelope and never mutate
  the brokerage account.

The implemented market-context boundary keeps these thresholds as review
parameters, produces only a candidate, and requires human approval before any
policy change. No broker or active policy mutation is performed.

### Phase 5 — Profit/loss and exceptional-review signals

**Purpose:** Combine performance evidence with allocation, concentration, risk,
and thesis integrity without turning gains or losses into automatic trades.

**Goals**

- Link tracked gains, losses, cost basis, and drawdown evidence to exact Toss
  snapshots and performance runs.
- Treat gains as review evidence only when combined with allocation or risk
  limits.
- Treat losses as thesis and holdability review evidence, never a standalone
  stop-loss instruction.
- Apply stricter overlap, management-burden, holdability, and ETF-substitution
  checks to satellite and experiment exposure.
- Reserve `Action` for broken thesis plus a hard limit breach.

**Exit criteria**

- Gain alone, loss alone, price decline alone, and cash deviation alone cannot
  produce direct trade language.
- Every item states the raw trigger, plain meaning, and human verification task.
- Outputs contain no order quantity, proposed trade amount, or execution flag.

Phase 5 is implemented as an offline, read-only review gate. Its persisted
evaluation path remains deliberately separate from operational rollout: live
Toss market sync, structured factor population, policy activation, and the
first `phase5-v1` persisted run require explicit operator approval.

### Phase 6 — Write-enabled Toss operational workbench

**Purpose:** Extend the Phase 3B read-only dashboard with authenticated IPS
intent editing and human-decision records.

**Goals**

- Add write contracts for Toss-instrument profiles, policy candidates,
  activation approvals, evaluation reviews, and human decisions.
- Preserve source freshness and reconciliation before any judgment.
- Manage Toss-instrument IPS profiles and target ranges without editing broker
  facts.
- Link every evaluation and decision to source snapshot and policy IDs.
- Keep one backend status engine; the API and frontend render rather than
  reclassify.

**Exit criteria**

- No manual portfolio entry, upload, editable quantity/value, or generic broker
  mode appears in the UI.
- Historical decisions replay with their original evidence and policy version.
- Authentication and secret boundaries prevent browser access to credentials or
  account identifiers.
- Generated explanations cannot introduce trade execution or order sizing.

## Failure Safety

- Authentication failure is machine-readable and never promotes cached data to
  current.
- Rate limiting uses bounded retry and respects Toss response headers.
- One missing required endpoint produces a partial observation, not fabricated
  values.
- Missing exchange rates block KRW aggregation for foreign-currency values.
- New or unknown source enums remain visible and fail closed.
- Unclassified holdings remain reviewable data-quality issues.
- There is no manual or yfinance fallback in any phase.
- Order creation, modification, cancellation, and conditional-order endpoints
  remain blocked locally even though Toss exposes them.

## Verification Program

Every phase passes an adversarial design gate before implementation and a
verification gate before the next phase starts.

Required coverage includes:

- Toss response normalization, reconciliation, pagination, idempotency, and
  redaction;
- destructive v3-to-v4 migration that proves Toss preservation and generic-data
  removal;
- snapshot state, freshness, and denominator invariants;
- baseline, cash-flow, execution, realized-profit/loss, and TWR replay;
- Toss-symbol profile identity and unclassified-holding behavior;
- deterministic policy versioning and evaluation classification;
- official Toss market-data fixtures without live network requirements;
- repository and lockfile scans proving removed data sources do not return;
- guardrail fixtures proving no direct buy/sell or order sizing;
- API, CLI, and frontend contract consistency from Phase 3B onward.

Live Toss verification is explicit and optional. Automated tests use local
fixtures and never require real credentials.

## Delivery and Approval Gates

- Phases 0–5 are complete Toss-only evidence, inspection, dashboard,
  market-context, and offline profit/loss review infrastructure.
- Phase 6 adds authenticated IPS intent editing and human decisions to the
  read-only browser foundation.

Before any phase advances:

1. Its design is approved.
2. Adversarial review has no unresolved critical finding.
3. Source snapshot and policy inputs are immutable and replayable.
4. Missing or ambiguous data fails safely.
5. The status vocabulary and action-language guardrails are preserved.
6. No broker mutation, order sizing, or automatic trade field is introduced.
7. The smallest complete verification suite passes.
