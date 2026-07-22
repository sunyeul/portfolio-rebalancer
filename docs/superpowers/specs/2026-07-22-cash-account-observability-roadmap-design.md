# Cash and Account Observability Roadmap Design

**Date:** 2026-07-22  
**Status:** Approved roadmap; implementation requires phase-specific plans  
**Product:** IPS Pilot

## Context

The user must continue operating the portfolio without relying on new external contributions. The account therefore needs an explicit, durable cash reserve so that market declines do not leave all capital committed. The user also needs to observe principal, account value, and profit/loss history and to inspect whether gains, losses, allocation drift, or thesis damage warrant human review.

IPS Pilot remains an IPS inspection workbench. It must not become an automated trading recommender. All outputs remain inspection signals using exactly `OK`, `Watch`, `Review`, and `Action`. `Action` means inspect possible exceptional intervention, not permission to trade.

## Product Outcome

IPS Pilot will connect to Toss Securities through an official, read-only Open API integration and provide a reproducible view of:

- total account value;
- invested market value;
- deployable cash and cash weight;
- current holdings cost basis;
- unrealized profit/loss;
- realized profit/loss only for the period the app can substantiate;
- principal and account-value history from an explicit tracking baseline;
- cash-policy inspection against a normal `10% / 15% / 20%` minimum, target, and maximum;
- evidence-backed review signals that combine account state, IPS limits, market context, and thesis status.

The integration never creates, modifies, cancels, sizes, or submits an order.

## Confirmed Decisions

- Broker: Toss Securities.
- Authentication credentials have already been issued.
- Integration mode: official Open API, read-only application surface.
- Managed cash: brokerage-account deployable cash only; living and emergency funds are excluded.
- Base currency: KRW.
- Currency handling: KRW and foreign-currency cash are converted to KRW, but they may be aggregated only after proving that Toss purchasing-power responses do not overlap.
- Normal cash policy: minimum 10%, target 15%, maximum 20%.
- Dynamic policy: composite market and portfolio signals produce a candidate policy change; the user must approve it.
- Cash is a portfolio reserve, not a fourth investment layer.
- Existing `core`, `satellite`, and `experiment` layers remain first-class.

## Non-Goals

- Brokerage order execution or order preparation.
- Automatic buy, sell, take-profit, stop-loss, or rebalance instructions.
- Order quantities, security-level cash amounts, or execution flags.
- A claim to reconstruct lifetime performance when the broker API cannot substantiate all historical cash flows and corporate actions.
- Including non-brokerage emergency or household cash.
- Phase 4 market-regime thresholds or Phase 5 profit/loss review rules in the first implementation delivery. Those phases require separate approved designs.

## Core Definitions

### Portfolio value

`total_account_value_krw = invested_market_value_krw + validated_deployable_cash_krw`

The calculation is valid only for a complete account snapshot whose holdings, purchasing power, and exchange-rate observations are mutually consistent.

### Principal and cost basis

These values must remain distinct:

- **tracking principal:** the user-confirmed account capital at the first complete tracking snapshot, adjusted only by verified later external deposits or withdrawals;
- **current holdings cost basis:** the broker-reported purchase amount for positions currently held;
- **unrealized profit/loss:** broker-reported or reproducibly calculated current holdings profit/loss;
- **tracked realized profit/loss:** realized result that can be supported from complete executions after the tracking baseline.

Current holdings cost basis must never be labeled as lifetime principal.

### Weight denominators

- **gross weight:** weight against total account value, including cash;
- **invested weight:** weight against invested market value, excluding cash.

Cash policy uses gross weight. Existing layer and asset evaluation uses invested weight. The existing generic `weight` field must not silently change meaning.

### Data quality

Every account snapshot has one of four states:

- `complete`: all required observations succeeded and passed reconciliation;
- `partial`: at least one required observation is missing or inconsistent;
- `stale`: source data or call timestamps exceed the allowed freshness window;
- `failed`: no usable snapshot can be produced.

Only `complete` snapshots may produce cash-policy or profit/loss review signals.

## Target Architecture

```text
Toss Securities Open API
  -> allowlisted GET-only Toss adapter
  -> raw response validation and redaction
  -> normalized account observation
  -> reconciliation and data-quality classification
  -> immutable account snapshot
  -> valuation and performance projection
  -> v2 inspection evaluation
  -> dashboard, Review Queue, and journal evidence
```

### Toss adapter

The adapter uses OAuth 2.0 Client Credentials and the required account header. It exposes only account, holdings, purchasing-power, exchange-rate, and order-history reads supported by the official OpenAPI document.

The transport layer rejects non-GET requests to the Toss domain. The application does not generate a full OpenAPI client that would expose order mutation methods. Client secrets and access tokens remain in environment-backed server configuration and process memory; they are never persisted or returned to the browser.

### Immutable account snapshots

Each successful synchronization creates an immutable observation set with:

- account alias rather than raw account number;
- source call timestamps and synchronization timestamp;
- holdings with native-currency and KRW-normalized values;
- purchasing-power observations by currency;
- applied exchange rates and rate timestamps;
- reconciliation result and data-quality state;
- a stable payload fingerprint for idempotency.

Repeated synchronization of the same source state must not create conflicting duplicate history. Partial data is stored as diagnostic evidence but is not promoted to the current evaluable snapshot.

### Existing portfolio snapshots

Existing position snapshots and append-only evaluation runs remain intact. A broker account snapshot may become the source for a new portfolio snapshot, but it does not mutate an existing position snapshot. Each evaluation records the exact account snapshot and position snapshot used.

## Roadmap

### Phase 0 — Trust foundation

**Purpose:** Protect existing local state and ensure the broker integration cannot mutate the account.

**Goals**

- Introduce explicit SQLite schema versioning and forward-only transactional migrations.
- Back up the existing database before the first schema migration.
- Remove destructive initialization patterns from the migration path.
- Add server-only credential configuration and systematic secret/account-number redaction.
- Implement an allowlisted GET-only broker transport with mutation-method rejection.

**Exit criteria**

- Existing portfolios, snapshots, evaluation runs, configuration, and journal entries survive migration tests.
- No Toss request using POST, PUT, PATCH, or DELETE can leave the process.
- Secrets, tokens, and raw account numbers are absent from persistence, logs, API responses, and frontend state.

### Phase 1 — Toss account observation

**Purpose:** Establish a trustworthy, reproducible account snapshot.

**Goals**

- Read account metadata, holdings, KRW/USD purchasing power, exchange rate, and paginated closed-order history.
- Normalize quantities, prices, costs, market values, currencies, executions, commissions, and taxes.
- Empirically verify whether KRW and USD purchasing power are independent before aggregation.
- Reconcile normalized totals against broker responses within documented rounding tolerances.
- Persist immutable `complete`, `partial`, `stale`, and `failed` synchronization results.

**Exit criteria**

- Holdings and market values reconcile with Toss account data.
- Cash is not double counted across currencies.
- A partial or stale response cannot replace the latest complete evaluable snapshot.
- Repeated synchronization is idempotent and pagination does not duplicate executions.

### Phase 2 — Account-value and performance history

**Purpose:** Track account value and profit/loss without overstating what the available data proves.

**Goals**

- Establish a one-time, user-confirmed tracking-principal baseline at the first complete snapshot.
- Store total account value, invested market value, deployable cash, current cost basis, unrealized profit/loss, and exchange-rate effect per snapshot.
- Calculate realized profit/loss only for executions that are complete after the tracking baseline.
- Detect later external cash flows instead of treating them as investment return; unresolved flows make the affected period non-evaluable.
- Build history series whose points link back to immutable source snapshots.

**Exit criteria**

- Every displayed value is reproducible from a stored complete snapshot.
- Tracking principal and current holdings cost basis are never conflated.
- Performance history states its tracking start and coverage limits.
- Missing flows, corporate actions, or executions cannot silently become profit/loss.

### Phase 3 — Fixed cash-policy MVP

**Purpose:** Prevent cash depletion using a deterministic IPS reserve policy before introducing market-regime logic.

**Goals**

- Model cash as a separate portfolio reserve.
- Preserve invested-weight semantics for `core`, `satellite`, and `experiment` evaluation.
- Evaluate gross cash weight against minimum 10%, target 15%, and maximum 20%.
- Use `OK`, `Watch`, and `Review` for reserve-policy inspection; cash deviation alone never produces `Action`.
- Explain the current cash state, triggered threshold, data quality, and next verification task without creating an order-sized gap.

**Exit criteria**

- Adding cash leaves existing invested-layer allocation semantics and risk contribution intact.
- The same complete snapshot and policy always produce the same status and triggers.
- Evaluation, API, CLI, frontend, and Copilot contain no direct buy/sell or order-sized output.

Phase 3 is the first operational MVP boundary.

### Phase 4 — Hybrid market-context review

**Purpose:** Inspect whether the normal 15% cash target should move within the approved 10%–20% envelope without turning a market decline into a standalone purchase reason.

**Design boundary**

- Candidate signals combine benchmark drawdown, realized volatility, medium/long-term trend, core underweight, thesis validity, and data freshness.
- A candidate change requires multiple confirming signals, hysteresis, and a cooling period.
- The user approves a policy candidate; approval changes only future inspection policy and never broker state.
- Missing, stale, or ambiguous data yields data verification rather than a policy-change candidate.

Phase 4 receives a separate design that fixes exact signal definitions, thresholds, hysteresis, evaluation fixtures, and offline validation before implementation.

### Phase 5 — Profit/loss review signals

**Purpose:** Use gains and losses as evidence for allocation, risk, and thesis review rather than as standalone trade triggers.

**Design boundary**

- A gain alone does not trigger reduction review.
- A loss alone does not trigger exceptional intervention.
- Overweight, concentration, risk contribution, reserve shortfall, overlap, and burden may produce `Review` when supported by complete data.
- `Action` retains its existing meaning: broken thesis plus a hard limit breach.
- Satellite and experiment positions receive stricter thesis, overlap, holdability, burden, and ETF-substitution checks.

Phase 5 receives a separate adversarially reviewed design before implementation.

### Phase 6 — Operational workbench

**Purpose:** Make source data, policy, inspection results, and human decisions traceable in one workflow.

**Goals**

- Show synchronization health and source timestamps.
- Plot tracking principal, account value, invested value, cash, and supported profit/loss history.
- Show reserve-policy state and any approved policy candidate.
- Link Review Queue items and journal decisions to exact account and evaluation snapshots.
- Keep Review Copilot explanations within the same status vocabulary and guardrails.

**Exit criteria**

- Historical decisions can be replayed with their original evidence and policy versions.
- User approval and deferral actions are auditable.
- Generated UI and Copilot output cannot introduce trade execution or order sizing.

## Error Handling and Failure Safety

- Authentication failure produces a machine-readable integration error and never falls back to cached data as current.
- Rate limiting uses bounded backoff and does not create retry storms.
- A single endpoint failure marks the synchronization partial; it does not fabricate missing values.
- Exchange-rate absence blocks KRW aggregation for foreign-currency values.
- Timestamp skew beyond the configured limit blocks evaluation.
- Unexpected API enum values are retained as unknown source values and surfaced for verification.
- Reconciliation failure preserves the raw diagnostic record but does not create a current account snapshot.
- Existing manual portfolio workflows continue to operate when Toss integration is unavailable.

## Verification Strategy

Every phase must pass its own adversarial design gate before implementation and its own verification gate before the next phase begins.

Required verification categories are:

- migration from a real prior-schema fixture without data loss;
- credential and account-identifier redaction;
- network tests proving broker mutation methods are rejected;
- official OpenAPI response-contract fixtures without live market dependency;
- KRW/USD purchasing-power overlap and rounding reconciliation cases;
- partial, stale, time-skewed, paginated, duplicated, and out-of-order responses;
- gross-weight and invested-weight denominator invariants;
- deterministic reserve status classification;
- guardrail cases proving gains, losses, and cash deviations do not generate direct trade instructions;
- API, CLI, frontend, and Copilot schema consistency.

Live Toss verification is optional and explicit. Automated tests must not require real credentials or live market data.

## Delivery Slicing

Phases 0–3 form the first delivery program because they provide a complete, useful cash-observability MVP without depending on unvalidated market-timing rules. Each phase receives its own implementation plan and completion gate.

Phases 4–6 are follow-on programs. They cannot be pulled into the Phase 0–3 implementation plan without a new approved design and adversarial review.

## Approval Gates

Before a phase advances, all of the following must be true:

1. The phase design is approved.
2. Adversarial review has no unresolved critical finding.
3. Existing snapshot and CLI contracts remain compatible or have an approved migration.
4. Missing or ambiguous data fails safely.
5. No direct buy/sell, order sizing, or execution field is introduced.
6. The smallest complete verification suite passes.

