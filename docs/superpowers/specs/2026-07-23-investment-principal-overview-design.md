# Investment Principal Overview Design

**Created:** 2026-07-23  
**Status:** Approved direction; implementation pending

## Purpose

Align the Overview with the user's account-level mental model:

```text
investment principal -> account value -> profit and return
```

The principal must remain stable when capital moves between brokerage cash and
holdings. Profit-taking, regular purchases, and internal sales therefore do not
change principal by themselves.

## Semantic Contract

`investment_principal_krw` is the confirmed initial account principal adjusted
only by classified external cash flows:

```text
investment_principal_krw
  = initial_principal_krw + cumulative_external_flow_krw
```

- Confirmed external deposits increase investment principal.
- Confirmed external withdrawals decrease investment principal.
- Cash-to-holding and holding-to-cash movements do not change it.
- Realized and unrealized profit or loss do not change it.
- Current holdings cost basis is a separate position-level fact and is not
  account investment principal.

The account-level derived values are:

```text
account_profit_krw = total_value_krw - investment_principal_krw
account_return = account_profit_krw / investment_principal_krw
```

`account_return` is unavailable when the principal is missing, non-finite, or
non-positive. Missing evidence must never render as numeric zero.

YTD TWR remains separate. It is the annual 10% objective metric and continues
to remove the effect of external cash-flow timing. The principal-relative
return is the intuitive account-state comparison shown on Overview; it does not
replace YTD, trailing-12-month, or cumulative TWR in performance history.

## Data and Migration

Rename the existing principal concept end to end from
`tracking_principal_krw` to `investment_principal_krw`:

- migrate the SQLite performance-point column without changing stored values;
- rename calculation and persistence payload fields;
- expose only `investment_principal_krw` in current service, CLI, API, and
  frontend contracts;
- update tests, fixtures, and current documentation;
- do not keep a duplicate public compatibility alias.

The baseline and classified external-flow workflow remains unchanged. This is
a semantic contract correction and derived-account-summary addition, not a new
cash-flow classifier.

## Overview Presentation

The primary Overview facts appear in this order:

1. `투자 원금`: `investment_principal_krw`.
2. `계좌 평가금`: `total_value_krw`, including brokerage cash.
3. `원금 대비 계좌 수익률`: `account_return`, with
   `account_profit_krw` as supporting text.
4. `YTD 계좌 TWR`: the annual-objective metric.
5. `예외 검토`: persisted backend queue summary.

The current-holdings unrealized return remains available in Performance and
instrument evidence. It is removed from the primary Overview sequence because
its denominator is holdings cost basis rather than account investment
principal.

## Backend Source of Truth

The backend produces `investment_principal_krw`, `account_profit_krw`, and
`account_return`. The frontend formats these values and does not independently
recalculate account return. This keeps CLI, API, stored evaluation results, and
the dashboard consistent.

Existing IPS status behavior is unchanged. Profit amount and principal-relative
return are descriptive account facts and cannot create `Action` or direct trade
language.

## Failure Handling

- A non-evaluable performance point exposes no fabricated principal or return.
- Missing or invalid principal produces null profit/return evidence and a
  missing-data presentation.
- Migration preserves existing values and fails rather than creating parallel
  old/new principal columns with conflicting values.
- Existing persisted evaluation JSON remains immutable historical evidence;
  new evaluations use the new contract.

## Verification

- Migration test proves stored principal values survive the column rename.
- Performance calculation and storage tests prove only external flows change
  investment principal.
- Inspection contract tests prove profit and return are emitted from the
  backend and remain descriptive facts.
- Frontend tests distinguish zero profit/return from unavailable evidence.
- Browser verification checks the five-card order and the separate YTD label.
- Guardrail tests confirm the change adds no order sizing, buy, sell, or
  execution semantics.

## Non-goals

- Redefining investment principal as current holdings cost basis.
- Changing the YTD TWR formula or annual 10% objective.
- Reclassifying external cash-flow candidates automatically.
- Changing IPS status thresholds or exceptional-review behavior.
