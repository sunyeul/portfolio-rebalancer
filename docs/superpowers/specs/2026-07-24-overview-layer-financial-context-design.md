# Overview Layer Financial Context Design

**Created:** 2026-07-24
**Status:** Approved for implementation

## Purpose

Remove the duplicate `레이어 점검` table from Overview. The existing `허용 범위와
현재 비중` section already presents the same layer status and allocation bands,
so it becomes the single layer overview surface.

## Selected Design

Keep the `허용 범위와 현재 비중` rows for 현금, core, satellite, and experiment.
For each invested layer, add three descriptive financial facts from the existing
Toss-backed instrument evidence:

- 지원 평가금 합계 (`market_value_krw`)
- 지원 매입원가 합계 (`cost_basis_krw`)
- 지원 평가손익 합계 (`unrealized_pnl_krw`)

These are position-level layer aggregates. They must not be presented as the
account's `investment_principal_krw`, and they do not affect status, priority, or
adjustment suggestions. The cash row keeps its current cash-value presentation;
its position cost basis is not applicable.

## Data and Missing Evidence

The frontend derives layer totals from the existing `instruments[].evidence`
payload and does not invent a new API field. Only finite numeric values are
included in each aggregate. If no value is supported for a layer, render `자료
없음` rather than numeric zero. The UI labels the values as 지원 values so a
partial evidence set is not mistaken for a complete accounting ledger.

## Scope

- Extend the existing allocation row presentation to include financial facts.
- Remove the Overview-only `LayerReviewTable` rendering and its unused component.
- Remove CSS used only by that table while keeping the Review Queue and other
  allocation surfaces intact.
- Do not modify backend evaluation, Review Queue contents, snapshot state, or
  IPS status semantics.

## Verification

Run `npm run typecheck` from `frontend` and `git diff --check` on the touched
frontend files. Confirm that Overview has one layer allocation surface, that
all four layer rows render the new financial context, and that missing values
show `자료 없음`.
