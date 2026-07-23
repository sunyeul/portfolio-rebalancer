# Hybrid Operating Dashboard Design

**Date:** 2026-07-23
**Status:** Approved
**Product:** IPS Pilot

## Goal

Combine the mockup's fast monthly-review hierarchy with the current workbench's
real Toss data, navigation, traceability, and detailed inspection surfaces.

## Product boundary

The dashboard remains read-only. It renders the existing evaluation, policy,
performance, profile, snapshot, and market-context responses without changing
backend status classification or introducing orders, trade sizing, or direct
buy/sell language.

## Shell and navigation

- Keep the dark left sidebar and the existing functional tabs.
- Allow the desktop sidebar to collapse into an icon rail. Persist this
  presentation preference in browser storage, expose an accessible toggle, and
  keep the narrow-screen horizontal navigation unchanged.
- Add a dedicated `Review Queue` tab for the complete queue.
- Remove the globally fixed right queue rail so the main content can use the
  width required by the mockup-style overview.
- Preserve the refresh control. Keep source, snapshot, evaluation, performance,
  and policy identifiers in `Source health` instead of repeating them above
  every workbench tab.

## Overview hierarchy

The Overview becomes a monthly operating summary in this order:

1. Header: `이번 달 계좌 운용 점검`, refresh, and a read-only inspection notice.
2. Four primary facts: tracking principal, account value, current-holdings
   unrealized return, and cumulative account TWR. Principal-relative change and
   unrealized return are shown only when their supporting values exist.
3. Source metadata is absent during normal operation. A source warning appears
   only when the latest evaluation reports an incomplete source state.
4. Allocation comparison: cash plus core, satellite, and experiment current
   weights, target markers, status, and explicit denominator labels.
5. Annual-return target card: actual YTD account TWR when supported, target,
   cumulative TWR, supported history, and the availability of the January 1
   anchor. Recent 12-month TWR remains a secondary view.
6. Cash-reserve panel: current amount/weight plus approved minimum, target, and
   maximum.
7. Layer review table: current, target, gap, aggregated supported unrealized
   profit/loss, and backend status.
8. Review summary: first three items in backend queue order. The complete list
   is available in the `Review Queue` tab.

The full instrument table moves to the existing `Allocation` tab, keeping the
Overview concise without hiding detail.

## Presentation rules

- Use existing API fields only. Missing values render as `—` or an explicit
  evidence-collection message.
- Render all KRW amounts as rounded whole won with no visible fractional part.
- Label current-holdings unrealized return separately from cumulative and YTD
  account TWR. Calculate it from the latest persisted performance point's
  `unrealized_pnl_krw / current_cost_basis_krw`; never present it as account
  return.
- Toss `GET /api/v1/holdings` also supplies aggregate and per-instrument
  `profitLoss.rate`, but this UI change does not add a snapshot schema migration
  because the persisted normalized cost and profit/loss amounts already support
  the required display.
- Cash uses gross account value; layers and instruments use invested account
  value. These denominator meanings remain visible.
- Allocation bars clamp visual width to 0–100%, but displayed values remain the
  raw backend values.
- Layer profit/loss is a presentation-only sum of supported instrument
  `unrealized_pnl_krw` values in that layer. If no supported values exist, show
  `—`.
- Preserve `OK`, `Watch`, `Review`, and `Action` exactly. The UI does not
  promote or demote statuses.

## Responsive behavior

- Desktop uses a two-column shell: sidebar and wide content.
- Collapsed desktop mode uses a compact icon rail and leaves tooltips or
  accessible labels for every destination.
- The allocation/return hero and layer/review section collapse to one column
  below tablet width.
- The sidebar becomes horizontal navigation on narrow screens.
- Tables retain horizontal scrolling rather than compressing numeric columns.

## Verification

- `bun run --cwd frontend typecheck`
- `bun run --cwd frontend build`
- Browser verification of Overview and every tab at desktop width.
- Narrow viewport verification that cards stack, navigation remains usable,
  and tables scroll without clipping the page.
