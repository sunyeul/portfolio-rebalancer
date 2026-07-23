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
- Add a dedicated `Review Queue` tab for the complete queue.
- Remove the globally fixed right queue rail so the main content can use the
  width required by the mockup-style overview.
- Preserve the refresh control and the source/snapshot identifiers.

## Overview hierarchy

The Overview becomes a monthly operating summary in this order:

1. Header: `이번 달 계좌 운용 점검`, refresh, and a read-only inspection notice.
2. Compact source strip: source state, snapshot, evaluation, performance,
   policy version, synchronization time, and annual target.
3. Four primary facts: tracking principal, account value, cumulative TWR, and
   annual TWR/readiness. Principal-relative change is shown only when both
   supported numbers exist.
4. Allocation comparison: cash plus core, satellite, and experiment current
   weights, target markers, status, and explicit denominator labels.
5. Annual-return target card: target, supported history, and remaining days
   before the 365-day comparison gate.
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
