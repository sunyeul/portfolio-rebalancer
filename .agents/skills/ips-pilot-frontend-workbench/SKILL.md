---
name: ips-pilot-frontend-workbench
description: Use when changing IPS Pilot React workbench views, evaluation tables, Review Queue presentation, client-side filters, sorting, loading states, or inspection-result wording.
---

# IPS Pilot Frontend Workbench

The frontend is a read-only IPS inspection view. Improve visibility and
ergonomics without changing the meaning, order, or persistence of backend
evaluation data.

## Current boundaries

- The stack is React, TypeScript, Vite, `lucide-react`, and plain global CSS in
  `frontend/src/styles.css`; do not assume Tailwind or a component framework.
- Use the existing shapes and fetch helper in `frontend/src/lib/api.ts`.
- The current screen reads inspection and performance data. If a future surface
  adds portfolio or snapshot editing, preserve its pending, error, disabled,
  and confirmation states; do not assume those controls already exist.
- Keep layer and instrument surfaces separate and preserve backend Review Queue
  order whenever the user has not chosen a local table sort.

## Inspection semantics

`OK`, `Watch`, `Review`, and `Action` are inspection labels, not trading
signals. Show status, raw triggers, and human-readable meaning when useful,
but do not derive a new status, priority, suggestion, or action from them.

`suggestion` is the single action-level label. Render it prominently as the
adjustment direction—for example, secure cash, increase future
regular-purchase pace, increase allocation, normalize overweight, observe, or
inspect an exceptional case. It never changes queue order, status, or priority,
and it must not be translated into a buy, sell, quantity, price, or execution
instruction.

Do not mutate snapshots, policies, journals, evaluation records, Review Queue
contents, or persistent app state for a presentation change. Return, risk,
drawdown, and IPS-fit data are explanatory only; no UI text may imply a buy,
sell, quantity, price, or execution action.

## Tables and verification

Keep filtering and sorting local to the table. Preserve source order when sort
is inactive, compare numeric columns by raw values, and place missing numeric
values after present values.

Run `npm run typecheck` from `frontend` for a typed presentation change; use
`npm run build` when the change affects the compiled application or layout.
For layout changes, inspect desktop and narrow viewports when feasible.
