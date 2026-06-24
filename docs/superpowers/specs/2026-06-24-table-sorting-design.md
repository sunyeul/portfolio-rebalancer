# Table Sorting Design

## Context

The evaluation result view has two read-only inspection tables:

- `LayerDashboard`, showing layer-level evaluation rows.
- `AssetEvaluationTable`, showing asset-level evaluation rows.

Both tables currently render in API order only. Users need sortable columns for faster inspection without changing the underlying evaluation output or adding trading/action semantics.

## Scope

Add client-side sorting to every visible column in both tables.

Sorting must:

- Stay local to each table.
- Preserve the original API row order when no sort is active.
- Cycle per header click through ascending, descending, and unsorted.
- Sort numeric columns by raw numeric values, not formatted strings.
- Sort text-like columns by the displayed label where applicable, including layer, thesis, and status labels.
- Treat missing numeric values consistently by placing them after present values.

## Architecture

Keep the implementation in `frontend/src/App.tsx` because both target tables already live there and the behavior is small. Add a tiny shared sort state type, row comparison helper, and reusable sortable header button. Each table defines column metadata with a label and value accessor, then derives sorted rows with `useMemo`.

## UI

Each sortable header appears as a compact button inside the table header. The active header shows direction with an icon. Inactive headers show a neutral sort icon on hover-ready controls. Table layout and density should remain close to the current UI.

## Guardrails

This is presentation-only. It must not alter evaluation records, statuses, review queue contents, snapshots, config, journal entries, or persistent app state.

## Testing

Run the frontend build or typecheck to verify TypeScript and JSX correctness. Broader backend tests are not required because the change is isolated to client-side rendering.
