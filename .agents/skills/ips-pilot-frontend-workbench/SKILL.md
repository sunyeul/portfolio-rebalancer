---
name: ips-pilot-frontend-workbench
description: Use when changing IPS Pilot frontend workbench surfaces, especially evaluation tables, portfolio or snapshot controls, and client-side presentation behavior.
---

# IPS Pilot Frontend Workbench

## Purpose

Keep frontend changes aligned with IPS Pilot's inspection-only contract. UI changes may improve visibility, layout, sorting, and ergonomics; they must not create trading semantics.

## Required Rules

- Keep layer and asset evaluation surfaces first-class and separate.
- Preserve portfolio and snapshot semantics: select, create, save/load, copy/edit/delete, errors, pending states, and disabled states.
- Use existing API helpers and response shapes from `frontend/src/lib/api.ts`; do not invent backend fields for UI-only changes.
- Treat `OK`, `Watch`, `Review`, and `Action` as inspection labels only.
- Make return, efficiency, risk, and IPS-fit signals explanatory, not buy/sell triggers.
- Show internal status and trigger labels with human-readable meaning when space allows; do not make users infer meaning from raw codes alone.
- Do not mutate snapshots, config, journal entries, evaluation records, Review Queue contents, or persistent app state for presentation-only work.
- Use existing `lucide-react` icons and local Tailwind/component patterns.

## Sorting

Keep table sorting client-side and local to the table. Preserve API order when sorting is inactive, sort numeric columns by raw values, and place missing numeric values after present values.

## Verification

Run the smallest frontend check that covers the change, usually `npm run typecheck` or `npm run build` from `frontend`. For layout work, also inspect desktop and narrow viewport behavior when feasible.
