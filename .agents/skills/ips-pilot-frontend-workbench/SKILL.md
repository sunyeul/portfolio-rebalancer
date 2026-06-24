---
name: ips-pilot-frontend-workbench
description: Use when changing IPS Pilot frontend workbench surfaces, especially evaluation tables, supporting-score visibility, portfolio or snapshot controls, and client-side presentation behavior in frontend/src/App.tsx.
---

# IPS Pilot Frontend Workbench

## Overview

Use this skill to keep frontend workbench changes aligned with IPS Pilot's inspection-only product contract. Frontend changes may improve visibility, sorting, layout, and ergonomics, but must not create trading semantics or mutate evaluation output.

## Scope Guardrails

- Keep workbench-only changes local to the existing frontend surface unless the user asks for a broader API, schema, or storage change.
- Preserve portfolio and snapshot behavior when changing layout: selection, creation, save/load, copy/edit/delete, errors, pending states, and disabled states should keep their existing semantics.
- Do not change snapshots, config, journal entries, evaluation records, Review Queue contents, or persistent app state for presentation-only features.
- Use existing API helpers and response shapes from `frontend/src/lib/api.ts`; do not invent new backend fields for UI-only changes.
- Use existing `lucide-react` icons and the app's Tailwind/component-layer styling patterns.
- Verify responsive layouts so controls, table headers, and action buttons do not overlap on narrow viewports.

## Evaluation UI Rules

- Keep layers and assets as separate first-class inspection tables or sections.
- Treat status labels as inspection labels only: `OK`, `Watch`, `Review`, and `Action`.
- Make supporting performance and efficiency signals explanatory, not standalone buy/sell triggers.
- If adding chips, summaries, or tables for return, efficiency, IPS fit, or risk contribution, preserve language that says why the item needs attention rather than what to trade.
- If adding table sorting, keep it client-side and local to the table. Preserve original API order when sorting is inactive, sort numeric columns by raw values, and place missing numeric values after present values.

## Verification

Run the smallest frontend check that covers the change, usually `npm run typecheck` or `npm run build` from `frontend`. For visual layout work, also inspect desktop and narrow viewport behavior in the browser when feasible.
