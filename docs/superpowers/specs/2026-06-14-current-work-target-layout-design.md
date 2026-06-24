# Current Work Target Layout Design

## Goal

Change the workbench "현재 작업 대상" panel to match the selected left-hand reference layout. The panel should read as one clear dashboard card with a header, a portfolio-management row, and a snapshot-management area.

## Scope

- Update only the workbench current-target panel and the CSS needed to support it.
- Preserve all existing portfolio and snapshot behavior.
- Keep the existing data loading, mutations, error display, and disabled states.
- Use existing `lucide-react` icons instead of inline SVG.
- Do not change backend APIs, storage models, analysis flow, evaluation flow, or settings screens.

## Layout

The panel becomes a full-width card constrained to the existing `max-w-6xl` workspace width.

The first section is a header row:

- Left: folder icon, `현재 작업 대상` title, and existing explanatory copy.
- Right: rounded recent-save badge.
- The badge shows the active portfolio's latest snapshot name and short date when available, otherwise the existing empty-state text.

The second section is the portfolio-management row:

- Left field group: uppercase label `포트폴리오 선택`, then the portfolio select.
- Right field group: uppercase label `신규 생성`, then the new portfolio input and create button.
- On desktop, both groups sit in one horizontal row aligned to the bottom.
- On narrow screens, the groups stack.

The third section is the snapshot-management area:

- Top row: snapshot name input and `현재 상태 저장` button.
- Below: snapshot list as larger card-like rows.
- Each snapshot row shows name, short created date, position count, and note if present.
- The right action area contains the status label (`평가`, `분석`, or `입력`) followed by copy, edit, and delete icon buttons.
- Empty state remains visible when there are no saved snapshots.

## Interaction

All current interactions remain the same:

- Selecting a portfolio updates `selectedPortfolioId` and clears `activeSnapshotId`.
- Creating a portfolio uses `createNamedPortfolio`.
- Saving a snapshot uses `saveCurrentSnapshot`.
- Clicking a snapshot loads it.
- Copy, edit, and delete keep their current mutations and pending indicators.
- Disabled states continue to reflect pending work, missing portfolio selection, and missing valid rows.

## Styling

The visual direction follows the approved left reference:

- Clear card border and subtle shadow.
- Header separated by a light divider.
- Portfolio-management row separated from snapshot-management with another light divider.
- Slightly larger rounded fields and buttons than the old compact panel.
- Snapshot rows use white backgrounds, soft borders, hover border emphasis, and icon-only secondary actions.
- The active editing snapshot still gets a blue-accented selected treatment.

The implementation should remain compatible with the existing Tailwind v4 setup and `frontend/src/styles/app.css` component layer. Existing app colors can be used where they keep the reference layout consistent.

## Testing

After implementation:

- Run the frontend typecheck or build script.
- Open the workbench in the in-app browser.
- Verify desktop layout shows the three approved sections.
- Verify narrow viewport stacks the portfolio controls and snapshot actions without text overlap.
- Verify snapshot actions remain reachable and disabled states still render clearly.
