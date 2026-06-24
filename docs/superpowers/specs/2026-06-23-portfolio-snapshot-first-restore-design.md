# Portfolio Snapshot First Restore Design

## Goal

Restore the first slice of saved portfolio and snapshot management in the frontend workbench.

## Scope

- Show saved portfolio selection and creation in the workbench.
- Allow saving the current session state to the selected portfolio.
- Show snapshots for the selected portfolio.
- Allow loading a snapshot back into the current workbench state.
- Do not add snapshot edit, copy, or delete controls in this pass.
- Do not change backend APIs, storage schema, CLI behavior, analysis logic, or evaluation logic.

## Design

Add a compact "작업 대상" management panel to `frontend/src/App.tsx` near the existing portfolio preview and evaluation settings. The panel uses the existing API helpers from `frontend/src/lib/api.ts`: `listPortfolios`, `createPortfolio`, `getCurrentState`, `saveCurrentState`, `listSnapshots`, and `loadSnapshot`.

The app owns a small set of server-state values: saved portfolios, selected portfolio id, snapshots for the selected portfolio, new portfolio name, active snapshot id, and a management error string. On initial render it loads portfolio summaries. When the user selects a portfolio, it loads the portfolio's saved current state if present and always refreshes that portfolio's snapshots. A missing current state is treated as an empty draft rather than a fatal error.

Loading current state or a snapshot updates the visible portfolio, analysis, evaluation, and editable input rows. New portfolio creation refreshes the portfolio list and selects the created portfolio. Saving current state refreshes the current state and portfolio list so the latest metadata stays visible.

## Error Handling

API errors are displayed inside the management panel. A selected portfolio with no current state shows a neutral helper message. Snapshot loading failures do not clear the existing workbench state.

## Testing

- Run the frontend TypeScript build.
- Run targeted backend persistence/API tests if frontend integration exposes response-shape issues.
- Manually inspect that selecting, creating, saving, listing snapshots, and loading snapshots are visible and wired.
