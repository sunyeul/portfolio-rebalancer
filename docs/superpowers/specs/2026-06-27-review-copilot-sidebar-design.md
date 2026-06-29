# Review Copilot Sidebar Design

## Context

Review Copilot is currently rendered through CopilotKit's `CopilotPopup` inside `frontend/src/copilot/ReviewCopilot.tsx`. The same component also registers the current portfolio/evaluation context and frontend tools, so the UI change should avoid moving those registrations.

## Goal

Render Review Copilot as a collapsible right-side sidebar instead of a floating popup.

## Approach

Use CopilotKit v2's exported `CopilotSidebar` component. Keep the existing labels, guardrail disclaimer, agent context, and frontend tool registrations unchanged. Replace only the chat shell returned by `ReviewCopilot`.

The sidebar should:

- Appear on the right side.
- Start open by default.
- Remain collapsible through CopilotKit's built-in toggle.
- Keep the existing 420px chat width unless local visual testing shows layout problems.

## Out Of Scope

- No changes to IPS evaluation logic, status vocabulary, Review Queue behavior, snapshots, config, journal entries, or persistent state.
- No custom order-sizing, buy/sell, or execution behavior.
- No custom chat implementation unless `CopilotSidebar` fails typecheck or runtime verification.

## Verification

Run the frontend validation available in the repo, starting with TypeScript/build-level checks. If a local dev server is available, visually confirm the chat is on the right and can be collapsed.
