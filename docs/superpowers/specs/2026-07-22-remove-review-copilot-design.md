# Remove Review Copilot Design

## Context

IPS Pilot's web workbench is intended to collect portfolio data and display deterministic inspection results. Review Copilot, its Bun sidecar runtime, and the A2UI layer add a chat, generated explanations, and generated presentation surfaces that are no longer needed because Codex can inspect the same data through the CLI.

## Goal

Remove the entire Review Copilot and A2UI feature set while preserving the ordinary portfolio, snapshot, analysis, evaluation, and deterministic result-display workflows.

## Approach

- Remove the frontend CopilotKit integration, Review Copilot component, A2UI providers, generated-surface code, schemas, renderers, utilities, and their dedicated tests.
- Render the existing evaluation graph and Review Queue directly from the API evaluation response. The Review Queue continues to use its existing inspection-only status vocabulary and locally managed decision controls; it no longer requests AI-generated explanations.
- Remove the Bun `agent-runtime` package and all CopilotKit/OpenAI runtime configuration, Vite proxying, styles, task commands, and documentation references.
- Keep the Python API and CLI unchanged. Codex analysis continues through the existing CLI commands and saved snapshots rather than an embedded web assistant.

## Data Flow

The browser submits portfolio and snapshot changes to the existing FastAPI endpoints, then renders the returned analysis and evaluation responses directly. No request is made to `/copilotkit`, no browser state accepts agent-generated patches, and no OpenAI key or sidecar process is required for the web workbench.

## Error Handling

Existing API errors, pending states, disabled controls, and snapshot-management errors remain unchanged. The optional agent-explanation failure path is deleted rather than replaced, so base evaluation screens no longer depend on a second service.

## Out of Scope

- Changes to the v2 evaluation logic, status vocabulary (`OK`, `Watch`, `Review`, `Action`), or IPS inspection-only guardrails.
- Changes to SQLite data, snapshots, portfolio inputs, journal persistence, or CLI response contracts.
- A replacement in-browser chat, agent runtime, or generated UI system.

## Verification

- Run the frontend typecheck/build and frontend tests after deletion.
- Run the smallest relevant backend/CLI test set to confirm the unchanged inspection interface remains intact.
- Search the repository for CopilotKit, A2UI, and agent-runtime references, excluding retained historical documentation only if explicitly intended.
