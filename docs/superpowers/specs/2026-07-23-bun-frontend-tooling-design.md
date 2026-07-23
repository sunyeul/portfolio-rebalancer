# Bun Frontend Tooling Design

**Date:** 2026-07-23
**Status:** Approved
**Product:** IPS Pilot

## Goal

Make Bun the package-management and script-execution standard for the Toss
dashboard frontend while preserving the existing React/Vite application and
Python API boundary.

## Decisions

- Use Bun for dependency installation and all frontend scripts.
- Replace `frontend/package-lock.json` with Bun's `frontend/bun.lock`.
- Keep Vite as the frontend bundler and React as the UI runtime.
- Keep the Python API on `uv`/FastAPI; Bun does not replace the backend.
- Keep the production screen served by `task toss-dashboard-api` after the
  frontend bundle is built.
- Keep the Vite development proxy from `/api` to `127.0.0.1:8000`.

## User-facing commands

```bash
bun install --cwd frontend
bun run --cwd frontend build
task toss-dashboard-api
```

Development mode keeps the API and Vite dev server in separate terminals:

```bash
task toss-dashboard-api
bun run --cwd frontend dev
```

## Non-goals

- No migration from Vite to Bun's bundler.
- No frontend component or API contract changes.
- No changes to Python dependencies or Toss integration behavior.
- No root-level JavaScript workspace is introduced.

## Verification

- `bun install --cwd frontend --frozen-lockfile`
- `bun run --cwd frontend typecheck`
- `bun run --cwd frontend build`
- Confirm the production bundle remains available through the FastAPI static
  mount and that no frontend command in README/Taskfile instructs users to use
  npm.
