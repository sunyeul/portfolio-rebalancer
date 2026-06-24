# IPS Pilot Agent Guide

## Purpose

IPS Pilot is an IPS inspection workbench, not a trading recommender. Keep changes aligned with the v2 evaluation frame: inspect portfolio layers and assets, classify them as `OK`, `Watch`, `Review`, or `Action`, and preserve the guardrails against automatic buy/sell instructions.

## Product Guardrails

- Treat output as inspection signals only; never add automatic order sizing, execution flags, or direct buy/sell recommendations.
- Preserve the v2 status vocabulary exactly: `OK`, `Watch`, `Review`, `Action`.
- Interpret `Action` as "inspect possible exceptional intervention", not permission to trade.
- Prefer regular-purchase policy changes over immediate trades when describing IPS behavior.
- When data is missing, stale, or ambiguous, prefer observe, review, or data-verification language.
- Apply stricter thesis, overlap, burden, and ETF-substitution checks to `satellite` and `experiment` exposure.
- Keep core/satellite/experiment layers first-class; do not collapse them into ticker-only logic.

## Local Skills

- Use `ips-judgment-filter` before designing, reviewing, or changing IPS judgment behavior, especially language around buying, selling, DCA, drawdowns, core/satellite decisions, or conservative holds.
- Use `ips-pilot-cli-review` when reviewing a saved portfolio snapshot through the v2 CLI. Pair it with `ips-judgment-filter` whenever the response includes investment action language.
- Do not edit snapshots, config, journal entries, or persistent app state unless the user explicitly asks for that change.

## CLI Contract

- Treat the CLI as an agent-facing interface; preserve one JSON object on stdout.
- Keep CLI errors machine-readable instead of falling back to prose tracebacks.
- Do not add `buy`, `sell`, `execute`, or order-sized fields to v2 inspection outputs.
- For `latest` snapshot requests, list snapshots first and choose the newest `created_at`; if timestamps are missing or tied, use the largest `id`. Always state the actual `snapshot_id`.

## Working Rules

- Discover commands, file layout, and API shape from the repo instead of relying on this guide.
- Use the smallest verification set that covers the change; broaden only when touching shared contracts or cross-surface behavior.
- Be careful with local SQLite schema and snapshot changes because user state may already exist.
- Tests should not require live market data unless the user explicitly asks for live-data verification.
- Keep status rules in one backend source of truth and avoid re-implementing classification logic in API, CLI, or frontend surfaces.

## Project Memory Maintenance

Follow the Claude.md management pattern for this `AGENTS.md`: keep it concise, current, project-specific, and useful beyond what `rg`, README, package metadata, tests, generated schemas, or source inspection can reveal.

When asked to update or audit this file:

1. Discover current repo state first.
2. Remove facts that are easy to recover from files, package metadata, tests, or generated schemas.
3. Report concrete gaps before editing when the request is an audit or broad improvement.
4. Apply targeted additions only; avoid generic agent advice and repeated README/code content.
5. Prefer one line per durable lesson. Do not record one-off debugging details unless they would prevent likely future mistakes.

Good additions are durable product guardrails, repo-specific gotchas, safety-critical workflow rules, and decisions that are easy to miss even after reading the code. Bad additions are command lists, directory maps, stale paths, broad investment advice, or implementation details obvious from nearby source.
