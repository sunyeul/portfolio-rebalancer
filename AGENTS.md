# IPS Pilot Agent Guide

## Purpose

IPS Pilot is an IPS inspection workbench, not a trading recommender. Keep changes aligned with the v2 evaluation frame: inspect portfolio layers and assets, classify them as `OK`, `Watch`, `Review`, or `Action`, and preserve the guardrails against automatic buy/sell instructions.

## Product Guardrails

- Treat output as inspection signals only; never add automatic order sizing, execution flags, or direct buy/sell recommendations.
- Preserve the v2 status vocabulary exactly: `OK`, `Watch`, `Review`, `Action`.
- Interpret `Action` as "inspect possible exceptional intervention", not permission to trade.
- Prefer regular-purchase policy changes over immediate trades when describing IPS behavior.
- When data is missing, stale, or ambiguous, prefer observe, review, or data-verification language.
- Apply stricter role, allocation-range, and drawdown checks to `satellite` and `experiment` exposure; do not invent retired manual thesis, overlap, burden, or ETF-substitution metadata.
- Keep core/satellite/experiment layers first-class; do not collapse them into ticker-only logic.
- Treat normalized Toss account snapshots as the only source of holdings, cash, cost, price, order, and execution facts; never reintroduce manual portfolio, yfinance, generic broker, or Japan-account fallback paths.

## Local Skills

- Use `ips-judgment-filter` before designing, reviewing, or changing IPS judgment behavior, especially language around buying, selling, DCA, drawdowns, core/satellite decisions, or conservative holds.
- Use `ips-pilot-cli-review` when reviewing a saved portfolio snapshot through the v2 CLI. Pair it with `ips-judgment-filter` whenever the response includes investment action language.
- Do not edit snapshots, config, journal entries, or persistent app state unless the user explicitly asks for that change.

## CLI Contract

- Treat the CLI as an agent-facing interface; preserve one JSON object on stdout.
- Keep CLI errors machine-readable instead of falling back to prose tracebacks.
- Do not add `buy`, `sell`, `execute`, or order-sized fields to v2 inspection outputs.
- For `latest` snapshot requests, list snapshots first and choose the newest `synced_at`; if timestamps are missing or tied, use the largest `id`. Always state the actual `snapshot_id`.

## Working Rules

- Discover commands, file layout, and API shape from the repo instead of relying on this guide.
- For codebase navigation, use TokenSave MCP semantic tools first when available; treat `.tokensave/` as local derived data.
- Use the smallest verification set that covers the change; broaden only when touching shared contracts or cross-surface behavior.
- Treat adversarial review as a mandatory design gate before implementation; challenge omissions, guardrail regressions, stale references, destructive steps, and verification gaps.
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

## ACE 협업 운영

- 작업 시작, 목표·범위 변경, 사용자의 협업 선호·정정 수신, 예상 밖의 재사용 가능한 실패, 메모리 후보가 있는 작업의 마무리에서 `.agents/skills/ace-collaboration-memory/SKILL.md`를 읽고 ACE 절차를 적용한다.
- 매번 `PURPOSE`, `GOAL`, `ALIGNMENT`, `WORKING LOG`를 함께 확인한다. 훅이 없거나 신뢰되지 않으면 같은 순서를 수동으로 수행한다.
- 현재 시스템·개발자·사용자 지시와 기존 저장소 계약은 ACE 메모리보다 우선한다.
- 공유 교훈은 `.agents/playbooks/collaboration-lessons.md`에 Git 추적하고, 개인 선호는 `.serena/memories/local/user_preferences.md`에만 기록한다.

@RTK.md
