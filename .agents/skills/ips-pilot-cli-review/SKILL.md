---
name: ips-pilot-cli-review
description: Use when Codex needs to review an IPS Pilot portfolio snapshot through the existing CLI, especially for latest snapshot checks, monthly or weekly IPS reviews, DCA Plan and Review Queue summaries, risk/data-quality flag interpretation, or safe agent-facing summaries without adding new CLI commands.
---

# IPS Pilot CLI Review

## Overview

Use this repo-local skill to run the existing IPS Pilot CLI and turn its JSON output into conservative operating language for the user. Pair this skill with `ips-judgment-filter` whenever the response includes investment action language.

Do not add CLI commands, edit app code, modify snapshots/config/journal, or create reports unless the user explicitly asks for that separate work.

## Workflow

1. If the portfolio is unclear, list portfolios:

```bash
uv run ips-pilot portfolios list
```

2. If the snapshot is unclear, list snapshots for the selected portfolio:

```bash
uv run ips-pilot snapshots list --portfolio-id <portfolio_id>
```

3. If the user asks for `latest`, choose the newest snapshot from `snapshots list`. Prefer the newest `created_at`; if timestamps are missing or tied, use the largest `id`. Always include the actual `snapshot_id` in the final response.

4. Evaluate with the current CLI contract:

```bash
uv run ips-pilot evaluate --snapshot-id <snapshot_id>
```

Use optional existing flags only when the user asks for them or the review context requires them, for example:

```bash
uv run ips-pilot evaluate --snapshot-id <snapshot_id> --decision-context regular_review
```

If a requested context is not accepted by the CLI, rerun with the closest supported context and state the fallback.

## JSON Fields To Read

Read these fields first:

- `evaluation.playbook`: current review frame, confidence, reasons, and steps.
- `evaluation.ips_actions`: primary IPS action table and action metadata.
- `evaluation.proposal`: weights, target gaps, data quality, numeric candidates, and action reasons.
- `evaluation.rc_violations`: explicit risk-contribution cap warnings.
- `agent_summary.data_quality_warnings`: missing tickers and low-quality rows.
- `analysis.portfolio_metrics`: portfolio-level context only, not a trading directive.

Prefer `evaluation.ips_actions` for action labels, summaries, next steps, reason codes, risk notes, and blocked reasons. Use `evaluation.proposal` to add current/target/gap context when useful.

## Interpretation Buckets

Build the user-facing summary from `ips_action` values:

| `ips_action` | Bucket | Language |
| --- | --- | --- |
| `increase_dca` | DCA Plan / increase | "다음 정기매수에서 늘릴 후보" |
| `reduce_or_pause_dca` | DCA Plan / reduce_or_pause | "다음 정기매수에서 줄이거나 멈출 후보" |
| `hold_observe` | DCA Plan / hold | "유지/관찰" |
| `review_before_action` | Review Queue | "실행 전 사람이 확인할 항목" |
| `risk_control_review` | Review Queue and Risk Flags | "위험 관리 점검 항목" |
| `rebalance_sell_review` | Review Queue | "예외적 리밸런싱 매도 검토 항목" |
| `block_action` | Review Queue and Data Quality Flags | "행동 보류/차단 항목" |

Risk Flags should include rows from `rc_violations`, any `risk_over` rows, and non-empty `risk_notes`.

Data Quality Flags should include `missing_tickers`, `low_quality_rows`, `data_quality_low`, high `missing_ratio`, low `observation_count`, and `block_action` rows caused by data quality.

## Response Contract

Use this order for a normal review:

1. State the snapshot actually evaluated, including `snapshot_id`.
2. Summarize `ips_status` from playbook and the presence of DCA, review, risk, or data-quality items.
3. Show DCA Plan as regular purchase adjustment candidates only.
4. Show Review Queue as human checks before action.
5. Show Risk Flags and Data Quality Flags.
6. End with what the user should verify next, without turning the output into trading instructions.

Use concise Korean by default when the user writes in Korean.

## Safety Rules

- Do not call CLI output investment advice.
- Do not generate buy or sell instructions.
- Treat DCA Plan as "next regular-purchase adjustment candidates" only.
- Keep Review Queue items as human verification tasks.
- Treat `rebalance_sell_review` as review only, never as a sell instruction.
- Treat immediate buy/sell language as exceptional and avoid it unless the CLI explicitly frames an item that way and the user asks for that framing.
- Ask for confirmation before changing snapshots, config, journal entries, files, or persistent app state.
- If data quality is weak, prefer hold, observe, thesis review, or data verification language.

## When Not To Use

Use `snapshot-parameter-sweep-report` instead when the user asks for sensitivity analysis, multiple periods/benchmarks/thresholds, parameter sweeps, generated reports, or scenario comparisons.

Use plain CLI/code work instead when the user asks to implement new CLI commands, change schemas, modify the frontend, or alter IPS domain logic.
