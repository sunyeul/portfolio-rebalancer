---
name: ips-pilot-cli-review
description: Use when Codex needs to review an IPS Pilot portfolio snapshot through the v2 CLI, especially for latest snapshot checks, monthly or weekly IPS reviews, Review Queue summaries, risk/status interpretation, or safe agent-facing summaries.
---

# IPS Pilot CLI Review

## Overview

Use this repo-local skill to run the existing IPS Pilot CLI and turn its v2 JSON output into conservative operating language for the user. Pair this skill with `ips-judgment-filter` whenever the response includes investment action language.

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

4. Use the compact brief by default:

```bash
uv run ips-pilot agent-brief --snapshot-id <snapshot_id>
```

5. Use raw evaluation only when the user asks for full data or CSV artifacts:

```bash
uv run ips-pilot evaluate --snapshot-id <snapshot_id> --period YTD
uv run ips-pilot evaluate --snapshot-id <snapshot_id> --output-dir /tmp/ips_pilot_eval_<snapshot_id>
```

6. Use narrower commands when the user asks for one surface:

```bash
uv run ips-pilot review-queue --snapshot-id <snapshot_id>
uv run ips-pilot risk --snapshot-id <snapshot_id>
```

## JSON Fields To Read

For `agent-brief`, read fields in this order:

- `summary`: period, layer status counts, asset status counts, and review count.
- `review_queue`: units that require Watch, Review, or Action inspection.
- `risk`: risk-focused review items derived from v2 triggers.
- `journal_draft`: prompts for recording the human review.
- `guardrails`: confirms this is not investment advice and follows the no immediate order rule.

For raw `evaluate`, read:

- `evaluation_period`
- `layer_evaluations`
- `asset_evaluations`
- `review_queue`
- `journal_draft`
- `warnings`
- `guardrails`

Do not infer buy, sell, execute, or regular-purchase instructions. The v2 schema intentionally has no execution flag or order-sized output.

## V2 CLI Contract

Treat the CLI as an agent-facing JSON interface. Successful commands should produce one JSON object with `ok`, `command`, `input`, evaluation or brief fields, `warnings`, `guardrails`, and `error`; failures should stay machine-readable.

The raw evaluation envelope includes:

- `evaluation_period`
- `layer_evaluations`
- `asset_evaluations`
- `review_queue`
- `journal_draft`
- `warnings`
- `guardrails`
- `error`

Review Queue items should be interpreted from `level`, `name`, `parent_layer`, `status`, `triggered_by`, `metrics_snapshot`, and `suggested_next_step`. Never add or expect `buy`, `sell`, `execute`, or order-sized fields in v2 inspection outputs.

## Interpretation Buckets

Build the user-facing summary from v2 status labels:

| Status | Meaning | Language |
| --- | --- | --- |
| `OK` | Inside the frame | "기준 안쪽입니다" |
| `Watch` | Soft warning | "다음 점검에서 관찰할 항목입니다" |
| `Review` | Hard threshold, data, thesis, or risk issue | "사람이 확인할 Review Queue 항목입니다" |
| `Action` | Broken thesis plus hard limit breach | "예외적 개입 여부를 검토할 신호입니다" |

Risk items come from `triggered_by` values related to MDD, volatility, concentration, risk contribution, or efficiency. Data-quality items are Review items whose triggers indicate insufficient data.

## Response Contract

Use this order for a normal review:

1. State the snapshot actually evaluated, including `snapshot_id`.
2. State the evaluation period.
3. Summarize layer status first, then asset status.
4. List Review Queue items with `level`, `name`, `status`, `triggered_by`, and `suggested_next_step`.
5. Mention Journal Draft prompts when they help the next human review.
6. End with what the user should verify next, without turning the output into trading instructions.

Use concise Korean by default when the user writes in Korean.

## Safety Rules

- Do not call CLI output investment advice.
- Follow the no immediate buy/sell rule: do not generate buy or sell instructions.
- Treat `Action` as an inspection signal, not permission to trade.
- Keep Review Queue items as human verification tasks.
- Do not treat price drops as standalone buy reasons.
- Ask for confirmation before changing snapshots, config, journal entries, files, or persistent app state.
- If data quality is weak, prefer observe, thesis review, or data verification language.

## Prompt Templates

Monthly review:

```text
Run `uv run ips-pilot agent-brief --snapshot-id <id>`.
Summarize layer status, asset status, and Review Queue.
Do not give immediate buy/sell instructions.
```

Weekly check:

```text
Run `uv run ips-pilot agent-brief --snapshot-id <id> --period 1M`.
Look only for new Watch/Review/Action items and risk triggers.
```

Quarterly IPS review:

```text
Run `uv run ips-pilot evaluate --snapshot-id <id> --period 3M`.
Compare layer_evaluations and asset_evaluations, then summarize Review Queue and Journal Draft prompts.
```

Sharp-drop review:

```text
Run `uv run ips-pilot risk --snapshot-id <id> --period 1M`.
Check whether MDD, volatility, concentration, thesis, or data-quality triggers require human review.
```

## When Not To Use

Use plain CLI/code work instead when the user asks to implement commands, change schemas, modify the frontend, or alter IPS domain logic.
