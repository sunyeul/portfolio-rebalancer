---
name: snapshot-parameter-sweep-report
description: Use when evaluating a saved portfolio snapshot across multiple analysis/evaluation parameter sets, sensitivity scenarios, benchmarks, risk thresholds, or decision contexts, especially when the user asks for a report or to clear old snapshot analysis data first.
---

# Snapshot Parameter Sweep Report

## Overview

Use this repo-local skill to run repeatable multi-parameter evaluations for a saved portfolio snapshot and produce an IPS-aware report. Pair this skill with `ips-judgment-filter` whenever the output includes investment action language.

## When to Use

Use for requests such as:

- "14번 스냅샷 평가해줘"
- "특정 스냅샷을 여러 파라미터로 분석"
- "민감도 평가 보고서 작성"
- "기존 데이터 지우고 재평가"
- "기간/벤치마크/임계값별 스냅샷 비교"

Do not use for one-off CLI evaluation unless the user wants parameter sensitivity, scenario comparison, or a report.

## Workflow

1. Confirm the snapshot exists with `uv run python cli.py snapshots list --portfolio-id <id>` when the portfolio id is known, or rely on the sweep script's snapshot lookup.
2. If the user asks to clear old data, pass `--clean`; it only removes the selected output directory's generated artifacts and reports.
3. Run the bundled script from the repo root:

```bash
uv run python .agents/skills/snapshot-parameter-sweep-report/scripts/run_snapshot_sweep.py --snapshot-id <id> --clean
```

4. Inspect `summary.json` and the report for consistency:
   - scenario count matches generated artifact directories
   - recommended counts match the scenario table
   - "common to all scenarios" and "frequent" lists match computed frequencies
   - low-quality data maps to hold/verify language
5. Summarize the output to the user with the report path and one short "한눈에 읽기용" sentence.

## Outputs

Default output path:

```text
reports/snapshot<id>_parameter_sweep/
```

Generated files:

- `summary.json`
- `snapshot<id>_parameter_sweep_report_ko.md`
- `snapshot<id>_parameter_sweep_report.md` when `--language en` or `--language both`
- `artifacts/<scenario>/{stdout.json,metrics.csv,proposal.csv,ips_actions.csv,group_summary.csv,rc_violations.csv}`

## Report Rules

Read `references/report_style.md` before editing report language manually or changing the script's narrative rules.

Core interpretation rule: never translate numerical underweight/overweight signals directly into immediate buy/sell recommendations. Use regular purchase adjustment, thesis review, hold/observe, or data verification unless the IPS gate clearly supports an exceptional action.
