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
- "위성 매수 중단 가정으로 비교"
- "IPS 전략 백테스트도 같이 봐줘"

Do not use for one-off CLI evaluation unless the user wants parameter sensitivity, scenario comparison, or a report.

## Workflow

1. Confirm the snapshot exists with `uv run portfolio-rebalancer snapshots list --portfolio-id <id>` when the portfolio id is known, or rely on the sweep script's snapshot lookup.
2. If the user asks to clear old data, pass `--clean`; it only removes generated artifacts and reports in the selected output directory.
3. Run the bundled script from the repo root. The default output is Korean-only:

```bash
uv run python .agents/skills/snapshot-parameter-sweep-report/scripts/run_snapshot_sweep.py --snapshot-id <id> --clean
```

4. Use these options when the request calls for them:
   - `--output-dir <path>` to write outside the default report directory
   - `--language ko|en|both` to choose report language
   - `--scenario-set default` for the currently bundled scenario set
5. Inspect `summary.json` and the report for consistency:
   - scenario count matches generated artifact directories
   - recommended counts match the scenario table
   - "common to all scenarios" and "frequent" lists match computed frequencies
   - low-quality data maps to hold/verify language
   - `summary.json` and the emitted script JSON point to the same report directory
6. If the user asks for what-if assumptions or strategy history, run optional extension analyses from the repo CLI and keep them separate from the core parameter sweep:
   - counterfactual for current-decision what-if questions
   - backtest for policy/strategy behavior over historical returns
7. Summarize the output to the user with the report path and one short "한눈에 읽기용" sentence.

## Current CLI Contract

The sweep script runs each scenario through the repo CLI. The equivalent public command shape is:

```bash
uv run portfolio-rebalancer evaluate --snapshot-id <id> --output-dir <artifact-dir>
```

Keep the scenario arguments aligned with the CLI's current `evaluate` options:

- `--period <months|YTD|Max>`
- `--rf <float>`
- `--bench <ticker|weighted benchmark>`
- `--rc-threshold <float>`
- `--e-threshold <float>`
- `--decision-context regular_review|market_correction|sharp_drop_review|rebalance_review`
- `--counterfactual-scenario core_reinforcement|pause_satellite_new_buys|dca_shift_to_core`
- `--backtest-strategy current_ips|core_first_dca|pause_overweight_satellite|return_chasing_reference`

If the CLI changes, update both `scripts/run_snapshot_sweep.py` and this section together.

## Optional Extensions

Do not include counterfactual or backtest runs in the default scenario table. They answer different questions and should be reported in separate sections or a separate short addendum.

Use counterfactual analysis when the user asks about assumptions such as:

- pausing satellite new buys
- reinforcing core exposure
- shifting DCA toward core assets
- comparing a current recommendation with a what-if policy

Example:

```bash
uv run portfolio-rebalancer evaluate --snapshot-id <id> --output-dir <output-dir>/extensions/counterfactual_pause_satellite --counterfactual-scenario pause_satellite_new_buys
```

Use backtesting when the user asks whether the IPS or a policy would have behaved well historically, or asks to compare strategies. Pass `--backtest-strategy` more than once when comparing strategies.

Example:

```bash
uv run portfolio-rebalancer evaluate --snapshot-id <id> --output-dir <output-dir>/extensions/backtest_ips --backtest-strategy current_ips --backtest-strategy core_first_dca --backtest-strategy pause_overweight_satellite
```

When summarizing optional extensions:

- Treat counterfactual output as decision sensitivity, not a replacement for the parameter sweep.
- Treat backtest output as policy behavior evidence, not proof of future returns.
- Keep IPS language conservative: regular purchase changes, pause/reduce new buys, hold/observe, thesis review, or exceptional action only.
- Mention any simulation warnings before drawing conclusions.

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
- optional extension artifacts under `<output-dir>/extensions/<name>/` when counterfactual or backtest runs are requested

The script also writes `artifacts/<scenario>/stderr.txt` when a scenario emits stderr.

## Report Rules

Read `references/report_style.md` before editing report language manually or changing the script's narrative rules.

Core interpretation rule: never translate numerical underweight/overweight signals directly into immediate buy/sell recommendations. Use regular purchase adjustment, thesis review, hold/observe, or data verification unless the IPS gate clearly supports an exceptional action.
