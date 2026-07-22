# IPS Pilot

IPS Pilot is an IPS inspection workbench built around a single v2 evaluation framework. It evaluates portfolio layers and assets with the same model, then classifies each unit as `OK`, `Watch`, `Review`, or `Action`. The output is an inspection signal, not investment advice and not an order instruction.

## Stack

- Backend: FastAPI, pandas, numpy, scipy, yfinance, Pydantic
- Frontend: Bun, Vite, React, TypeScript
- Frontend library: lucide-react
- Storage: SQLite

## Product Shape

- First-class layers: `core`, `satellite`, `experiment`
- First-class layer assignment through `layer`
- Evaluation period support: `1M`, `3M`, `6M`, `YTD`, `1Y`, `Max`, or custom start/end dates
- Shared layer/asset outputs: weight, target gap, return, layer benchmark return, benchmark excess return, CAGR, MDD, volatility, concentration, risk contribution, efficiency, thesis status, burden, status
- Main result surfaces: Layer Dashboard, Asset Evaluation Table, Review Queue, Journal Draft
- Guardrails: no automatic buy/sell output, no execution flags, and no order-sizing recommendation surface

## Development

```bash
uv sync

cd frontend
bun install
cd ..

task run
task frontend-dev
```

During development, Vite proxies `/api` requests to `http://localhost:8000`. Production builds are served by FastAPI from `frontend/dist`.

The default SQLite database is `data/portfolio_rebalancer.sqlite3`. Override it with `PORTFOLIO_DB_PATH`.

### Toss Securities integration boundary

The Toss Securities integration is server-only and read-only. Phase 0 permits the OAuth token request and allowlisted account observations, but it exposes no sync API or user-facing account data yet. Order creation, modification, cancellation, sizing, and execution are outside IPS Pilot's product boundary.

Configure credentials only through the local environment variables listed in `.env.example`. Do not place real credentials in source files, SQLite, logs, browser storage, screenshots, or test fixtures.

## Common Commands

```bash
task run             # FastAPI API server
task frontend-dev    # Vite dev server
task dev             # API + frontend dev servers
task build-frontend  # React production build
uv run pytest        # backend tests
```

## CLI

The CLI prints a single JSON object to stdout. `evaluate` is the canonical v2 evaluation command.

```bash
uv run ips-pilot evaluate --text "VOO 40
QQQ 60" --period YTD
uv run ips-pilot evaluate --snapshot-id 14 --period 3M
uv run ips-pilot evaluate --snapshot-id 14 --start-date 2026-01-01 --end-date 2026-06-30
uv run ips-pilot evaluate --snapshot-id 14 --layer-benchmark core=SPY:80,QQQ:20 --layer-benchmark satellite=QQQ --layer-benchmark experiment=QQQ
uv run ips-pilot evaluate --file portfolio.csv --output-dir out
uv run ips-pilot agent-brief --snapshot-id 14
uv run ips-pilot review-queue --snapshot-id 14
uv run ips-pilot risk --snapshot-id 14
uv run ips-pilot portfolios list
uv run ips-pilot snapshots list --portfolio-id 1
```

`evaluate` returns:

```json
{
  "ok": true,
  "command": "evaluate",
  "input": {
    "source": "snapshot",
    "snapshot_id": 14,
    "portfolio_id": 1,
    "period": "3M",
    "start_date": "2026-03-24",
    "end_date": "2026-06-24",
    "bench": "SPY:80,QQQ:20",
    "layer_benchmarks": {
      "core": "SPY:80,QQQ:20",
      "satellite": "QQQ",
      "experiment": "QQQ"
    },
    "database_path": "data/portfolio_rebalancer.sqlite3"
  },
  "evaluation_period": {},
  "layer_evaluations": [],
  "asset_evaluations": [],
  "review_queue": [],
  "journal_draft": [],
  "warnings": [],
  "guardrails": {
    "not_investment_advice": true,
    "no_immediate_order_instruction": true
  },
  "error": null,
  "artifacts": {},
  "saved": {
    "saved": false
  }
}
```

Layer benchmarks are the canonical CLI benchmark setting. Use repeated `--layer-benchmark layer=BENCHMARK` options for `core`, `satellite`, and `experiment`; omitted layers default to `SPY:80,QQQ:20` for core and `QQQ` for satellite and experiment. The analysis benchmark is derived from the `core` layer benchmark. Each layer evaluation reports benchmark return and excess return against that layer's benchmark over the same evaluation period.

Legacy experimental commands and scenario-comparison options have been removed from the product surface.

## API

All application APIs live under `/api/v1`.

Workbench:

- `POST /api/v1/portfolio/manual`
- `POST /api/v1/portfolio/csv`
- `POST /api/v1/analysis/run`
- `POST /api/v1/evaluation/run`
- `GET /api/v1/evaluation/download-csv`

Saved portfolios and snapshots:

- `GET /api/v1/portfolios`
- `POST /api/v1/portfolios`
- `PATCH /api/v1/portfolios/{portfolio_id}`
- `GET /api/v1/portfolios/{portfolio_id}/current-state`
- `POST /api/v1/portfolios/{portfolio_id}/current-state`
- `GET /api/v1/portfolios/{portfolio_id}/snapshots`
- `POST /api/v1/portfolios/{portfolio_id}/snapshots`
- `GET /api/v1/portfolios/snapshots/{snapshot_id}`
- `POST /api/v1/portfolios/snapshots/{snapshot_id}/load`
- `PATCH /api/v1/portfolios/snapshots/{snapshot_id}`
- `DELETE /api/v1/portfolios/snapshots/{snapshot_id}`

Config and journal endpoints remain available. Saved snapshots store portfolio positions and metadata; analysis and v2 evaluation outputs are recomputed after loading a snapshot.

CSV download supports only:

- `metrics`
- `layer_evaluations`
- `asset_evaluations`
- `review_queue`

## Input Format

Paste input accepts ticker and allocation:

```text
VOO 40
QQQ 25
SOXX 15
```

CSV/TSV and manual rows support:

- `ticker`
- `allocation`
- `return_total`
- `layer`: `core`, `satellite`, `experiment`
- `thesis_status`: `valid`, `watch`, `broken`, `unknown`

## Evaluation Status

- `OK`: no threshold, data, thesis, or burden warning
- `Watch`: soft warning such as thesis watch/unknown, target gap, low efficiency, or elevated burden
- `Review`: hard threshold breach, risk overage, broken thesis, or insufficient data
- `Action`: broken thesis plus at least one hard limit breach

Status labels are inspection signals only.

## Verification

```bash
uv run pytest

cd frontend
bun run typecheck
bun run build
```

## Guardrail

This app is for education and prototyping. Results are not investment advice. Market data depends on `yfinance` and network availability.
