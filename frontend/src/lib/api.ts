import type { PortfolioRowInput } from './schemas';

export type AssetRow = {
  ticker: string;
  allocation: number;
  return_total: number | null;
  layer?: string | null;
  thesis_status: string;
  weight: number;
};

export type MetricRow = {
  ticker: string;
  cagr: number | null;
  volatility: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  information_ratio: number | null;
  beta: number | null;
  alpha: number | null;
  data_start: string | null;
  data_end: string | null;
  observation_count: number | null;
  missing_ratio: number | null;
  risk_contribution: number | null;
  return_contribution: number | null;
  weight: number;
  efficiency_score: number | null;
  return_total: number | null;
  layer?: string | null;
  thesis_status: string;
};

export type MetricsSummary = {
  cagr: number | null;
  volatility: number | null;
  sharpe: number | null;
  max_drawdown?: number | null;
};

export type EvaluationStatus = 'OK' | 'Watch' | 'Review' | 'Action';

export type EvaluationPeriod = {
  label: '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max' | 'custom';
  start_date: string;
  end_date: string;
};

export type EvaluationUnit = {
  level: 'layer' | 'asset';
  name: string;
  parent_layer: string | null;
  benchmark?: unknown;
  target_weight: number | null;
  allowed_mdd: number | null;
  allowed_volatility: number | null;
  max_weight: number | null;
  min_efficiency: number | null;
};

export type EvaluationOutput = {
  current_weight: number;
  weight_gap: number | null;
  layer_internal_weight: number | null;
  period_return: number | null;
  cagr: number | null;
  benchmark_return: number | null;
  benchmark_excess_return: number | null;
  mdd: number | null;
  volatility: number | null;
  concentration: number | null;
  risk_contribution: number | null;
  return_mdd_ratio: number | null;
  cagr_mdd_ratio: number | null;
  thesis_status: 'valid' | 'watch' | 'broken' | 'unknown';
  burden: 'low' | 'medium' | 'high';
  status: EvaluationStatus;
  triggered_by: string[];
};

export type EvaluationRecord = {
  unit: EvaluationUnit;
  output: EvaluationOutput;
};

export type ReviewItem = {
  level: 'layer' | 'asset';
  name: string;
  parent_layer: string | null;
  status: Exclude<EvaluationStatus, 'OK'>;
  triggered_by: string[];
  metrics_snapshot: Record<string, unknown>;
  thesis: string | null;
  counter_scenario: string | null;
  suggested_next_step: string;
};

export type EvaluationResponse = {
  evaluation_period: EvaluationPeriod;
  layer_evaluations: EvaluationRecord[];
  asset_evaluations: EvaluationRecord[];
  review_queue: ReviewItem[];
  journal_draft: Array<Record<string, unknown>>;
  warnings: string[];
  guardrails: {
    not_investment_advice: boolean;
    no_immediate_order_instruction: boolean;
  };
};

export type EvaluationRun = {
  id: number;
  snapshot_id: number;
  settings: {
    period?: string | null;
    start_date?: string | null;
    end_date?: string | null;
    as_of_date?: string | null;
    bench?: string | null;
    layer_benchmarks?: Record<string, string>;
  };
  schema_version: number;
  engine_version: string;
  ips_config_hash: string;
  status: 'active' | 'superseded';
  created_at: string | null;
  superseded_by_run_id: number | null;
  is_stale: boolean;
};

export type PortfolioResponse = {
  assets: AssetRow[];
  warnings: string[];
};

export type AnalysisResponse = {
  metrics: MetricRow[];
  portfolio_metrics: MetricsSummary;
  benchmark_metrics: MetricsSummary | null;
  missing_tickers: string[];
};

export type SnapshotSummary = {
  id: number;
  portfolio_id: number;
  name: string;
  note: string;
  created_at: string | null;
  updated_at: string | null;
  position_count: number;
};

export type SavedPortfolio = {
  id: number;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  latest_snapshot: {
    id: number;
    name: string;
    created_at: string | null;
    updated_at: string | null;
    position_count: number;
  } | null;
};

export type SnapshotLoadResponse = {
  snapshot: SnapshotSummary;
  portfolio: PortfolioResponse;
  analysis: AnalysisResponse | null;
  evaluation: EvaluationResponse | null;
  evaluation_run: EvaluationRun | null;
};

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: 'include',
    headers: init.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...init
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `요청 실패: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function normalizePortfolioRows(rows: PortfolioRowInput[]) {
  return rows.map((row) => ({
    ...row,
    ticker: String(row.ticker ?? '').trim().toUpperCase(),
    thesis_status: row.thesis_status || 'valid'
  }));
}

export function submitPortfolio(rows: PortfolioRowInput[]) {
  return requestJson<PortfolioResponse>('/api/v1/portfolio/manual', {
    method: 'POST',
    body: JSON.stringify({ rows: normalizePortfolioRows(rows) })
  });
}

export function runAnalysis(payload: {
  period: number | 'YTD' | 'Max';
  as_of_date?: string;
  rf: number;
  bench: string;
  layer_benchmarks?: Record<string, string>;
}, signal?: AbortSignal) {
  return requestJson<AnalysisResponse>('/api/v1/analysis/run', {
    method: 'POST',
    signal,
    body: JSON.stringify(payload)
  });
}

export function runEvaluation(payload: {
  period?: '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max';
  start_date?: string;
  end_date?: string;
  as_of_date?: string;
  bench?: string;
  layer_benchmarks?: Record<string, string>;
}, signal?: AbortSignal) {
  return requestJson<EvaluationResponse>('/api/v1/evaluation/run', {
    method: 'POST',
    signal,
    body: JSON.stringify(payload)
  });
}

export function runSnapshotEvaluation(
  snapshotId: number,
  payload: {
    period?: '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max';
    start_date?: string;
    end_date?: string;
    as_of_date?: string;
    bench?: string;
    layer_benchmarks?: Record<string, string>;
  },
  signal?: AbortSignal
) {
  return requestJson<{
    analysis: AnalysisResponse;
    evaluation: EvaluationResponse;
    evaluation_run: EvaluationRun | null;
  }>(`/api/v1/portfolios/snapshots/${snapshotId}/evaluations/run`, {
    method: 'POST',
    signal,
    body: JSON.stringify(payload)
  });
}

export function saveSnapshotEvaluation(snapshotId: number) {
  return requestJson<{ evaluation: EvaluationResponse; evaluation_run: EvaluationRun }>(
    `/api/v1/portfolios/snapshots/${snapshotId}/evaluations`,
    {
      method: 'POST'
    }
  );
}

export function listPortfolios() {
  return requestJson<{ portfolios: SavedPortfolio[] }>('/api/v1/portfolios', {
    method: 'GET'
  });
}

export function createPortfolio(payload: { name: string; description?: string }) {
  return requestJson<{ portfolio: SavedPortfolio }>('/api/v1/portfolios', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function listSnapshots(portfolioId: number) {
  return requestJson<{ snapshots: SnapshotSummary[] }>(`/api/v1/portfolios/${portfolioId}/snapshots`, {
    method: 'GET'
  });
}

export function saveSnapshot(
  portfolioId: number,
  payload: { name?: string; note?: string; rows?: PortfolioRowInput[]; source_snapshot_id?: number }
) {
  return requestJson<{ snapshot: SnapshotSummary }>(`/api/v1/portfolios/${portfolioId}/snapshots`, {
    method: 'POST',
    body: JSON.stringify({
      ...payload,
      rows: payload.rows ? normalizePortfolioRows(payload.rows) : undefined
    })
  });
}

export function loadSnapshot(snapshotId: number) {
  return requestJson<SnapshotLoadResponse>(`/api/v1/portfolios/snapshots/${snapshotId}/load`, {
    method: 'POST'
  });
}

export function deleteSnapshot(snapshotId: number) {
  return requestJson<{ ok: boolean }>(`/api/v1/portfolios/snapshots/${snapshotId}`, {
    method: 'DELETE'
  });
}

export function updateSnapshot(
  snapshotId: number,
  payload: { name?: string; note?: string }
) {
  return requestJson<{ snapshot: SnapshotSummary }>(`/api/v1/portfolios/snapshots/${snapshotId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload)
  });
}
