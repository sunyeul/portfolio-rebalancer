import type { PortfolioRowInput } from './schemas';

export type AssetRow = {
  ticker: string;
  allocation: number;
  return_total: number | null;
  group: string;
  role: string;
  dca_enabled: boolean;
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
  risk_contribution: number | null;
  return_contribution: number | null;
  weight: number;
  efficiency_score: number | null;
  efficiency_score_prime: number | null;
  dca_intensity_score: number | null;
  return_total: number | null;
  group: string;
  role: string;
  dca_enabled: boolean;
  thesis_status: string;
};

export type MetricsSummary = {
  cagr: number | null;
  volatility: number | null;
  sharpe: number | null;
  max_drawdown?: number | null;
};

export type ProposalRow = {
  ticker: string;
  current_weight_pct: number;
  target_weight_pct: number;
  gap_pct: number;
  efficiency_score: number | null;
  efficiency_score_prime: number | null;
  dca_intensity_score: number | null;
  rc_over_pct: number;
  rc_target_pct: number;
  return_total_pct: number | null;
  group: string;
  role: string;
  dca_enabled: boolean;
  thesis_status: string;
  risk_over: boolean;
  efficiency_good: boolean;
  within_hysteresis: boolean;
  below_min_trade: boolean;
  should_execute: boolean;
  adjusted_gap_pct?: number | null;
};

export type EvaluationResponse = {
  proposal: ProposalRow[];
  ips_actions: Array<Record<string, unknown>>;
  group_summary: Array<Record<string, unknown>>;
  sell_list: ProposalRow[];
  buy_list: ProposalRow[];
  fine_tune_list: ProposalRow[];
  rc_violations: Array<Record<string, unknown>>;
  ips_config_snapshot?: Record<string, unknown> | null;
};

export type ConfigOption = {
  value: string;
  label: string;
  is_active: boolean;
  sort_order: number;
  group_type?: string;
};

export type ConfigOptionsResponse = {
  groups: ConfigOption[];
  roles: ConfigOption[];
  thesis_statuses: ConfigOption[];
};

export type TargetAllocation = {
  group_type: string;
  min: number;
  target: number;
  max: number;
};

export type ActionPriority = {
  action_code: string;
  label: string;
  priority: number;
  is_active: boolean;
};

export type IpsRule = {
  key: string;
  value: unknown;
};

export type IpsConfigResponse = {
  target_allocations: TargetAllocation[];
  action_priorities: ActionPriority[];
  rules: IpsRule[];
  ips_config: Record<string, unknown>;
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
  created_at: string;
  position_count: number;
  has_analysis: boolean;
  has_evaluation: boolean;
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
    created_at: string;
    position_count: number;
  } | null;
};

export type SnapshotLoadResponse = {
  snapshot: SnapshotSummary;
  portfolio: PortfolioResponse;
  analysis: AnalysisResponse | null;
  evaluation: EvaluationResponse | null;
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

export function submitPortfolio(rows: PortfolioRowInput[]) {
  return requestJson<PortfolioResponse>('/api/v1/portfolio/manual', {
    method: 'POST',
    body: JSON.stringify({ rows })
  });
}

export function uploadPortfolioCsv(file: File) {
  const body = new FormData();
  body.append('file', file);
  return requestJson<PortfolioResponse>('/api/v1/portfolio/csv', {
    method: 'POST',
    body
  });
}

export function runAnalysis(payload: {
  period: number | 'YTD' | 'Max';
  rf: number;
  bench: string;
  momentum_weight: number;
}) {
  return requestJson<AnalysisResponse>('/api/v1/analysis/run', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function runEvaluation(payload: {
  rc_over_thresh_pct: number;
  e_thresh: number;
  target_weights?: Record<string, number>;
}) {
  return requestJson<EvaluationResponse>('/api/v1/evaluation/run', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function csvDownloadUrl(type: 'metrics' | 'proposal' | 'ips_actions' | 'group_summary') {
  return `/api/v1/evaluation/download-csv?type=${type}`;
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
  payload: { name?: string; note?: string; rows?: PortfolioRowInput[] }
) {
  return requestJson<{ snapshot: SnapshotSummary }>(`/api/v1/portfolios/${portfolioId}/snapshots`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function updateSnapshot(
  snapshotId: number,
  payload: { name?: string; note?: string; rows?: PortfolioRowInput[] }
) {
  return requestJson<{ snapshot: SnapshotSummary }>(`/api/v1/portfolios/snapshots/${snapshotId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload)
  });
}

export function getSnapshot(snapshotId: number) {
  return requestJson<SnapshotLoadResponse>(`/api/v1/portfolios/snapshots/${snapshotId}`, {
    method: 'GET'
  });
}

export function deleteSnapshot(snapshotId: number) {
  return requestJson<{ ok: true }>(`/api/v1/portfolios/snapshots/${snapshotId}`, {
    method: 'DELETE'
  });
}

export function loadSnapshot(snapshotId: number) {
  return requestJson<SnapshotLoadResponse>(`/api/v1/portfolios/snapshots/${snapshotId}/load`, {
    method: 'POST'
  });
}

export function getConfigOptions() {
  return requestJson<ConfigOptionsResponse>('/api/v1/config/options', {
    method: 'GET'
  });
}

export function getIpsConfig() {
  return requestJson<IpsConfigResponse>('/api/v1/config/ips', {
    method: 'GET'
  });
}

export function saveConfigOption(
  table: 'groups' | 'roles' | 'thesis_statuses',
  payload: {
    code: string;
    label: string;
    sort_order?: number;
    is_active?: boolean;
    group_type?: string;
  }
) {
  return requestJson<{ option: ConfigOption }>(`/api/v1/config/${table}`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function setConfigOptionActive(
  table: 'groups' | 'roles' | 'thesis_statuses',
  code: string,
  is_active: boolean
) {
  return requestJson<{ option: ConfigOption }>(`/api/v1/config/${table}/${code}/active`, {
    method: 'PATCH',
    body: JSON.stringify({ is_active })
  });
}

export function saveTargetAllocations(rows: TargetAllocation[]) {
  return requestJson<{ target_allocations: TargetAllocation[] }>('/api/v1/config/ips/target-allocations', {
    method: 'PUT',
    body: JSON.stringify(rows)
  });
}

export function saveActionPriorities(rows: ActionPriority[]) {
  return requestJson<{ action_priorities: ActionPriority[] }>('/api/v1/config/ips/action-priorities', {
    method: 'PUT',
    body: JSON.stringify(rows)
  });
}

export function saveIpsRules(rows: IpsRule[]) {
  return requestJson<{ rules: IpsRule[] }>('/api/v1/config/ips/rules', {
    method: 'PUT',
    body: JSON.stringify(rows)
  });
}
