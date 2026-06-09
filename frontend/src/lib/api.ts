import type { PortfolioRowInput } from './schemas';

export type AssetRow = {
  ticker: string;
  allocation: number;
  return_total: number | null;
  group: string;
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
  data_start: string | null;
  data_end: string | null;
  observation_count: number | null;
  missing_ratio: number | null;
  risk_contribution: number | null;
  return_contribution: number | null;
  weight: number;
  efficiency_score: number | null;
  return_total: number | null;
  group: string;
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
  rc_gap_pct: number;
  rc_over_pct: number;
  rc_target_pct: number;
  return_total_pct: number | null;
  ips_fit_score: number | null;
  ips_fit_band: string | null;
  ips_score_role: number | null;
  ips_score_allocation: number | null;
  ips_score_thesis: number | null;
  ips_score_risk: number | null;
  ips_score_action: number | null;
  ips_score_efficiency: number | null;
  ips_score_data_quality: number | null;
  group: string;
  dca_enabled: boolean;
  thesis_status: string;
  risk_over: boolean;
  efficiency_warning: boolean;
  within_hysteresis: boolean;
  below_min_trade: boolean;
  numeric_candidate: boolean;
  reference_trade_pct: number;
  should_execute: boolean;
  suggested_trade_pct: number;
  action_reason: string;
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
  playbook?: PlaybookRecommendation | null;
};

export type PlaybookRecommendation = {
  code: DecisionContext;
  label: string;
  confidence: 'high' | 'medium' | 'low';
  reasons: string[];
  steps: string[];
  manual_context: DecisionContext;
  is_manual_override: boolean;
};

export type CounterfactualScenario =
  | 'current_proposal'
  | 'core_reinforcement'
  | 'pause_satellite_new_buys'
  | 'dca_shift_to_core';

export type DecisionContext =
  | 'regular_review'
  | 'market_correction'
  | 'sharp_drop_review'
  | 'rebalance_review';

export type CounterfactualAssetDelta = {
  ticker: string;
  baseline_weight: number;
  scenario_weight: number;
  delta_weight_pct: number;
  baseline_risk_contribution: number;
  scenario_risk_contribution: number;
  delta_risk_contribution_pct: number;
  baseline_gap_pct: number;
  scenario_gap_pct: number;
};

export type CounterfactualResponse = {
  baseline: {
    weights: Record<string, number>;
    risk_contributions: Record<string, number>;
    target_gaps_pct: Record<string, number>;
    group_weights: Record<string, number>;
    actions: Record<string, string>;
    action_labels: Record<string, string>;
  };
  scenario: {
    weights: Record<string, number>;
    risk_contributions: Record<string, number>;
    target_gaps_pct: Record<string, number>;
    group_weights: Record<string, number>;
    actions: Record<string, string>;
    action_labels: Record<string, string>;
  };
  deltas: {
    assets: CounterfactualAssetDelta[];
    groups: Record<string, { baseline: number; scenario: number; delta_pct: number }>;
  };
  action_changes: Array<{
    ticker: string;
    baseline_action: string;
    scenario_action: string;
    baseline_label: string;
    scenario_label: string;
  }>;
  warnings: string[];
  interpretation: string[];
};

export type BacktestStrategy =
  | 'current_ips'
  | 'core_first_dca'
  | 'pause_overweight_satellite'
  | 'return_chasing_reference';

export type BacktestStrategySummary = {
  strategy: string;
  strategy_label: string;
  decision_context: string;
  cagr: number | null;
  volatility: number | null;
  max_drawdown: number | null;
  sharpe: number | null;
  ips_violation_count: number;
  satellite_over_periods: number;
  risk_contribution_over_count: number;
  adjustment_count: number;
  avg_core_gap: number;
  months_to_core_target_recovery: number | null;
};

export type BacktestResponse = {
  strategy_summaries: BacktestStrategySummary[];
  ips_fit_summary: Array<Record<string, unknown>>;
  performance_summary: Array<Record<string, unknown>>;
  timeline: Array<Record<string, unknown>>;
};

export type ConfigOption = {
  value: string;
  label: string;
  is_active: boolean;
  sort_order: number;
};

export type ConfigOptionsResponse = {
  thesis_statuses: ConfigOption[];
};

export type TargetAllocation = {
  group: string;
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

export type CurrentStateResponse = {
  portfolio: PortfolioResponse;
  analysis: AnalysisResponse | null;
  evaluation: EvaluationResponse | null;
  updated_at: string;
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
}) {
  return requestJson<AnalysisResponse>('/api/v1/analysis/run', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function runEvaluation(payload: {
  rc_over_thresh_pct: number;
  e_thresh: number;
  decision_context: DecisionContext;
  target_weights?: Record<string, number>;
}) {
  return requestJson<EvaluationResponse>('/api/v1/evaluation/run', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function runCounterfactual(payload: {
  scenario: CounterfactualScenario;
  rc_over_thresh_pct: number;
  e_thresh: number;
  decision_context: DecisionContext;
}) {
  return requestJson<CounterfactualResponse>('/api/v1/simulation/counterfactual', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function runBacktest(payload: {
  strategies: BacktestStrategy[];
  frequency: 'monthly';
  decision_context: DecisionContext;
  rf?: number;
}) {
  return requestJson<BacktestResponse>('/api/v1/simulation/backtest', {
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

export function getCurrentState(portfolioId: number) {
  return requestJson<CurrentStateResponse>(`/api/v1/portfolios/${portfolioId}/current-state`, {
    method: 'GET'
  });
}

export function saveCurrentState(portfolioId: number) {
  return requestJson<CurrentStateResponse>(`/api/v1/portfolios/${portfolioId}/current-state`, {
    method: 'POST'
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
