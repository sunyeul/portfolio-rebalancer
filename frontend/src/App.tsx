import type { ColumnDef } from '@tanstack/react-table';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  Edit3,
  FileUp,
  FolderOpen,
  LineChart,
  Loader2,
  Play,
  Plus,
  RefreshCcw,
  Save,
  ShieldCheck,
  Trash2,
  X
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

import { DataTable } from './components/DataTable';
import { MetricCard } from './components/MetricCard';
import {
  type AnalysisResponse,
  type ActionPriority,
  type AssetRow,
  type BacktestResponse,
  type BacktestStrategy,
  type ConfigOption,
  type CounterfactualResponse,
  type CounterfactualScenario,
  type EvaluationResponse,
  type IpsRule,
  type MetricRow,
  type ProposalRow,
  type SnapshotLoadResponse,
  type SnapshotSummary,
  type TargetAllocation,
  createPortfolio,
  csvDownloadUrl,
  deleteSnapshot,
  getConfigOptions,
  getCurrentState,
  getIpsConfig,
  getSnapshot,
  listPortfolios,
  listSnapshots,
  loadSnapshot,
  runAnalysis,
  runBacktest,
  runCounterfactual,
  runEvaluation,
  saveCurrentState,
  saveIpsRules,
  saveSnapshot,
  saveTargetAllocations,
  submitPortfolio,
  updateSnapshot,
  uploadPortfolioCsv
} from './lib/api';
import { blankRow, parsePortfolioText } from './lib/parser';
import { type PortfolioRowInput, type SettingsValues, settingsSchema } from './lib/schemas';

const sampleText = 'VOO 40\nQQQ 25\nSOXX 15\nUFO 3\nIONQ 2';
const DEFAULT_RF_PCT = 2.5;
const DEFAULT_BENCHMARK = 'SPY:80,QQQ:20';
const fixedGroupOptions = [
  { value: 'core', label: '코어' },
  { value: 'satellite', label: '위성' },
  { value: 'cash', label: '현금' },
  { value: 'unclassified', label: '미분류' }
] as const;
const decisionContextOptions = [
  { value: 'regular_review', label: '일반 점검' },
  { value: 'market_correction', label: '시장 조정 대응' },
  { value: 'sharp_drop_review', label: '급락 후 추매 검토' },
  { value: 'rebalance_review', label: '비중 리밸런싱 점검' }
] as const;
const counterfactualScenarioOptions: Array<{ value: CounterfactualScenario; label: string; description: string }> = [
  { value: 'core_reinforcement', label: '코어 보강', description: '부족한 코어 목표 비중을 정기매수 조정으로 더 우선합니다.' },
  { value: 'pause_satellite_new_buys', label: '위성 신규매수 중단', description: '위성의 추가 매수분만 막고, 초과 위성의 감액 검토는 유지합니다.' },
  { value: 'dca_shift_to_core', label: '정기매수 코어 이동', description: '위성으로 향할 신규 정기매수 여력을 코어 쪽으로 돌려봅니다.' }
];
const backtestStrategyOptions: Array<{ value: BacktestStrategy; label: string; description: string }> = [
  { value: 'current_ips', label: '현재 IPS 유지', description: '현재 IPS 목표와 평가 모드를 그대로 적용하는 기준 정책입니다.' },
  { value: 'core_first_dca', label: '코어 부족분 우선', description: '코어가 목표보다 낮으면 정기매수 여력을 코어 회복에 먼저 둡니다.' },
  { value: 'pause_overweight_satellite', label: '위성 초과 신규매수 중단', description: '위성이 IPS 상한을 넘는 기간에는 위성 신규매수를 막습니다.' },
  { value: 'return_chasing_reference', label: '수익률 중심 참고', description: '최근 수익률을 우선한 비교용 정책이며 IPS 적합성 판단의 반례로 봅니다.' }
];
type AppView = 'workbench' | 'settings';

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function optionLabel(options: ConfigOption[], value: string | null | undefined) {
  if (!value) return '미정';
  const option = options.find((item) => item.value === value);
  if (!option) return value;
  return option.is_active ? option.label : `${option.label} (비활성)`;
}

function groupLabel(value: string | null | undefined) {
  const option = fixedGroupOptions.find((item) => item.value === value);
  return option?.label ?? '미분류';
}

function pct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined) return 'N/A';
  return `${(fromUnit ? value * 100 : value).toFixed(2)}%`;
}

function signedPct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined) return 'N/A';
  const numeric = fromUnit ? value * 100 : value;
  const sign = numeric > 0 ? '+' : '';
  return `${sign}${numeric.toFixed(2)}%`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined) return 'N/A';
  return value.toFixed(2);
}

function valueFromRecord(row: Record<string, unknown>, key: string) {
  const value = row[key];
  if (typeof value === 'number') return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function textFromRecord(row: Record<string, unknown>, key: string) {
  const value = row[key];
  return typeof value === 'string' ? value : 'unknown';
}

function nullableNumberColumn<T>(
  id: string,
  header: string,
  getValue: (row: T) => number | null | undefined,
  formatValue: (value: number | null | undefined) => string
): ColumnDef<T> {
  return {
    id,
    header,
    accessorFn: (row) => getValue(row) ?? undefined,
    cell: ({ row }) => formatValue(getValue(row.original)),
    sortUndefined: 'last'
  };
}

function shortDate(value: string | null | undefined) {
  if (!value) return '없음';
  return new Date(value).toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function rowsFromAssets(assets: AssetRow[]): PortfolioRowInput[] {
  return assets.map((asset) => ({
    ticker: asset.ticker,
    allocation: asset.allocation,
    return_total: asset.return_total === null ? '' : Number((asset.return_total * 100).toFixed(4)),
    group: asset.group,
    dca_enabled: asset.dca_enabled,
    thesis_status: asset.thesis_status
  }));
}

function rowInputLine(row: PortfolioRowInput) {
  return [row.ticker, row.allocation].filter(Boolean).join(' ');
}

function rowsSignature(rows: PortfolioRowInput[]) {
  return JSON.stringify(
    rows.map((row) => ({
      ticker: row.ticker,
      allocation: row.allocation,
      return_total: row.return_total ?? '',
      group: row.group ?? '',
      dca_enabled: row.dca_enabled ?? true,
      thesis_status: row.thesis_status ?? ''
    }))
  );
}

export function App() {
  const queryClient = useQueryClient();
  const [activeView, setActiveView] = useState<AppView>('workbench');
  const [text, setText] = useState(sampleText);
  const [rows, setRows] = useState<PortfolioRowInput[]>(() => parsePortfolioText(sampleText));
  const [portfolio, setPortfolio] = useState<AssetRow[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [counterfactualScenario, setCounterfactualScenario] = useState<CounterfactualScenario>('core_reinforcement');
  const [counterfactual, setCounterfactual] = useState<CounterfactualResponse | null>(null);
  const [backtestStrategies, setBacktestStrategies] = useState<BacktestStrategy[]>([
    'current_ips',
    'core_first_dca',
    'pause_overweight_satellite'
  ]);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [runCounterfactualAfterEvaluation, setRunCounterfactualAfterEvaluation] = useState(true);
  const [runBacktestAfterEvaluation, setRunBacktestAfterEvaluation] = useState(false);
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [snapshotName, setSnapshotName] = useState('');
  const [editingSnapshotId, setEditingSnapshotId] = useState<number | null>(null);
  const [editingSnapshotName, setEditingSnapshotName] = useState('');
  const [editingSnapshotNote, setEditingSnapshotNote] = useState('');
  const [editingSnapshotRows, setEditingSnapshotRows] = useState<PortfolioRowInput[]>([]);
  const [deletingSnapshotId, setDeletingSnapshotId] = useState<number | null>(null);
  const [appliedRowsSignature, setAppliedRowsSignature] = useState(() => rowsSignature(parsePortfolioText(sampleText)));
  const [targetAllocationRows, setTargetAllocationRows] = useState<TargetAllocation[]>([]);
  const [actionPriorityRows, setActionPriorityRows] = useState<ActionPriority[]>([]);
  const [rulesJson, setRulesJson] = useState('[]');

  const configOptionsQuery = useQuery({
    queryKey: ['config-options'],
    queryFn: getConfigOptions
  });

  const ipsConfigQuery = useQuery({
    queryKey: ['ips-config'],
    queryFn: getIpsConfig
  });

  const portfoliosQuery = useQuery({
    queryKey: ['portfolios'],
    queryFn: listPortfolios
  });

  const savedPortfolios = portfoliosQuery.data?.portfolios ?? [];
  const activePortfolio = savedPortfolios.find((item) => item.id === selectedPortfolioId) ?? null;

  const snapshotsQuery = useQuery({
    queryKey: ['portfolio-snapshots', selectedPortfolioId],
    queryFn: () => listSnapshots(selectedPortfolioId as number),
    enabled: selectedPortfolioId !== null
  });
  const currentStateQuery = useQuery({
    queryKey: ['portfolio-current-state', selectedPortfolioId],
    queryFn: () => getCurrentState(selectedPortfolioId as number),
    enabled: selectedPortfolioId !== null,
    retry: false
  });
  const savedSnapshots = snapshotsQuery.data?.snapshots ?? [];
  const thesisStatusOptions = configOptionsQuery.data?.thesis_statuses ?? [];
  const activeThesisStatusOptions = thesisStatusOptions.filter((option) => option.is_active);

  useEffect(() => {
    if (selectedPortfolioId === null && savedPortfolios.length > 0) {
      setSelectedPortfolioId(savedPortfolios[0].id);
    }
  }, [savedPortfolios, selectedPortfolioId]);

  useEffect(() => {
    if (!ipsConfigQuery.data) return;
    setTargetAllocationRows(ipsConfigQuery.data.target_allocations);
    setActionPriorityRows(ipsConfigQuery.data.action_priorities);
    setRulesJson(JSON.stringify(ipsConfigQuery.data.rules, null, 2));
  }, [ipsConfigQuery.data]);

  useEffect(() => {
    if (!currentStateQuery.data) return;
    const nextRows = rowsFromAssets(currentStateQuery.data.portfolio.assets);
    setRows(nextRows);
    setText(nextRows.map(rowInputLine).join('\n'));
    setAppliedRowsSignature(rowsSignature(nextRows));
    setPortfolio(currentStateQuery.data.portfolio.assets);
    setAnalysis(currentStateQuery.data.analysis);
    setEvaluation(currentStateQuery.data.evaluation);
    setCounterfactual(null);
    setBacktest(null);
  }, [currentStateQuery.data]);

  const { register, watch } = useForm<SettingsValues>({
    defaultValues: {
      periodMode: 'months',
      months: 12,
      rfPct: DEFAULT_RF_PCT,
      bench: DEFAULT_BENCHMARK,
      rcOverThreshPct: 1.5,
      eThresh: 0.5,
      decisionContext: 'regular_review'
    }
  });
  const settings = watch();

  async function persistCurrentState() {
    if (selectedPortfolioId === null) return;
    try {
      await saveCurrentState(selectedPortfolioId);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-current-state', selectedPortfolioId] })
      ]);
    } catch {
      // Current-state persistence is best-effort so the workflow step stays usable.
    }
  }

  async function restoreSnapshotToWorkbench(data: SnapshotLoadResponse) {
    const nextRows = rowsFromAssets(data.portfolio.assets);
    setRows(nextRows);
    setText(nextRows.map(rowInputLine).join('\n'));
    setAppliedRowsSignature(rowsSignature(nextRows));
    setPortfolio(data.portfolio.assets);
    setAnalysis(data.analysis);
    setEvaluation(data.evaluation);
    setCounterfactual(null);
    setBacktest(null);
    setSelectedPortfolioId(data.snapshot.portfolio_id);
    await queryClient.invalidateQueries({
      queryKey: ['portfolio-current-state', data.snapshot.portfolio_id]
    });
  }

  const portfolioMutation = useMutation({
    mutationFn: submitPortfolio,
    onSuccess: async (data) => {
      const nextRows = rowsFromAssets(data.assets);
      setRows(nextRows);
      setAppliedRowsSignature(rowsSignature(nextRows));
      setPortfolio(data.assets);
      setAnalysis(null);
      setEvaluation(null);
      setCounterfactual(null);
      setBacktest(null);
      await persistCurrentState();
    }
  });

  const csvMutation = useMutation({
    mutationFn: uploadPortfolioCsv,
    onSuccess: async (data) => {
      const nextRows = rowsFromAssets(data.assets);
      setRows(nextRows);
      setText(nextRows.map(rowInputLine).join('\n'));
      setAppliedRowsSignature(rowsSignature(nextRows));
      setPortfolio(data.assets);
      setAnalysis(null);
      setEvaluation(null);
      await persistCurrentState();
    }
  });

  const analysisMutation = useMutation({
    mutationFn: runAnalysis,
    onSuccess: async (data) => {
      setAnalysis(data);
      setEvaluation(null);
      setCounterfactual(null);
      setBacktest(null);
      await persistCurrentState();
    }
  });

  const evaluationMutation = useMutation({
    mutationFn: runEvaluation,
    onSuccess: async (data) => {
      setEvaluation(data);
      setCounterfactual(null);
      setBacktest(null);
      const parsedSettings = settingsSchema.parse(settings);
      if (runCounterfactualAfterEvaluation) {
        counterfactualMutation.mutate({
          scenario: counterfactualScenario,
          rc_over_thresh_pct: parsedSettings.rcOverThreshPct,
          e_thresh: parsedSettings.eThresh,
          decision_context: parsedSettings.decisionContext
        });
      }
      if (runBacktestAfterEvaluation && backtestStrategies.length > 0) {
        backtestMutation.mutate({
          strategies: backtestStrategies,
          frequency: 'monthly',
          decision_context: parsedSettings.decisionContext,
          rf: parsedSettings.rfPct / 100
        });
      }
      await persistCurrentState();
    }
  });

  const counterfactualMutation = useMutation({
    mutationFn: runCounterfactual,
    onSuccess: (data) => {
      setCounterfactual(data);
    }
  });

  const backtestMutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (data) => {
      setBacktest(data);
    }
  });

  const createPortfolioMutation = useMutation({
    mutationFn: createPortfolio,
    onSuccess: async (data) => {
      setNewPortfolioName('');
      setSelectedPortfolioId(data.portfolio.id);
      await queryClient.invalidateQueries({ queryKey: ['portfolios'] });
    }
  });

  const saveSnapshotMutation = useMutation({
    mutationFn: async () => {
      if (selectedPortfolioId === null) throw new Error('저장할 포트폴리오를 선택해주세요.');
      if ((rowsDirty || !portfolio.length) && !validRows.length) {
        throw new Error('저장할 포트폴리오 입력이 없습니다.');
      }
      if (rowsDirty || !portfolio.length) {
        await portfolioMutation.mutateAsync(validRows);
      }
      return saveSnapshot(selectedPortfolioId, {
        name: snapshotName || undefined
      });
    },
    onSuccess: async () => {
      setSnapshotName('');
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-snapshots', selectedPortfolioId] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-current-state', selectedPortfolioId] })
      ]);
    }
  });

  const updateSnapshotMutation = useMutation({
    mutationFn: ({
      snapshotId,
      payload
    }: {
      snapshotId: number;
      payload: { name?: string; note?: string; rows?: PortfolioRowInput[] };
    }) => updateSnapshot(snapshotId, payload),
    onSuccess: async (_data, variables) => {
      setEditingSnapshotId(null);
      setEditingSnapshotName('');
      setEditingSnapshotNote('');
      setEditingSnapshotRows([]);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-snapshots', selectedPortfolioId] })
      ]);
      if (variables.payload.rows !== undefined) {
        const snapshot = await loadSnapshot(variables.snapshotId);
        await restoreSnapshotToWorkbench(snapshot);
      }
    }
  });

  const editSnapshotMutation = useMutation({
    mutationFn: getSnapshot,
    onSuccess: (data) => {
      setEditingSnapshotId(data.snapshot.id);
      setEditingSnapshotName(data.snapshot.name);
      setEditingSnapshotNote(data.snapshot.note);
      setEditingSnapshotRows(rowsFromAssets(data.portfolio.assets));
    }
  });

  const deleteSnapshotMutation = useMutation({
    mutationFn: deleteSnapshot,
    onSuccess: async () => {
      setEditingSnapshotId(null);
      setEditingSnapshotName('');
      setEditingSnapshotNote('');
      setEditingSnapshotRows([]);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-snapshots', selectedPortfolioId] })
      ]);
    },
    onSettled: () => {
      setDeletingSnapshotId(null);
    }
  });

  const loadSnapshotMutation = useMutation({
    mutationFn: loadSnapshot,
    onSuccess: async (data) => {
      await restoreSnapshotToWorkbench(data);
    }
  });

  const saveTargetsMutation = useMutation({
    mutationFn: () => saveTargetAllocations(targetAllocationRows),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ips-config'] })
  });

  const saveRulesMutation = useMutation({
    mutationFn: () => saveIpsRules(JSON.parse(rulesJson) as IpsRule[]),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ips-config'] })
  });

  const validRows = rows.filter((row) => row.ticker && row.allocation !== '');
  const totalAllocation = validRows.reduce(
    (sum, row) => sum + (Number.parseFloat(String(row.allocation)) || 0),
    0
  );
  const duplicateTickers = validRows
    .map((row) => row.ticker)
    .filter((ticker, index, tickers) => tickers.indexOf(ticker) !== index);
  const duplicateCount = new Set(duplicateTickers).size;
  const rowsDirty = rowsSignature(validRows) !== appliedRowsSignature;
  const shouldApplyRowsBeforeAnalysis = rowsDirty || !portfolio.length;
  const snapshotActionPending =
    editSnapshotMutation.isPending ||
    loadSnapshotMutation.isPending ||
    updateSnapshotMutation.isPending ||
    deleteSnapshotMutation.isPending;

  const assetColumns = useMemo<ColumnDef<AssetRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'allocation', header: '입력 비중', cell: ({ row }) => pct(row.original.allocation, false) },
      { accessorKey: 'weight', header: '정규화 비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'group', header: '그룹', cell: ({ row }) => groupLabel(row.original.group) }
    ],
    []
  );

  const metricColumns = useMemo<ColumnDef<MetricRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'weight', header: '비중', cell: ({ row }) => pct(row.original.weight) },
      nullableNumberColumn('cagr', 'CAGR', (row) => row.cagr, pct),
      nullableNumberColumn('volatility', '변동성', (row) => row.volatility, pct),
      nullableNumberColumn('sharpe', '샤프', (row) => row.sharpe, num),
      nullableNumberColumn('information_ratio', 'IR', (row) => row.information_ratio, num),
      nullableNumberColumn('beta', '베타', (row) => row.beta, num),
      nullableNumberColumn('alpha', '알파', (row) => row.alpha, pct),
      nullableNumberColumn('risk_contribution', '위험기여도', (row) => row.risk_contribution, pct),
      nullableNumberColumn('return_total', '기간 수익률', (row) => row.return_total, pct),
      nullableNumberColumn('efficiency_score', 'E', (row) => row.efficiency_score, num),
      { accessorKey: 'data_start', header: '시작일' },
      { accessorKey: 'data_end', header: '종료일' },
      nullableNumberColumn('observation_count', '관측수', (row) => row.observation_count, (value) => (value === null || value === undefined ? 'N/A' : String(value))),
      nullableNumberColumn('missing_ratio', '결측률', (row) => row.missing_ratio, pct)
    ],
    []
  );

  const proposalColumns = useMemo<ColumnDef<ProposalRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'current_weight_pct', header: '현재', cell: ({ row }) => pct(row.original.current_weight_pct, false) },
      { accessorKey: 'target_weight_pct', header: '목표', cell: ({ row }) => pct(row.original.target_weight_pct, false) },
      { accessorKey: 'gap_pct', header: '갭', cell: ({ row }) => pct(row.original.gap_pct, false) },
      { accessorKey: 'target_preference_score', header: '목표점수', cell: ({ row }) => num(row.original.target_preference_score) },
      { accessorKey: 'suggested_trade_pct', header: '최종조정', cell: ({ row }) => pct(row.original.suggested_trade_pct, false) },
      { accessorKey: 'should_execute', header: '최종실행', cell: ({ row }) => (row.original.should_execute ? '실행' : '보류') },
      { accessorKey: 'action_reason', header: '사유' }
    ],
    []
  );

  const actionColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'action_label', header: '액션' },
      { accessorKey: 'decision_summary', header: '판단 요약' },
      { accessorKey: 'next_step', header: '다음 행동' }
    ],
    []
  );

  const counterfactualDeltaColumns = useMemo<ColumnDef<CounterfactualResponse['deltas']['assets'][number]>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'baseline_weight', header: '기준 비중', cell: ({ row }) => pct(row.original.baseline_weight) },
      { accessorKey: 'scenario_weight', header: '정책 후 비중', cell: ({ row }) => pct(row.original.scenario_weight) },
      { accessorKey: 'delta_weight_pct', header: '비중 변화', cell: ({ row }) => pct(row.original.delta_weight_pct, false) },
      { accessorKey: 'delta_risk_contribution_pct', header: 'RC 변화', cell: ({ row }) => pct(row.original.delta_risk_contribution_pct, false) },
      { accessorKey: 'baseline_gap_pct', header: '기준 갭', cell: ({ row }) => pct(row.original.baseline_gap_pct, false) },
      { accessorKey: 'scenario_gap_pct', header: '정책 후 갭', cell: ({ row }) => pct(row.original.scenario_gap_pct, false) }
    ],
    []
  );

  const actionChangeColumns = useMemo<ColumnDef<CounterfactualResponse['action_changes'][number]>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'baseline_label', header: '기준 액션' },
      { accessorKey: 'scenario_label', header: '정책 후 액션' }
    ],
    []
  );

  const backtestColumns = useMemo<ColumnDef<BacktestResponse['strategy_summaries'][number]>[]>(
    () => [
      { accessorKey: 'strategy_label', header: '정책' },
      { accessorKey: 'ips_violation_count', header: 'IPS 위반' },
      { accessorKey: 'satellite_over_periods', header: '위성 초과' },
      { accessorKey: 'risk_contribution_over_count', header: 'RC 초과' },
      { accessorKey: 'adjustment_count', header: '조정 빈도' },
      { accessorKey: 'avg_core_gap', header: '평균 코어 갭', cell: ({ row }) => pct(row.original.avg_core_gap) },
      nullableNumberColumn('cagr', 'CAGR', (row) => row.cagr, pct),
      nullableNumberColumn('volatility', '변동성', (row) => row.volatility, pct),
      nullableNumberColumn('max_drawdown', 'MDD', (row) => row.max_drawdown, pct),
      nullableNumberColumn('sharpe', 'Sharpe', (row) => row.sharpe, num)
    ],
    []
  );

  function syncText(nextText: string) {
    setText(nextText);
    setRows(parsePortfolioText(nextText));
  }

  function updateRow(index: number, field: keyof PortfolioRowInput, value: string | boolean) {
    setRows((current) =>
      current.map((row, rowIndex) =>
        rowIndex === index
          ? { ...row, [field]: field === 'ticker' ? String(value).toUpperCase() : value }
          : row
      )
    );
  }

  function submitPortfolioFile(file: File | undefined) {
    if (file) csvMutation.mutate(file);
  }

  function runCurrentAnalysis() {
    const parsedSettings = settingsSchema.parse(settings);
    analysisMutation.mutate({
      period:
        parsedSettings.periodMode === 'months'
          ? parsedSettings.months
          : parsedSettings.periodMode,
      rf: parsedSettings.rfPct / 100,
      bench: parsedSettings.bench
    });
  }

  async function applyRowsAndRunAnalysis() {
    const parsedSettings = settingsSchema.parse(settings);
    await portfolioMutation.mutateAsync(validRows);
    await analysisMutation.mutateAsync({
      period:
        parsedSettings.periodMode === 'months'
          ? parsedSettings.months
          : parsedSettings.periodMode,
      rf: parsedSettings.rfPct / 100,
      bench: parsedSettings.bench
    });
  }

  function runCurrentEvaluation() {
    const parsedSettings = settingsSchema.parse(settings);
    evaluationMutation.mutate({
      rc_over_thresh_pct: parsedSettings.rcOverThreshPct,
      e_thresh: parsedSettings.eThresh,
      decision_context: parsedSettings.decisionContext
    });
  }

  function runCurrentCounterfactual() {
    const parsedSettings = settingsSchema.parse(settings);
    counterfactualMutation.mutate({
      scenario: counterfactualScenario,
      rc_over_thresh_pct: parsedSettings.rcOverThreshPct,
      e_thresh: parsedSettings.eThresh,
      decision_context: parsedSettings.decisionContext
    });
  }

  function toggleBacktestStrategy(strategy: BacktestStrategy) {
    setBacktestStrategies((current) =>
      current.includes(strategy)
        ? current.filter((item) => item !== strategy)
        : [...current, strategy]
    );
  }

  function runCurrentBacktest() {
    const parsedSettings = settingsSchema.parse(settings);
    backtestMutation.mutate({
      strategies: backtestStrategies,
      frequency: 'monthly',
      decision_context: parsedSettings.decisionContext,
      rf: parsedSettings.rfPct / 100
    });
  }

  function createNamedPortfolio() {
    const name = newPortfolioName.trim();
    if (!name) return;
    createPortfolioMutation.mutate({ name });
  }

  function saveCurrentSnapshot() {
    saveSnapshotMutation.mutate();
  }

  function startEditingSnapshot(snapshot: SnapshotSummary) {
    editSnapshotMutation.mutate(snapshot.id);
  }

  function cancelEditingSnapshot() {
    setEditingSnapshotId(null);
    setEditingSnapshotName('');
    setEditingSnapshotNote('');
    setEditingSnapshotRows([]);
  }

  function updateEditingSnapshotRow(index: number, field: keyof PortfolioRowInput, value: string | boolean) {
    setEditingSnapshotRows((current) =>
      current.map((row, rowIndex) =>
        rowIndex === index
          ? { ...row, [field]: field === 'ticker' ? String(value).toUpperCase() : value }
          : row
      )
    );
  }

  function saveEditedSnapshot() {
    if (editingSnapshotId === null) return;
    updateSnapshotMutation.mutate({
      snapshotId: editingSnapshotId,
      payload: {
        name: editingSnapshotName,
        note: editingSnapshotNote,
        rows: editingSnapshotRows
      }
    });
  }

  function removeSnapshot(snapshot: SnapshotSummary) {
    if (!window.confirm('이 스냅샷을 삭제할까요?')) return;
    setDeletingSnapshotId(snapshot.id);
    deleteSnapshotMutation.mutate(snapshot.id);
  }

  function updateTargetAllocation(index: number, field: keyof TargetAllocation, value: string) {
    setTargetAllocationRows((current) =>
      current.map((row, rowIndex) =>
        rowIndex === index
          ? {
              ...row,
              [field]: field === 'group' ? value : Number(value)
            }
          : row
      )
    );
  }

  const chartData = analysis?.metrics.map((row) => ({
    ticker: row.ticker,
    weight: Number(((row.weight ?? 0) * 100).toFixed(2)),
    risk: Number(((row.risk_contribution ?? 0) * 100).toFixed(2))
  })) ?? [];
  const riskBudgetData = evaluation?.proposal.map((row) => ({
    ticker: row.ticker,
    risk: Number((row.rc_target_pct + row.rc_gap_pct).toFixed(2)),
    target: Number(row.rc_target_pct.toFixed(2))
  })) ?? [];
  const actionChartData = evaluation?.proposal.map((row) => ({
    ticker: row.ticker,
    suggestedTrade: Number(row.suggested_trade_pct.toFixed(2))
  })) ?? [];
  const efficiencyRiskData = evaluation?.proposal.map((row) => ({
    ticker: row.ticker,
    efficiency: Number((row.efficiency_score ?? 0).toFixed(2)),
    rcGap: Number(row.rc_gap_pct.toFixed(2))
  })) ?? [];
  const groupAllocationData = evaluation?.group_summary.length
    ? [
        evaluation.group_summary.reduce<Record<string, string | number>>(
          (acc, row) => {
            const group = textFromRecord(row, 'group');
            acc[groupLabel(group)] = Number((valueFromRecord(row, 'weight') * 100).toFixed(2));
            return acc;
          },
          { label: '비중' }
        )
      ]
    : [];
  const selectedCounterfactualOption =
    counterfactualScenarioOptions.find((option) => option.value === counterfactualScenario) ??
    counterfactualScenarioOptions[0];
  const selectedBacktestOptions = backtestStrategyOptions.filter((option) =>
    backtestStrategies.includes(option.value)
  );
  const counterfactualReadout = counterfactual
    ? [
        `코어 ${signedPct(counterfactual.deltas.groups.core?.delta_pct ?? 0, false)}, 위성 ${signedPct(counterfactual.deltas.groups.satellite?.delta_pct ?? 0, false)}`,
        counterfactual.action_changes.length
          ? `액션 변화 ${counterfactual.action_changes.length}건은 아래 표에서 기준 액션과 정책 후 액션을 비교합니다.`
          : '액션 변화가 없으면 이 대안은 현재 IPS 판단을 크게 바꾸지 않습니다.',
        counterfactual.warnings.length
          ? '경고가 있으면 비중 변화보다 데이터 품질과 투자 논리 확인을 먼저 봅니다.'
          : '경고가 없으면 비중, 위험기여도, 목표 갭 변화 순서로 보면 됩니다.'
      ]
    : [];
  const bestIpsBacktest = backtest?.strategy_summaries[0] ?? null;
  const bestReturnBacktest = backtest?.strategy_summaries.reduce(
    (best, row) => {
      if (!best) return row;
      return (row.cagr ?? -Infinity) > (best.cagr ?? -Infinity) ? row : best;
    },
    null as BacktestResponse['strategy_summaries'][number] | null
  ) ?? null;
  const backtestReadout = backtest
    ? [
        bestIpsBacktest
          ? `IPS 적합성 우선 정렬 기준의 첫 정책은 ${bestIpsBacktest.strategy_label}입니다.`
          : '비교할 정책 결과가 없습니다.',
        bestReturnBacktest && bestIpsBacktest && bestReturnBacktest.strategy !== bestIpsBacktest.strategy
          ? `성과 지표만 보면 ${bestReturnBacktest.strategy_label}가 앞설 수 있지만, 먼저 IPS 위반과 위성/RC 초과를 확인합니다.`
          : '성과 지표와 IPS 적합성이 크게 충돌하지 않는지 확인합니다.',
        'CAGR, 변동성, MDD, Sharpe는 보조 지표이며 정책 채택 순위가 아닙니다.'
      ]
    : [];

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <p className="eyebrow">Portfolio Rebalancer</p>
          <h1>리밸런싱 워크벤치</h1>
        </div>
        <nav className="view-tabs" aria-label="주요 화면">
          <button
            type="button"
            className={cx(activeView === 'workbench' && 'active')}
            onClick={() => setActiveView('workbench')}
          >
            워크벤치
          </button>
          <button
            type="button"
            className={cx(activeView === 'settings' && 'active')}
            onClick={() => setActiveView('settings')}
          >
            설정
          </button>
        </nav>
      </header>

      <div className="app-body">
        <section className="workspace">
        <header className="topbar">
          <div>
            <h2>{activeView === 'workbench' ? '워크벤치' : '설정 관리'}</h2>
            <p>
              {activeView === 'workbench'
                ? 'Python 계산 코어를 JSON API로 호출하고, React에서 결과를 검토합니다.'
                : 'IPS 목표와 투자 논리 옵션을 관리하고 다음 평가에 적용합니다.'}
            </p>
          </div>
          {activeView === 'workbench' && (
            <div className="status-strip">
              <span className={portfolio.length ? 'done' : ''}>1 입력</span>
              <span className={analysis ? 'done' : ''}>2 분석</span>
              <span className={evaluation ? 'done' : ''}>3 평가</span>
            </div>
          )}
        </header>

        <section
          className="mx-auto w-full max-w-6xl rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
          hidden={activeView !== 'workbench'}
        >
          <div className="grid gap-4 xl:grid-cols-[1fr_1.3fr]">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FolderOpen className="h-5 w-5 text-blue-700" />
                <h3 className="text-lg font-semibold text-slate-950">저장된 포트폴리오</h3>
              </div>
              <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                <input
                  className="table-input"
                  value={newPortfolioName}
                  placeholder="새 포트폴리오 이름"
                  onChange={(event) => setNewPortfolioName(event.target.value)}
                />
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                  disabled={!newPortfolioName.trim() || createPortfolioMutation.isPending}
                  onClick={createNamedPortfolio}
                >
                  {createPortfolioMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Plus className="h-4 w-4" />}
                  만들기
                </button>
              </div>
              <select
                className="table-input"
                value={selectedPortfolioId ?? ''}
                onChange={(event) => setSelectedPortfolioId(event.target.value ? Number(event.target.value) : null)}
              >
                <option value="">포트폴리오 선택</option>
                {savedPortfolios.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
              <div className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">
                {activePortfolio?.latest_snapshot
                  ? `최근 저장: ${activePortfolio.latest_snapshot.name} · ${shortDate(activePortfolio.latest_snapshot.created_at)}`
                  : '아직 저장된 스냅샷이 없습니다.'}
              </div>
              <ErrorLine error={createPortfolioMutation.error} />
            </div>

            <div className="space-y-3">
              <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                <input
                  className="table-input"
                  value={snapshotName}
                  placeholder="스냅샷 이름"
                  onChange={(event) => setSnapshotName(event.target.value)}
                />
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm font-bold text-blue-800 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
                  disabled={selectedPortfolioId === null || !validRows.length || portfolioMutation.isPending || saveSnapshotMutation.isPending}
                  onClick={saveCurrentSnapshot}
                >
                  {saveSnapshotMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Save className="h-4 w-4" />}
                  현재 상태 저장
                </button>
              </div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {savedSnapshots.map((snapshot) => (
                    <div
                      key={snapshot.id}
                      className={cx(
                        'flex items-center gap-2 rounded-lg border px-2 py-2 text-sm transition hover:border-blue-300 hover:bg-blue-50',
                        editingSnapshotId === snapshot.id ? 'border-blue-300 bg-blue-50' : 'border-slate-200'
                      )}
                    >
                      <button
                        type="button"
                        className="min-w-0 flex-1 px-1 text-left"
                        onClick={() => loadSnapshotMutation.mutate(snapshot.id)}
                        disabled={snapshotActionPending}
                      >
                        <span>
                          <strong className="block truncate text-slate-900">{snapshot.name}</strong>
                          <span className="block truncate text-xs text-slate-500">
                            {shortDate(snapshot.created_at)} · {snapshot.position_count}종목
                            {snapshot.note ? ` · ${snapshot.note}` : ''}
                          </span>
                        </span>
                      </button>
                      <span className="shrink-0 text-xs font-bold text-blue-800">
                        {snapshot.has_evaluation ? '평가' : snapshot.has_analysis ? '분석' : '입력'}
                      </span>
                      <button
                        type="button"
                        className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-slate-400 transition hover:bg-white hover:text-blue-700 disabled:cursor-not-allowed disabled:text-slate-300"
                        title="스냅샷 편집"
                        disabled={snapshotActionPending}
                        onClick={() => startEditingSnapshot(snapshot)}
                      >
                        {editSnapshotMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Edit3 className="h-4 w-4" />}
                      </button>
                      <button
                        type="button"
                        className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-slate-400 transition hover:bg-white hover:text-red-700 disabled:cursor-not-allowed disabled:text-slate-300"
                        title="스냅샷 삭제"
                        disabled={snapshotActionPending}
                        onClick={() => removeSnapshot(snapshot)}
                      >
                        {deletingSnapshotId === snapshot.id ? <Loader2 className="spin h-4 w-4" /> : <Trash2 className="h-4 w-4" />}
                      </button>
                    </div>
                ))}
                {!savedSnapshots.length && (
                  <div className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-center text-sm text-slate-500">
                    저장 이력이 없습니다.
                  </div>
                )}
              </div>
              <ErrorLine
                error={
                  saveSnapshotMutation.error ??
                  editSnapshotMutation.error ??
                  loadSnapshotMutation.error ??
                  updateSnapshotMutation.error ??
                  deleteSnapshotMutation.error
                }
              />
            </div>
          </div>
        </section>

        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm" hidden={activeView !== 'settings'}>
            <div className="mb-5 flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
              <div>
                <h3 className="flex items-center gap-2 text-xl font-semibold text-slate-950">
                  <ShieldCheck className="h-5 w-5 text-blue-700" />
                  설정 관리
                </h3>
                <p className="mt-2 text-sm text-slate-500">옵션과 IPS 정책은 DB에 저장되고 다음 평가부터 바로 적용됩니다.</p>
              </div>
              {(configOptionsQuery.isLoading || ipsConfigQuery.isLoading) && <Loader2 className="spin h-5 w-5 text-blue-700" />}
            </div>
            <div className="grid gap-6 xl:grid-cols-2">
              <div className="space-y-4">
                <div className="grid gap-3">
                  <div>
                    <div className="mb-2 text-sm font-bold text-slate-700">고정 그룹</div>
                    <div className="grid gap-2">
                      {fixedGroupOptions.map((option) => (
                        <div key={option.value} className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700">
                          <strong>{option.label}</strong> · {option.value}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="mb-2 text-sm font-bold text-slate-700">투자 논리 상태</div>
                    <div className="grid gap-2">
                      {thesisStatusOptions.map((option) => (
                        <div key={`thesis-${option.value}`} className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700">
                          <strong>{option.label}</strong> · {option.value}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-5">
                <div>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h4 className="font-semibold text-slate-950">타입별 목표 비중</h4>
                    <div className="flex gap-2">
                      <button type="button" className="rounded-lg bg-blue-800 px-3 py-2 text-xs font-bold text-white disabled:bg-slate-300" disabled={saveTargetsMutation.isPending} onClick={() => saveTargetsMutation.mutate()}>
                        저장
                      </button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {targetAllocationRows.map((row, index) => (
                      <div key={row.group} className="grid grid-cols-4 gap-2">
                        <input className="table-input" value={row.group} readOnly />
                        <input className="table-input" type="number" step="0.01" value={row.min} onChange={(event) => updateTargetAllocation(index, 'min', event.target.value)} />
                        <input className="table-input" type="number" step="0.01" value={row.target} onChange={(event) => updateTargetAllocation(index, 'target', event.target.value)} />
                        <input className="table-input" type="number" step="0.01" value={row.max} onChange={(event) => updateTargetAllocation(index, 'max', event.target.value)} />
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h4 className="font-semibold text-slate-950">액션 우선순위</h4>
                  </div>
                  <div className="space-y-2">
                    {actionPriorityRows.map((row) => (
                      <div key={row.action_code} className="grid grid-cols-[1fr_1fr_72px_56px] items-center gap-2">
                        <span className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">{row.action_code}</span>
                        <span className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">{row.label}</span>
                        <span className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-center text-sm text-slate-700">{row.priority}</span>
                        <span className={cx('justify-self-center rounded-full px-2 py-0.5 text-xs font-bold', row.is_active ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-500')}>
                          {row.is_active ? '활성' : '비활성'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h4 className="font-semibold text-slate-950">IPS 룰 JSON</h4>
                    <button type="button" className="rounded-lg bg-blue-800 px-3 py-2 text-xs font-bold text-white disabled:bg-slate-300" disabled={saveRulesMutation.isPending} onClick={() => saveRulesMutation.mutate()}>
                      저장
                    </button>
                  </div>
                  <textarea className="table-input min-h-32 w-full font-mono text-xs" value={rulesJson} onChange={(event) => setRulesJson(event.target.value)} />
                </div>
                <ErrorLine error={saveTargetsMutation.error ?? saveRulesMutation.error ?? configOptionsQuery.error ?? ipsConfigQuery.error} />
              </div>
            </div>
          </section>

          {activeView === 'workbench' && (
            <>
              <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
            <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm xl:col-span-5">
              <div className="mb-4 flex items-start justify-between gap-4">
                <div>
                  <h3 className="flex items-center gap-2 text-xl font-semibold text-slate-950">
                    <Edit3 className="h-5 w-5 text-blue-700" />
                    포트폴리오 붙여넣기
                  </h3>
                  <p className="mt-2 text-sm text-slate-500">
                    티커와 비중만 한 줄씩 입력하세요. 그룹은 오른쪽 표에서 선택합니다.
                  </p>
                </div>
                <button
                  type="button"
                  className="inline-flex shrink-0 items-center gap-1 rounded-lg px-2 py-1.5 text-sm font-semibold text-blue-700 transition hover:bg-blue-50"
                  onClick={() => syncText(sampleText)}
                >
                  <RefreshCcw className="h-4 w-4" />
                  예시
                </button>
              </div>

              <textarea
                className="h-80 w-full resize-none rounded-lg border-0 bg-slate-100 p-4 font-mono text-sm leading-6 text-slate-900 outline-none transition placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-blue-700"
                value={text}
                placeholder={'VOO 40\nQQQ 25.5\nAAPL 10\nMSFT 10\nNVDA 2.0'}
                onChange={(event) => syncText(event.target.value)}
              />

              <label
                className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-white p-6 text-center transition hover:border-blue-300 hover:bg-blue-50/50"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  submitPortfolioFile(event.dataTransfer.files[0]);
                }}
              >
                <FileUp className="mb-2 h-8 w-8 text-slate-400" />
                <span className="text-sm font-semibold text-slate-800">CSV/TSV 파일을 여기에 놓거나 선택하세요</span>
                <span className="mt-1 text-xs text-slate-500">Excel, Google Sheets 데이터 지원</span>
                <input
                  className="hidden"
                  type="file"
                  accept=".csv,.tsv,text/csv,text/tab-separated-values"
                  onChange={(event) => {
                    submitPortfolioFile(event.target.files?.[0]);
                    event.target.value = '';
                  }}
                />
              </label>
            </section>

            <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm xl:col-span-7">
              <div className="flex flex-col gap-4 border-b border-slate-200 bg-slate-50/80 p-6 md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="flex items-center gap-2 text-xl font-semibold text-slate-950">
                    <Database className="h-5 w-5 text-blue-700" />
                    자동 인식 내용
                  </h3>
                  <p className="mt-2 text-sm text-slate-500">붙여넣기는 티커와 비중만 받고, IPS 그룹은 여기서 선택합니다.</p>
                </div>
                <div className="grid grid-cols-3 gap-3 text-right">
                  <SummaryStat label="유효 행" value={`${validRows.length}개`} />
                  <SummaryStat label="합계 비중" value={`${totalAllocation.toFixed(2)}%`} />
                  <SummaryStat label="중복" value={`${duplicateCount}개`} tone={duplicateCount ? 'warn' : 'default'} />
                </div>
              </div>

              <div className="overflow-x-auto p-4">
                <div className="min-w-[620px] space-y-2">
                  <div className="grid grid-cols-[0.9fr_0.75fr_1.3fr_44px] gap-2 px-1 text-xs font-bold uppercase text-slate-500">
                    <span>티커</span>
                    <span>비중</span>
                    <span>그룹</span>
                    <span />
                  </div>
                  {rows.map((row, index) => (
                    <div
                      className="grid grid-cols-[0.9fr_0.75fr_1.3fr_44px] items-center gap-2 rounded-lg border border-slate-200 bg-white p-2 transition hover:bg-slate-50"
                      key={`${row.ticker}-${index}`}
                    >
                      <input className="table-input font-bold text-blue-700" value={String(row.ticker ?? '')} placeholder="VOO" onChange={(event) => updateRow(index, 'ticker', event.target.value)} />
                      <input className="table-input text-right" value={String(row.allocation ?? '')} placeholder="40" type="number" onChange={(event) => updateRow(index, 'allocation', event.target.value)} />
                      <select className="table-input" value={String(row.group ?? '')} onChange={(event) => updateRow(index, 'group', event.target.value)}>
                        <option value="">그룹 선택</option>
                        {fixedGroupOptions.map((group) => (
                          <option key={group.value} value={group.value}>
                            {group.label}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className="grid h-9 w-9 place-items-center rounded-lg text-slate-400 transition hover:bg-red-50 hover:text-red-700"
                        aria-label="행 삭제"
                        onClick={() => setRows((current) => current.filter((_, rowIndex) => rowIndex !== index))}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-3 border-t border-slate-200 bg-slate-50 p-4 sm:flex-row sm:items-center sm:justify-between">
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
                  onClick={() => setRows((current) => [...current, blankRow()])}
                >
                  <Plus className="h-4 w-4" />
                  행 추가
                </button>
                <span className="text-sm text-slate-500">DCA와 투자 논리는 분석 후 세부 판단값에서 입력합니다.</span>
              </div>
            </section>
          </div>

          <ErrorLine error={portfolioMutation.error ?? csvMutation.error} />

          <section className="relative overflow-hidden rounded-xl bg-blue-800 p-6 shadow-lg">
            <div className="relative z-10 flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
              <div className="text-white">
                <h3 className="text-2xl font-semibold">분석에 사용할 포트폴리오 확정</h3>
                <p className="mt-2 text-sm text-blue-100">이 입력값으로 데이터 조회와 평가 워크플로우를 시작합니다.</p>
              </div>
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-white px-6 py-3 text-sm font-bold text-blue-800 shadow-xl transition hover:bg-blue-100 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={!validRows.length || portfolioMutation.isPending}
                onClick={() => portfolioMutation.mutate(validRows)}
              >
                {portfolioMutation.isPending ? <Loader2 className="spin h-5 w-5" /> : <CheckCircle2 className="h-5 w-5" />}
                이 포트폴리오 사용
              </button>
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-blue-700" />
              <h3 className="text-xl font-semibold text-slate-950">정규화된 포트폴리오</h3>
            </div>
            <DataTable data={portfolio} columns={assetColumns} emptyLabel="정규화된 포트폴리오가 아직 없습니다." />
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <div className="grid h-8 w-8 place-items-center rounded-lg bg-violet-100 text-sm font-bold text-violet-800">2</div>
                  <h3 className="text-xl font-semibold text-slate-950">데이터 조회 & 보강</h3>
                </div>
                <p className="mt-2 text-sm text-slate-500">가격 데이터를 조회하고 포트폴리오/벤치마크/자산별 지표를 계산합니다. 이후 DCA와 투자 논리를 보정합니다.</p>
              </div>
              <div className="run-controls analysis-controls">
                <label className="control-field">
                  <span>평가 기간</span>
                  <select className="table-input" {...register('periodMode')}>
                    <option value="months">개월</option>
                    <option value="YTD">YTD</option>
                    <option value="Max">Max</option>
                  </select>
                </label>
                {settings.periodMode === 'months' && (
                  <label className="control-field compact">
                    <span>개월 수</span>
                    <input className="table-input" type="number" min="1" max="120" {...register('months')} />
                  </label>
                )}
                <label className="control-field compact">
                  <span>무위험 수익률 (%)</span>
                  <input className="table-input" type="number" step="0.1" {...register('rfPct')} />
                </label>
                <label className="control-field wide">
                  <span>벤치마크</span>
                  <input className="table-input" type="text" {...register('bench')} />
                </label>
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                  disabled={(!portfolio.length && !validRows.length) || portfolioMutation.isPending || analysisMutation.isPending}
                  onClick={shouldApplyRowsBeforeAnalysis ? applyRowsAndRunAnalysis : runCurrentAnalysis}
                >
                  {portfolioMutation.isPending || analysisMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <BarChart3 className="h-4 w-4" />}
                  {shouldApplyRowsBeforeAnalysis ? '입력 반영 후 분석 실행' : '분석 실행'}
                </button>
              </div>
            </div>
            <ErrorLine error={analysisMutation.error} />
            {analysis && (
              <>
                <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricCard label="포트폴리오 CAGR" value={analysis.portfolio_metrics.cagr} />
                  <MetricCard label="포트폴리오 변동성" value={analysis.portfolio_metrics.volatility} />
                  <MetricCard label="포트폴리오 샤프" value={analysis.portfolio_metrics.sharpe} format="number" />
                  <MetricCard label="벤치마크 샤프" value={analysis.benchmark_metrics?.sharpe} format="number" />
                </div>
                <ChartBlock data={chartData} />
              </>
            )}
            <DataTable data={analysis?.metrics ?? []} columns={metricColumns} emptyLabel="분석 결과가 아직 없습니다." />
            {analysis?.metrics.length ? (
              <a className="download-link" href={csvDownloadUrl('metrics')}>
                <Download className="h-4 w-4" /> 메트릭 CSV
              </a>
            ) : null}
          </section>

          {analysis && (
            <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
              <div className="flex flex-col gap-4 border-b border-slate-200 bg-slate-50/80 p-6 md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="text-xl font-semibold text-slate-950">세부 판단값 보정</h3>
                  <p className="mt-2 text-sm text-slate-500">
                    기간 수익률은 자동 계산값입니다. 계좌 기준 수익률을 반영하려면 override를 입력한 뒤 분석을 다시 실행하세요.
                  </p>
                </div>
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                  disabled={!validRows.length || !rowsDirty || portfolioMutation.isPending || analysisMutation.isPending}
                  onClick={applyRowsAndRunAnalysis}
                >
                  {portfolioMutation.isPending || analysisMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <RefreshCcw className="h-4 w-4" />}
                  보정값 반영 후 분석 재실행
                </button>
              </div>
              <div className="overflow-x-auto p-4">
                <div className="min-w-[820px] space-y-2">
                  <div className="grid grid-cols-[0.8fr_0.85fr_1fr_56px_1fr] gap-2 px-1 text-xs font-bold uppercase text-slate-500">
                    <span>티커</span>
                    <span className="text-right">계좌 수익률 override</span>
                    <span>그룹</span>
                    <span className="text-center">DCA</span>
                    <span>투자 논리</span>
                  </div>
                  {rows.map((row, index) => (
                    <div
                      className="grid grid-cols-[0.8fr_0.85fr_1fr_56px_1fr] items-center gap-2 rounded-lg border border-slate-200 bg-white p-2"
                      key={`detail-${row.ticker}-${index}`}
                    >
                      <div className="px-2 text-sm font-bold text-blue-700">{row.ticker || '미입력'}</div>
                      <input className="table-input text-right" value={String(row.return_total ?? '')} placeholder="자동 계산" type="number" onChange={(event) => updateRow(index, 'return_total', event.target.value)} />
                      <select className="table-input" value={String(row.group ?? '')} onChange={(event) => updateRow(index, 'group', event.target.value)}>
                        <option value="">그룹 선택</option>
                        {fixedGroupOptions.map((group) => (
                          <option key={group.value} value={group.value}>
                            {group.label}
                          </option>
                        ))}
                      </select>
                      <label className="grid place-items-center">
                        <input className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700" checked={Boolean(row.dca_enabled)} type="checkbox" onChange={(event) => updateRow(index, 'dca_enabled', event.target.checked)} />
                      </label>
                      <select className="table-input" value={String(row.thesis_status ?? '')} onChange={(event) => updateRow(index, 'thesis_status', event.target.value)}>
                        <option value="">논리 선택</option>
                        {activeThesisStatusOptions.map((status) => (
                          <option key={status.value} value={status.value}>
                            {status.label}
                          </option>
                        ))}
                        {row.thesis_status && !thesisStatusOptions.some((status) => status.value === row.thesis_status) ? (
                          <option value={String(row.thesis_status)}>기타: {row.thesis_status}</option>
                        ) : null}
                      </select>
                    </div>
                  ))}
                </div>
              </div>
              {rowsDirty && (
                <div className="border-t border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
                  보정값이 아직 분석 결과에 반영되지 않았습니다.
                </div>
              )}
            </section>
          )}

              <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="flex items-center gap-3">
                      <div className="grid h-8 w-8 place-items-center rounded-lg bg-cyan-100 text-sm font-bold text-cyan-900">3</div>
                      <h3 className="text-xl font-semibold text-slate-950">평가 & 실행 계획 제안</h3>
                    </div>
                    <p className="mt-2 text-sm text-slate-500">IPS 기준으로 실행 후보, 위험 초과, DCA 조정 신호를 확인합니다.</p>
                  </div>
                  <div className="run-controls evaluation-controls">
                    <label className="control-field compact">
                      <span>RC Over (%)</span>
                      <input className="table-input" type="number" step="0.1" {...register('rcOverThreshPct')} />
                    </label>
                    <label className="control-field compact">
                      <span>E 임계값</span>
                      <input className="table-input" type="number" step="0.05" min="0" max="1" {...register('eThresh')} />
                    </label>
                    <label className="control-field">
                      <span>판단 모드</span>
                      <select className="table-input" {...register('decisionContext')}>
                        {decisionContextOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div className="grid min-w-[230px] gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                      <span className="text-xs font-bold text-slate-500">평가 후 검증</span>
                      <label className="inline-flex items-center gap-2 font-semibold">
                        <input
                          className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700"
                          type="checkbox"
                          checked={runCounterfactualAfterEvaluation}
                          onChange={(event) => setRunCounterfactualAfterEvaluation(event.target.checked)}
                        />
                        Counterfactual 자동 실행
                      </label>
                      <label className="inline-flex items-center gap-2 font-semibold">
                        <input
                          className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700"
                          type="checkbox"
                          checked={runBacktestAfterEvaluation}
                          onChange={(event) => setRunBacktestAfterEvaluation(event.target.checked)}
                        />
                        제한 백테스트 자동 실행
                      </label>
                    </div>
                    <button
                      type="button"
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                      disabled={!analysis || rowsDirty || evaluationMutation.isPending || counterfactualMutation.isPending || backtestMutation.isPending}
                      onClick={runCurrentEvaluation}
                    >
                      {evaluationMutation.isPending || counterfactualMutation.isPending || backtestMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Play className="h-4 w-4" />}
                      평가 실행
                    </button>
                  </div>
                </div>
                <ErrorLine error={evaluationMutation.error} />
                <ErrorLine error={counterfactualMutation.error ?? backtestMutation.error} />
                <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                  Counterfactual와 백테스트는 평가 결과의 후속 검증입니다. 자동 실행을 켜면 현재 판단 모드와 임계값을 그대로 사용합니다.
                </div>
                {analysis && rowsDirty && (
                  <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
                    세부 판단값 변경사항을 먼저 분석 결과에 반영해야 평가를 실행할 수 있습니다.
                  </div>
                )}
                {evaluation && (
                  <EvaluationCharts
                    actionData={actionChartData}
                    efficiencyRiskData={efficiencyRiskData}
                    groupAllocationData={groupAllocationData}
                    riskBudgetData={riskBudgetData}
                  />
                )}
                <div className="mt-4">
                  <DataTable data={evaluation?.ips_actions ?? []} columns={actionColumns} emptyLabel="평가 결과가 아직 없습니다." />
                </div>
                {evaluation && (
                  <div className="mt-6">
                    <h4 className="mb-3 text-sm font-bold text-slate-700">수치 제안</h4>
                    <DataTable data={evaluation.proposal} columns={proposalColumns} emptyLabel="수치 제안이 없습니다." />
                  </div>
                )}
                {evaluation && (
                  <div className="mt-4 flex flex-wrap gap-3">
                    <a className="download-link" href={csvDownloadUrl('proposal')}><Download className="h-4 w-4" /> 제안 CSV</a>
                    <a className="download-link" href={csvDownloadUrl('ips_actions')}><Download className="h-4 w-4" /> IPS CSV</a>
                    <a className="download-link" href={csvDownloadUrl('group_summary')}><Download className="h-4 w-4" /> 그룹 CSV</a>
                  </div>
                )}
              </section>

              <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="flex items-center gap-3">
                      <div className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-100 text-sm font-bold text-emerald-900">4</div>
                      <h3 className="text-xl font-semibold text-slate-950">Counterfactual 비교</h3>
                    </div>
                    <p className="mt-2 text-sm text-slate-500">현재 평가와 preset 정책 적용 결과를 IPS 변화량 중심으로 비교합니다.</p>
                    <p className="mt-2 text-sm font-semibold text-slate-600">{selectedCounterfactualOption.description}</p>
                  </div>
                  <div className="run-controls evaluation-controls">
                    <label className="control-field wide">
                      <span>대안 정책</span>
                      <select
                        className="table-input"
                        value={counterfactualScenario}
                        onChange={(event) => setCounterfactualScenario(event.target.value as CounterfactualScenario)}
                      >
                        {counterfactualScenarioOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                      disabled={!evaluation || rowsDirty || counterfactualMutation.isPending}
                      onClick={runCurrentCounterfactual}
                    >
                      {counterfactualMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <LineChart className="h-4 w-4" />}
                      정책 비교
                    </button>
                  </div>
                </div>
                <ErrorLine error={counterfactualMutation.error} />
                <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-950">
                  <strong className="block font-bold">먼저 볼 것</strong>
                  <span className="mt-1 block">코어/위성 비중 변화, 목표 갭 변화, 액션 변화, 경고 순서로 봅니다. 기대수익률 비교가 아니라 IPS 정책 차이 확인입니다.</span>
                </div>
                {counterfactual && (
                  <>
                    <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                      <MetricCard label="코어 변화" value={(counterfactual.deltas.groups.core?.delta_pct ?? 0) / 100} />
                      <MetricCard label="위성 변화" value={(counterfactual.deltas.groups.satellite?.delta_pct ?? 0) / 100} />
                      <MetricCard label="액션 변화" value={counterfactual.action_changes.length} format="number" />
                      <MetricCard label="경고" value={counterfactual.warnings.length} format="number" />
                    </div>
                    <div className="mt-5 grid gap-3 lg:grid-cols-2">
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <h4 className="text-sm font-bold text-slate-800">결과 요약</h4>
                        <ul className="mt-3 space-y-2 text-sm text-slate-600">
                          {[...counterfactualReadout, ...counterfactual.interpretation].map((line) => (
                            <li key={line}>{line}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <h4 className="text-sm font-bold text-slate-800">경고</h4>
                        <ul className="mt-3 space-y-2 text-sm text-slate-600">
                          {(counterfactual.warnings.length ? counterfactual.warnings : ['데이터 품질/투자 논리 경고가 없습니다.']).map((line) => (
                            <li key={line}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    <div className="mt-6">
                      <h4 className="mb-3 text-sm font-bold text-slate-700">비중·위험 변화</h4>
                      <p className="mb-3 text-sm text-slate-500">양수는 대안 정책 적용 후 늘어난 값, 음수는 줄어든 값입니다.</p>
                      <DataTable data={counterfactual.deltas.assets} columns={counterfactualDeltaColumns} emptyLabel="변화량이 없습니다." />
                    </div>
                    <div className="mt-6">
                      <h4 className="mb-3 text-sm font-bold text-slate-700">액션 변화</h4>
                      <DataTable data={counterfactual.action_changes} columns={actionChangeColumns} emptyLabel="액션 변화가 없습니다." />
                    </div>
                  </>
                )}
              </section>

              <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="flex items-center gap-3">
                      <div className="grid h-8 w-8 place-items-center rounded-lg bg-amber-100 text-sm font-bold text-amber-900">5</div>
                      <h3 className="text-xl font-semibold text-slate-950">제한된 IPS 백테스트</h3>
                    </div>
                    <p className="mt-2 text-sm text-slate-500">월간 점검 기준으로 정책별 IPS 위반, 위성 초과, 위험기여도 초과, 조정 빈도를 비교합니다.</p>
                  </div>
                  <div className="run-controls analysis-controls">
                    <div className="grid min-w-[260px] gap-2">
                      <span className="text-xs font-bold text-slate-500">비교 정책</span>
                      <div className="flex flex-wrap gap-2">
                        {backtestStrategyOptions.map((option) => (
                          <label key={option.value} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700" title={option.description}>
                            <input
                              className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700"
                              type="checkbox"
                              checked={backtestStrategies.includes(option.value)}
                              onChange={() => toggleBacktestStrategy(option.value)}
                            />
                            {option.label}
                          </label>
                        ))}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                      disabled={!analysis || rowsDirty || backtestStrategies.length === 0 || backtestMutation.isPending}
                      onClick={runCurrentBacktest}
                    >
                      {backtestMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Play className="h-4 w-4" />}
                      월간 검증
                    </button>
                  </div>
                </div>
                <ErrorLine error={backtestMutation.error} />
                <div className="mt-4 grid gap-3 lg:grid-cols-2">
                  <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
                    <strong className="block font-bold">먼저 볼 것</strong>
                    <span className="mt-1 block">IPS 위반, 위성 초과, RC 초과, 조정 빈도를 먼저 보고 성과 지표는 나중에 봅니다.</span>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                    <strong className="block font-bold text-slate-800">선택한 정책</strong>
                    <span className="mt-1 block">
                      {selectedBacktestOptions.length
                        ? selectedBacktestOptions.map((option) => `${option.label}: ${option.description}`).join(' / ')
                        : '비교할 정책을 1개 이상 선택하세요.'}
                    </span>
                  </div>
                </div>
                {backtest && (
                  <>
                    <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                      <MetricCard label="비교 정책" value={backtest.strategy_summaries.length} format="number" />
                      <MetricCard label="최소 IPS 위반" value={Math.min(...backtest.strategy_summaries.map((row) => row.ips_violation_count))} format="number" />
                      <MetricCard label="최소 위성 초과" value={Math.min(...backtest.strategy_summaries.map((row) => row.satellite_over_periods))} format="number" />
                      <MetricCard label="최소 RC 초과" value={Math.min(...backtest.strategy_summaries.map((row) => row.risk_contribution_over_count))} format="number" />
                    </div>
                    <div className="mt-5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-900">
                      {backtestReadout.map((line) => (
                        <span key={line} className="block">{line}</span>
                      ))}
                    </div>
                    <div className="mt-6">
                      <DataTable data={backtest.strategy_summaries} columns={backtestColumns} emptyLabel="백테스트 결과가 없습니다." />
                    </div>
                  </>
                )}
              </section>
            </>
          )}
        </div>
      </section>
      </div>
      {editingSnapshotId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4" role="dialog" aria-modal="true" aria-labelledby="snapshot-edit-title" onClick={cancelEditingSnapshot}>
          <div className="flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex flex-col gap-3 border-b border-slate-200 p-5 md:flex-row md:items-start md:justify-between">
              <div>
                <h3 id="snapshot-edit-title" className="text-xl font-semibold text-slate-950">스냅샷 편집</h3>
                <p className="mt-1 text-sm text-slate-500">저장하면 기존 분석/평가 결과는 초기화되고 스냅샷 상태가 입력으로 돌아갑니다.</p>
              </div>
              <button
                type="button"
                className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
                title="닫기"
                disabled={updateSnapshotMutation.isPending}
                onClick={cancelEditingSnapshot}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
              <div className="grid gap-3 md:grid-cols-[1fr_1.4fr]">
                <input
                  className="table-input w-full"
                  value={editingSnapshotName}
                  placeholder="스냅샷 이름"
                  onChange={(event) => setEditingSnapshotName(event.target.value)}
                />
                <input
                  className="table-input w-full"
                  value={editingSnapshotNote}
                  placeholder="메모"
                  onChange={(event) => setEditingSnapshotNote(event.target.value)}
                />
              </div>
              <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
                <div className="min-w-[780px] space-y-2 p-3">
                  <div className="grid grid-cols-[0.8fr_0.7fr_0.8fr_1.1fr_0.7fr_1fr_36px] gap-2 px-1 text-xs font-bold uppercase text-slate-500">
                    <span>티커</span>
                    <span className="text-right">비중</span>
                    <span className="text-right">수익률</span>
                    <span>그룹</span>
                    <span>DCA</span>
                    <span>논리 상태</span>
                    <span />
                  </div>
                  {editingSnapshotRows.map((row, index) => (
                    <div key={`${row.ticker}-${index}`} className="grid grid-cols-[0.8fr_0.7fr_0.8fr_1.1fr_0.7fr_1fr_36px] items-center gap-2">
                      <input className="table-input font-bold text-blue-700" value={String(row.ticker ?? '')} placeholder="VOO" onChange={(event) => updateEditingSnapshotRow(index, 'ticker', event.target.value)} />
                      <input className="table-input text-right" value={String(row.allocation ?? '')} placeholder="40" type="number" onChange={(event) => updateEditingSnapshotRow(index, 'allocation', event.target.value)} />
                      <input className="table-input text-right" value={String(row.return_total ?? '')} placeholder="%" type="number" onChange={(event) => updateEditingSnapshotRow(index, 'return_total', event.target.value)} />
                      <select className="table-input" value={String(row.group ?? '')} onChange={(event) => updateEditingSnapshotRow(index, 'group', event.target.value)}>
                        <option value="">그룹 선택</option>
                        {fixedGroupOptions.map((group) => (
                          <option key={group.value} value={group.value}>
                            {group.label}
                          </option>
                        ))}
                      </select>
                      <select className="table-input" value={String(row.dca_enabled ?? true)} onChange={(event) => updateEditingSnapshotRow(index, 'dca_enabled', event.target.value === 'true')}>
                        <option value="true">ON</option>
                        <option value="false">OFF</option>
                      </select>
                      <select className="table-input" value={String(row.thesis_status ?? '')} onChange={(event) => updateEditingSnapshotRow(index, 'thesis_status', event.target.value)}>
                        <option value="">상태 선택</option>
                        {activeThesisStatusOptions.map((status) => (
                          <option key={status.value} value={status.value}>
                            {status.label}
                          </option>
                        ))}
                        {row.thesis_status && !thesisStatusOptions.some((status) => status.value === row.thesis_status) ? (
                          <option value={String(row.thesis_status)}>기타: {row.thesis_status}</option>
                        ) : null}
                      </select>
                      <button
                        type="button"
                        className="grid h-9 w-9 place-items-center rounded-lg text-slate-400 transition hover:bg-red-50 hover:text-red-700"
                        title="행 삭제"
                        onClick={() => setEditingSnapshotRows((current) => current.filter((_, rowIndex) => rowIndex !== index))}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                  {!editingSnapshotRows.length && (
                    <div className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-center text-sm text-slate-500">
                      저장할 포지션이 없습니다.
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-3 border-t border-slate-200 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
                onClick={() => setEditingSnapshotRows((current) => [...current, blankRow()])}
              >
                <Plus className="h-4 w-4" />
                행 추가
              </button>
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
                  disabled={updateSnapshotMutation.isPending}
                  onClick={cancelEditingSnapshot}
                >
                  <X className="h-4 w-4" />
                  취소
                </button>
                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                  disabled={updateSnapshotMutation.isPending || !editingSnapshotRows.some((row) => row.ticker && row.allocation !== '')}
                  onClick={saveEditedSnapshot}
                >
                  {updateSnapshotMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Save className="h-4 w-4" />}
                  저장
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function SummaryStat({
  label,
  value,
  tone = 'default'
}: {
  label: string;
  value: string;
  tone?: 'default' | 'warn';
}) {
  return (
    <div>
      <span className="block text-xs font-bold uppercase text-slate-500">{label}</span>
      <strong className={cx('block text-sm font-bold', tone === 'warn' ? 'text-amber-700' : 'text-blue-800')}>
        {value}
      </strong>
    </div>
  );
}

function ErrorLine({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="mt-3 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
      <AlertCircle className="h-4 w-4" /> {error.message}
    </div>
  );
}

function ChartBlock({ data }: { data: Array<{ ticker: string; weight: number; risk: number }> }) {
  if (!data.length) return null;
  return (
    <div className="mt-5 h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
        <LineChart className="h-4 w-4 text-blue-700" />
        비중 대비 위험기여도
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ticker" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="weight" fill="#1d4ed8" name="비중 %" />
          <Bar dataKey="risk" fill="#0f766e" name="위험기여도 %" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function EvaluationCharts({
  actionData,
  efficiencyRiskData,
  groupAllocationData,
  riskBudgetData
}: {
  actionData: Array<{ ticker: string; suggestedTrade: number }>;
  efficiencyRiskData: Array<{ ticker: string; efficiency: number; rcGap: number }>;
  groupAllocationData: Array<Record<string, string | number>>;
  riskBudgetData: Array<{ ticker: string; risk: number; target: number }>;
}) {
  const groupKeys = groupAllocationData.length
    ? Object.keys(groupAllocationData[0]).filter((key) => key !== 'label')
    : [];

  return (
    <div className="mt-5 grid gap-4 xl:grid-cols-2">
      {riskBudgetData.length ? (
        <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
            <BarChart3 className="h-4 w-4 text-blue-700" />
            위험 예산
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={riskBudgetData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ticker" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="risk" fill="#1d4ed8" name="위험기여도 %" />
              <Bar dataKey="target" fill="#0f766e" name="RC Target %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : null}

      {actionData.length ? (
        <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
            <RefreshCcw className="h-4 w-4 text-blue-700" />
            리밸런싱 액션
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={actionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ticker" />
              <YAxis />
              <Tooltip />
              <ReferenceLine y={0} stroke="#64748b" />
              <Bar dataKey="suggestedTrade" fill="#7c3aed" name="제안조정 %p" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : null}

      {efficiencyRiskData.length ? (
        <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
            <LineChart className="h-4 w-4 text-blue-700" />
            E vs RC Gap
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="efficiency" name="E" type="number" domain={[0, 1]} />
              <YAxis dataKey="rcGap" name="RC Gap" type="number" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine y={0} stroke="#64748b" />
              <Scatter data={efficiencyRiskData} fill="#0f766e" name="자산" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : null}

      {groupAllocationData.length && groupKeys.length ? (
        <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
            <ShieldCheck className="h-4 w-4 text-blue-700" />
            그룹 비중
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={groupAllocationData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="label" type="category" />
              <Tooltip />
              <Legend />
              {groupKeys.map((key, index) => (
                <Bar
                  dataKey={key}
                  fill={['#1d4ed8', '#0f766e', '#7c3aed', '#ca8a04', '#64748b'][index % 5]}
                  key={key}
                  name={`${key} %`}
                  stackId="group"
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : null}
    </div>
  );
}
