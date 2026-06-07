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
import { Controller, useForm } from 'react-hook-form';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

import { DataTable } from './components/DataTable';
import { MetricCard } from './components/MetricCard';
import {
  type AnalysisResponse,
  type AssetRow,
  type ConfigOption,
  type EvaluationResponse,
  type IpsRule,
  type MetricRow,
  type ProposalRow,
  type SnapshotSummary,
  type TargetAllocation,
  createPortfolio,
  csvDownloadUrl,
  deleteSnapshot,
  getConfigOptions,
  getIpsConfig,
  listPortfolios,
  listSnapshots,
  loadSnapshot,
  runAnalysis,
  runEvaluation,
  saveActionPriorities,
  saveConfigOption,
  saveIpsRules,
  saveSnapshot,
  saveTargetAllocations,
  setConfigOptionActive,
  submitPortfolio,
  updateSnapshot,
  uploadPortfolioCsv
} from './lib/api';
import { blankRow, parsePortfolioText } from './lib/parser';
import { type PortfolioRowInput, type SettingsValues, settingsSchema } from './lib/schemas';

const sampleText = 'VOO 40\nQQQ 25\nSOXX 15\nUFO 3\nIONQ 2';
const groupTypes = ['core', 'satellite', 'defensive', 'cash', 'unknown'];
type OptionTable = 'groups' | 'roles' | 'thesis_statuses';

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function optionLabel(options: ConfigOption[], value: string | null | undefined) {
  if (!value) return '미정';
  const option = options.find((item) => item.value === value);
  if (!option) return value;
  return option.is_active ? option.label : `${option.label} (비활성)`;
}

function pct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined) return 'N/A';
  return `${(fromUnit ? value * 100 : value).toFixed(2)}%`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined) return 'N/A';
  return value.toFixed(2);
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
    role: asset.role,
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
      role: row.role ?? '',
      dca_enabled: row.dca_enabled ?? true,
      thesis_status: row.thesis_status ?? ''
    }))
  );
}

export function App() {
  const queryClient = useQueryClient();
  const [text, setText] = useState(sampleText);
  const [rows, setRows] = useState<PortfolioRowInput[]>(() => parsePortfolioText(sampleText));
  const [portfolio, setPortfolio] = useState<AssetRow[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [snapshotName, setSnapshotName] = useState('');
  const [editingSnapshotId, setEditingSnapshotId] = useState<number | null>(null);
  const [editingSnapshotName, setEditingSnapshotName] = useState('');
  const [editingSnapshotNote, setEditingSnapshotNote] = useState('');
  const [deletingSnapshotId, setDeletingSnapshotId] = useState<number | null>(null);
  const [appliedRowsSignature, setAppliedRowsSignature] = useState(() => rowsSignature(parsePortfolioText(sampleText)));
  const [newOptionTable, setNewOptionTable] = useState<OptionTable>('groups');
  const [newOptionCode, setNewOptionCode] = useState('');
  const [newOptionLabel, setNewOptionLabel] = useState('');
  const [newOptionGroupType, setNewOptionGroupType] = useState('satellite');
  const [targetAllocationRows, setTargetAllocationRows] = useState<TargetAllocation[]>([]);
  const [actionPriorityRows, setActionPriorityRows] = useState<Array<{ action_code: string; label: string; priority: number; is_active: boolean }>>([]);
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
  const savedSnapshots = snapshotsQuery.data?.snapshots ?? [];
  const groupOptions = configOptionsQuery.data?.groups ?? [];
  const roleOptions = configOptionsQuery.data?.roles ?? [];
  const thesisStatusOptions = configOptionsQuery.data?.thesis_statuses ?? [];
  const activeGroupOptions = groupOptions.filter((option) => option.is_active);
  const activeRoleOptions = roleOptions.filter((option) => option.is_active);
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

  const { register, control, watch } = useForm<SettingsValues>({
    defaultValues: {
      periodMode: 'months',
      months: 12,
      rfPct: 0,
      bench: 'SPY',
      momentumWeight: 0.2,
      rcOverThreshPct: 1.5,
      eThresh: 0.5
    }
  });
  const settings = watch();

  const portfolioMutation = useMutation({
    mutationFn: submitPortfolio,
    onSuccess: (data) => {
      const nextRows = rowsFromAssets(data.assets);
      setRows(nextRows);
      setAppliedRowsSignature(rowsSignature(nextRows));
      setPortfolio(data.assets);
      setAnalysis(null);
      setEvaluation(null);
    }
  });

  const csvMutation = useMutation({
    mutationFn: uploadPortfolioCsv,
    onSuccess: (data) => {
      const nextRows = rowsFromAssets(data.assets);
      setRows(nextRows);
      setText(nextRows.map(rowInputLine).join('\n'));
      setAppliedRowsSignature(rowsSignature(nextRows));
      setPortfolio(data.assets);
      setAnalysis(null);
      setEvaluation(null);
    }
  });

  const analysisMutation = useMutation({
    mutationFn: runAnalysis,
    onSuccess: (data) => {
      setAnalysis(data);
      setEvaluation(null);
    }
  });

  const evaluationMutation = useMutation({
    mutationFn: runEvaluation,
    onSuccess: setEvaluation
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
    mutationFn: () => {
      if (selectedPortfolioId === null) throw new Error('저장할 포트폴리오를 선택해주세요.');
      return saveSnapshot(selectedPortfolioId, {
        name: snapshotName || undefined
      });
    },
    onSuccess: async () => {
      setSnapshotName('');
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-snapshots', selectedPortfolioId] })
      ]);
    }
  });

  const updateSnapshotMutation = useMutation({
    mutationFn: ({
      snapshotId,
      payload
    }: {
      snapshotId: number;
      payload: { name?: string; note?: string };
    }) => updateSnapshot(snapshotId, payload),
    onSuccess: async () => {
      setEditingSnapshotId(null);
      setEditingSnapshotName('');
      setEditingSnapshotNote('');
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
        queryClient.invalidateQueries({ queryKey: ['portfolio-snapshots', selectedPortfolioId] })
      ]);
    }
  });

  const deleteSnapshotMutation = useMutation({
    mutationFn: deleteSnapshot,
    onSuccess: async () => {
      setEditingSnapshotId(null);
      setEditingSnapshotName('');
      setEditingSnapshotNote('');
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
    onSuccess: (data) => {
      const nextRows = rowsFromAssets(data.portfolio.assets);
      setRows(nextRows);
      setText(nextRows.map(rowInputLine).join('\n'));
      setAppliedRowsSignature(rowsSignature(nextRows));
      setPortfolio(data.portfolio.assets);
      setAnalysis(data.analysis);
      setEvaluation(data.evaluation);
      setSelectedPortfolioId(data.snapshot.portfolio_id);
    }
  });

  const saveOptionMutation = useMutation({
    mutationFn: () =>
      saveConfigOption(newOptionTable, {
        code: newOptionCode,
        label: newOptionLabel,
        group_type: newOptionTable === 'groups' ? newOptionGroupType : undefined
      }),
    onSuccess: async () => {
      setNewOptionCode('');
      setNewOptionLabel('');
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['config-options'] }),
        queryClient.invalidateQueries({ queryKey: ['ips-config'] })
      ]);
    }
  });

  const optionActiveMutation = useMutation({
    mutationFn: ({ table, code, isActive }: { table: OptionTable; code: string; isActive: boolean }) =>
      setConfigOptionActive(table, code, isActive),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['config-options'] }),
        queryClient.invalidateQueries({ queryKey: ['ips-config'] })
      ]);
    }
  });

  const saveTargetsMutation = useMutation({
    mutationFn: () => saveTargetAllocations(targetAllocationRows),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['ips-config'] })
  });

  const savePrioritiesMutation = useMutation({
    mutationFn: () => saveActionPriorities(actionPriorityRows),
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
    loadSnapshotMutation.isPending ||
    updateSnapshotMutation.isPending ||
    deleteSnapshotMutation.isPending;

  const assetColumns = useMemo<ColumnDef<AssetRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'allocation', header: '입력 비중', cell: ({ row }) => pct(row.original.allocation, false) },
      { accessorKey: 'weight', header: '정규화 비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'group', header: '그룹', cell: ({ row }) => optionLabel(groupOptions, row.original.group) }
    ],
    [groupOptions]
  );

  const metricColumns = useMemo<ColumnDef<MetricRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'weight', header: '비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'cagr', header: 'CAGR', cell: ({ row }) => pct(row.original.cagr) },
      { accessorKey: 'volatility', header: '변동성', cell: ({ row }) => pct(row.original.volatility) },
      { accessorKey: 'sharpe', header: '샤프', cell: ({ row }) => num(row.original.sharpe) },
      { accessorKey: 'risk_contribution', header: '위험기여도', cell: ({ row }) => pct(row.original.risk_contribution) },
      { accessorKey: 'return_total', header: '기간 수익률', cell: ({ row }) => pct(row.original.return_total) },
      { accessorKey: 'efficiency_score', header: 'E', cell: ({ row }) => num(row.original.efficiency_score) },
      { accessorKey: 'efficiency_score_prime', header: "E'", cell: ({ row }) => num(row.original.efficiency_score_prime) }
    ],
    []
  );

  const proposalColumns = useMemo<ColumnDef<ProposalRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'current_weight_pct', header: '현재', cell: ({ row }) => pct(row.original.current_weight_pct, false) },
      { accessorKey: 'target_weight_pct', header: '목표', cell: ({ row }) => pct(row.original.target_weight_pct, false) },
      { accessorKey: 'gap_pct', header: '갭', cell: ({ row }) => pct(row.original.gap_pct, false) },
      { accessorKey: 'adjusted_gap_pct', header: '조정갭', cell: ({ row }) => pct(row.original.adjusted_gap_pct, false) },
      { accessorKey: 'rc_over_pct', header: 'RC Over', cell: ({ row }) => pct(row.original.rc_over_pct, false) },
      { accessorKey: 'efficiency_score', header: 'E', cell: ({ row }) => num(row.original.efficiency_score) },
      { accessorKey: 'should_execute', header: '실행', cell: ({ row }) => (row.original.should_execute ? '실행' : '보류') }
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
      bench: parsedSettings.bench,
      momentum_weight: parsedSettings.momentumWeight
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
      bench: parsedSettings.bench,
      momentum_weight: parsedSettings.momentumWeight
    });
  }

  function runCurrentEvaluation() {
    const parsedSettings = settingsSchema.parse(settings);
    evaluationMutation.mutate({
      rc_over_thresh_pct: parsedSettings.rcOverThreshPct,
      e_thresh: parsedSettings.eThresh
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
    setEditingSnapshotId(snapshot.id);
    setEditingSnapshotName(snapshot.name);
    setEditingSnapshotNote(snapshot.note);
  }

  function cancelEditingSnapshot() {
    setEditingSnapshotId(null);
    setEditingSnapshotName('');
    setEditingSnapshotNote('');
  }

  function saveEditedSnapshot() {
    if (editingSnapshotId === null) return;
    updateSnapshotMutation.mutate({
      snapshotId: editingSnapshotId,
      payload: {
        name: editingSnapshotName,
        note: editingSnapshotNote
      }
    });
  }

  function removeSnapshot(snapshot: SnapshotSummary) {
    if (!window.confirm('이 스냅샷을 삭제할까요?')) return;
    setDeletingSnapshotId(snapshot.id);
    deleteSnapshotMutation.mutate(snapshot.id);
  }

  function saveNewOption() {
    if (!newOptionCode.trim() || !newOptionLabel.trim()) return;
    saveOptionMutation.mutate();
  }

  function updateTargetAllocation(index: number, field: keyof TargetAllocation, value: string) {
    setTargetAllocationRows((current) =>
      current.map((row, rowIndex) =>
        rowIndex === index
          ? {
              ...row,
              [field]: field === 'group_type' ? value : Number(value)
            }
          : row
      )
    );
  }

  function updateActionPriority(
    index: number,
    field: 'action_code' | 'label' | 'priority' | 'is_active',
    value: string | boolean
  ) {
    setActionPriorityRows((current) =>
      current.map((row, rowIndex) =>
        rowIndex === index
          ? {
              ...row,
              [field]: field === 'priority' ? Number(value) : value
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

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div>
          <p className="eyebrow">Portfolio Rebalancer</p>
          <h1>리밸런싱 워크벤치</h1>
        </div>
        <form className="settings-form">
          <label>
            평가 기간
            <select {...register('periodMode')}>
              <option value="months">개월</option>
              <option value="YTD">YTD</option>
              <option value="Max">Max</option>
            </select>
          </label>
          {settings.periodMode === 'months' && (
            <label>
              개월 수
              <input type="number" min="1" max="120" {...register('months')} />
            </label>
          )}
          <label>
            무위험 수익률 (%)
            <input type="number" step="0.1" {...register('rfPct')} />
          </label>
          <label>
            벤치마크
            <input type="text" {...register('bench')} />
          </label>
          <label>
            RC Over 임계값 (%)
            <input type="number" step="0.1" {...register('rcOverThreshPct')} />
          </label>
          <label>
            E 임계값
            <input type="number" step="0.05" min="0" max="1" {...register('eThresh')} />
          </label>
          <Controller
            control={control}
            name="momentumWeight"
            render={({ field }) => (
              <label>
                모멘텀 가중치 <strong>{field.value}</strong>
                <input type="range" min="0" max="0.5" step="0.05" {...field} />
              </label>
            )}
          />
        </form>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>입력 → 분석 → 평가</h2>
            <p>Python 계산 코어를 JSON API로 호출하고, React에서 결과를 검토합니다.</p>
          </div>
          <div className="status-strip">
            <span className={portfolio.length ? 'done' : ''}>1 입력</span>
            <span className={analysis ? 'done' : ''}>2 분석</span>
            <span className={evaluation ? 'done' : ''}>3 평가</span>
          </div>
        </header>

        <section className="mx-auto w-full max-w-6xl rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
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
                  disabled={selectedPortfolioId === null || !portfolio.length || saveSnapshotMutation.isPending}
                  onClick={saveCurrentSnapshot}
                >
                  {saveSnapshotMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Save className="h-4 w-4" />}
                  현재 상태 저장
                </button>
              </div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {savedSnapshots.map((snapshot) =>
                  editingSnapshotId === snapshot.id ? (
                    <div key={snapshot.id} className="space-y-2 rounded-lg border border-blue-200 bg-blue-50/70 p-3">
                      <input
                        className="table-input w-full"
                        value={editingSnapshotName}
                        placeholder="스냅샷 이름"
                        onChange={(event) => setEditingSnapshotName(event.target.value)}
                      />
                      <textarea
                        className="table-input min-h-16 w-full resize-none"
                        value={editingSnapshotNote}
                        placeholder="메모"
                        onChange={(event) => setEditingSnapshotNote(event.target.value)}
                      />
                      <div className="flex justify-end gap-2">
                        <button
                          type="button"
                          className="grid h-9 w-9 place-items-center rounded-lg text-slate-500 transition hover:bg-white hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
                          title="취소"
                          disabled={updateSnapshotMutation.isPending}
                          onClick={cancelEditingSnapshot}
                        >
                          <X className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          className="grid h-9 w-9 place-items-center rounded-lg bg-blue-800 text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                          title="저장"
                          disabled={updateSnapshotMutation.isPending}
                          onClick={saveEditedSnapshot}
                        >
                          {updateSnapshotMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Save className="h-4 w-4" />}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div
                      key={snapshot.id}
                      className="flex items-center gap-2 rounded-lg border border-slate-200 px-2 py-2 text-sm transition hover:border-blue-300 hover:bg-blue-50"
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
                        <Edit3 className="h-4 w-4" />
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
                  )
                )}
                {!savedSnapshots.length && (
                  <div className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-center text-sm text-slate-500">
                    저장 이력이 없습니다.
                  </div>
                )}
              </div>
              <ErrorLine
                error={
                  saveSnapshotMutation.error ??
                  loadSnapshotMutation.error ??
                  updateSnapshotMutation.error ??
                  deleteSnapshotMutation.error
                }
              />
            </div>
          </div>
        </section>

        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
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
                <div className="grid gap-2 sm:grid-cols-[1fr_1fr_1fr_auto]">
                  <select className="table-input" value={newOptionTable} onChange={(event) => setNewOptionTable(event.target.value as OptionTable)}>
                    <option value="groups">그룹</option>
                    <option value="roles">역할</option>
                    <option value="thesis_statuses">투자 논리 상태</option>
                  </select>
                  <input className="table-input" value={newOptionCode} placeholder="code" onChange={(event) => setNewOptionCode(event.target.value)} />
                  <input className="table-input" value={newOptionLabel} placeholder="표시명" onChange={(event) => setNewOptionLabel(event.target.value)} />
                  <button
                    type="button"
                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                    disabled={!newOptionCode.trim() || !newOptionLabel.trim() || saveOptionMutation.isPending}
                    onClick={saveNewOption}
                  >
                    {saveOptionMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Plus className="h-4 w-4" />}
                    저장
                  </button>
                </div>
                {newOptionTable === 'groups' && (
                  <select className="table-input max-w-xs" value={newOptionGroupType} onChange={(event) => setNewOptionGroupType(event.target.value)}>
                    {groupTypes.map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                )}
                <div className="grid gap-3">
                  {[
                    ['groups', '그룹', groupOptions],
                    ['roles', '역할', roleOptions],
                    ['thesis_statuses', '투자 논리 상태', thesisStatusOptions]
                  ].map(([table, label, options]) => (
                    <div key={String(table)}>
                      <div className="mb-2 text-sm font-bold text-slate-700">{String(label)}</div>
                      <div className="grid gap-2">
                        {(options as ConfigOption[]).map((option) => (
                          <div key={`${table}-${option.value}`} className="grid grid-cols-[1fr_auto] items-center gap-2 rounded-lg border border-slate-200 px-3 py-2">
                            <span className="min-w-0 truncate text-sm text-slate-700">
                              <strong>{option.label}</strong> · {option.value}
                              {option.group_type ? ` · ${option.group_type}` : ''}
                            </span>
                            <button
                              type="button"
                              className={cx(
                                'rounded-lg px-3 py-1.5 text-xs font-bold transition disabled:cursor-not-allowed disabled:opacity-50',
                                option.is_active ? 'bg-blue-50 text-blue-800 hover:bg-blue-100' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                              )}
                              disabled={optionActiveMutation.isPending}
                              onClick={() =>
                                optionActiveMutation.mutate({
                                  table: table as OptionTable,
                                  code: option.value,
                                  isActive: !option.is_active
                                })
                              }
                            >
                              {option.is_active ? '활성' : '비활성'}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-5">
                <div>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h4 className="font-semibold text-slate-950">타입별 목표 비중</h4>
                    <div className="flex gap-2">
                      <button type="button" className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-bold text-slate-700" onClick={() => setTargetAllocationRows((current) => [...current, { group_type: 'unknown', min: 0, target: 0, max: 0 }])}>
                        행 추가
                      </button>
                      <button type="button" className="rounded-lg bg-blue-800 px-3 py-2 text-xs font-bold text-white disabled:bg-slate-300" disabled={saveTargetsMutation.isPending} onClick={() => saveTargetsMutation.mutate()}>
                        저장
                      </button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {targetAllocationRows.map((row, index) => (
                      <div key={row.group_type} className="grid grid-cols-4 gap-2">
                        <input className="table-input" value={row.group_type} onChange={(event) => updateTargetAllocation(index, 'group_type', event.target.value)} />
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
                    <div className="flex gap-2">
                      <button type="button" className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-bold text-slate-700" onClick={() => setActionPriorityRows((current) => [...current, { action_code: '', label: '', priority: 99, is_active: true }])}>
                        행 추가
                      </button>
                      <button type="button" className="rounded-lg bg-blue-800 px-3 py-2 text-xs font-bold text-white disabled:bg-slate-300" disabled={savePrioritiesMutation.isPending} onClick={() => savePrioritiesMutation.mutate()}>
                        저장
                      </button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {actionPriorityRows.map((row, index) => (
                      <div key={row.action_code} className="grid grid-cols-[1fr_1fr_72px_56px] items-center gap-2">
                        <input className="table-input" value={row.action_code} onChange={(event) => updateActionPriority(index, 'action_code', event.target.value)} />
                        <input className="table-input" value={row.label} onChange={(event) => updateActionPriority(index, 'label', event.target.value)} />
                        <input className="table-input" type="number" value={row.priority} onChange={(event) => updateActionPriority(index, 'priority', event.target.value)} />
                        <input className="h-4 w-4 justify-self-center" type="checkbox" checked={row.is_active} onChange={(event) => updateActionPriority(index, 'is_active', event.target.checked)} />
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
                <ErrorLine error={saveOptionMutation.error ?? optionActiveMutation.error ?? saveTargetsMutation.error ?? savePrioritiesMutation.error ?? saveRulesMutation.error ?? configOptionsQuery.error ?? ipsConfigQuery.error} />
              </div>
            </div>
          </section>

          <WorkflowStepper
            steps={[
              { label: '포트폴리오 입력', complete: portfolio.length > 0, active: !portfolio.length },
              { label: '데이터 분석', complete: Boolean(analysis), active: portfolio.length > 0 && !analysis },
              { label: '평가 및 제안', complete: Boolean(evaluation), active: Boolean(analysis) && !evaluation }
            ]}
          />

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
                        {activeGroupOptions.map((group) => (
                          <option key={group.value} value={group.value}>
                            {group.label}
                          </option>
                        ))}
                        {row.group && !groupOptions.some((group) => group.value === row.group) ? (
                          <option value={String(row.group)}>기타: {row.group}</option>
                        ) : null}
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
                <span className="text-sm text-slate-500">역할, DCA, 투자 논리는 분석 후 세부 판단값에서 입력합니다.</span>
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
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <div className="grid h-8 w-8 place-items-center rounded-lg bg-violet-100 text-sm font-bold text-violet-800">2</div>
                  <h3 className="text-xl font-semibold text-slate-950">데이터 조회 & 보강</h3>
                </div>
                <p className="mt-2 text-sm text-slate-500">가격 데이터를 조회하고 포트폴리오/벤치마크/자산별 지표를 계산합니다. 이후 역할, DCA, 투자 논리를 보정합니다.</p>
              </div>
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
                <div className="min-w-[980px] space-y-2">
                  <div className="grid grid-cols-[0.8fr_0.85fr_1fr_1fr_56px_1fr] gap-2 px-1 text-xs font-bold uppercase text-slate-500">
                    <span>티커</span>
                    <span className="text-right">계좌 수익률 override</span>
                    <span>그룹</span>
                    <span>역할</span>
                    <span className="text-center">DCA</span>
                    <span>투자 논리</span>
                  </div>
                  {rows.map((row, index) => (
                    <div
                      className="grid grid-cols-[0.8fr_0.85fr_1fr_1fr_56px_1fr] items-center gap-2 rounded-lg border border-slate-200 bg-white p-2"
                      key={`detail-${row.ticker}-${index}`}
                    >
                      <div className="px-2 text-sm font-bold text-blue-700">{row.ticker || '미입력'}</div>
                      <input className="table-input text-right" value={String(row.return_total ?? '')} placeholder="자동 계산" type="number" onChange={(event) => updateRow(index, 'return_total', event.target.value)} />
                      <select className="table-input" value={String(row.group ?? '')} onChange={(event) => updateRow(index, 'group', event.target.value)}>
                        <option value="">그룹 선택</option>
                        {activeGroupOptions.map((group) => (
                          <option key={group.value} value={group.value}>
                            {group.label}
                          </option>
                        ))}
                        {row.group && !groupOptions.some((group) => group.value === row.group) ? (
                          <option value={String(row.group)}>기타: {row.group}</option>
                        ) : null}
                      </select>
                      <select className="table-input" value={String(row.role ?? '')} onChange={(event) => updateRow(index, 'role', event.target.value)}>
                        <option value="">역할 선택</option>
                        {activeRoleOptions.map((role) => (
                          <option key={role.value} value={role.value}>
                            {role.label}
                          </option>
                        ))}
                        {row.role && !roleOptions.some((role) => role.value === row.role) ? (
                          <option value={String(row.role)}>기타: {row.role}</option>
                        ) : null}
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
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <div className="grid h-8 w-8 place-items-center rounded-lg bg-cyan-100 text-sm font-bold text-cyan-900">3</div>
                  <h3 className="text-xl font-semibold text-slate-950">평가 & 실행 계획 제안</h3>
                </div>
                <p className="mt-2 text-sm text-slate-500">IPS 기준으로 실행 후보, 위험 초과, DCA 조정 신호를 확인합니다.</p>
              </div>
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                disabled={!analysis || rowsDirty || evaluationMutation.isPending}
                onClick={runCurrentEvaluation}
              >
                {evaluationMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Play className="h-4 w-4" />}
                평가 실행
              </button>
            </div>
            <ErrorLine error={evaluationMutation.error} />
            {analysis && rowsDirty && (
              <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
                세부 판단값 변경사항을 먼저 분석 결과에 반영해야 평가를 실행할 수 있습니다.
              </div>
            )}
            <DataTable data={evaluation?.proposal ?? []} columns={proposalColumns} emptyLabel="평가 결과가 아직 없습니다." />
            {evaluation && (
              <div className="mt-4 flex flex-wrap gap-3">
                <a className="download-link" href={csvDownloadUrl('proposal')}><Download className="h-4 w-4" /> 제안 CSV</a>
                <a className="download-link" href={csvDownloadUrl('ips_actions')}><Download className="h-4 w-4" /> IPS CSV</a>
                <a className="download-link" href={csvDownloadUrl('group_summary')}><Download className="h-4 w-4" /> 그룹 CSV</a>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

function WorkflowStepper({
  steps
}: {
  steps: Array<{ label: string; complete: boolean; active: boolean }>;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-5 shadow-sm">
      <div className="mx-auto grid max-w-3xl grid-cols-[auto_1fr_auto_1fr_auto] items-start gap-3">
        {steps.map((step, index) => (
          <StepItem key={step.label} index={index} step={step} isLast={index === steps.length - 1} />
        ))}
      </div>
    </div>
  );
}

function StepItem({
  index,
  step,
  isLast
}: {
  index: number;
  step: { label: string; complete: boolean; active: boolean };
  isLast: boolean;
}) {
  const stateClass = step.complete
    ? 'border-blue-800 bg-blue-800 text-white'
    : step.active
      ? 'border-blue-800 bg-white text-blue-800'
      : 'border-slate-300 bg-white text-slate-400';

  return (
    <>
      <div className="flex min-w-0 flex-col items-center gap-2 text-center">
        <div className={cx('grid h-10 w-10 place-items-center rounded-full border-2 text-sm font-bold', stateClass)}>
          {step.complete ? <CheckCircle2 className="h-5 w-5" /> : index + 1}
        </div>
        <span className={cx('text-sm font-bold', step.complete || step.active ? 'text-blue-800' : 'text-slate-500')}>
          {step.label}
        </span>
      </div>
      {!isLast && (
        <div className={cx('mt-5 h-0.5 min-w-8', step.complete ? 'bg-blue-800' : 'bg-slate-200')} />
      )}
    </>
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
