import type { ColumnDef } from '@tanstack/react-table';
import { useMutation } from '@tanstack/react-query';
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  Edit3,
  FileUp,
  LineChart,
  Loader2,
  Play,
  Plus,
  RefreshCcw,
  ShieldCheck,
  Trash2
} from 'lucide-react';
import { useMemo, useState } from 'react';
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
  type EvaluationResponse,
  type MetricRow,
  type ProposalRow,
  csvDownloadUrl,
  runAnalysis,
  runEvaluation,
  submitPortfolio,
  uploadPortfolioCsv
} from './lib/api';
import { blankRow, parsePortfolioText } from './lib/parser';
import { type PortfolioRowInput, type SettingsValues, settingsSchema } from './lib/schemas';

const sampleText = 'VOO 40\nQQQ 25\nSOXX 15\nUFO 3 watch\nIONQ 2 -12 watch';
const groupOptions = [
  { value: 'ungrouped', label: '미분류' },
  { value: 'core', label: '핵심 자산' },
  { value: 'satellite_ai_infra', label: '위성: AI 인프라' },
  { value: 'satellite_ai_software', label: '위성: AI 소프트웨어' },
  { value: 'satellite_space', label: '위성: 우주/항공' },
  { value: 'satellite_quantum', label: '위성: 양자' },
  { value: 'korea_equity', label: '한국 주식' },
  { value: 'bond_mixed', label: '채권/혼합' },
  { value: 'cash', label: '현금' }
];
const roleOptions = [
  { value: 'unknown', label: '미정' },
  { value: 'broad_etf', label: '광범위 ETF' },
  { value: 'theme_etf', label: '테마 ETF' },
  { value: 'individual', label: '개별 종목' },
  { value: 'duplicate', label: '중복 포지션' },
  { value: 'small_position', label: '소액 포지션' }
];
const thesisStatusOptions = [
  { value: 'unknown', label: '미정' },
  { value: 'intact', label: '유효' },
  { value: 'watch', label: '관찰' },
  { value: 'broken', label: '훼손' }
];

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function optionLabel(options: Array<{ value: string; label: string }>, value: string | null | undefined) {
  if (!value) return '미정';
  return options.find((option) => option.value === value)?.label ?? value;
}

function pct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined) return 'N/A';
  return `${(fromUnit ? value * 100 : value).toFixed(2)}%`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined) return 'N/A';
  return value.toFixed(2);
}

export function App() {
  const [text, setText] = useState(sampleText);
  const [rows, setRows] = useState<PortfolioRowInput[]>(() => parsePortfolioText(sampleText));
  const [portfolio, setPortfolio] = useState<AssetRow[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);

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
      setPortfolio(data.assets);
      setAnalysis(null);
      setEvaluation(null);
    }
  });

  const csvMutation = useMutation({
    mutationFn: uploadPortfolioCsv,
    onSuccess: (data) => {
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

  const validRows = rows.filter((row) => row.ticker && row.allocation !== '');
  const totalAllocation = validRows.reduce(
    (sum, row) => sum + (Number.parseFloat(String(row.allocation)) || 0),
    0
  );
  const duplicateTickers = validRows
    .map((row) => row.ticker)
    .filter((ticker, index, tickers) => tickers.indexOf(ticker) !== index);
  const duplicateCount = new Set(duplicateTickers).size;

  const assetColumns = useMemo<ColumnDef<AssetRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'allocation', header: '입력 비중', cell: ({ row }) => pct(row.original.allocation, false) },
      { accessorKey: 'weight', header: '정규화 비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'group', header: '그룹', cell: ({ row }) => optionLabel(groupOptions, row.original.group) },
      { accessorKey: 'role', header: '역할', cell: ({ row }) => optionLabel(roleOptions, row.original.role) },
      { accessorKey: 'dca_enabled', header: 'DCA', cell: ({ row }) => (row.original.dca_enabled ? '대상' : '제외') },
      { accessorKey: 'thesis_status', header: '투자 논리', cell: ({ row }) => optionLabel(thesisStatusOptions, row.original.thesis_status) }
    ],
    []
  );

  const metricColumns = useMemo<ColumnDef<MetricRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'weight', header: '비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'cagr', header: 'CAGR', cell: ({ row }) => pct(row.original.cagr) },
      { accessorKey: 'volatility', header: '변동성', cell: ({ row }) => pct(row.original.volatility) },
      { accessorKey: 'sharpe', header: '샤프', cell: ({ row }) => num(row.original.sharpe) },
      { accessorKey: 'risk_contribution', header: '위험기여도', cell: ({ row }) => pct(row.original.risk_contribution) },
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

  function runCurrentEvaluation() {
    const parsedSettings = settingsSchema.parse(settings);
    evaluationMutation.mutate({
      rc_over_thresh_pct: parsedSettings.rcOverThreshPct,
      e_thresh: parsedSettings.eThresh
    });
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

        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
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
                    티커와 비중을 한 줄씩 입력하세요. 예: VOO 40, QQQ 25%
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
                  <p className="mt-2 text-sm text-slate-500">붙여넣은 내용을 확인하고 필요한 값만 직접 수정하세요.</p>
                </div>
                <div className="grid grid-cols-3 gap-3 text-right">
                  <SummaryStat label="유효 행" value={`${validRows.length}개`} />
                  <SummaryStat label="합계 비중" value={`${totalAllocation.toFixed(2)}%`} />
                  <SummaryStat label="중복" value={`${duplicateCount}개`} tone={duplicateCount ? 'warn' : 'default'} />
                </div>
              </div>

              <div className="overflow-x-auto p-4">
                <div className="min-w-[900px] space-y-2">
                  <div className="grid grid-cols-[0.8fr_0.75fr_0.8fr_1fr_1fr_56px_1fr_44px] gap-2 px-1 text-xs font-bold uppercase text-slate-500">
                    <span>티커</span>
                    <span>비중</span>
                    <span>수익률</span>
                    <span>그룹</span>
                    <span>역할</span>
                    <span className="text-center">DCA</span>
                    <span>투자 논리</span>
                    <span />
                  </div>
                  {rows.map((row, index) => (
                    <div
                      className="grid grid-cols-[0.8fr_0.75fr_0.8fr_1fr_1fr_56px_1fr_44px] items-center gap-2 rounded-lg border border-slate-200 bg-white p-2 transition hover:bg-slate-50"
                      key={`${row.ticker}-${index}`}
                    >
                      <input className="table-input font-bold text-blue-700" value={String(row.ticker ?? '')} placeholder="VOO" onChange={(event) => updateRow(index, 'ticker', event.target.value)} />
                      <input className="table-input text-right" value={String(row.allocation ?? '')} placeholder="40" type="number" onChange={(event) => updateRow(index, 'allocation', event.target.value)} />
                      <input className="table-input text-right" value={String(row.return_total ?? '')} placeholder="-12" type="number" onChange={(event) => updateRow(index, 'return_total', event.target.value)} />
                      <select className="table-input" value={String(row.group ?? '')} onChange={(event) => updateRow(index, 'group', event.target.value)}>
                        <option value="">그룹 선택</option>
                        {groupOptions.map((group) => (
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
                        {roleOptions.map((role) => (
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
                        {thesisStatusOptions.map((status) => (
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
                <span className="text-sm text-slate-500">확정 후 정규화된 결과가 아래 표에 반영됩니다.</span>
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
                <p className="mt-2 text-sm text-slate-500">가격 데이터를 조회하고 포트폴리오/벤치마크/자산별 지표를 계산합니다.</p>
              </div>
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
                disabled={!portfolio.length || analysisMutation.isPending}
                onClick={runCurrentAnalysis}
              >
                {analysisMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <BarChart3 className="h-4 w-4" />}
                분석 실행
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
                disabled={!analysis || evaluationMutation.isPending}
                onClick={runCurrentEvaluation}
              >
                {evaluationMutation.isPending ? <Loader2 className="spin h-4 w-4" /> : <Play className="h-4 w-4" />}
                평가 실행
              </button>
            </div>
            <ErrorLine error={evaluationMutation.error} />
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
