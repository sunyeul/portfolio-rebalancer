import type { ColumnDef } from '@tanstack/react-table';
import { useMutation } from '@tanstack/react-query';
import {
  AlertCircle,
  BarChart3,
  Download,
  FileUp,
  Loader2,
  Play,
  Plus,
  RefreshCcw,
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

  const assetColumns = useMemo<ColumnDef<AssetRow>[]>(
    () => [
      { accessorKey: 'ticker', header: '티커' },
      { accessorKey: 'allocation', header: '입력 비중', cell: ({ row }) => pct(row.original.allocation, false) },
      { accessorKey: 'weight', header: '정규화 비중', cell: ({ row }) => pct(row.original.weight) },
      { accessorKey: 'group', header: '그룹' },
      { accessorKey: 'role', header: '역할' },
      { accessorKey: 'dca_enabled', header: 'DCA', cell: ({ row }) => (row.original.dca_enabled ? '대상' : '제외') },
      { accessorKey: 'thesis_status', header: '투자 논리' }
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

        <section className="panel input-panel">
          <div className="panel-header">
            <div>
              <h3>포트폴리오 입력</h3>
              <p>티커와 비중을 붙여넣거나 CSV/TSV 파일을 업로드하세요.</p>
            </div>
            <button type="button" className="ghost-button" onClick={() => syncText(sampleText)}>
              <RefreshCcw size={16} /> 예시
            </button>
          </div>

          <div className="input-grid">
            <div className="paste-box">
              <textarea value={text} onChange={(event) => syncText(event.target.value)} />
              <label className="file-button">
                <FileUp size={16} /> CSV/TSV 업로드
                <input
                  type="file"
                  accept=".csv,.tsv,text/csv,text/tab-separated-values"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) csvMutation.mutate(file);
                    event.target.value = '';
                  }}
                />
              </label>
            </div>

            <div className="edit-box">
              <div className="summary-row">
                <span>유효 행 {validRows.length}개</span>
                <span>합계 {totalAllocation.toFixed(2)}%</span>
                <span>중복 {new Set(duplicateTickers).size}개</span>
              </div>
              <div className="editable-table">
                {rows.map((row, index) => (
                  <div className="editable-row" key={`${row.ticker}-${index}`}>
                    <input value={String(row.ticker ?? '')} placeholder="VOO" onChange={(event) => updateRow(index, 'ticker', event.target.value)} />
                    <input value={String(row.allocation ?? '')} placeholder="40" type="number" onChange={(event) => updateRow(index, 'allocation', event.target.value)} />
                    <input value={String(row.return_total ?? '')} placeholder="-12" type="number" onChange={(event) => updateRow(index, 'return_total', event.target.value)} />
                    <input value={String(row.group ?? '')} placeholder="core" onChange={(event) => updateRow(index, 'group', event.target.value)} />
                    <input value={String(row.role ?? '')} placeholder="role" onChange={(event) => updateRow(index, 'role', event.target.value)} />
                    <label className="check-cell">
                      <input checked={Boolean(row.dca_enabled)} type="checkbox" onChange={(event) => updateRow(index, 'dca_enabled', event.target.checked)} />
                    </label>
                    <input value={String(row.thesis_status ?? '')} placeholder="intact" onChange={(event) => updateRow(index, 'thesis_status', event.target.value)} />
                    <button type="button" aria-label="행 삭제" onClick={() => setRows((current) => current.filter((_, rowIndex) => rowIndex !== index))}>
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="action-row">
                <button type="button" className="ghost-button" onClick={() => setRows((current) => [...current, blankRow()])}>
                  <Plus size={16} /> 행 추가
                </button>
                <button type="button" className="primary-button" disabled={!validRows.length || portfolioMutation.isPending} onClick={() => portfolioMutation.mutate(validRows)}>
                  {portfolioMutation.isPending ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
                  포트폴리오 확정
                </button>
              </div>
            </div>
          </div>
          <ErrorLine error={portfolioMutation.error ?? csvMutation.error} />
          <DataTable data={portfolio} columns={assetColumns} emptyLabel="정규화된 포트폴리오가 아직 없습니다." />
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>데이터 분석</h3>
              <p>가격 데이터를 조회하고 포트폴리오/벤치마크/자산별 지표를 계산합니다.</p>
            </div>
            <button type="button" className="primary-button" disabled={!portfolio.length || analysisMutation.isPending} onClick={runCurrentAnalysis}>
              {analysisMutation.isPending ? <Loader2 className="spin" size={16} /> : <BarChart3 size={16} />}
              분석 실행
            </button>
          </div>
          <ErrorLine error={analysisMutation.error} />
          {analysis && (
            <>
              <div className="metric-grid">
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
              <Download size={16} /> 메트릭 CSV
            </a>
          ) : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>평가 및 제안</h3>
              <p>IPS 기준으로 실행 후보, 위험 초과, DCA 조정 신호를 확인합니다.</p>
            </div>
            <button type="button" className="primary-button" disabled={!analysis || evaluationMutation.isPending} onClick={runCurrentEvaluation}>
              {evaluationMutation.isPending ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              평가 실행
            </button>
          </div>
          <ErrorLine error={evaluationMutation.error} />
          <DataTable data={evaluation?.proposal ?? []} columns={proposalColumns} emptyLabel="평가 결과가 아직 없습니다." />
          {evaluation && (
            <div className="download-row">
              <a href={csvDownloadUrl('proposal')}><Download size={16} /> 제안 CSV</a>
              <a href={csvDownloadUrl('ips_actions')}><Download size={16} /> IPS CSV</a>
              <a href={csvDownloadUrl('group_summary')}><Download size={16} /> 그룹 CSV</a>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function ErrorLine({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="error-line">
      <AlertCircle size={16} /> {error.message}
    </div>
  );
}

function ChartBlock({ data }: { data: Array<{ ticker: string; weight: number; risk: number }> }) {
  if (!data.length) return null;
  return (
    <div className="chart-block">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ticker" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="weight" fill="#2563eb" name="비중 %" />
          <Bar dataKey="risk" fill="#d97706" name="위험기여도 %" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

