import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BarChart3,
  ClipboardList,
  PencilLine,
  Loader2,
  Play,
  Plus,
  ShieldCheck,
  Trash2,
  X
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  type AnalysisResponse,
  type AssetRow,
  type EvaluationRecord,
  type EvaluationRun,
  type EvaluationResponse,
  type SavedPortfolio,
  type SnapshotLoadResponse,
  type SnapshotSummary,
  createPortfolio,
  deleteSnapshot,
  listPortfolios,
  listSnapshots,
  loadSnapshot,
  runAnalysis,
  runEvaluation,
  runSnapshotEvaluation,
  saveSnapshotEvaluation,
  saveSnapshot,
  submitPortfolio,
  updateSnapshot
} from './lib/api';
import {
  layerValues,
  thesisStatusValues,
  type LayerType,
  type PortfolioRowInput,
  type ThesisStatusInput
} from './lib/schemas';
import { ReviewCopilot, ReviewCopilotHost } from './copilot/ReviewCopilot';
import { GeneratedSurfaceHost } from './a2ui/GeneratedSurfaceHost';
import { GenerativeUiProvider, useGenerativeUi } from './a2ui/GenerativeUiContext';
import { ReviewQueueTriageSurface } from './a2ui/renderers/ReviewQueueTriageSurface';
import { requestReviewQueueAgentExplanations } from './a2ui/utils/agentExplanations';
import {
  buildDefaultEvaluationGraphSurface,
  buildReviewQueueTriageSurface,
  defaultReviewDecisionsFromEvaluation,
  mergeReviewQueueAgentExplanations
} from './a2ui/utils/builders';
import { validateReviewA2UISurface } from './a2ui/utils/validation';

const DEFAULT_BENCHMARK = 'SPY:80,QQQ:20';

const layerLabels: Record<LayerType, string> = {
  core: '코어',
  satellite: '위성',
  experiment: '실험'
};

const thesisLabels: Record<ThesisStatusInput, string> = {
  valid: '유효',
  watch: '관찰',
  broken: '훼손',
  unknown: '미정'
};

const statusLabels: Record<string, string> = {
  OK: '정상',
  Watch: '관찰',
  Review: '점검',
  Action: '조치 검토'
};

const DEFAULT_LAYER_BENCHMARKS: Record<LayerType, string> = {
  core: DEFAULT_BENCHMARK,
  satellite: 'QQQ',
  experiment: 'QQQ'
};
const ANALYSIS_DEFAULT_RF = 0.025;

type PortfolioInputRow = {
  id: string;
  ticker: string;
  allocation: string;
  layer: LayerType;
  thesis_status: ThesisStatusInput;
};

type WorkflowPending = 'portfolio' | 'analysis' | 'evaluation' | null;
type ManagementPending = 'portfolios' | 'portfolio' | 'save' | 'snapshot' | 'delete' | 'update' | 'evaluation-save' | null;
type SnapshotModalMode = 'create' | 'edit';
type SortDirection = 'asc' | 'desc';
type SortState<ColumnId extends string> = { column: ColumnId; direction: SortDirection } | null;
type SortValue = string | number | null | undefined;

type SortableColumn<Row, ColumnId extends string> = {
  id: ColumnId;
  label: string;
  getValue: (row: Row) => SortValue;
};

const initialRows: PortfolioInputRow[] = [
  {
    id: 'row-voo',
    ticker: 'VOO',
    allocation: '70',
    layer: 'core',
    thesis_status: 'valid'
  },
  {
    id: 'row-smh',
    ticker: 'SMH',
    allocation: '8',
    layer: 'satellite',
    thesis_status: 'valid'
  },
  {
    id: 'row-qqq',
    ticker: 'QQQ',
    allocation: '12',
    layer: 'satellite',
    thesis_status: 'watch'
  },
  {
    id: 'row-gld',
    ticker: 'GLD',
    allocation: '10',
    layer: 'core',
    thesis_status: 'valid'
  }
];

function pct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  const numeric = fromUnit ? value * 100 : value;
  const sign = numeric > 0 ? '+' : '';
  return `${sign}${numeric.toFixed(2)}%`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return value.toFixed(2);
}

function statusClass(status: string) {
  if (status === 'OK') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  if (status === 'Watch') return 'border-amber-200 bg-amber-50 text-amber-800';
  if (status === 'Review') return 'border-blue-200 bg-blue-50 text-blue-800';
  return 'border-red-200 bg-red-50 text-red-800';
}

function statusLabel(status: string) {
  return statusLabels[status] ?? status;
}

function layerLabel(value: string | null | undefined) {
  return layerLabels[value as LayerType] ?? value ?? '미지정';
}

function thesisLabel(value: string | null | undefined) {
  return thesisLabels[value as ThesisStatusInput] ?? value ?? '미정';
}

function nextSortState<ColumnId extends string>(current: SortState<ColumnId>, column: ColumnId): SortState<ColumnId> {
  if (current?.column !== column) return { column, direction: 'asc' };
  if (current.direction === 'asc') return { column, direction: 'desc' };
  return null;
}

function isMissingSortValue(value: SortValue) {
  return value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value));
}

function compareSortValues(left: SortValue, right: SortValue, direction: SortDirection) {
  const leftMissing = isMissingSortValue(left);
  const rightMissing = isMissingSortValue(right);
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;

  const result =
    typeof left === 'number' && typeof right === 'number'
      ? left - right
      : String(left).localeCompare(String(right), 'ko', { numeric: true, sensitivity: 'base' });

  return direction === 'asc' ? result : -result;
}

function sortRows<Row, ColumnId extends string>(
  rows: Row[],
  columns: Array<SortableColumn<Row, ColumnId>>,
  sortState: SortState<ColumnId>
) {
  if (!sortState) return rows;
  const column = columns.find((candidate) => candidate.id === sortState.column);
  if (!column) return rows;

  return rows
    .map((row, index) => ({ row, index }))
    .sort((left, right) => {
      const result = compareSortValues(column.getValue(left.row), column.getValue(right.row), sortState.direction);
      return result === 0 ? left.index - right.index : result;
    })
    .map(({ row }) => row);
}

function SortableHeader<ColumnId extends string>({
  id,
  label,
  sortState,
  onSort
}: {
  id: ColumnId;
  label: string;
  sortState: SortState<ColumnId>;
  onSort: (id: ColumnId) => void;
}) {
  const direction = sortState?.column === id ? sortState.direction : null;
  const Icon = direction === 'asc' ? ArrowUp : direction === 'desc' ? ArrowDown : ArrowUpDown;
  const ariaSort = direction === 'asc' ? 'ascending' : direction === 'desc' ? 'descending' : 'none';

  return (
    <th className="px-2 py-2" aria-sort={ariaSort}>
      <button
        type="button"
        className="inline-flex items-center gap-1 rounded text-left font-bold text-slate-500 transition hover:text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-700 focus:ring-offset-2"
        onClick={() => onSort(id)}
      >
        <span>{label}</span>
        <Icon className={`h-3.5 w-3.5 ${direction ? 'text-slate-900' : 'text-slate-400'}`} aria-hidden="true" />
      </button>
    </th>
  );
}

function toNumber(value: string) {
  const numeric = Number(value.replace(/,/g, '').trim());
  return Number.isFinite(numeric) ? numeric : Number.NaN;
}

function isMeaningfulRow(row: PortfolioInputRow) {
  return row.ticker.trim() !== '' || row.allocation.trim() !== '';
}

function makeInputRow(id: string): PortfolioInputRow {
  return {
    id,
    ticker: '',
    allocation: '',
    layer: 'core',
    thesis_status: 'valid'
  };
}

function isLayerType(value: string | null | undefined): value is LayerType {
  return layerValues.includes(value as LayerType);
}

function editableThesisStatus(value: string | null | undefined): PortfolioInputRow['thesis_status'] {
  if (value === 'watch' || value === 'broken' || value === 'unknown') return value;
  return 'valid';
}

function inputRowsFromAssets(assets: AssetRow[]): PortfolioInputRow[] {
  if (assets.length === 0) return initialRows;
  return assets.map((asset, index) => {
    const layer = isLayerType(asset.layer) ? asset.layer : 'core';
    const allocation = Number.isFinite(asset.allocation) ? asset.allocation : asset.weight * 100;
    return {
      id: `loaded-${asset.ticker}-${index}`,
      ticker: asset.ticker,
      allocation: String(Number.isFinite(allocation) ? allocation : ''),
      layer,
      thesis_status: editableThesisStatus(asset.thesis_status)
    };
  });
}

function normalizedLayerBenchmarks(layerBenchmarks: Record<LayerType, string>) {
  return Object.fromEntries(
    layerValues.map((layer) => {
      const value = layerBenchmarks[layer].trim().toUpperCase();
      return [layer, value || DEFAULT_LAYER_BENCHMARKS[layer]];
    })
  ) as Record<LayerType, string>;
}

function isEvaluationPeriod(value: string | null | undefined): value is '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max' {
  return value === '1M' || value === '3M' || value === '6M' || value === 'YTD' || value === '1Y' || value === 'Max';
}

function layerBenchmarksFromEvaluationRun(evaluationRun: EvaluationRun | null) {
  const savedBenchmarks = evaluationRun?.settings.layer_benchmarks ?? {};
  return Object.fromEntries(
    layerValues.map((layer) => {
      const value = savedBenchmarks[layer];
      return [layer, value ? String(value).trim().toUpperCase() : DEFAULT_LAYER_BENCHMARKS[layer]];
    })
  ) as Record<LayerType, string>;
}

function todayIsoDate() {
  const date = new Date();
  const offsetDate = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return offsetDate.toISOString().slice(0, 10);
}

function formatSnapshotTimestamp(value: string | null | undefined) {
  if (!value) return 'N/A';
  return value.replace('T', ' ').slice(0, 16);
}

function analysisPeriodFromEvaluationPeriod(period: '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max') {
  if (period === '1Y') return 12;
  if (period === 'YTD' || period === 'Max') return period;
  return Number(period.replace('M', ''));
}

function toPortfolioPayload(rows: PortfolioInputRow[]): PortfolioRowInput[] {
  return rows
    .filter(isMeaningfulRow)
    .map((row) => ({
      ticker: row.ticker.trim().toUpperCase(),
      allocation: row.allocation.trim(),
      layer: row.layer,
      thesis_status: row.thesis_status
    }));
}

function inputRowErrors(rows: PortfolioInputRow[]) {
  const errors = new Map<string, string>();
  rows.filter(isMeaningfulRow).forEach((row) => {
    if (row.ticker.trim() === '') {
      errors.set(row.id, '티커를 입력해주세요.');
      return;
    }
    if (row.allocation.trim() === '') {
      errors.set(row.id, '비중을 입력해주세요.');
      return;
    }
    const allocation = toNumber(row.allocation);
    if (Number.isNaN(allocation)) {
      errors.set(row.id, '비중은 숫자로 입력해주세요.');
      return;
    }
    if (allocation < 0) {
      errors.set(row.id, '비중은 0 이상이어야 합니다.');
    }
  });
  return errors;
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-flex rounded-md border px-2 py-1 text-xs font-bold ${statusClass(status)}`}>
      {statusLabel(status)}
    </span>
  );
}

function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-800">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

function EvaluationRunHeader({ evaluationRun }: { evaluationRun: EvaluationRun | null }) {
  if (evaluationRun === null) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600">
        임시 평가 결과입니다. 보유현황 스냅샷을 선택한 뒤 다시 평가하면 평가 기록으로 저장됩니다.
      </section>
    );
  }

  const statusText = evaluationRun.status === 'active' ? '최신 평가' : '이전 평가';

  return (
    <section className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 md:flex-row md:items-center md:justify-between">
      <div>
        평가 기록 #{evaluationRun.id} · {statusText} · 생성 {formatSnapshotTimestamp(evaluationRun.created_at)}
      </div>
      {evaluationRun.is_stale ? (
        <span className="inline-flex w-fit rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs font-bold text-amber-800">
          재평가 권장
        </span>
      ) : (
        <span className="inline-flex w-fit rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-bold text-emerald-800">
          현재 설정과 호환
        </span>
      )}
    </section>
  );
}

function PortfolioInputTable({
  rows,
  allocationTotal,
  rowErrors,
  onAdd,
  onDelete,
  onChange
}: {
  rows: PortfolioInputRow[];
  allocationTotal: number;
  rowErrors: Map<string, string>;
  onAdd: () => void;
  onDelete: (id: string) => void;
  onChange: (id: string, patch: Partial<PortfolioInputRow>) => void;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white">
      <div className="flex flex-col gap-3 border-b border-slate-100 p-4 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="text-xs font-bold uppercase text-slate-500">포트폴리오 입력</div>
          <h2 className="mt-1 text-lg font-bold text-slate-950">계층/종목 평가 워크벤치</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-bold text-slate-700">
            입력 합계 {allocationTotal.toFixed(2)}%
          </span>
          <button
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-bold text-white transition hover:bg-slate-800"
            type="button"
            onClick={onAdd}
          >
            <Plus className="h-4 w-4" />
            행 추가
          </button>
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="m-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm font-semibold text-slate-500">
          입력 행이 없습니다. 행을 추가한 뒤 티커와 비중을 입력해주세요.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full table-fixed border-collapse text-left text-sm">
            <colgroup>
              <col className="w-[16%]" />
              <col className="w-[13%]" />
              <col className="w-[15%]" />
              <col className="w-[34%]" />
              <col className="w-[12%]" />
            </colgroup>
            <thead className="bg-slate-50 text-xs font-bold uppercase text-slate-500">
              <tr>
                <th className="border-b border-slate-200 px-3 py-3">티커</th>
                <th className="border-b border-slate-200 px-3 py-3">비중(%)</th>
                <th className="border-b border-slate-200 px-3 py-3">계층</th>
                <th className="border-b border-slate-200 px-3 py-3">논리 상태</th>
                <th className="border-b border-slate-200 px-3 py-3 text-center">삭제</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const rowError = rowErrors.get(row.id);
                return (
                  <tr key={row.id} className="border-b border-slate-100 transition hover:bg-slate-50/70">
                    <td className="min-w-0 px-3 py-3 align-top">
                      <input
                        aria-label="티커"
                        className="w-full min-w-0 rounded-md border border-slate-200 bg-white px-2.5 py-2 font-bold uppercase text-slate-900 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                        placeholder="VOO"
                        value={row.ticker}
                        onChange={(event) => onChange(row.id, { ticker: event.target.value.toUpperCase() })}
                      />
                      {rowError ? <div className="mt-1 text-xs font-semibold text-red-600">{rowError}</div> : null}
                    </td>
                    <td className="min-w-0 px-3 py-3 align-top">
                      <input
                        aria-label="비중"
                        className="w-full min-w-0 rounded-md border border-slate-200 bg-white px-2.5 py-2 text-right font-semibold text-slate-900 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                        inputMode="decimal"
                        placeholder="0"
                        value={row.allocation}
                        onChange={(event) => onChange(row.id, { allocation: event.target.value })}
                      />
                    </td>
                    <td className="min-w-0 px-3 py-3 align-top">
                      <select
                        aria-label="계층"
                        className="w-full min-w-0 rounded-md border border-slate-200 bg-white px-2.5 py-2 font-semibold text-slate-800 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                        value={row.layer}
                        onChange={(event) => onChange(row.id, { layer: event.target.value as LayerType })}
                      >
                        {layerValues.map((value) => (
                          <option key={value} value={value}>
                            {layerLabels[value]}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="min-w-0 px-3 py-3 align-top">
                      <select
                        aria-label="논리 상태"
                        className="w-full min-w-0 rounded-md border border-slate-200 bg-white px-2.5 py-2 font-semibold text-slate-800 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                        value={row.thesis_status}
                        onChange={(event) => onChange(row.id, { thesis_status: event.target.value as PortfolioInputRow['thesis_status'] })}
                      >
                        {thesisStatusValues.map((value) => (
                          <option key={value} value={value}>
                            {thesisLabels[value]}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-3 py-3 text-center align-top">
                      <button
                        aria-label={`${row.ticker || '입력 행'} 삭제`}
                        className="inline-grid h-9 w-9 place-items-center rounded-md text-slate-400 transition hover:bg-red-50 hover:text-red-600"
                        type="button"
                        onClick={() => onDelete(row.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function PortfolioInputModal({
  allocationTotal,
  allocationWarning,
  canRunWorkflow,
  isOpen,
  pending,
  rowErrors,
  rows,
  workflowButtonLabel,
  onAdd,
  onChange,
  onClose,
  onDelete,
  onRun
}: {
  allocationTotal: number;
  allocationWarning: string | null;
  canRunWorkflow: boolean;
  isOpen: boolean;
  pending: 'portfolio' | 'analysis' | 'evaluation' | null;
  rowErrors: Map<string, string>;
  rows: PortfolioInputRow[];
  workflowButtonLabel: string;
  onAdd: () => void;
  onChange: (id: string, patch: Partial<PortfolioInputRow>) => void;
  onClose: () => void;
  onDelete: (id: string) => void;
  onRun: () => void;
}) {
  if (!isOpen) return null;
  const isBusy = pending !== null;

  return (
    <div
      aria-labelledby="portfolio-input-title"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      role="dialog"
      onClick={() => {
        if (!isBusy) onClose();
      }}
    >
      <div
        className="flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex flex-col gap-3 border-b border-slate-200 p-5 md:flex-row md:items-start md:justify-between">
          <div>
            <h2 id="portfolio-input-title" className="text-xl font-bold text-slate-950">
              포트폴리오 구성 편집
            </h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">
              티커와 비중을 입력하면 평가 대상 포트폴리오로 적용됩니다. 계층과 논리 상태는 기본값으로 보강할 수 있습니다.
            </p>
          </div>
          <button
            aria-label="입력 모달 닫기"
            className="inline-grid h-9 w-9 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50 p-5">
          <PortfolioInputTable
            rows={rows}
            allocationTotal={allocationTotal}
            rowErrors={rowErrors}
            onAdd={onAdd}
            onDelete={onDelete}
            onChange={onChange}
          />
          {allocationWarning ? (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
              {allocationWarning}
            </div>
          ) : null}
        </div>

        <div className="flex flex-col gap-3 border-t border-slate-200 bg-white p-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm font-semibold text-slate-500">
            입력 합계 {allocationTotal.toFixed(2)}%
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
            <button
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
              disabled={isBusy}
              type="button"
              onClick={onClose}
            >
              닫기
            </button>
            <button
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={!canRunWorkflow}
              type="button"
              onClick={onRun}
            >
              {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {workflowButtonLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function PortfolioPreview({ portfolio }: { portfolio: AssetRow[] }) {
  if (!portfolio.length) {
    return (
      <section className="grid justify-items-center gap-3 rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center">
        <div>
          <h2 className="text-base font-bold text-slate-950">아직 적용된 포트폴리오가 없습니다.</h2>
          <p className="mt-2 text-sm font-semibold text-slate-500">
            먼저 포트폴리오를 입력하면 평가를 실행할 수 있습니다.
          </p>
        </div>
      </section>
    );
  }
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-bold text-slate-950">적용된 포트폴리오</h2>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              <th className="px-2 py-2">티커</th>
              <th className="px-2 py-2">비중</th>
              <th className="px-2 py-2">계층</th>
              <th className="px-2 py-2">논리 상태</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.map((asset) => (
              <tr key={asset.ticker} className="border-t border-slate-100">
                <td className="px-2 py-2 font-bold text-slate-900">{asset.ticker}</td>
                <td className="px-2 py-2">{pct(asset.weight)}</td>
                <td className="px-2 py-2">{layerLabel(asset.layer)}</td>
                <td className="px-2 py-2">{thesisLabel(asset.thesis_status)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetricsStrip({ analysis }: { analysis: AnalysisResponse | null }) {
  if (!analysis) return null;
  const metrics = [
    ['포트폴리오 CAGR', pct(analysis.portfolio_metrics.cagr)],
    ['변동성', pct(analysis.portfolio_metrics.volatility)],
    ['최대낙폭', pct(analysis.portfolio_metrics.max_drawdown ?? null)],
    ['누락 티커', String(analysis.missing_tickers.length)]
  ];
  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {metrics.map(([label, value]) => (
        <div key={label} className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="text-xs font-bold uppercase text-slate-500">{label}</div>
          <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
        </div>
      ))}
    </section>
  );
}

type LayerDashboardColumnId =
  | 'layer'
  | 'current'
  | 'gap'
  | 'return'
  | 'benchmarkReturn'
  | 'excessReturn'
  | 'mdd'
  | 'efficiency'
  | 'status';

const layerDashboardColumns: Array<SortableColumn<EvaluationRecord, LayerDashboardColumnId>> = [
  { id: 'layer', label: '계층', getValue: (row) => layerLabel(row.unit.name) },
  { id: 'current', label: '현재', getValue: (row) => row.output.current_weight },
  { id: 'gap', label: '목표-현재', getValue: (row) => row.output.weight_gap },
  { id: 'return', label: '수익률', getValue: (row) => row.output.period_return },
  { id: 'benchmarkReturn', label: '벤치', getValue: (row) => row.output.benchmark_return },
  { id: 'excessReturn', label: '초과', getValue: (row) => row.output.benchmark_excess_return },
  { id: 'mdd', label: 'MDD', getValue: (row) => row.output.mdd },
  { id: 'efficiency', label: '효율', getValue: (row) => row.output.cagr_mdd_ratio },
  { id: 'status', label: '상태', getValue: (row) => statusLabel(row.output.status) }
];

type AssetEvaluationColumnId =
  | 'layer'
  | 'ticker'
  | 'weight'
  | 'layerInternalWeight'
  | 'return'
  | 'cagr'
  | 'mdd'
  | 'riskContribution'
  | 'thesis'
  | 'status';

const assetEvaluationColumns: Array<SortableColumn<EvaluationRecord, AssetEvaluationColumnId>> = [
  { id: 'layer', label: '계층', getValue: (row) => layerLabel(row.unit.parent_layer) },
  { id: 'ticker', label: '티커', getValue: (row) => row.unit.name },
  { id: 'weight', label: '비중', getValue: (row) => row.output.current_weight },
  { id: 'layerInternalWeight', label: '계층 내 비중', getValue: (row) => row.output.layer_internal_weight },
  { id: 'return', label: '수익률', getValue: (row) => row.output.period_return },
  { id: 'cagr', label: 'CAGR', getValue: (row) => row.output.cagr },
  { id: 'mdd', label: 'MDD', getValue: (row) => row.output.mdd },
  { id: 'riskContribution', label: '위험 기여', getValue: (row) => row.output.risk_contribution },
  { id: 'thesis', label: '논리', getValue: (row) => thesisLabel(row.output.thesis_status) },
  { id: 'status', label: '상태', getValue: (row) => statusLabel(row.output.status) }
];

function localRowId(value: string) {
  return value.replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase();
}

function LayerDashboard({
  focusedLayer,
  rows
}: {
  focusedLayer: string | null;
  rows: EvaluationRecord[];
}) {
  const [sortState, setSortState] = useState<SortState<LayerDashboardColumnId>>(null);
  const sortedRows = useMemo(() => sortRows(rows, layerDashboardColumns, sortState), [rows, sortState]);

  if (!rows.length) return null;
  return (
    <section id="layer-dashboard" className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-cyan-800" />
        <h2 className="text-base font-bold text-slate-950">계층 대시보드</h2>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              {layerDashboardColumns.map((column) => (
                <SortableHeader
                  key={column.id}
                  id={column.id}
                  label={column.label}
                  sortState={sortState}
                  onSort={(columnId) => setSortState((current) => nextSortState(current, columnId))}
                />
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map(({ unit, output }) => {
              const isFocused = focusedLayer === unit.name;
              const isDimmed = focusedLayer !== null && !isFocused;
              return (
                <tr
                  key={unit.name}
                  id={`layer-row-${localRowId(unit.name)}`}
                  className={`border-t border-slate-100 transition ${
                    isFocused ? 'bg-cyan-50 ring-1 ring-inset ring-cyan-200' : isDimmed ? 'opacity-50' : ''
                  }`}
                >
                  <td className="px-2 py-2 font-bold text-slate-900">{layerLabel(unit.name)}</td>
                  <td className="px-2 py-2">{pct(output.current_weight)}</td>
                  <td className="px-2 py-2">{pct(output.weight_gap)}</td>
                  <td className="px-2 py-2">{pct(output.period_return)}</td>
                  <td className="px-2 py-2">{pct(output.benchmark_return)}</td>
                  <td className="px-2 py-2">{pct(output.benchmark_excess_return)}</td>
                  <td className="px-2 py-2">{pct(output.mdd)}</td>
                  <td className="px-2 py-2">{num(output.cagr_mdd_ratio)}</td>
                  <td className="px-2 py-2"><StatusBadge status={output.status} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function AssetEvaluationTable({
  focusedLayer,
  focusedTicker,
  onTickerFocus,
  onTickerSelect,
  rows
}: {
  focusedLayer: string | null;
  focusedTicker: string | null;
  onTickerFocus: (ticker: string | null) => void;
  onTickerSelect: (ticker: string) => void;
  rows: EvaluationRecord[];
}) {
  const [sortState, setSortState] = useState<SortState<AssetEvaluationColumnId>>(null);
  const sortedRows = useMemo(() => sortRows(rows, assetEvaluationColumns, sortState), [rows, sortState]);

  if (!rows.length) return null;
  return (
    <section id="asset-evaluation-table" className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-bold text-slate-950">종목 평가 테이블</h2>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              {assetEvaluationColumns.map((column) => (
                <SortableHeader
                  key={column.id}
                  id={column.id}
                  label={column.label}
                  sortState={sortState}
                  onSort={(columnId) => setSortState((current) => nextSortState(current, columnId))}
                />
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map(({ unit, output }) => {
              const isTickerFocused = focusedTicker === unit.name;
              const isLayerFocused = focusedLayer === unit.parent_layer;
              const isDimmed =
                (focusedTicker !== null && !isTickerFocused) ||
                (focusedTicker === null && focusedLayer !== null && !isLayerFocused);
              return (
                <tr
                  key={unit.name}
                  id={`asset-row-${localRowId(unit.name)}`}
                  className={`cursor-pointer border-t border-slate-100 transition hover:bg-slate-50 ${
                    isTickerFocused ? 'bg-cyan-50 ring-1 ring-inset ring-cyan-200' : isLayerFocused ? 'bg-cyan-50/50' : isDimmed ? 'opacity-50' : ''
                  }`}
                  onClick={() => onTickerSelect(unit.name)}
                  onMouseEnter={() => onTickerFocus(unit.name)}
                  onMouseLeave={() => onTickerFocus(null)}
                >
                  <td className="px-2 py-2">{layerLabel(unit.parent_layer)}</td>
                  <td className="px-2 py-2 font-bold text-slate-900">{unit.name}</td>
                  <td className="px-2 py-2">{pct(output.current_weight)}</td>
                  <td className="px-2 py-2">{pct(output.layer_internal_weight)}</td>
                  <td className="px-2 py-2">{pct(output.period_return)}</td>
                  <td className="px-2 py-2">{pct(output.cagr)}</td>
                  <td className="px-2 py-2">{pct(output.mdd)}</td>
                  <td className="px-2 py-2">{pct(output.risk_contribution)}</td>
                  <td className="px-2 py-2">{thesisLabel(output.thesis_status)}</td>
                  <td className="px-2 py-2"><StatusBadge status={output.status} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ReviewQueue({ evaluation }: { evaluation: EvaluationResponse }) {
  const {
    applySurfacePatch,
    clearSurface,
    reviewDecisions,
    reviewQueueSurface,
    setReviewDecisions,
    updateReviewDecision
  } = useGenerativeUi();
  const defaultSurface = useMemo(() => buildReviewQueueTriageSurface(evaluation), [evaluation]);
  const defaultReviewDecisions = useMemo(() => defaultReviewDecisionsFromEvaluation(evaluation), [evaluation]);
  const currentReviewItemIds = useMemo(
    () => new Set(defaultReviewDecisions.map((decision) => decision.review_item_id)),
    [defaultReviewDecisions]
  );
  const defaultSurfaceSignature = useMemo(
    () =>
      defaultSurface.groups
        .flatMap((group) => group.items.map((item) => `${item.id}:${item.status}`))
        .join('|'),
    [defaultSurface]
  );
  const generatedSurfaceSignature = useMemo(
    () =>
      reviewQueueSurface?.groups
        .flatMap((group) => group.items.map((item) => `${item.id}:${item.status}`))
        .join('|'),
    [reviewQueueSurface]
  );
  const hasCurrentReviewDecisions =
    reviewDecisions.length === defaultReviewDecisions.length &&
    reviewDecisions.every((decision) => currentReviewItemIds.has(decision.review_item_id));
  const generatedSurfaceMatchesEvaluation =
    reviewQueueSurface?.evaluation_period.label === evaluation.evaluation_period.label &&
    reviewQueueSurface.evaluation_period.start_date === evaluation.evaluation_period.start_date &&
    reviewQueueSurface.evaluation_period.end_date === evaluation.evaluation_period.end_date &&
    generatedSurfaceSignature === defaultSurfaceSignature;

  useEffect(() => {
    clearSurface('review_queue');
    setReviewDecisions(defaultReviewDecisions);
  }, [clearSurface, defaultReviewDecisions, evaluation, setReviewDecisions]);

  useEffect(() => {
    if (evaluation.review_queue.length === 0) return;
    const controller = new AbortController();

    requestReviewQueueAgentExplanations({
      signal: controller.signal,
      source: 'automatic',
      surface: defaultSurface
    })
      .then((patch) => {
        const surface = mergeReviewQueueAgentExplanations(defaultSurface, patch);
        const validation = validateReviewA2UISurface(surface);
        if (!validation.ok || validation.surface.component !== 'ReviewQueueTriageSurface') return;
        applySurfacePatch({
          target: 'review_queue',
          surface: 'ReviewQueueTriageSurface',
          mode: 'replace',
          payload: validation.surface
        });
      })
      .catch(() => {
        // Review Copilot runtime is optional for the base inspection board.
      });

    return () => controller.abort();
  }, [applySurfacePatch, defaultSurface, evaluation.review_queue.length]);

  return (
    <section id="review-queue" className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2">
        <ClipboardList className="h-5 w-5 text-cyan-800" />
        <h2 className="text-base font-bold text-slate-950">점검 큐</h2>
      </div>
      {evaluation.review_queue.length === 0 ? (
        <p className="mt-3 text-sm font-semibold text-slate-500">점검 큐가 비어 있습니다.</p>
      ) : (
        <div className="mt-3">
          <ReviewQueueTriageSurface
            surface={generatedSurfaceMatchesEvaluation ? reviewQueueSurface : defaultSurface}
            decisions={hasCurrentReviewDecisions ? reviewDecisions : defaultReviewDecisions}
            onDecisionChange={updateReviewDecision}
          />
        </div>
      )}
    </section>
  );
}

function EvaluationGraphs({
  evaluation,
  focusedLayer,
  focusedTicker,
  onLayerFocus,
  onTickerFocus,
  onTickerSelect
}: {
  evaluation: EvaluationResponse;
  focusedLayer: string | null;
  focusedTicker: string | null;
  onLayerFocus: (layer: string | null) => void;
  onTickerFocus: (ticker: string | null) => void;
  onTickerSelect: (ticker: string) => void;
}) {
  const { applySurfacePatch } = useGenerativeUi();
  const defaultSurface = useMemo(() => buildDefaultEvaluationGraphSurface(evaluation), [evaluation]);

  useEffect(() => {
    applySurfacePatch({
      target: 'evaluation_graphs',
      surface: 'EvaluationGraphSurface',
      mode: 'replace',
      payload: defaultSurface
    });
  }, [applySurfacePatch, defaultSurface]);

  return (
    <GeneratedSurfaceHost
      evaluation={evaluation}
      focusedLayer={focusedLayer}
      focusedTicker={focusedTicker}
      onLayerFocus={onLayerFocus}
      onTickerFocus={onTickerFocus}
      onTickerSelect={onTickerSelect}
      target="evaluation_graphs"
    />
  );
}

function EvaluationResults({
  evaluation,
  evaluationRun
}: {
  evaluation: EvaluationResponse;
  evaluationRun: EvaluationRun | null;
}) {
  const [focusedLayer, setFocusedLayer] = useState<string | null>(null);
  const [hoveredTicker, setHoveredTicker] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const focusedTicker = hoveredTicker ?? selectedTicker;

  useEffect(() => {
    setFocusedLayer(null);
    setHoveredTicker(null);
    setSelectedTicker(null);
  }, [evaluation]);

  function handleLayerFocus(layer: string | null) {
    setFocusedLayer(layer);
    if (layer !== null) {
      setHoveredTicker(null);
      setSelectedTicker(null);
    }
  }

  function handleTickerFocus(ticker: string | null) {
    setHoveredTicker(ticker);
    if (ticker !== null) {
      setFocusedLayer(null);
    }
  }

  function handleTickerSelect(ticker: string) {
    setFocusedLayer(null);
    setHoveredTicker(null);
    setSelectedTicker((current) => (current === ticker ? null : ticker));

    window.requestAnimationFrame(() => {
      document.getElementById(`asset-row-${localRowId(ticker)}`)?.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    });
  }

  return (
    <>
      <EvaluationRunHeader evaluationRun={evaluationRun} />
      <EvaluationGraphs
        evaluation={evaluation}
        focusedLayer={focusedLayer}
        focusedTicker={focusedTicker}
        onLayerFocus={handleLayerFocus}
        onTickerFocus={handleTickerFocus}
        onTickerSelect={handleTickerSelect}
      />
      <LayerDashboard focusedLayer={focusedLayer} rows={evaluation.layer_evaluations} />
      <AssetEvaluationTable
        focusedLayer={focusedLayer}
        focusedTicker={focusedTicker}
        rows={evaluation.asset_evaluations}
        onTickerFocus={handleTickerFocus}
        onTickerSelect={handleTickerSelect}
      />
      <ReviewQueue evaluation={evaluation} />
      <JournalDraft rows={evaluation.journal_draft} />
    </>
  );
}

function JournalDraft({ rows }: { rows: Array<Record<string, unknown>> }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-bold text-slate-950">저널 초안</h2>
      {rows.length === 0 ? (
        <p className="mt-3 text-sm font-semibold text-slate-500">초안 항목이 없습니다.</p>
      ) : (
        <ul className="mt-3 grid gap-2 text-sm text-slate-700">
          {rows.map((row) => (
            <li key={String(row.title)} className="rounded-lg bg-slate-50 p-3">
              <strong className="block text-slate-950">{String(row.title)}</strong>
              <span>{Array.isArray(row.prompts) ? row.prompts.join(' / ') : ''}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-3">
        <GeneratedSurfaceHost target="journal_draft" />
      </div>
    </section>
  );
}

function ManagementError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-semibold text-red-800">
      {message}
    </div>
  );
}

function Toast({ message, onDismiss }: { message: string | null; onDismiss: () => void }) {
  if (!message) return null;
  return (
    <div
      className="fixed bottom-4 right-4 z-50 flex max-w-sm items-start gap-3 rounded-lg border border-slate-800 bg-slate-950 px-4 py-3 text-sm font-bold text-white shadow-2xl"
      role="status"
    >
      <span className="min-w-0 flex-1">{message}</span>
      <button
        aria-label="알림 닫기"
        className="inline-grid h-6 w-6 shrink-0 place-items-center rounded-md text-slate-300 transition hover:bg-white/10 hover:text-white"
        type="button"
        onClick={onDismiss}
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

function CreatePortfolioModal({
  error,
  isOpen,
  name,
  pending,
  onChange,
  onClose,
  onCreate
}: {
  error: string | null;
  isOpen: boolean;
  name: string;
  pending: ManagementPending;
  onChange: (value: string) => void;
  onClose: () => void;
  onCreate: () => void;
}) {
  if (!isOpen) return null;
  const isBusy = pending === 'portfolios';

  return (
    <div
      aria-labelledby="create-portfolio-title"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      role="dialog"
      onClick={() => {
        if (!isBusy) onClose();
      }}
    >
      <div
        className="w-full max-w-md rounded-xl bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 p-5">
          <div>
            <h2 id="create-portfolio-title" className="text-lg font-bold text-slate-950">
              새 포트폴리오
            </h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">
              저장할 포트폴리오 이름을 입력해주세요.
            </p>
          </div>
          <button
            aria-label="새 포트폴리오 모달 닫기"
            className="inline-grid h-9 w-9 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-5">
          <label className="block text-sm font-bold text-slate-700">
            이름
            <input
              autoFocus
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
              disabled={isBusy}
              placeholder="예: 장기 투자 포트폴리오"
              value={name}
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  onCreate();
                }
              }}
            />
          </label>
          {error ? (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-semibold text-red-800">
              {error}
            </div>
          ) : null}
        </div>
        <div className="flex justify-end gap-2 border-t border-slate-200 bg-slate-50 p-5">
          <button
            className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onClose}
          >
            닫기
          </button>
          <button
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isBusy || name.trim() === ''}
            type="button"
            onClick={onCreate}
          >
            {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            생성
          </button>
        </div>
      </div>
    </div>
  );
}

function DeleteSnapshotConfirmModal({
  isOpen,
  pending,
  snapshot,
  onCancel,
  onConfirm
}: {
  isOpen: boolean;
  pending: ManagementPending;
  snapshot: SnapshotSummary | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  if (!isOpen || snapshot === null) return null;
  const isBusy = pending === 'delete';

  return (
    <div
      aria-labelledby="delete-snapshot-title"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      role="dialog"
      onClick={() => {
        if (!isBusy) onCancel();
      }}
    >
      <div
        className="w-full max-w-md rounded-xl bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-4 border-b border-slate-200 p-5">
          <div className="inline-grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-red-50 text-red-600">
            <AlertCircle className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <h2 id="delete-snapshot-title" className="text-lg font-bold text-slate-950">
              보유현황 스냅샷을 정말 삭제할까요?
            </h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">
              삭제하면 이 보유현황 스냅샷과 저장된 평가 기록을 되돌릴 수 없습니다.
            </p>
          </div>
          <button
            aria-label="보유현황 스냅샷 삭제 확인 닫기"
            className="inline-grid h-9 w-9 shrink-0 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onCancel}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-5">
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-bold text-slate-800">
            {snapshot.name}
          </div>
        </div>
        <div className="flex justify-end gap-2 border-t border-slate-200 bg-slate-50 p-5">
          <button
            className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onCancel}
          >
            취소
          </button>
          <button
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-red-600 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isBusy}
            type="button"
            onClick={onConfirm}
          >
            {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            삭제
          </button>
        </div>
      </div>
    </div>
  );
}

function SnapshotModal({
  allocationTotal,
  allocationWarning,
  canSubmit,
  error,
  isOpen,
  mode,
  name,
  note,
  pending,
  rowErrors,
  rows,
  onAdd,
  onChange,
  onClose,
  onDelete,
  onNameChange,
  onNoteChange,
  onSubmit
}: {
  allocationTotal: number;
  allocationWarning: string | null;
  canSubmit: boolean;
  error: string | null;
  isOpen: boolean;
  mode: SnapshotModalMode;
  name: string;
  note: string;
  pending: ManagementPending;
  rowErrors: Map<string, string>;
  rows: PortfolioInputRow[];
  onAdd: () => void;
  onChange: (id: string, patch: Partial<PortfolioInputRow>) => void;
  onClose: () => void;
  onDelete: (id: string) => void;
  onNameChange: (value: string) => void;
  onNoteChange: (value: string) => void;
  onSubmit: () => void;
}) {
  if (!isOpen) return null;
  const isBusy = pending === 'save' || pending === 'update';
  const title = mode === 'create' ? '보유현황 스냅샷 추가' : '보유현황 스냅샷 정보 수정';
  const showPositionEditor = mode === 'create';

  return (
    <div
      aria-labelledby="snapshot-modal-title"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      role="dialog"
      onClick={() => {
        if (!isBusy) onClose();
      }}
    >
      <div
        className="flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 p-5">
          <div>
            <h2 id="snapshot-modal-title" className="text-xl font-bold text-slate-950">
              {title}
            </h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">
              {showPositionEditor
                ? '현재 포트폴리오 행을 고정된 보유현황 기록으로 저장합니다.'
                : '보유현황 스냅샷의 이름과 메모만 수정합니다.'}
            </p>
          </div>
          <button
            aria-label="보유현황 스냅샷 모달 닫기"
            className="inline-grid h-9 w-9 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50 p-5">
          <div className="mb-4 grid gap-3 rounded-lg border border-slate-200 bg-white p-4 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
            <label className="block text-sm font-bold text-slate-700">
              이름
              <input
                autoFocus
                className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                disabled={isBusy}
                placeholder="저장된 보유현황 스냅샷"
                value={name}
                onChange={(event) => onNameChange(event.target.value)}
              />
            </label>
            <label className="block text-sm font-bold text-slate-700">
              메모
              <input
                className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                disabled={isBusy}
                placeholder="선택 사항"
                value={note}
                onChange={(event) => onNoteChange(event.target.value)}
              />
            </label>
          </div>

          {showPositionEditor ? (
            <PortfolioInputTable
              rows={rows}
              allocationTotal={allocationTotal}
              rowErrors={rowErrors}
              onAdd={onAdd}
              onDelete={onDelete}
              onChange={onChange}
            />
          ) : null}
          {showPositionEditor && allocationWarning ? (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
              {allocationWarning}
            </div>
          ) : null}
          {error ? (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-800">
              {error}
            </div>
          ) : null}
        </div>

        <div className="flex flex-col gap-3 border-t border-slate-200 bg-white p-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm font-semibold text-slate-500">
            {showPositionEditor ? `입력 합계 ${allocationTotal.toFixed(2)}%` : '포지션 변경은 새 보유현황 스냅샷으로 저장합니다.'}
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
          <button
            className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={isBusy}
            type="button"
            onClick={onClose}
          >
            닫기
          </button>
          <button
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canSubmit}
            type="button"
            onClick={onSubmit}
          >
            {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : mode === 'create' ? <Plus className="h-4 w-4" /> : <PencilLine className="h-4 w-4" />}
            {mode === 'create' ? '추가' : '정보 수정'}
          </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function PortfolioContextBar({
  canCreateSnapshot,
  managementPending,
  onDeleteSnapshot,
  onEditSnapshot,
  onOpenCreatePortfolio,
  onOpenSnapshotCreate,
  onSelectPortfolio,
  onSelectSnapshot,
  savedPortfolios,
  selectedPortfolioId,
  selectedSnapshotId,
  snapshots
}: {
  canCreateSnapshot: boolean;
  managementPending: ManagementPending;
  onDeleteSnapshot: (snapshot: SnapshotSummary) => void;
  onEditSnapshot: (snapshot: SnapshotSummary) => void;
  onOpenCreatePortfolio: () => void;
  onOpenSnapshotCreate: () => void;
  onSelectPortfolio: (id: number | null) => void;
  onSelectSnapshot: (snapshot: SnapshotSummary | null) => void;
  savedPortfolios: SavedPortfolio[];
  selectedPortfolioId: number | null;
  selectedSnapshotId: number | null;
  snapshots: SnapshotSummary[];
}) {
  const selectedPortfolio = savedPortfolios.find((portfolio) => portfolio.id === selectedPortfolioId);
  const selectedSnapshot = snapshots.find((snapshot) => snapshot.id === selectedSnapshotId);
  const isBusy = managementPending !== null;
  const canOpenSnapshotCreate = canCreateSnapshot && !isBusy;
  const canEditSnapshot = selectedSnapshot !== undefined && !isBusy;
  const canDeleteSnapshot = selectedSnapshot !== undefined && !isBusy;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="grid gap-3 lg:grid-cols-[minmax(180px,0.75fr)_minmax(320px,1.25fr)] lg:items-end">
        <div className="grid min-w-0 gap-1">
          <span className="text-sm font-bold text-slate-700">포트폴리오 선택</span>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
            <select
              className="min-w-0 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
              disabled={isBusy && managementPending !== 'snapshot'}
              value={selectedPortfolioId ?? ''}
              onChange={(event) => onSelectPortfolio(event.target.value ? Number(event.target.value) : null)}
            >
              <option value="">
                {savedPortfolios.length === 0 ? '저장된 포트폴리오 없음' : '포트폴리오 선택 안 함'}
              </option>
              {savedPortfolios.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
            <button
              aria-label="새 포트폴리오 만들기"
              className="inline-grid h-[42px] w-[42px] place-items-center rounded-lg bg-slate-900 text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isBusy}
              title="새 포트폴리오 만들기"
              type="button"
              onClick={onOpenCreatePortfolio}
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <p className="min-h-4 text-xs font-semibold text-slate-500">
            {selectedPortfolio?.latest_snapshot
              ? `최근 보유현황 저장 ${formatSnapshotTimestamp(selectedPortfolio.latest_snapshot.updated_at)}`
              : '\u00A0'}
          </p>
        </div>

        <div className="grid min-w-0 gap-1">
          <span className="text-sm font-bold text-slate-700">보유현황 스냅샷 선택</span>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_repeat(3,42px)]">
            <select
              className="min-w-0 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800 outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100 disabled:text-slate-400"
              disabled={isBusy || selectedPortfolioId === null || snapshots.length === 0}
              value={selectedSnapshotId ?? ''}
              onChange={(event) => {
                const snapshot = snapshots.find((item) => item.id === Number(event.target.value));
                onSelectSnapshot(snapshot ?? null);
              }}
            >
              <option value="">{snapshots.length === 0 ? '보유현황 없음' : '보유현황 선택 안 함'}</option>
              {snapshots.map((snapshot) => (
                <option key={snapshot.id} value={snapshot.id}>
                  {snapshot.name}
                </option>
              ))}
            </select>
            <button
              aria-label="보유현황 스냅샷 추가"
              className="inline-grid h-[42px] w-full place-items-center rounded-lg bg-slate-900 text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={!canOpenSnapshotCreate}
              title="보유현황 스냅샷 추가"
              type="button"
              onClick={onOpenSnapshotCreate}
            >
              <Plus className="h-4 w-4" />
            </button>
            <button
              aria-label="선택한 보유현황 스냅샷 정보 수정"
              className="inline-grid h-[42px] w-full place-items-center rounded-lg border border-slate-300 bg-white text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
              disabled={!canEditSnapshot}
              title="선택한 보유현황 스냅샷 정보 수정"
              type="button"
              onClick={() => {
                if (selectedSnapshot) onEditSnapshot(selectedSnapshot);
              }}
            >
              <PencilLine className="h-4 w-4" />
            </button>
            <button
              aria-label="선택한 보유현황 스냅샷 삭제"
              className="inline-grid h-[42px] w-full place-items-center rounded-lg border border-red-200 bg-white text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300"
              disabled={!canDeleteSnapshot}
              title="선택한 보유현황 스냅샷 삭제"
              type="button"
              onClick={() => {
                if (selectedSnapshot) onDeleteSnapshot(selectedSnapshot);
              }}
            >
              {managementPending === 'delete' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            </button>
          </div>
          <p className="min-h-4 text-xs font-semibold text-slate-500">
            {selectedSnapshot
              ? `생성 ${formatSnapshotTimestamp(selectedSnapshot.created_at)} · 최근 저장 ${formatSnapshotTimestamp(selectedSnapshot.updated_at)}`
              : '\u00A0'}
          </p>
        </div>
      </div>
    </section>
  );
}

export function App() {
  const [inputRows, setInputRows] = useState<PortfolioInputRow[]>(initialRows);
  const [isInputModalOpen, setInputModalOpen] = useState(false);
  const [isCreatePortfolioModalOpen, setCreatePortfolioModalOpen] = useState(false);
  const [nextRowId, setNextRowId] = useState(1);
  const [period, setPeriod] = useState<'1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max'>('3M');
  const [asOfDate, setAsOfDate] = useState(todayIsoDate);
  const [layerBenchmarks, setLayerBenchmarks] = useState<Record<LayerType, string>>(DEFAULT_LAYER_BENCHMARKS);
  const [portfolio, setPortfolio] = useState<AssetRow[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<WorkflowPending>(null);
  const [savedPortfolios, setSavedPortfolios] = useState<SavedPortfolio[]>([]);
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<number | null>(null);
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [isSnapshotModalOpen, setSnapshotModalOpen] = useState(false);
  const [snapshotModalMode, setSnapshotModalMode] = useState<SnapshotModalMode>('create');
  const [snapshotModalName, setSnapshotModalName] = useState('');
  const [snapshotModalNote, setSnapshotModalNote] = useState('');
  const [snapshotRows, setSnapshotRows] = useState<PortfolioInputRow[]>(initialRows);
  const [nextSnapshotRowId, setNextSnapshotRowId] = useState(1);
  const [snapshotPendingDelete, setSnapshotPendingDelete] = useState<SnapshotSummary | null>(null);
  const [managementPending, setManagementPending] = useState<ManagementPending>(null);
  const [managementError, setManagementError] = useState<string | null>(null);
  const [managementNotice, setManagementNotice] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  function applyLoadedState(response: SnapshotLoadResponse) {
    const loadedRows = inputRowsFromAssets(response.portfolio.assets);
    const loadedEvaluationRun = response.evaluation_run;
    setPortfolio(response.portfolio.assets);
    setAnalysis(response.analysis);
    setEvaluation(response.evaluation);
    setEvaluationRun(loadedEvaluationRun);
    if (loadedEvaluationRun !== null) {
      const settings = loadedEvaluationRun.settings;
      if (isEvaluationPeriod(settings.period)) {
        setPeriod(settings.period);
      }
      const loadedAsOfDate = settings.as_of_date || settings.end_date;
      if (loadedAsOfDate) {
        setAsOfDate(loadedAsOfDate);
      }
      setLayerBenchmarks(layerBenchmarksFromEvaluationRun(loadedEvaluationRun));
    }
    setInputRows(loadedRows);
    setNextRowId(loadedRows.length + 1);
  }

  function clearAppliedState() {
    setPortfolio([]);
    setAnalysis(null);
    setEvaluation(null);
    setEvaluationRun(null);
    setInputRows(initialRows);
    setNextRowId(initialRows.length + 1);
  }

  async function refreshPortfolioList(preferredId?: number | null) {
    const response = await listPortfolios();
    setSavedPortfolios(response.portfolios);
    if (preferredId !== undefined) {
      setSelectedPortfolioId(preferredId);
      return;
    }
    setSelectedPortfolioId((currentId) => {
      if (currentId && response.portfolios.some((item) => item.id === currentId)) return currentId;
      return response.portfolios[0]?.id ?? null;
    });
  }

  async function refreshSnapshots(portfolioId: number) {
    const response = await listSnapshots(portfolioId);
    setSnapshots(response.snapshots);
    return response.snapshots;
  }

  useEffect(() => {
    let ignore = false;
    async function loadInitialPortfolios() {
      setManagementPending('portfolios');
      setManagementError(null);
      try {
        const response = await listPortfolios();
        if (ignore) return;
        setSavedPortfolios(response.portfolios);
        setSelectedPortfolioId((currentId) => currentId ?? response.portfolios[0]?.id ?? null);
      } catch (err) {
        if (!ignore) setManagementError(err instanceof Error ? err.message : '포트폴리오 목록을 불러오지 못했습니다.');
      } finally {
        if (!ignore) setManagementPending(null);
      }
    }
    loadInitialPortfolios();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (selectedPortfolioId === null) {
      setSnapshots([]);
      setSelectedSnapshotId(null);
      setManagementNotice(null);
      clearAppliedState();
      return;
    }

    let ignore = false;
    const portfolioId = selectedPortfolioId;
    async function loadSelectedPortfolio() {
      setManagementPending('portfolio');
      setManagementError(null);
      setManagementNotice(null);
      setSelectedSnapshotId(null);
      clearAppliedState();
      try {
        const loadedSnapshots = await refreshSnapshots(portfolioId);
        if (ignore) return;
        if (loadedSnapshots.length === 0) {
          setManagementNotice('선택한 포트폴리오에 저장된 보유현황 스냅샷이 없습니다.');
        }
      } catch (err) {
        if (!ignore) setManagementError(err instanceof Error ? err.message : '포트폴리오 상태를 불러오지 못했습니다.');
      } finally {
        if (!ignore) setManagementPending(null);
      }
    }
    loadSelectedPortfolio();
    return () => {
      ignore = true;
    };
  }, [selectedPortfolioId]);

  useEffect(() => {
    if (!toastMessage) return;
    const timeoutId = window.setTimeout(() => setToastMessage(null), 3500);
    return () => window.clearTimeout(timeoutId);
  }, [toastMessage]);

  const meaningfulRows = useMemo(() => inputRows.filter(isMeaningfulRow), [inputRows]);
  const allocationTotal = useMemo(
    () => meaningfulRows.reduce((sum, row) => {
      const value = toNumber(row.allocation);
      return Number.isNaN(value) ? sum : sum + value;
    }, 0),
    [meaningfulRows]
  );

  const rowErrors = useMemo(() => inputRowErrors(inputRows), [inputRows]);

  const allocationWarning =
    meaningfulRows.length > 0 && Math.abs(allocationTotal - 100) > 0.01
      ? `입력 합계가 100%와 ${(allocationTotal - 100).toFixed(2)}%p 차이납니다. 백엔드는 입력 비중을 기준으로 정규화합니다.`
      : null;

  const canRunWorkflow = pending === null && meaningfulRows.length > 0 && rowErrors.size === 0;
  const canRunEvaluationOnly = pending === null && portfolio.length > 0;
  const selectedSnapshot = snapshots.find((snapshot) => snapshot.id === selectedSnapshotId) ?? null;
  const canSaveEvaluationRun =
    selectedSnapshotId !== null && evaluation !== null && evaluationRun === null && pending === null && managementPending === null;
  const snapshotMeaningfulRows = useMemo(() => snapshotRows.filter(isMeaningfulRow), [snapshotRows]);
  const snapshotAllocationTotal = useMemo(
    () => snapshotMeaningfulRows.reduce((sum, row) => {
      const value = toNumber(row.allocation);
      return Number.isNaN(value) ? sum : sum + value;
    }, 0),
    [snapshotMeaningfulRows]
  );
  const snapshotRowErrors = useMemo(() => inputRowErrors(snapshotRows), [snapshotRows]);
  const snapshotAllocationWarning =
    snapshotMeaningfulRows.length > 0 && Math.abs(snapshotAllocationTotal - 100) > 0.01
      ? `입력 합계가 100%와 ${(snapshotAllocationTotal - 100).toFixed(2)}%p 차이납니다. 백엔드는 입력 비중을 기준으로 정규화합니다.`
      : null;
  const canSubmitSnapshot =
    managementPending === null &&
    snapshotModalName.trim() !== '' &&
    (snapshotModalMode === 'edit' || (snapshotMeaningfulRows.length > 0 && snapshotRowErrors.size === 0));
  const canCreateSnapshot = selectedPortfolioId !== null && meaningfulRows.length > 0;
  const copilotLayerBenchmarks = useMemo(() => normalizedLayerBenchmarks(layerBenchmarks), [layerBenchmarks]);
  const copilotSettings = useMemo(
    () => ({
      period,
      asOfDate,
      layerBenchmarks: copilotLayerBenchmarks,
      analysisBenchmark: copilotLayerBenchmarks.core
    }),
    [period, asOfDate, copilotLayerBenchmarks]
  );
  const workflowButtonLabel =
    pending === 'portfolio'
      ? '포트폴리오 적용 중'
      : pending === 'analysis'
        ? '분석 실행 중'
        : pending === 'evaluation'
          ? '평가 실행 중'
          : '적용하고 평가 실행';

  function updateInputRow(id: string, patch: Partial<PortfolioInputRow>) {
    setInputRows((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  function addInputRow() {
    setInputRows((rows) => [...rows, makeInputRow(`row-new-${nextRowId}`)]);
    setNextRowId((value) => value + 1);
  }

  function deleteInputRow(id: string) {
    setInputRows((rows) => rows.filter((row) => row.id !== id));
  }

  function updateSnapshotRow(id: string, patch: Partial<PortfolioInputRow>) {
    setSnapshotRows((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  function addSnapshotRow() {
    setSnapshotRows((rows) => [...rows, makeInputRow(`snapshot-row-new-${nextSnapshotRowId}`)]);
    setNextSnapshotRowId((value) => value + 1);
  }

  function deleteSnapshotRow(id: string) {
    setSnapshotRows((rows) => rows.filter((row) => row.id !== id));
  }

  function updateLayerBenchmark(layer: LayerType, value: string) {
    setLayerBenchmarks((current) => ({ ...current, [layer]: value }));
  }

  async function createNamedPortfolio() {
    const name = newPortfolioName.trim();
    if (!name) {
      setManagementError('포트폴리오 이름을 입력해주세요.');
      return;
    }
    setManagementPending('portfolios');
    setManagementError(null);
    setManagementNotice(null);
    try {
      const response = await createPortfolio({ name });
      setNewPortfolioName('');
      setCreatePortfolioModalOpen(false);
      await refreshPortfolioList(response.portfolio.id);
      setSnapshots([]);
      setSelectedSnapshotId(null);
    } catch (err) {
      setManagementError(err instanceof Error ? err.message : '포트폴리오를 생성하지 못했습니다.');
    } finally {
      setManagementPending(null);
    }
  }

  function openSnapshotCreateModal() {
    const baseSnapshot = selectedSnapshot;
    const baseRows = inputRows.length > 0 ? inputRows : initialRows;
    setManagementError(null);
    setManagementNotice(null);
    setSnapshotModalMode('create');
    setSnapshotModalName(baseSnapshot?.name ?? '저장된 보유현황 스냅샷');
    setSnapshotModalNote(baseSnapshot?.note ?? '');
    setSnapshotRows(baseRows.map((row, index) => ({ ...row, id: `snapshot-create-${index}` })));
    setNextSnapshotRowId(baseRows.length + 1);
    setSnapshotModalOpen(true);
  }

  function openSnapshotEditModal(snapshot: SnapshotSummary) {
    const baseRows = inputRows.length > 0 ? inputRows : initialRows;
    setManagementError(null);
    setManagementNotice(null);
    setSnapshotModalMode('edit');
    setSnapshotModalName(snapshot.name);
    setSnapshotModalNote(snapshot.note ?? '');
    setSnapshotRows(baseRows.map((row, index) => ({ ...row, id: `snapshot-edit-${index}` })));
    setNextSnapshotRowId(baseRows.length + 1);
    setSnapshotModalOpen(true);
  }

  async function submitSnapshotModal() {
    if (selectedPortfolioId === null) return;
    if (snapshotModalName.trim() === '') {
      setManagementError('보유현황 스냅샷 이름을 입력해주세요.');
      return;
    }
    if (snapshotModalMode === 'create' && !canSubmitSnapshot) {
      setManagementError(snapshotMeaningfulRows.length === 0 ? '최소 1개 이상의 티커와 비중을 입력해주세요.' : '입력 오류를 먼저 확인해주세요.');
      return;
    }

    const payload: { name: string; note: string; rows?: PortfolioRowInput[] } = {
      name: snapshotModalName.trim(),
      note: snapshotModalNote
    };
    if (snapshotModalMode === 'create') {
      payload.rows = toPortfolioPayload(snapshotRows);
    }

    setManagementPending(snapshotModalMode === 'create' ? 'save' : 'update');
    setManagementError(null);
    setManagementNotice(null);
    setToastMessage(null);
    try {
      const response =
        snapshotModalMode === 'create'
          ? await saveSnapshot(selectedPortfolioId, payload)
          : selectedSnapshotId !== null
            ? await updateSnapshot(selectedSnapshotId, payload)
            : null;

      if (response === null) return;
      const loadedResponse = await loadSnapshot(response.snapshot.id);
      applyLoadedState(loadedResponse);
      await refreshSnapshots(selectedPortfolioId);
      await refreshPortfolioList(selectedPortfolioId);
      setSelectedSnapshotId(response.snapshot.id);
      setSnapshotModalOpen(false);
      setToastMessage(`${response.snapshot.name}을 ${snapshotModalMode === 'create' ? '추가' : '수정'}했습니다.`);
    } catch (err) {
      setManagementError(err instanceof Error ? err.message : '보유현황 스냅샷을 저장하지 못했습니다.');
    } finally {
      setManagementPending(null);
    }
  }

  async function loadSavedSnapshot(snapshot: SnapshotSummary) {
    setManagementPending('snapshot');
    setManagementError(null);
    setManagementNotice(null);
    setToastMessage(null);
    try {
      const response = await loadSnapshot(snapshot.id);
      applyLoadedState(response);
      setSelectedSnapshotId(snapshot.id);
      await refreshPortfolioList(response.snapshot.portfolio_id);
      setToastMessage(`${snapshot.name}을 불러왔습니다.`);
    } catch (err) {
      setManagementError(err instanceof Error ? err.message : '보유현황 스냅샷을 불러오지 못했습니다.');
    } finally {
      setManagementPending(null);
    }
  }

  async function deleteSavedSnapshot(snapshot: SnapshotSummary) {
    setManagementPending('delete');
    setManagementError(null);
    setManagementNotice(null);
    setToastMessage(null);
    try {
      await deleteSnapshot(snapshot.id);
      await refreshSnapshots(snapshot.portfolio_id);
      await refreshPortfolioList(snapshot.portfolio_id);
      setSelectedSnapshotId(null);
      setSnapshotPendingDelete(null);
      setToastMessage(`${snapshot.name}을 삭제했습니다.`);
    } catch (err) {
      setManagementError(err instanceof Error ? err.message : '보유현황 스냅샷을 삭제하지 못했습니다.');
    } finally {
      setManagementPending(null);
    }
  }

  async function runPortfolioWorkflow() {
    if (!canRunWorkflow) {
      setError(meaningfulRows.length === 0 ? '최소 1개 이상의 티커와 비중을 입력해주세요.' : '입력 오류를 먼저 확인해주세요.');
      return;
    }

    setError(null);
    try {
      const evaluationLayerBenchmarks = normalizedLayerBenchmarks(layerBenchmarks);
      const analysisBenchmark = evaluationLayerBenchmarks.core;
      setPending('portfolio');
      const portfolioData = await submitPortfolio(toPortfolioPayload(inputRows));
      setSelectedSnapshotId(null);
      setPortfolio(portfolioData.assets);
      setAnalysis(null);
      setEvaluation(null);
      setEvaluationRun(null);

      setPending('analysis');
      const analysisData = await runAnalysis({
        period: analysisPeriodFromEvaluationPeriod(period),
        as_of_date: asOfDate,
        rf: ANALYSIS_DEFAULT_RF,
        bench: analysisBenchmark,
        layer_benchmarks: evaluationLayerBenchmarks
      });
      setAnalysis(analysisData);
      setEvaluation(null);
      setEvaluationRun(null);

      setPending('evaluation');
      const evaluationData = await runEvaluation({
        period,
        as_of_date: asOfDate,
        bench: analysisBenchmark,
        layer_benchmarks: evaluationLayerBenchmarks
      });
      setEvaluation(evaluationData);
      setEvaluationRun(null);
      setInputModalOpen(false);
      if (portfolioData.warnings.length > 0) {
        setError(portfolioData.warnings.join(' '));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '적용·분석·평가 실행에 실패했습니다.');
    } finally {
      setPending(null);
    }
  }

  async function runEvaluationOnlyWorkflow() {
    if (!canRunEvaluationOnly) {
      setError('포트폴리오를 입력하면 평가를 실행할 수 있습니다.');
      return;
    }

    setError(null);
    try {
      const evaluationLayerBenchmarks = normalizedLayerBenchmarks(layerBenchmarks);
      const analysisBenchmark = evaluationLayerBenchmarks.core;
      if (selectedSnapshotId !== null) {
        setPending('evaluation');
        const snapshotEvaluation = await runSnapshotEvaluation(selectedSnapshotId, {
          period,
          as_of_date: asOfDate,
          bench: analysisBenchmark,
          layer_benchmarks: evaluationLayerBenchmarks
        });
        setAnalysis(snapshotEvaluation.analysis);
        setEvaluation(snapshotEvaluation.evaluation);
        setEvaluationRun(null);
      } else {
        setPending('analysis');
        const analysisData = await runAnalysis({
          period: analysisPeriodFromEvaluationPeriod(period),
          as_of_date: asOfDate,
          rf: ANALYSIS_DEFAULT_RF,
          bench: analysisBenchmark,
          layer_benchmarks: evaluationLayerBenchmarks
        });
        setAnalysis(analysisData);
        setEvaluation(null);
        setEvaluationRun(null);

        setPending('evaluation');
        const evaluationData = await runEvaluation({
          period,
          as_of_date: asOfDate,
          bench: analysisBenchmark,
          layer_benchmarks: evaluationLayerBenchmarks
        });
        setEvaluation(evaluationData);
        setEvaluationRun(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '평가 실행에 실패했습니다.');
    } finally {
      setPending(null);
    }
  }

  async function saveCurrentEvaluationRun() {
    if (selectedSnapshotId === null || evaluation === null) {
      setManagementError('보유현황 스냅샷과 평가 결과가 있어야 평가 기록을 저장할 수 있습니다.');
      return;
    }

    setManagementPending('evaluation-save');
    setManagementError(null);
    setManagementNotice(null);
    setToastMessage(null);
    try {
      const response = await saveSnapshotEvaluation(selectedSnapshotId);
      setEvaluation(response.evaluation);
      setEvaluationRun(response.evaluation_run);
      if (selectedPortfolioId !== null) {
        await refreshSnapshots(selectedPortfolioId);
        await refreshPortfolioList(selectedPortfolioId);
      }
      setToastMessage(`평가 기록 #${response.evaluation_run.id}을 최신 평가로 저장했습니다.`);
    } catch (err) {
      setManagementError(err instanceof Error ? err.message : '평가 기록을 저장하지 못했습니다.');
    } finally {
      setManagementPending(null);
    }
  }

  return (
    <ReviewCopilotHost>
    <GenerativeUiProvider>
    <main className="min-h-screen bg-slate-100 p-4 text-slate-900 md:p-8">
      <div className="mx-auto grid max-w-7xl gap-5">
        <header className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-6 w-6 text-cyan-800" />
              <h1 className="text-2xl font-bold text-slate-950">IPS Pilot v2 워크벤치</h1>
            </div>
            <p className="mt-2 text-sm font-semibold text-slate-500">
              계층과 종목을 같은 평가 프레임으로 점검합니다. 결과는 매매 지시가 아닙니다.
            </p>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm font-bold text-cyan-900">
            {evaluation ? `${evaluation.evaluation_period.label} · 점검 ${evaluation.review_queue.length}건` : 'v2 평가 준비'}
          </div>
        </header>

        <ErrorBanner message={error} />

        <PortfolioContextBar
          canCreateSnapshot={canCreateSnapshot}
          managementPending={managementPending}
          savedPortfolios={savedPortfolios}
          selectedPortfolioId={selectedPortfolioId}
          selectedSnapshotId={selectedSnapshotId}
          snapshots={snapshots}
          onDeleteSnapshot={(snapshot) => {
            setManagementError(null);
            setManagementNotice(null);
            setSnapshotPendingDelete(snapshot);
          }}
          onEditSnapshot={openSnapshotEditModal}
          onOpenCreatePortfolio={() => {
            setManagementError(null);
            setCreatePortfolioModalOpen(true);
          }}
          onOpenSnapshotCreate={openSnapshotCreateModal}
          onSelectPortfolio={(id) => {
            setSnapshotPendingDelete(null);
            setSelectedPortfolioId(id);
            setSelectedSnapshotId(null);
          }}
          onSelectSnapshot={(snapshot) => {
            if (!snapshot) {
              setSelectedSnapshotId(null);
              clearAppliedState();
              return;
            }
            loadSavedSnapshot(snapshot);
          }}
        />

        <ManagementError message={managementError} />
        {managementNotice ? (
          <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-600">
            {managementNotice}
          </div>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
          <PortfolioPreview portfolio={portfolio} />

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="text-base font-bold text-slate-950">평가 설정</h2>
            <label className="mt-4 block text-sm font-bold text-slate-700">
              기간
              <select className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2" value={period} onChange={(event) => setPeriod(event.target.value as typeof period)}>
                {['1M', '3M', '6M', 'YTD', '1Y', 'Max'].map((value) => <option key={value}>{value}</option>)}
              </select>
            </label>
            <label className="mt-4 block text-sm font-bold text-slate-700">
              기준일
              <input className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2" type="date" value={asOfDate} onChange={(event) => setAsOfDate(event.target.value)} />
            </label>
            <div className="mt-4 grid gap-3">
              <div className="text-sm font-bold text-slate-700">벤치마크</div>
              {layerValues.map((layer) => (
                <label key={layer} className="block text-sm font-bold text-slate-700">
                  {layerLabels[layer]}
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2"
                    value={layerBenchmarks[layer]}
                    onChange={(event) => updateLayerBenchmark(layer, event.target.value)}
                  />
                </label>
              ))}
            </div>

            <div className="mt-5 grid gap-2">
              <button
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!canRunEvaluationOnly}
                type="button"
                onClick={runEvaluationOnlyWorkflow}
              >
                {pending === 'analysis' || pending === 'evaluation' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {selectedSnapshotId === null ? '평가 실행' : '다시 평가'}
              </button>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
                disabled={!canSaveEvaluationRun}
                type="button"
                onClick={saveCurrentEvaluationRun}
              >
                {managementPending === 'evaluation-save' ? <Loader2 className="h-4 w-4 animate-spin" /> : <ClipboardList className="h-4 w-4" />}
                평가 기록 저장
              </button>
              {portfolio.length === 0 ? (
                <p className="text-sm font-semibold text-slate-500">
                  포트폴리오를 입력하면 평가를 실행할 수 있습니다.
                </p>
              ) : selectedSnapshotId === null ? (
                <p className="text-sm font-semibold text-slate-500">
                  평가 기록 저장은 보유현황 스냅샷을 선택한 뒤 사용할 수 있습니다.
                </p>
              ) : null}
            </div>
          </div>
        </section>

        <MetricsStrip analysis={analysis} />

        {evaluation ? (
          <EvaluationResults evaluation={evaluation} evaluationRun={evaluationRun} />
        ) : (
          <section className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm font-semibold text-slate-500">
            평가 결과는 적용·분석·평가 실행 버튼을 누르면 여기에 표시됩니다.
          </section>
        )}
      </div>
      <PortfolioInputModal
        allocationTotal={allocationTotal}
        allocationWarning={allocationWarning}
        canRunWorkflow={canRunWorkflow}
        isOpen={isInputModalOpen}
        pending={pending}
        rowErrors={rowErrors}
        rows={inputRows}
        workflowButtonLabel={workflowButtonLabel}
        onAdd={addInputRow}
        onChange={updateInputRow}
        onClose={() => setInputModalOpen(false)}
        onDelete={deleteInputRow}
        onRun={runPortfolioWorkflow}
      />
      <CreatePortfolioModal
        error={managementError}
        isOpen={isCreatePortfolioModalOpen}
        name={newPortfolioName}
        pending={managementPending}
        onChange={setNewPortfolioName}
        onClose={() => setCreatePortfolioModalOpen(false)}
        onCreate={createNamedPortfolio}
      />
      <SnapshotModal
        allocationTotal={snapshotAllocationTotal}
        allocationWarning={snapshotAllocationWarning}
        canSubmit={canSubmitSnapshot}
        error={managementError}
        isOpen={isSnapshotModalOpen}
        mode={snapshotModalMode}
        name={snapshotModalName}
        note={snapshotModalNote}
        pending={managementPending}
        rowErrors={snapshotRowErrors}
        rows={snapshotRows}
        onAdd={addSnapshotRow}
        onChange={updateSnapshotRow}
        onClose={() => setSnapshotModalOpen(false)}
        onDelete={deleteSnapshotRow}
        onNameChange={setSnapshotModalName}
        onNoteChange={setSnapshotModalNote}
        onSubmit={submitSnapshotModal}
      />
      <DeleteSnapshotConfirmModal
        isOpen={snapshotPendingDelete !== null}
        pending={managementPending}
        snapshot={snapshotPendingDelete}
        onCancel={() => setSnapshotPendingDelete(null)}
        onConfirm={() => {
          if (snapshotPendingDelete) deleteSavedSnapshot(snapshotPendingDelete);
        }}
      />
      <ReviewCopilot
        portfolio={portfolio}
        evaluation={evaluation}
        settings={copilotSettings}
        onAnalysis={setAnalysis}
        onEvaluation={setEvaluation}
      />
      <Toast message={toastMessage} onDismiss={() => setToastMessage(null)} />
    </main>
    </GenerativeUiProvider>
    </ReviewCopilotHost>
  );
}
