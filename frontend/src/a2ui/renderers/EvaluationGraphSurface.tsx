import { BarChart3, CircleDot } from 'lucide-react';
import { useState } from 'react';

import type { EvaluationRecord, EvaluationResponse } from '../../lib/api';
import type { EvaluationGraphSurfacePayload } from '../types';
import { GuardrailNotice } from './GuardrailNotice';

type GraphChart = EvaluationGraphSurfacePayload['charts'][number];
type GraphSource = GraphChart['source'];

const statusStyles: Record<string, { fill: string; stroke: string; text: string; label: string }> = {
  OK: { fill: '#10b981', stroke: '#047857', text: 'text-emerald-700', label: '정상' },
  Watch: { fill: '#f59e0b', stroke: '#b45309', text: 'text-amber-700', label: '관찰' },
  Review: { fill: '#ef4444', stroke: '#b91c1c', text: 'text-red-700', label: '점검' },
  Action: { fill: '#dc2626', stroke: '#7f1d1d', text: 'text-red-800', label: '조치 검토' }
};

const layerLabels: Record<string, string> = {
  core: '코어',
  satellite: '위성',
  experiment: '실험'
};

const layerShapeLabels: Record<string, string> = {
  core: '원',
  satellite: '마름모',
  experiment: '사각형'
};

const thesisLabels: Record<string, string> = {
  valid: '유효',
  watch: '관찰',
  broken: '훼손',
  unknown: '미정'
};

const thesisRank: Record<string, number> = {
  unknown: 0,
  valid: 1,
  watch: 2,
  broken: 3
};

const statusRank: Record<string, number> = {
  OK: 0,
  Watch: 1,
  Review: 2,
  Action: 3
};

const metricLabels: Record<string, string> = {
  current_weight: '현재 비중',
  target_weight: '목표 비중',
  weight_gap: '목표 대비 차이',
  layer_internal_weight: '계층 내 비중',
  period_return: '수익률',
  benchmark_excess_return: '초과 수익률',
  cagr: 'CAGR',
  mdd: 'MDD',
  volatility: '변동성',
  risk_contribution: '위험 기여',
  return_mdd_ratio: '수익/MDD',
  cagr_mdd_ratio: 'CAGR/MDD',
  thesis_status: '논리 상태',
  status: '상태'
};

function pct(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  const numeric = value * 100;
  const sign = numeric > 0 ? '+' : '';
  return `${sign}${numeric.toFixed(1)}%`;
}

function weightPct(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

function ppAbs(valuePp: number) {
  return `${Math.abs(valuePp).toFixed(1)}%p`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return value.toFixed(2);
}

function layerLabel(value: string | null | undefined) {
  return layerLabels[value ?? ''] ?? value ?? '미지정';
}

function metricLabel(metric: string) {
  return metricLabels[metric] ?? metric;
}

function thesisLabel(value: string | null | undefined) {
  return thesisLabels[value ?? ''] ?? value ?? '미정';
}

function evaluationStatusLabel(value: string | null | undefined) {
  if (value === 'Review') return '점검 필요';
  if (value === 'Action') return '조치 검토';
  return statusStyles[value ?? '']?.label ?? value ?? '미정';
}

function valueForMetric(row: EvaluationRecord, metric: string): number | string | null {
  if (metric === 'target_weight') return row.unit.target_weight;
  if (metric === 'weight_gap') {
    return row.unit.target_weight === null ? null : (row.output.current_weight - row.unit.target_weight);
  }
  if (metric === 'status') return row.output.status;
  if (metric === 'thesis_status') return row.output.thesis_status;
  return (row.output[metric as keyof EvaluationRecord['output']] as number | null | undefined) ?? null;
}

function numericValueForMetric(row: EvaluationRecord, metric: string): number | null {
  const value = valueForMetric(row, metric);
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (metric === 'status' && typeof value === 'string') return statusRank[value] ?? null;
  if (metric === 'thesis_status' && typeof value === 'string') return thesisRank[value] ?? null;
  return null;
}

function rowsForSource(evaluation: EvaluationResponse | null, source: GraphSource) {
  if (!evaluation) return [];
  return source === 'layer_evaluations' ? evaluation.layer_evaluations : evaluation.asset_evaluations;
}

function applyChartFilter(rows: EvaluationRecord[], chart: GraphChart) {
  const filter = 'filter' in chart ? chart.filter : undefined;
  if (!filter) return rows;

  return rows.filter((row) => {
    const rowLayer = row.unit.level === 'layer' ? row.unit.name : row.unit.parent_layer;
    if (filter.status && !filter.status.includes(row.output.status)) return false;
    if (filter.layer && !filter.layer.includes(rowLayer as never)) return false;
    if (filter.thesis_status && !filter.thesis_status.includes(row.output.thesis_status)) return false;
    return true;
  });
}

function sourceLabel(source: GraphSource) {
  return source === 'layer_evaluations' ? '계층 평가 테이블' : '종목 평가 테이블';
}

function criteriaText(chart: GraphChart) {
  const parts = [`source: ${sourceLabel(chart.source)}`];
  if (chart.chart_type === 'layer_weight_gap_bar') {
    parts.push('metrics: 현재 비중, 목표 비중, 현재-목표');
  } else if (chart.chart_type === 'asset_risk_scatter') {
    parts.push('metrics: 현재 비중, 위험 기여, 수익률, 상태, 계층');
  } else {
    parts.push(`metric: ${metricLabel(chart.metric)}`);
    if (chart.sort) parts.push(`sort: ${metricLabel(chart.sort.by)} ${chart.sort.direction}`);
    if (chart.limit) parts.push(`limit: ${chart.limit}`);
  }
  if ('filter' in chart && chart.filter) {
    const filters = Object.entries(chart.filter)
      .filter(([, value]) => Array.isArray(value) && value.length > 0)
      .map(([key, value]) => `${key}=${(value as string[]).join(',')}`);
    if (filters.length) parts.push(`filter: ${filters.join(' / ')}`);
  }
  return parts.join(' · ');
}

function EmptyGraphMessage({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-white px-3 py-8 text-center text-sm font-semibold text-slate-500">
      {text}
    </div>
  );
}

const TARGET_TOLERANCE_PP = 1.0;

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function layerGapState(current: number, target: number | null) {
  if (target === null) {
    return {
      action: '상세 영역에서 목표 비중을 확인하세요.',
      badgeClass: 'border-slate-300 bg-slate-100 text-slate-700',
      deltaPp: null,
      label: '목표 미설정',
      summary: '목표 미설정'
    };
  }

  const deltaPp = (current - target) * 100;
  const absGapPp = Math.abs(deltaPp);
  if (absGapPp <= TARGET_TOLERANCE_PP) {
    return {
      action: '현재 배분 정책 유지',
      badgeClass: 'border-emerald-200 bg-emerald-50 text-emerald-800',
      deltaPp,
      label: '균형',
      summary: '목표 범위 내'
    };
  }
  if (deltaPp < 0) {
    return {
      action: '향후 추가 배분 우선 검토',
      badgeClass: 'border-amber-200 bg-amber-50 text-amber-800',
      deltaPp,
      label: '부족',
      summary: `부족 ${ppAbs(deltaPp)}`
    };
  }
  return {
    action: target === 0 ? '보유 지속 여부 또는 목표 비중 재설정 리뷰' : '향후 신규 배분 중단 / 비중 축소 필요성 리뷰',
    badgeClass: 'border-red-200 bg-red-50 text-red-800',
    deltaPp,
    label: '초과',
    summary: `초과 ${ppAbs(deltaPp)}`
  };
}

function MarkerLabel({
  className,
  label,
  position,
  value,
  vertical
}: {
  className: string;
  label: string;
  position: number;
  value: string;
  vertical: 'above' | 'below';
}) {
  const transform = position <= 4 ? 'translateX(0)' : position >= 96 ? 'translateX(-100%)' : 'translateX(-50%)';
  return (
    <span
      className={`absolute z-10 whitespace-nowrap rounded-md bg-white px-1.5 py-0.5 text-[11px] font-extrabold shadow-sm ${className}`}
      style={{
        left: `${position}%`,
        top: vertical === 'above' ? 0 : 45,
        transform
      }}
    >
      {label} {value}
    </span>
  );
}

function AllocationComparisonBar({
  current,
  gapFillClass,
  gapState,
  target
}: {
  current: number;
  gapFillClass: string;
  gapState: ReturnType<typeof layerGapState>;
  target: number;
}) {
  const currentPosition = clampPercent(current * 100);
  const targetPosition = clampPercent(target * 100);
  const gapStart = Math.min(currentPosition, targetPosition);
  const gapWidth = Math.abs(currentPosition - targetPosition);

  return (
    <div className="grid gap-1">
      <div
        className="relative h-16"
        aria-label={`현재 ${weightPct(current)} · 목표 ${weightPct(target)} · 현재-목표 ${gapState.deltaPp?.toFixed(1)}%p`}
      >
        <MarkerLabel
          className="border border-cyan-100 text-cyan-800"
          label="현재"
          position={currentPosition}
          value={weightPct(current)}
          vertical="below"
        />
        <MarkerLabel
          className="border border-slate-200 text-slate-700"
          label="목표"
          position={targetPosition}
          value={weightPct(target)}
          vertical="above"
        />

        <div className="absolute left-0 right-0 top-8 h-5 -translate-y-1/2 rounded-full bg-slate-100">
          <div className="absolute left-0 top-0 h-full rounded-full bg-cyan-700" style={{ width: `${currentPosition}%` }} />
          {gapWidth > 0 ? (
            <div className={`absolute top-0 h-full ${gapFillClass}`} style={{ left: `${gapStart}%`, width: `${gapWidth}%` }} />
          ) : null}
          <div
            className="absolute top-1/2 h-8 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-slate-700"
            style={{ left: `${targetPosition}%` }}
            title={`목표 ${weightPct(target)}`}
          />
          <div
            className="absolute top-1/2 h-9 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-cyan-950"
            style={{ left: `${currentPosition}%` }}
            title={`현재 ${weightPct(current)}`}
          />
        </div>
      </div>
      <div className="flex items-center justify-between text-[11px] font-extrabold text-slate-500">
        <span>0%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function LayerWeightGapBar({
  chart,
  evaluation,
  focusedLayer,
  onLayerFocus
}: {
  chart: GraphChart;
  evaluation: EvaluationResponse | null;
  focusedLayer: string | null;
  onLayerFocus?: (layer: string | null) => void;
}) {
  const rows = applyChartFilter(rowsForSource(evaluation, 'layer_evaluations'), chart);

  if (chart.chart_type !== 'layer_weight_gap_bar') return null;
  if (rows.length === 0) return <EmptyGraphMessage text="표시할 계층 평가 값이 없습니다." />;

  return (
    <div className="grid gap-3">
      {rows.map((row) => {
        const current = row.output.current_weight ?? 0;
        const target = row.unit.target_weight;
        const gapState = layerGapState(current, target);
        const gapFillClass =
          gapState.label === '부족'
            ? 'bg-amber-400'
            : gapState.label === '초과'
              ? 'bg-red-400'
              : 'bg-emerald-400';
        const status = statusStyles[row.output.status] ?? statusStyles.OK;
        const targetIsZero = target === 0;
        const layerIsFocused = focusedLayer === row.unit.name;
        const layerIsDimmed = focusedLayer !== null && !layerIsFocused;
        return (
          <div
            key={row.unit.name}
            className={`grid cursor-pointer gap-3 rounded-md border bg-white px-3 py-3 transition ${
              layerIsFocused ? 'border-cyan-300 ring-2 ring-cyan-100' : 'border-slate-100'
            } ${layerIsDimmed ? 'bg-slate-50/60' : ''}`}
            onClick={() => onLayerFocus?.(row.unit.name)}
            onMouseEnter={() => onLayerFocus?.(row.unit.name)}
          >
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <strong className="text-slate-950">{layerLabel(row.unit.name)}</strong>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`rounded-full border px-2 py-0.5 text-xs font-extrabold ${gapState.badgeClass}`}>
                  비중: {gapState.label}
                </span>
                <span className={`rounded-full bg-slate-50 px-2 py-0.5 text-xs font-extrabold ${status.text}`}>
                  평가: {evaluationStatusLabel(row.output.status)}
                </span>
                <a
                  className="rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs font-extrabold text-slate-600 transition hover:border-cyan-300 hover:text-cyan-800"
                  href="#asset-evaluation-table"
                  onClick={() => onLayerFocus?.(row.unit.name)}
                >
                  테이블에서 보기
                </a>
              </div>
            </div>

            <div className="flex flex-wrap gap-x-3 gap-y-1 text-sm font-bold text-slate-700">
              <span>현재 {weightPct(current)}</span>
              <span>목표 {target === null ? '목표 미설정' : weightPct(target)}</span>
              <span>목표 대비 {gapState.summary}</span>
            </div>

            <div className="grid gap-2">
              {!targetIsZero && target !== null ? (
                <div className={layerIsDimmed ? 'opacity-45' : ''}>
                  <AllocationComparisonBar
                    current={current}
                    gapFillClass={gapFillClass}
                    gapState={gapState}
                    target={target}
                  />
                </div>
              ) : null}
              {targetIsZero ? (
                <div className="rounded-md border border-red-100 bg-red-50 px-3 py-2 text-xs font-bold leading-5 text-red-800">
                  목표가 0%로 설정되어 있습니다. 보유를 허용하려면 실험 목표 비중을 설정하세요.
                </div>
              ) : target === null ? (
                <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold leading-5 text-slate-600">
                  목표 미설정 상태입니다. 상세 영역에서 목표 비중을 확인하세요.
                </div>
              ) : null}
              <div className="text-sm font-bold text-slate-700">
                권장 조치: <span className={gapState.label === '초과' ? 'text-red-800' : gapState.label === '부족' ? 'text-amber-800' : 'text-emerald-800'}>
                  {gapState.action}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function LayerPoint({
  fill,
  isFocused,
  layer,
  onClick,
  onMouseEnter,
  onMouseLeave,
  stroke,
  x,
  y
}: {
  fill: string;
  isFocused: boolean;
  layer: string | null;
  onClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  stroke: string;
  x: number;
  y: number;
}) {
  const commonProps = {
    className: 'cursor-pointer transition',
    fill,
    onClick,
    onMouseEnter,
    onMouseLeave,
    stroke,
    strokeWidth: isFocused ? 3 : 1.5
  };

  if (layer === 'satellite') {
    return <rect {...commonProps} height="12" transform={`rotate(45 ${x} ${y})`} width="12" x={x - 6} y={y - 6} />;
  }
  if (layer === 'experiment') {
    return <rect {...commonProps} height="11" width="11" x={x - 5.5} y={y - 5.5} />;
  }
  return <circle {...commonProps} cx={x} cy={y} r={isFocused ? 7 : 6} />;
}

function SelectedGraphSummary({
  evaluation,
  focusedLayer,
  focusedTicker
}: {
  evaluation: EvaluationResponse | null;
  focusedLayer: string | null;
  focusedTicker: string | null;
}) {
  let summaryText: string | null = null;

  if (focusedLayer !== null) {
    const layerRow = evaluation?.layer_evaluations.find((row) => row.unit.name === focusedLayer);
    const gapState = layerGapState(layerRow?.output.current_weight ?? 0, layerRow?.unit.target_weight ?? null);
    const gapText = gapState.deltaPp === null ? gapState.summary : `${gapState.summary}`;
    summaryText = `선택: ${layerLabel(focusedLayer)} · ${gapText} · ${layerLabel(focusedLayer)} 종목 강조 중`;
  } else if (focusedTicker !== null) {
    const assetRow = evaluation?.asset_evaluations.find((row) => row.unit.name === focusedTicker);
    summaryText = `선택: ${focusedTicker}${
      assetRow ? ` · ${layerLabel(assetRow.unit.parent_layer)} · ${evaluationStatusLabel(assetRow.output.status)} 종목 강조 중` : ' · 종목 강조 중'
    }`;
  }

  return (
    <div className="h-9 min-w-0">
      {summaryText ? (
        <div
          className="flex h-9 min-w-0 items-center rounded-md border border-cyan-200 bg-cyan-50 px-3 text-xs font-extrabold text-cyan-950"
          title={summaryText}
        >
          <span className="min-w-0 truncate">{summaryText}</span>
        </div>
      ) : null}
    </div>
  );
}

type ScatterTooltip = {
  kind: 'risk' | 'return';
  metricLabel: string;
  metricValue: number | null | undefined;
  row: EvaluationRecord;
  x: number;
  y: number;
} | null;

function AssetRiskScatter({
  chart,
  evaluation,
  focusedLayer,
  focusedTicker,
  onTickerFocus,
  onTickerSelect
}: {
  chart: GraphChart;
  evaluation: EvaluationResponse | null;
  focusedLayer: string | null;
  focusedTicker: string | null;
  onTickerFocus?: (ticker: string | null) => void;
  onTickerSelect?: (ticker: string) => void;
}) {
  const [mobileChart, setMobileChart] = useState<'risk' | 'return'>('risk');
  const [tooltip, setTooltip] = useState<ScatterTooltip>(null);
  const rows = applyChartFilter(rowsForSource(evaluation, 'asset_evaluations'), chart);
  const width = 720;
  const height = 238;
  const pad = 40;
  const plotWidth = width - pad * 2;
  const plotHeight = height - pad * 2;
  const sharedWeightAxisMax = Math.max(0.01, ...rows.map((row) => row.output.current_weight ?? 0));
  const riskAxisMax = Math.max(
    0.01,
    ...rows.flatMap((row) => [row.output.current_weight ?? 0, row.output.risk_contribution ?? 0])
  );
  const returnValues = rows.map((row) => row.output.period_return ?? 0);
  const rawReturnMin = Math.min(0, ...returnValues);
  const rawReturnMax = Math.max(0, ...returnValues);
  const returnPadding = Math.max(0.01, (rawReturnMax - rawReturnMin) * 0.12);
  const returnMin = rawReturnMin - returnPadding;
  const returnMax = rawReturnMax + returnPadding;

  if (chart.chart_type !== 'asset_risk_scatter') return null;
  if (rows.length === 0) return <EmptyGraphMessage text="표시할 종목 평가 값이 없습니다." />;

  function labelTickersFor(kind: 'risk' | 'return') {
    const labelTickers = new Set(
      rows
        .filter((row) => row.output.status === 'Review' || row.output.status === 'Action')
        .map((row) => row.unit.name)
    );

    if (focusedLayer !== null) {
      rows
        .filter((row) => row.unit.parent_layer === focusedLayer)
        .sort((left, right) => {
          const rightMetric = kind === 'risk' ? right.output.risk_contribution : Math.abs(right.output.period_return ?? 0);
          const leftMetric = kind === 'risk' ? left.output.risk_contribution : Math.abs(left.output.period_return ?? 0);
          const rightScore = Math.max(rightMetric ?? 0, right.output.current_weight ?? 0);
          const leftScore = Math.max(leftMetric ?? 0, left.output.current_weight ?? 0);
          return rightScore - leftScore;
        })
        .slice(0, 4)
        .forEach((row) => labelTickers.add(row.unit.name));
    } else if (kind === 'risk') {
      [...rows]
        .sort((left, right) => (right.output.risk_contribution ?? -1) - (left.output.risk_contribution ?? -1))
        .slice(0, 5)
        .forEach((row) => labelTickers.add(row.unit.name));
      [...rows]
        .sort((left, right) => (right.output.current_weight ?? -1) - (left.output.current_weight ?? -1))
        .slice(0, 5)
        .forEach((row) => labelTickers.add(row.unit.name));
    } else {
      [...rows]
        .sort((left, right) => (right.output.period_return ?? -Infinity) - (left.output.period_return ?? -Infinity))
        .slice(0, 5)
        .forEach((row) => labelTickers.add(row.unit.name));
      [...rows]
        .sort((left, right) => (left.output.period_return ?? Infinity) - (right.output.period_return ?? Infinity))
        .slice(0, 3)
        .forEach((row) => labelTickers.add(row.unit.name));
    }

    if (focusedTicker !== null) labelTickers.add(focusedTicker);
    return labelTickers;
  }

  function renderScatter(kind: 'risk' | 'return') {
    const isRisk = kind === 'risk';
    const xMax = sharedWeightAxisMax;
    const labelTickers = labelTickersFor(kind);
    const yTicks = isRisk
      ? [0, 0.25, 0.5, 0.75, 1].map((tick) => riskAxisMax * tick)
      : [returnMin, 0, returnMax].filter((value, index, values) => values.findIndex((candidate) => Math.abs(candidate - value) < 0.0001) === index);
    const title = isRisk ? '비중 × 위험 기여' : '비중 × 수익률';
    const yLabel = isRisk ? '위험 기여' : '수익률';
    const baselineLabel = isRisk ? '위험 기여 = 비중' : '수익률 0%';
    const formatYValue = (value: number | null | undefined) => (isRisk ? weightPct(value) : pct(value));
    const focusedRow = focusedTicker === null ? null : rows.find((row) => row.unit.name === focusedTicker);
    const focusedX =
      focusedRow === undefined || focusedRow === null
        ? null
        : pad + ((focusedRow.output.current_weight ?? 0) / xMax) * plotWidth;

    const yForValue = (value: number) => {
      if (isRisk) return height - pad - (value / riskAxisMax) * plotHeight;
      return pad + ((returnMax - value) / (returnMax - returnMin)) * plotHeight;
    };

    return (
      <div className="relative min-w-[680px] rounded-md border border-slate-200 bg-white p-2">
        <div className="mb-1 flex items-center justify-between gap-2 px-1">
          <h5 className="m-0 text-xs font-extrabold text-slate-900">{title}</h5>
          <span className="text-[11px] font-bold text-slate-500">
            {isRisk ? '기준선 위 = 비중 대비 위험 기여 큼' : '기준선 위 = 수익, 아래 = 손실'}
          </span>
        </div>
        <svg aria-label={title} className="w-full" role="img" viewBox={`0 0 ${width} ${height}`}>
          <rect x="0" y="0" width={width} height={height} rx="8" fill="#ffffff" />
          {isRisk ? (
            <line
              x1={pad}
              x2={width - pad}
              y1={height - pad}
              y2={yForValue(sharedWeightAxisMax)}
              stroke="#94a3b8"
              strokeDasharray="5 5"
              strokeWidth="1.5"
            />
          ) : (
            <line
              x1={pad}
              x2={width - pad}
              y1={yForValue(0)}
              y2={yForValue(0)}
              stroke="#94a3b8"
              strokeDasharray="5 5"
              strokeWidth="1.5"
            />
          )}
          <text
            x={width - pad - 4}
            y={isRisk ? pad + 12 : yForValue(0) - 6}
            fill="#64748b"
            fontSize="11"
            fontWeight="700"
            textAnchor="end"
          >
            {baselineLabel}
          </text>
          <line x1={pad} x2={pad} y1={pad} y2={height - pad} stroke="#cbd5e1" />
          <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} stroke="#cbd5e1" />
          {focusedX !== null ? (
            <line
              x1={focusedX}
              x2={focusedX}
              y1={pad}
              y2={height - pad}
              stroke="#0f172a"
              strokeDasharray="3 4"
              strokeOpacity="0.35"
            />
          ) : null}
          <text x={pad} y={22} fill="#64748b" fontSize="12" fontWeight="700">{yLabel}</text>
          <text x={width - pad} y={height - 12} fill="#64748b" fontSize="12" fontWeight="700" textAnchor="end">비중</text>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const x = pad + plotWidth * tick;
            return (
              <g key={`x-${kind}-${tick}`}>
                <line x1={x} x2={x} y1={height - pad} y2={height - pad + 4} stroke="#cbd5e1" />
                <text x={x} y={height - pad + 18} fill="#64748b" fontSize="11" textAnchor="middle">{weightPct(xMax * tick)}</text>
              </g>
            );
          })}
          {yTicks.map((tick) => {
            const y = yForValue(tick);
            return (
              <g key={`y-${kind}-${tick}`}>
                <line x1={pad - 4} x2={pad} y1={y} y2={y} stroke="#cbd5e1" />
                <text x={pad - 8} y={y + 4} fill="#64748b" fontSize="11" textAnchor="end">{formatYValue(tick)}</text>
              </g>
            );
          })}
          {rows.map((row) => {
            const metricValue = isRisk ? row.output.risk_contribution : row.output.period_return;
            const x = pad + ((row.output.current_weight ?? 0) / xMax) * plotWidth;
            const y = yForValue(metricValue ?? 0);
            const status = statusStyles[row.output.status] ?? statusStyles.OK;
            const pointLayer = row.unit.parent_layer;
            const isFocused = focusedTicker === row.unit.name || focusedLayer === pointLayer;
            const isDimmed =
              (focusedTicker !== null && focusedTicker !== row.unit.name) ||
              (focusedTicker === null && focusedLayer !== null && focusedLayer !== pointLayer);
            const shouldLabel = labelTickers.has(row.unit.name);
            return (
              <g key={`${kind}:${row.unit.parent_layer}:${row.unit.name}`} opacity={isDimmed ? 0.25 : 1}>
                <LayerPoint
                  fill={status.fill}
                  isFocused={isFocused}
                  layer={pointLayer}
                  onClick={() => onTickerSelect?.(row.unit.name)}
                  onMouseEnter={() => {
                    setTooltip({ kind, metricLabel: yLabel, metricValue, row, x, y });
                    onTickerFocus?.(row.unit.name);
                  }}
                  onMouseLeave={() => {
                    setTooltip(null);
                    onTickerFocus?.(null);
                  }}
                  stroke={isFocused ? '#0f172a' : status.stroke}
                  x={x}
                  y={y}
                />
                <title>{`${row.unit.name} · 계층: ${layerLabel(row.unit.parent_layer)} · 비중: ${weightPct(row.output.current_weight)} · ${yLabel}: ${formatYValue(metricValue)} · 상태: ${status.label} · 논리: ${thesisLabel(row.output.thesis_status)}`}</title>
                {shouldLabel ? (
                  <text x={x + 8} y={y + 4} fill="#334155" fontSize="11" fontWeight="700">{row.unit.name}</text>
                ) : null}
              </g>
            );
          })}
        </svg>
        {tooltip?.kind === kind ? (
          <div
            className="pointer-events-none absolute z-20 w-44 rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-700 shadow-lg"
            style={{
              left: `${(tooltip.x / width) * 100}%`,
              top: `${(tooltip.y / height) * 100}%`,
              transform: tooltip.x > width * 0.72 ? 'translate(-105%, -96%)' : 'translate(12px, -96%)'
            }}
          >
            <div className="text-sm font-extrabold text-slate-950">{tooltip.row.unit.name}</div>
            <div>계층: {layerLabel(tooltip.row.unit.parent_layer)}</div>
            <div>비중: {weightPct(tooltip.row.output.current_weight)}</div>
            <div>{tooltip.metricLabel}: {tooltip.kind === 'risk' ? weightPct(tooltip.metricValue) : pct(tooltip.metricValue)}</div>
            <div>상태: {statusStyles[tooltip.row.output.status]?.label ?? tooltip.row.output.status}</div>
            <div>논리: {thesisLabel(tooltip.row.output.thesis_status)}</div>
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      <SelectedGraphSummary evaluation={evaluation} focusedLayer={focusedLayer} focusedTicker={focusedTicker} />
      <div className="flex rounded-md border border-slate-200 bg-white p-1 md:hidden">
        {[
          ['risk', '위험 기여'],
          ['return', '수익률']
        ].map(([value, label]) => (
          <button
            key={value}
            className={`flex-1 rounded px-3 py-2 text-xs font-extrabold transition ${
              mobileChart === value ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-50'
            }`}
            type="button"
            onClick={() => setMobileChart(value as 'risk' | 'return')}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="md:hidden">{renderScatter(mobileChart)}</div>
      <div className="hidden gap-3 md:grid">
        {renderScatter('risk')}
        {renderScatter('return')}
      </div>
      <div className="grid gap-2 text-xs font-bold text-slate-600">
        <div className="flex flex-wrap items-center gap-3">
          <span className="w-9 text-slate-500">상태:</span>
          {['OK', 'Watch', 'Review', 'Action'].map((statusKey) => (
            <span key={statusKey} className="inline-flex items-center gap-1">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: statusStyles[statusKey].fill }}
              />
              {statusStyles[statusKey].label}
            </span>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="w-9 text-slate-500">계층:</span>
          {Object.entries(layerShapeLabels).map(([layer, shape]) => (
            <span key={layer}>
              {layerLabel(layer)}={shape}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricBar({ chart, evaluation }: { chart: GraphChart; evaluation: EvaluationResponse | null }) {
  if (chart.chart_type !== 'metric_bar') return null;
  let rows = applyChartFilter(rowsForSource(evaluation, chart.source), chart);
  if (chart.sort) {
    rows = [...rows].sort((left, right) => {
      const leftValue = numericValueForMetric(left, chart.sort!.by);
      const rightValue = numericValueForMetric(right, chart.sort!.by);
      if (leftValue === null && rightValue === null) return 0;
      if (leftValue === null) return 1;
      if (rightValue === null) return -1;
      return chart.sort!.direction === 'asc' ? leftValue - rightValue : rightValue - leftValue;
    });
  }
  if (chart.limit) rows = rows.slice(0, chart.limit);

  const values = rows.map((row) => numericValueForMetric(row, chart.metric)).filter((value): value is number => value !== null);
  const max = Math.max(0.01, ...values.map((value) => Math.abs(value)));

  if (rows.length === 0) return <EmptyGraphMessage text="표시할 평가 값이 없습니다." />;

  return (
    <div className="grid gap-2">
      {rows.map((row) => {
        const numericValue = numericValueForMetric(row, chart.metric);
        const width = numericValue === null ? 0 : Math.min(100, Math.abs(numericValue / max) * 100);
        const status = statusStyles[row.output.status] ?? statusStyles.OK;
        const label = row.unit.level === 'layer' ? layerLabel(row.unit.name) : row.unit.name;
        const display =
          chart.metric === 'status' ? row.output.status :
          chart.metric === 'thesis_status' ? row.output.thesis_status :
          chart.metric.endsWith('ratio') ? num(numericValue) : pct(numericValue);
        return (
          <div key={`${chart.id}:${row.unit.name}`} className="grid grid-cols-[96px_minmax(0,1fr)_72px] items-center gap-2 text-xs font-bold text-slate-600">
            <span className="truncate text-slate-800">{label}</span>
            <div className="h-4 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full" style={{ width: `${width}%`, backgroundColor: status.fill }} />
            </div>
            <span className="text-right">{display}</span>
          </div>
        );
      })}
    </div>
  );
}

function ChartRenderer({
  chart,
  evaluation,
  focusedLayer,
  focusedTicker,
  onLayerFocus,
  onTickerFocus,
  onTickerSelect
}: {
  chart: GraphChart;
  evaluation: EvaluationResponse | null;
  focusedLayer: string | null;
  focusedTicker: string | null;
  onLayerFocus?: (layer: string | null) => void;
  onTickerFocus?: (ticker: string | null) => void;
  onTickerSelect?: (ticker: string) => void;
}) {
  return (
    <article className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <header className="flex items-start gap-2">
        {chart.chart_type === 'asset_risk_scatter' ? (
          <CircleDot className="mt-1 h-4 w-4 shrink-0 text-cyan-800" aria-hidden="true" />
        ) : (
          <BarChart3 className="mt-1 h-4 w-4 shrink-0 text-cyan-800" aria-hidden="true" />
        )}
        <div className="min-w-0">
          <h4 className="m-0 text-sm font-extrabold text-slate-950">{chart.title}</h4>
          {chart.description ? <p className="m-0 mt-1 text-xs font-semibold text-slate-500">{chart.description}</p> : null}
        </div>
      </header>
      {chart.chart_type === 'layer_weight_gap_bar' ? (
        <LayerWeightGapBar
          chart={chart}
          evaluation={evaluation}
          focusedLayer={focusedLayer}
          onLayerFocus={onLayerFocus}
        />
      ) : null}
      {chart.chart_type === 'asset_risk_scatter' ? (
        <AssetRiskScatter
          chart={chart}
          evaluation={evaluation}
          focusedLayer={focusedLayer}
          focusedTicker={focusedTicker}
          onTickerFocus={onTickerFocus}
          onTickerSelect={onTickerSelect}
        />
      ) : null}
      {chart.chart_type === 'metric_bar' ? <MetricBar chart={chart} evaluation={evaluation} /> : null}
      <p className="m-0 text-xs font-bold text-slate-500">표시 기준: {criteriaText(chart)}</p>
    </article>
  );
}

export function EvaluationGraphSurface({
  evaluation,
  focusedLayer,
  focusedTicker,
  onLayerFocus,
  onTickerFocus,
  onTickerSelect,
  surface
}: {
  evaluation: EvaluationResponse | null;
  focusedLayer: string | null;
  focusedTicker: string | null;
  onLayerFocus?: (layer: string | null) => void;
  onTickerFocus?: (ticker: string | null) => void;
  onTickerSelect?: (ticker: string) => void;
  surface: EvaluationGraphSurfacePayload;
}) {
  return (
    <section className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4">
      <header className="flex items-start gap-2">
        <BarChart3 className="mt-1 h-5 w-5 shrink-0 text-cyan-800" aria-hidden="true" />
        <div className="min-w-0">
          <h2 className="m-0 text-base font-bold text-slate-950">{surface.title}</h2>
          <p className="m-0 mt-1 text-sm font-semibold text-slate-500">
            {surface.evaluation_period.label}: {surface.evaluation_period.start_date} ~ {surface.evaluation_period.end_date}
          </p>
        </div>
      </header>

      <GuardrailNotice text={surface.guardrail_notice.text} />

      <div className="grid gap-3 xl:grid-cols-2">
        {surface.charts.map((chart) => (
          <ChartRenderer
            key={chart.id}
            chart={chart}
            evaluation={evaluation}
            focusedLayer={focusedLayer}
            focusedTicker={focusedTicker}
            onLayerFocus={onLayerFocus}
            onTickerFocus={onTickerFocus}
            onTickerSelect={onTickerSelect}
          />
        ))}
      </div>
    </section>
  );
}
