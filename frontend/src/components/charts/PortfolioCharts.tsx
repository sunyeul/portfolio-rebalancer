import { BarChart3, LineChart, RefreshCcw, ShieldCheck } from 'lucide-react';

import type { EvaluationResponse, ProposalRow } from '../../lib/api';

type ReliabilityRow = {
  ticker: string;
  weightPct: number | null;
  riskContributionPct: number | null;
  riskWeightGapPct: number | null;
};

const chartWidth = 640;
const chartHeight = 210;
const plotTop = 18;
const plotRight = 16;
const plotBottom = 36;
const plotLeft = 42;
const plotWidth = chartWidth - plotLeft - plotRight;
const plotHeight = chartHeight - plotTop - plotBottom;
const actionFamilyLabels: Record<string, string> = {
  buy_adjustment: '정기매수 조정',
  hold: '유지·관찰',
  thesis_review: '실행 전 검토',
  risk_review: '위험 점검',
  sell_review: '리밸런싱 검토',
  blocked: '판단 보류'
};
const actionFamilyColors: Record<string, string> = {
  buy_adjustment: '#1d4ed8',
  hold: '#0f766e',
  thesis_review: '#7c3aed',
  risk_review: '#dc2626',
  sell_review: '#ca8a04',
  blocked: '#64748b',
  unknown: '#94a3b8'
};

type BarSeries = {
  key: string;
  label: string;
  color: string;
};

type BarPoint = {
  label: string;
  values: Record<string, number | null>;
  colors?: Record<string, string>;
};

type ActionDistributionPoint = {
  family: string;
  familyLabel: string;
  count: number;
  totalWeightPct: number;
  actionLabels: string[];
};

type GroupBandPoint = {
  groupLabel: string;
  current: number;
  min: number | null;
  target: number | null;
  max: number | null;
};

type RiskBudgetPoint = {
  ticker: string;
  currentRc: number;
  targetRc: number;
  riskOver: boolean;
};

type EfficiencyRiskPoint = {
  ticker: string;
  efficiency: number;
  rcGap: number;
  family: string;
  familyLabel: string;
  actionLabel: string;
};

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function pct(value: number | null | undefined, fromUnit = true) {
  if (value === null || value === undefined) return 'N/A';
  return `${(fromUnit ? value * 100 : value).toFixed(2)}%`;
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

function groupLabel(value: string | null | undefined) {
  if (value === 'core') return '코어';
  if (value === 'satellite_ai_infra') return '위성_AI인프라';
  if (value === 'satellite_ai_software') return '위성_AI소프트웨어';
  if (value === 'satellite_nextgen') return '위성_차세대';
  return '코어';
}

function recordByTicker(rows: Array<Record<string, unknown>>) {
  return rows.reduce<Record<string, Record<string, unknown>>>((acc, row) => {
    const ticker = textFromRecord(row, 'ticker');
    if (ticker !== 'unknown') acc[ticker] = row;
    return acc;
  }, {});
}

function recordFromConfig(config: Record<string, unknown> | null | undefined, key: string) {
  const value = config?.[key];
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function numberFromConfig(config: Record<string, unknown>, key: string) {
  const value = config[key];
  return typeof value === 'number' && Number.isFinite(value) ? Number((value * 100).toFixed(2)) : null;
}

function actionFamilyLabel(value: string) {
  return actionFamilyLabels[value] ?? '기타 검토';
}

function actionFamilyColor(value: string) {
  return actionFamilyColors[value] ?? actionFamilyColors.unknown;
}

function scaleY(value: number, max: number) {
  return plotTop + plotHeight - (value / max) * plotHeight;
}

function niceMax(values: number[], floor = 1) {
  const max = Math.max(floor, ...values.map((value) => Math.abs(value)));
  const magnitude = 10 ** Math.floor(Math.log10(max));
  return Math.ceil(max / magnitude) * magnitude;
}

function ChartPanel({
  icon,
  title,
  children,
  legend
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  legend?: Array<{ label: string; color: string }>;
}) {
  return (
    <div className="h-72 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-600">
        {icon}
        {title}
      </div>
      <div className="h-[214px]">
        {children}
      </div>
      {legend?.length ? (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] font-bold text-slate-500">
          {legend.map((item) => (
            <span className="inline-flex items-center gap-1" key={item.label}>
              <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Axis({ max, min = 0 }: { max: number; min?: number }) {
  const zeroY = min < 0 ? scaleSignedY(0, min, max) : scaleY(0, max);
  return (
    <>
      <line stroke="#cbd5e1" x1={plotLeft} x2={chartWidth - plotRight} y1={zeroY} y2={zeroY} />
      <line stroke="#cbd5e1" x1={plotLeft} x2={plotLeft} y1={plotTop} y2={plotTop + plotHeight} />
      <text fill="#64748b" fontSize="11" x={8} y={plotTop + 4}>
        {max}
      </text>
      {min < 0 ? (
        <text fill="#64748b" fontSize="11" x={8} y={plotTop + plotHeight}>
          {min}
        </text>
      ) : null}
    </>
  );
}

function scaleSignedY(value: number, min: number, max: number) {
  return plotTop + ((max - value) / (max - min)) * plotHeight;
}

function GroupedBars({ data, series, max }: { data: BarPoint[]; series: BarSeries[]; max?: number }) {
  const chartMax = max ?? niceMax(data.flatMap((point) => series.map((item) => point.values[item.key] ?? 0)));
  const groupWidth = plotWidth / Math.max(data.length, 1);
  const barWidth = Math.max(5, Math.min(22, (groupWidth - 14) / series.length));

  return (
    <svg className="h-full w-full" role="img" viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
      <Axis max={chartMax} />
      {data.map((point, pointIndex) => {
        const groupX = plotLeft + pointIndex * groupWidth + groupWidth / 2;
        return (
          <g key={point.label}>
            {series.map((item, seriesIndex) => {
              const value = Math.max(0, point.values[item.key] ?? 0);
              const barHeight = (value / chartMax) * plotHeight;
              const x = groupX - (barWidth * series.length) / 2 + seriesIndex * barWidth;
              const y = plotTop + plotHeight - barHeight;
              return (
                <rect
                  fill={point.colors?.[item.key] ?? item.color}
                  height={barHeight}
                  key={item.key}
                  rx="2"
                  width={barWidth - 2}
                  x={x}
                  y={y}
                >
                  <title>{`${point.label} · ${item.label}: ${pct(value, false)}`}</title>
                </rect>
              );
            })}
            <text fill="#64748b" fontSize="10" textAnchor="middle" x={groupX} y={chartHeight - 10}>
              {point.label.length > 8 ? `${point.label.slice(0, 7)}...` : point.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function HorizontalBandChart({ data }: { data: GroupBandPoint[] }) {
  return (
    <svg className="h-full w-full" role="img" viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
      <Axis max={100} />
      {data.map((point, index) => {
        const y = plotTop + 20 + index * 52;
        const xFor = (value: number | null) => plotLeft + ((value ?? 0) / 100) * plotWidth;
        return (
          <g key={point.groupLabel}>
            <text fill="#334155" fontSize="12" fontWeight="700" x={0} y={y + 5}>
              {point.groupLabel}
            </text>
            <line stroke="#cbd5e1" strokeWidth="8" x1={plotLeft} x2={chartWidth - plotRight} y1={y} y2={y} />
            {point.min !== null && point.max !== null ? (
              <line stroke="#94a3b8" strokeWidth="8" x1={xFor(point.min)} x2={xFor(point.max)} y1={y} y2={y}>
                <title>{`${point.groupLabel} IPS 밴드 ${pct(point.min, false)} - ${pct(point.max, false)}`}</title>
              </line>
            ) : null}
            {point.target !== null ? (
              <line stroke="#0f766e" strokeWidth="3" x1={xFor(point.target)} x2={xFor(point.target)} y1={y - 16} y2={y + 16}>
                <title>{`${point.groupLabel} 목표 ${pct(point.target, false)}`}</title>
              </line>
            ) : null}
            <circle cx={xFor(point.current)} cy={y} fill="#1d4ed8" r="7">
              <title>{`${point.groupLabel} 현재 ${pct(point.current, false)}`}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}

function ScatterPanel({ data }: { data: EfficiencyRiskPoint[] }) {
  const absMax = niceMax(data.map((point) => point.rcGap), 1);
  const minY = -absMax;
  const maxY = absMax;

  return (
    <svg className="h-full w-full" role="img" viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
      <Axis max={maxY} min={minY} />
      <line stroke="#94a3b8" strokeDasharray="4 4" x1={plotLeft + plotWidth / 2} x2={plotLeft + plotWidth / 2} y1={plotTop} y2={plotTop + plotHeight} />
      {data.map((point) => {
        const x = plotLeft + point.efficiency * plotWidth;
        const y = scaleSignedY(point.rcGap, minY, maxY);
        return (
          <circle cx={x} cy={y} fill={actionFamilyColor(point.family)} key={point.ticker} r="6">
            <title>{`${point.ticker} · ${point.actionLabel} · E ${point.efficiency.toFixed(2)} · RC Gap ${pct(point.rcGap, false)}`}</title>
          </circle>
        );
      })}
      <text fill="#64748b" fontSize="11" textAnchor="end" x={chartWidth - plotRight} y={chartHeight - 10}>
        E
      </text>
    </svg>
  );
}

export function ReliabilityRiskChart({ data }: { data: ReliabilityRow[] }) {
  if (!data.length) return null;
  const rows = data.map((row) => ({
    label: row.ticker,
    values: {
      weightPct: row.weightPct ?? 0,
      riskContributionPct: row.riskContributionPct ?? 0
    },
    colors: {
      riskContributionPct: (row.riskWeightGapPct ?? 0) > 1 ? '#dc2626' : '#0f766e'
    }
  }));

  return (
    <ChartPanel
      icon={<LineChart className="h-4 w-4 text-blue-700" />}
      legend={[
        { label: '비중 %', color: '#1d4ed8' },
        { label: '위험기여도 %', color: '#0f766e' },
        { label: '위험 초과', color: '#dc2626' }
      ]}
      title="비중 대비 위험기여도 점검"
    >
      <GroupedBars
        data={rows}
        series={[
          { key: 'weightPct', label: '비중 %', color: '#1d4ed8' },
          { key: 'riskContributionPct', label: '위험기여도 %', color: '#0f766e' }
        ]}
      />
    </ChartPanel>
  );
}

export function EvaluationCharts({ evaluation }: { evaluation: EvaluationResponse }) {
  const actionByTicker = recordByTicker(evaluation.ips_actions);
  const proposalByTicker = evaluation.proposal.reduce<Record<string, ProposalRow>>((acc, row) => {
    acc[row.ticker] = row;
    return acc;
  }, {});

  const actionDistributionData = Object.values(
    evaluation.ips_actions.reduce<Record<string, ActionDistributionPoint>>((acc, row) => {
      const ticker = textFromRecord(row, 'ticker');
      const proposal = proposalByTicker[ticker];
      const family = textFromRecord(row, 'action_family');
      const normalizedFamily = family === 'unknown' ? 'unknown' : family;
      const actionLabel = textFromRecord(row, 'action_label');
      const currentWeight = proposal?.current_weight_pct ?? valueFromRecord(row, '현재%');

      if (!acc[normalizedFamily]) {
        acc[normalizedFamily] = {
          family: normalizedFamily,
          familyLabel: actionFamilyLabel(normalizedFamily),
          count: 0,
          totalWeightPct: 0,
          actionLabels: []
        };
      }

      acc[normalizedFamily].count += 1;
      acc[normalizedFamily].totalWeightPct = Number((acc[normalizedFamily].totalWeightPct + currentWeight).toFixed(2));
      if (actionLabel !== 'unknown' && !acc[normalizedFamily].actionLabels.includes(actionLabel)) {
        acc[normalizedFamily].actionLabels.push(actionLabel);
      }
      return acc;
    }, {})
  );

  const targetAllocation = recordFromConfig(evaluation.ips_config_snapshot, 'target_allocation');
  const groupRows = evaluation.group_summary.reduce<Record<string, Record<string, unknown>>>((acc, row) => {
    acc[textFromRecord(row, 'group')] = row;
    return acc;
  }, {});
  const groupBandData = ['core', 'satellite_ai_infra', 'satellite_ai_software', 'satellite_nextgen'].map((group) => {
    const groupConfig = recordFromConfig(targetAllocation, group);
    return {
      groupLabel: groupLabel(group),
      current: Number((valueFromRecord(groupRows[group] ?? {}, 'weight') * 100).toFixed(2)),
      min: numberFromConfig(groupConfig, 'min'),
      target: numberFromConfig(groupConfig, 'target'),
      max: numberFromConfig(groupConfig, 'max')
    };
  });

  const riskBudgetData = evaluation.proposal.map<RiskBudgetPoint>((row) => ({
    ticker: row.ticker,
    currentRc: Number((row.rc_target_pct + row.rc_gap_pct).toFixed(2)),
    targetRc: Number(row.rc_target_pct.toFixed(2)),
    riskOver: row.risk_over
  }));

  const efficiencyRiskData = evaluation.proposal.map<EfficiencyRiskPoint>((row) => {
    const action = actionByTicker[row.ticker] ?? {};
    const family = textFromRecord(action, 'action_family');
    const normalizedFamily = family === 'unknown' ? 'unknown' : family;
    return {
      ticker: row.ticker,
      efficiency: Number((row.efficiency_score ?? 0).toFixed(2)),
      rcGap: Number(row.rc_gap_pct.toFixed(2)),
      family: normalizedFamily,
      familyLabel: actionFamilyLabel(normalizedFamily),
      actionLabel: textFromRecord(action, 'action_label')
    };
  });
  const efficiencyRiskFamilies = Array.from(new Map(efficiencyRiskData.map((row) => [row.family, row.familyLabel])).entries());

  return (
    <div className="mt-5 grid gap-4 xl:grid-cols-2">
      {actionDistributionData.length ? (
        <ChartPanel
          icon={<RefreshCcw className="h-4 w-4 text-blue-700" />}
          legend={actionDistributionData.map((row) => ({ label: row.familyLabel, color: actionFamilyColor(row.family) }))}
          title="액션 분포 판단 근거"
        >
          <GroupedBars
            data={actionDistributionData.map((row) => ({
              label: row.familyLabel,
              values: { count: row.count },
              colors: { count: actionFamilyColor(row.family) }
            }))}
            series={[{ key: 'count', label: '종목 수', color: '#1d4ed8' }]}
          />
        </ChartPanel>
      ) : null}

      <ChartPanel
        icon={<ShieldCheck className="h-4 w-4 text-blue-700" />}
        legend={[
          { label: '현재', color: '#1d4ed8' },
          { label: 'IPS 밴드', color: '#94a3b8' },
          { label: '목표', color: '#0f766e' }
        ]}
        title="그룹 비중 vs IPS 밴드"
      >
        <HorizontalBandChart data={groupBandData} />
      </ChartPanel>

      {riskBudgetData.length ? (
        <ChartPanel
          icon={<BarChart3 className="h-4 w-4 text-blue-700" />}
          legend={[
            { label: 'Current RC %', color: '#1d4ed8' },
            { label: 'RC Target %', color: '#0f766e' },
            { label: '위험 초과', color: '#dc2626' }
          ]}
          title="위험 예산 점검 근거"
        >
          <GroupedBars
            data={riskBudgetData.map((row) => ({
              label: row.ticker,
              values: { currentRc: row.currentRc, targetRc: row.targetRc },
              colors: { currentRc: row.riskOver ? '#dc2626' : '#1d4ed8' }
            }))}
            series={[
              { key: 'currentRc', label: 'Current RC %', color: '#1d4ed8' },
              { key: 'targetRc', label: 'RC Target %', color: '#0f766e' }
            ]}
          />
        </ChartPanel>
      ) : null}

      {efficiencyRiskData.length ? (
        <ChartPanel
          icon={<LineChart className="h-4 w-4 text-blue-700" />}
          legend={efficiencyRiskFamilies.map(([family, label]) => ({ label, color: actionFamilyColor(family) }))}
          title="효율-위험 액션 맵"
        >
          <ScatterPanel data={efficiencyRiskData} />
        </ChartPanel>
      ) : null}
    </div>
  );
}
