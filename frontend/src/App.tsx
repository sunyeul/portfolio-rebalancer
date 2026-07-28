import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CircleAlert, CircleCheck, ChevronDown, ChevronUp, Layers3, ListFilter, PanelLeftClose, PanelLeftOpen, RefreshCw, RotateCcw, Search, ShieldCheck, type LucideIcon } from "lucide-react";
import { Evaluation, getJson, type InspectionData, type InspectionItem, type JsonObject, type PerformanceRun } from "./lib/api";
import { evidenceValue, finiteNumber, formatAccountReturn, formatAllocationReason, formatKrw as money, formatPercent as percent, formatQueuePriority, formatSignedKrw as signedMoney, supportedRate } from "./lib/presentation";
import { filterAndSortRows, toggleSort, uniqueFilterValues, type SortState } from "./lib/tableControls";

const statusLabel: Record<string, string> = { OK: "OK", Watch: "Watch", Review: "Review", Action: "Action" };
const sidebarStorageKey = "ips-pilot.sidebar-collapsed";
type TabKey = "overview" | "performance" | "allocation" | "review";
type PerformanceView = "ytd" | "trailing_12m" | "cumulative" | "holding";
type NavItem = [LucideIcon, string, TabKey];

const navItems: NavItem[] = [[Activity, "Overview", "overview"], [BarChart3, "Performance", "performance"], [Layers3, "Allocation", "allocation"], [CircleAlert, "Review Queue", "review"]];
const panelCopy: Record<TabKey, { eyebrow: string; title: string; description: string }> = {
  overview: { eyebrow: "MONTHLY IPS INSPECTION", title: "이번 달 계좌 운용 점검", description: "현재 상황, 목표 비중, 수익률과 다음 확인 항목을 한 화면에서 확인합니다." },
  performance: { eyebrow: "PERFORMANCE HISTORY", title: "성과 이력", description: "YTD·최근 1년·누적 TWR과 원금·평가금 추이를 확인하는 화면입니다." },
  allocation: { eyebrow: "ALLOCATION REVIEW", title: "비중 점검", description: "현금과 core·satellite·experiment 레이어의 목표 갭을 확인하는 화면입니다." },
  review: { eyebrow: "REVIEW QUEUE", title: "확인할 항목", description: "백엔드 평가 순서를 유지한 전체 읽기 전용 점검 목록입니다." },
};

function initialSidebarCollapsed() {
  if (typeof window === "undefined") return false;
  try { return window.localStorage.getItem(sidebarStorageKey) === "true"; }
  catch { return false; }
}

function visualPercent(value: unknown) {
  const number = finiteNumber(value);
  return number === null ? 0 : Math.min(100, Math.max(0, number * 100));
}

function denominatorLabel(value: unknown) {
  if (value === "gross_account_value") return "총계좌 평가금 기준";
  if (value === "invested_account_value") return "투자금 평가금 기준";
  return "분모 근거 —";
}

function Status({ value }: { value?: unknown }) {
  const status = typeof value === "string" && statusLabel[value] ? value : null;
  return status
    ? <span className={`status status-${status.toLowerCase()}`}>{statusLabel[status]}</span>
    : <span className="status status-missing">—</span>;
}

type TableFilterOption = { key: string; label: string; value: string; options: string[]; onChange: (value: string) => void };

function TableToolbar({ query, onQueryChange, filters, onReset, visibleCount, totalCount, sortActive = false }: { query: string; onQueryChange: (value: string) => void; filters?: TableFilterOption[]; onReset: () => void; visibleCount: number; totalCount: number; sortActive?: boolean }) {
  const hasFilters = query.length > 0 || filters?.some(filter => filter.value.length > 0) === true || sortActive;
  return <div className="table-toolbar" role="search">
    <label className="table-search"><Search size={14} aria-hidden="true" /><span className="sr-only">테이블 검색</span><input value={query} onChange={event => onQueryChange(event.target.value)} placeholder="검색" aria-label="테이블 검색" /></label>
    {filters?.map(filter => <label className="table-filter" key={filter.key}><span>{filter.label}</span><select value={filter.value} onChange={event => filter.onChange(event.target.value)} aria-label={`${filter.label} 필터`}><option value="">전체</option>{filter.options.map(option => <option value={option} key={option}>{option}</option>)}</select></label>)}
    <button type="button" className="table-reset" onClick={onReset} disabled={!hasFilters}><RotateCcw size={13} aria-hidden="true" />초기화</button>
    <span className="table-count"><ListFilter size={13} aria-hidden="true" />{visibleCount} / {totalCount}개 표시</span>
  </div>;
}

function SortableHeader({ label, column, sort, onSort }: { label: string; column: string; sort: SortState; onSort: (column: string) => void }) {
  const active = sort?.key === column;
  const direction = active ? sort.direction : null;
  const ariaSort = direction === "asc" ? "ascending" : direction === "desc" ? "descending" : "none";
  const directionLabel = direction === "asc" ? "내림차순으로 변경" : direction === "desc" ? "정렬 해제" : "오름차순으로 정렬";
  return <th aria-sort={ariaSort}><button type="button" className={`sortable-header${active ? " active" : ""}`} onClick={() => onSort(column)} aria-label={`${label}, ${directionLabel}`}><span>{label}</span>{direction === "desc" ? <ChevronDown size={13} aria-hidden="true" /> : <ChevronUp size={13} aria-hidden="true" />}</button></th>;
}

function TableEmptyRow({ colSpan }: { colSpan: number }) {
  return <tr><td className="table-empty" colSpan={colSpan}>조건에 맞는 항목이 없습니다.</td></tr>;
}

function AdjustmentSuggestions({ items, contractSupported, allocationState, allocationReason }: { items: InspectionItem[]; contractSupported: boolean; allocationState?: unknown; allocationReason?: unknown }) {
  return <section className="adjustment-suggestions" aria-labelledby="adjustment-suggestions-title">
    <div className="section-title"><div><p className="eyebrow">NEXT ALLOCATION REVIEW</p><h2 id="adjustment-suggestions-title">다음 조정 검토</h2></div><span className="denominator">백엔드 우선순위 순</span></div>
    {!contractSupported && <div className="contract-notice"><CircleAlert size={18} /><div><strong>새 조정 계약을 아직 적용하지 않았습니다.</strong><span>현재 저장된 평가는 이전 엔진 결과이므로 우선순위와 조정 제안을 추정하지 않습니다. 새 Toss 평가를 저장하면 이 영역이 활성화됩니다.</span></div></div>}
    {contractSupported && allocationState === "not_evaluable" && <div className="allocation-blocked"><CircleAlert size={18} /><div><strong>비중 조정 판단 보류</strong><span>{formatAllocationReason(allocationReason)} 원천 상태와 정책 근거를 확인하세요.</span></div></div>}
    {contractSupported && allocationState !== "not_evaluable" && items.length === 0 && <div className="suggestion-empty"><CircleCheck size={18} /><span>현재 우선 검토할 비중 조정 항목이 없습니다. 허용 범위 안에서 관찰을 유지합니다.</span></div>}
    {contractSupported && allocationState !== "not_evaluable" && items.length > 0 && <div className="suggestion-list">{items.slice(0, 3).map((item, index) => {
      const suggestion = item.suggestion;
      return <article className="suggestion-card" key={`${String(item.kind)}-${String(item.identity)}-${index}`}>
        <div className="suggestion-meta"><span className="priority-chip">{String(item.priority ?? "—")} · {String(item.priority_label ?? "검토 시점 미상")}</span><Status value={item.status} /></div>
        <div className="suggestion-title-row"><strong>{String(item.identity ?? "—")}</strong><small>{String(item.kind ?? "allocation")}</small></div>
        <div className="suggestion-band"><span>현재 {percent(item.current)}</span><span>허용 {percent(item.minimum)}–{percent(item.maximum)}</span><span>목표 {percent(item.target)}</span></div>
        <p className="suggestion-label">조정 방향 · {String(suggestion?.label ?? "확인 필요")}</p>
        <p className="suggestion-meaning">{String(item.meaning ?? "점검 의미 근거가 필요합니다.")}</p>
        <details><summary>근거와 확인 과제</summary><p>{String(item.verification_task ?? "확인 과제 근거가 필요합니다.")}</p>{Array.isArray(item.triggers) && item.triggers.length > 0 && <small>트리거: {item.triggers.map(String).join(" · ")}</small>}</details>
      </article>;
    })}</div>}
  </section>;
}

function GapCard({ title, row }: { title: string; row?: Record<string, unknown> | null }) {
  return <article className="metric-card">
    <div className="card-heading"><span>{title}</span><Status value={row?.status} /></div>
    <div className="metric-main">{percent(row?.current)}</div>
    <div className="metric-caption">목표 {percent(row?.target)} · 범위 {percent(row?.minimum)}–{percent(row?.maximum)}</div>
    <div className="metric-gap">갭 {percent(row?.gap)} · {String(row?.denominator ?? "자료 확인 필요")}</div>
  </article>;
}

type LayerFinancials = { marketValue: number | null; costBasis: number | null; unrealizedPnl: number | null };

function aggregateLayerFinancials(instruments: JsonObject[], layerByIdentity?: Map<string, string>) {
  const totals = new Map<string, { marketValue: number; costBasis: number; unrealizedPnl: number; hasMarketValue: boolean; hasCostBasis: boolean; hasUnrealizedPnl: boolean }>();
  for (const instrument of instruments) {
    const resolvedLayer = instrument.layer ?? layerByIdentity?.get(String(instrument.identity));
    const layer = typeof resolvedLayer === "string" ? resolvedLayer.toLowerCase() : null;
    if (!layer) continue;
    const total = totals.get(layer) ?? { marketValue: 0, costBasis: 0, unrealizedPnl: 0, hasMarketValue: false, hasCostBasis: false, hasUnrealizedPnl: false };
    const evidence = typeof instrument.evidence === "object" && instrument.evidence !== null ? instrument.evidence as JsonObject : undefined;
    const marketValue = finiteNumber(evidence?.market_value_krw);
    const costBasis = finiteNumber(evidence?.cost_basis_krw);
    const unrealizedPnl = finiteNumber(evidence?.unrealized_pnl_krw);
    if (marketValue !== null) { total.marketValue += marketValue; total.hasMarketValue = true; }
    if (costBasis !== null) { total.costBasis += costBasis; total.hasCostBasis = true; }
    if (unrealizedPnl !== null) { total.unrealizedPnl += unrealizedPnl; total.hasUnrealizedPnl = true; }
    totals.set(layer, total);
  }
  return new Map<string, LayerFinancials>([...totals].map(([layer, total]) => [layer, {
    marketValue: total.hasMarketValue ? total.marketValue : null,
    costBasis: total.hasCostBasis ? total.costBasis : null,
    unrealizedPnl: total.hasUnrealizedPnl ? total.unrealizedPnl : null,
  }]));
}

function AllocationOverview({ cash, layers, instruments, layerByIdentity }: { cash?: InspectionItem | null; layers: InspectionItem[]; instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  const rowsByLayer = new Map(layers.map(row => [String(row.identity ?? "").toLowerCase(), row]));
  const financialsByLayer = aggregateLayerFinancials(instruments, layerByIdentity);
  const rows = [
    { label: "현금", row: cash, financials: null },
    { label: "core", row: rowsByLayer.get("core"), financials: financialsByLayer.get("core") },
    { label: "satellite", row: rowsByLayer.get("satellite"), financials: financialsByLayer.get("satellite") },
    { label: "experiment", row: rowsByLayer.get("experiment"), financials: financialsByLayer.get("experiment") },
  ];
  return <section className="allocation-overview">
    <div className="section-title"><div><p className="eyebrow">ALLOCATION BANDS</p><h2>허용 범위와 현재 비중</h2></div><span className="denominator">현금은 총계좌 · 레이어는 투자금 평가금 기준</span></div>
    <div className="allocation-overview-list">{rows.map(({ label, row, financials }) => {
      const target = finiteNumber(row?.target);
      const marketValue = financials?.marketValue ?? null;
      const costBasis = financials?.costBasis ?? null;
      const unrealizedPnl = financials?.unrealizedPnl ?? null;
      return <article className="allocation-overview-row" key={label}>
        <div className="allocation-row-heading"><div><strong>{label}</strong><small>{denominatorLabel(row?.denominator)}</small></div><Status value={row?.status} /></div>
        <div className="allocation-values"><strong>{percent(row?.current)}</strong><span>허용 {percent(row?.minimum)}–{percent(row?.maximum)} · 목표 {percent(row?.target)}</span></div>
        <div className="allocation-track">
          <span className="allocation-fill" style={{ width: `${visualPercent(row?.current)}%` }} />
          {target !== null && <span className="allocation-target" style={{ left: `${visualPercent(target)}%` }} />}
        </div>
        {label === "현금" ? <dl className="allocation-financials">
          <div><dt>현재 현금</dt><dd>{money(row?.cash_value_krw)}</dd></div>
          <div><dt>매입원가</dt><dd>해당 없음</dd></div>
          <div><dt>평가손익</dt><dd>해당 없음</dd></div>
        </dl> : <dl className="allocation-financials">
          <div><dt>지원 평가금</dt><dd>{marketValue === null ? "자료 없음" : money(marketValue)}</dd></div>
          <div><dt>지원 매입원가</dt><dd>{costBasis === null ? "자료 없음" : money(costBasis)}</dd></div>
          <div><dt>지원 평가손익</dt><dd>{unrealizedPnl === null ? "자료 없음" : signedMoney(unrealizedPnl)}</dd></div>
        </dl>}
      </article>;
    })}</div>
  </section>;
}

function AnnualTargetCard({ performance }: { performance?: JsonObject }) {
  const historyDays = finiteNumber(performance?.history_days);
  const measurement = String(performance?.measurement ?? "ytd_twr");
  const ytdTwr = finiteNumber(performance?.ytd_twr) ?? (measurement === "ytd_twr" ? finiteNumber(performance?.annual_twr) : null);
  const trailingTwr = finiteNumber(performance?.trailing_12m_twr);
  return <article className="annual-target-card">
    <div className="annual-target-heading"><div><p className="eyebrow">YTD ACCOUNT TARGET</p><h2>연간 목표 점검</h2></div><Status value={performance?.status} /></div>
    <div className="annual-target-comparison">
      <div><span>현재 YTD 계좌 TWR</span><strong>{ytdTwr === null ? "산출 전" : percent(ytdTwr)}</strong></div>
      <div><span>연간 목표</span><strong>{percent(performance?.annual_target)}</strong></div>
    </div>
    <dl className="annual-target-facts">
      <div><dt>누적 계좌 TWR</dt><dd>{percent(performance?.cumulative_twr)}</dd></div>
      <div><dt>지원 이력</dt><dd>{historyDays === null ? "—" : `${historyDays.toLocaleString()}일`}</dd></div>
      <div><dt>최근 1년 보조</dt><dd>{trailingTwr === null ? "산출 전" : percent(trailingTwr)}</dd></div>
    </dl>
    <small>{measurement === "ytd_twr" ? "YTD는 1월 1일 기준 스냅샷과 이후 평가 포인트가 있어야 산출합니다." : "최근 1년 TWR은 365일 근거가 쌓인 뒤 산출합니다."} 보유 평가손익률과는 다른 계좌 성과 지표입니다.</small>
  </article>;
}

function CashReserveOverview({ cash, cashValue }: { cash?: JsonObject | null; cashValue?: unknown }) {
  const supportedCashValue = finiteNumber(cashValue) ?? finiteNumber(cash?.cash_value_krw);
  return <section className="section cash-reserve-overview">
    <div className="section-title"><div><p className="eyebrow">CASH RESERVE</p><h2>현금 리저브</h2></div><div className="section-status"><span className="denominator">총계좌 평가금 기준</span><Status value={cash?.status} /></div></div>
    <div className="cash-reserve-grid">
      <article className="cash-reserve-current"><span>현재 현금</span><strong>{money(supportedCashValue)}</strong><small>현재 비중 {percent(cash?.current)}</small></article>
      <article><span>최소</span><strong>{percent(cash?.minimum)}</strong><small>승인 범위 하단</small></article>
      <article><span>목표</span><strong>{percent(cash?.target)}</strong><small>정책 기준점</small></article>
      <article><span>최대</span><strong>{percent(cash?.maximum)}</strong><small>승인 범위 상단</small></article>
    </div>
  </section>;
}

const queueKindLabel: Record<string, string> = { source: "원천", cash: "현금", layer: "레이어", instrument: "종목", performance: "성과", account_risk: "계좌 리스크" };

function ReviewQueue({ items }: { items: InspectionItem[] }) {
  return <section className="review-queue-section full">
    <div className="queue-heading"><div><p className="eyebrow">REVIEW QUEUE</p><h2>전체 확인 항목</h2></div><span className="queue-count">{items.length}</span></div>
    {items.length === 0 && <div className="queue-empty"><CircleCheck size={18} />현재 확인 항목 없음</div>}
    <div className="review-queue-list">{items.map((item, index) => <article className="queue-item" key={`${String(item.kind)}-${String(item.identity)}-${index}`}><div className="queue-top"><div className="queue-axis"><Status value={item.status} />{(item.priority || item.priority_label) && <span className="priority-label">{formatQueuePriority(item.priority, item.priority_label)}</span>}</div><small>{queueKindLabel[String(item.kind)] ?? String(item.kind ?? "근거")}</small></div><strong>{String(item.identity ?? "—")}</strong>{item.suggestion?.label && <p className="queue-suggestion">조정 방향 · {String(item.suggestion.label)}</p>}<p>{String(item.meaning ?? "점검 의미 근거가 필요합니다.")}</p><span>{String(item.verification_task ?? "확인 과제 근거가 필요합니다.")}</span>{item.evidence_refs !== undefined && <details className="evidence-detail"><summary>근거 연결</summary><pre>{String(JSON.stringify(item.evidence_refs as JsonObject, null, 2) ?? "")}</pre></details>}</article>)}</div>
    <div className="queue-footer">이 목록은 읽기 전용 검사 신호이며, 상태와 순서는 백엔드 평가를 그대로 따릅니다.</div>
  </section>;
}

function TrendChart({ points, valueKey, label }: { points: JsonObject[]; valueKey: string; label: string }) {
  const values = points.map(point => typeof point[valueKey] === "number" ? Number(point[valueKey]) : null).filter((value): value is number => value !== null && Number.isFinite(value));
  if (values.length < 2) return <div className="chart-empty">지원되는 값이 2개 이상 쌓이면 추이를 표시합니다.</div>;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = maximum - minimum || 1;
  const chartPoints = values.map((value, index) => `${(index / (values.length - 1)) * 640},${160 - ((value - minimum) / spread) * 140}`).join(" ");
  return <div className="trend-chart">
    <svg viewBox="0 0 640 180" role="img" aria-label={`${label} 추이`} preserveAspectRatio="none"><line x1="0" y1="160" x2="640" y2="160" /><polyline points={chartPoints} /></svg>
    <div className="chart-caption"><span>최저 {money(minimum)}</span><strong>{label}</strong><span>최고 {money(maximum)}</span></div>
  </div>;
}

function PerformancePanel({ summary, run, accountProfitLoss }: { summary?: JsonObject; run: PerformanceRun | null; accountProfitLoss?: JsonObject }) {
  const [view, setView] = useState<PerformanceView>("ytd");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortState>(null);
  const points = Array.isArray(run?.points) ? run.points : [];
  const evaluablePoints = points.filter(point => point.evaluation_state === "evaluable");
  const baselinePoints = points.slice(-8).reverse();
  const visiblePoints = useMemo(() => filterAndSortRows({
    rows: baselinePoints,
    query,
    searchText: point => `${String(point.point_at ?? "")} ${String(point.snapshot_id ?? "")} ${String(point.total_value_krw ?? "")} ${String(point.investment_principal_krw ?? "")} ${String(point.interval_twr ?? "")} ${String(point.evaluation_state ?? "")}`,
    sort,
    columns: {
      pointAt: point => point.point_at,
      value: point => point.invested_value_krw,
      principal: point => point.investment_principal_krw,
      intervalTwr: point => point.interval_twr,
      state: point => point.evaluation_state,
    },
  }), [baselinePoints, query, sort]);
  const measurement = String(summary?.measurement ?? "ytd_twr");
  const ytdTwr = finiteNumber(summary?.ytd_twr) ?? (measurement === "ytd_twr" ? finiteNumber(summary?.annual_twr) : null);
  const trailingTwr = finiteNumber(summary?.trailing_12m_twr);
  const latestPoint = [...evaluablePoints].sort((left, right) => String(left.point_at ?? "").localeCompare(String(right.point_at ?? ""))).at(-1);
  const holdingReturn = supportedRate(latestPoint?.unrealized_pnl_krw, latestPoint?.current_cost_basis_krw);
  const viewDetails: Record<PerformanceView, { label: string; value: number | null; description: string }> = {
    ytd: { label: "YTD 계좌 TWR", value: ytdTwr, description: "연초 기준 계좌 시간가중수익률" },
    trailing_12m: { label: "최근 1년 TWR", value: trailingTwr, description: "최근 365일 계좌 시간가중수익률" },
    cumulative: { label: "누적 계좌 TWR", value: finiteNumber(summary?.cumulative_twr), description: "추적 시작 이후 계좌 시간가중수익률" },
    holding: { label: "보유 평가손익률", value: holdingReturn, description: "최근 평가 포인트의 매입원가 기준 보유 손익률" },
  };
  const selectedView = viewDetails[view];
  const drawdown = (accountProfitLoss?.drawdown ?? {}) as JsonObject;
  const realizedSupported = accountProfitLoss?.realized_pnl_supported === true;
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">ACCOUNT PERFORMANCE</p><h2>성과 추이</h2></div><span className="denominator">Toss 스냅샷 기반</span></div><div className="performance-view-selector" role="tablist" aria-label="성과 보기"><span className="performance-view-label">보기</span>{(Object.keys(viewDetails) as PerformanceView[]).map(key => <button key={key} type="button" role="tab" aria-selected={view === key} className={`performance-view-button${view === key ? " active" : ""}`} onClick={() => setView(key)}>{viewDetails[key].label}</button>)}</div><div className="chart-grid"><article className="chart-card"><div className="card-heading"><span>투자자산 평가금</span><span>{String(summary?.history_days ?? 0)}일</span></div><TrendChart points={evaluablePoints} valueKey="invested_value_krw" label="투자자산 평가금" /><small>매입원가 기반 보유 손익과 분리해 투자자산 평가금 추이를 표시합니다.</small></article><article className="chart-card"><div className="card-heading"><span>{selectedView.label}</span><Status value={summary?.status} /></div><div className="chart-number">{selectedView.value === null ? "산출 전" : percent(selectedView.value)}</div><small>{selectedView.description}{view === "ytd" ? ` · 목표 ${percent(summary?.annual_target)}` : ""}</small></article></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">PERFORMANCE POINTS</p><h2>관측 포인트</h2></div><span className="denominator">최근 8개 · 매입원가·투자자산 평가금·TWR</span></div><TableToolbar query={query} onQueryChange={setQuery} onReset={() => { setQuery(""); setSort(null); }} visibleCount={visiblePoints.length} totalCount={baselinePoints.length} sortActive={sort !== null} /><div className="table-wrap"><table><thead><tr><SortableHeader label="시각" column="pointAt" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="투자자산 평가금" column="value" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="매입원가" column="principal" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="구간 TWR" column="intervalTwr" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="상태" column="state" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /></tr></thead><tbody>{visiblePoints.length === 0 ? <TableEmptyRow colSpan={5} /> : visiblePoints.map((point, index) => <tr key={`${String(point.snapshot_id)}-${index}`}><td>{String(point.point_at ?? "—")}</td><td>{money(point.invested_value_krw)}</td><td>{money(point.investment_principal_krw)}</td><td>{percent(point.interval_twr)}</td><td>{String(point.evaluation_state ?? "—")}</td></tr>)}</tbody></table></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">RISK EVIDENCE</p><h2>손익·drawdown 근거</h2></div><span className="denominator">손익만으로 상태를 승격하지 않음</span></div><div className="risk-fact-grid"><article className="fact-card"><span>계좌 현재 drawdown</span><strong>{evidenceValue(drawdown.current, drawdown.state, percent)}</strong><small>최대 {evidenceValue(drawdown.maximum, drawdown.state, percent)} · {String(drawdown.state ?? "자료 없음")}</small></article><article className="fact-card"><span>실현손익</span><strong>{realizedSupported ? money(accountProfitLoss?.actual_realized_pnl_krw) : "자료 없음"}</strong><small>{realizedSupported ? "Toss 체결 근거 지원" : "체결 근거 없음 · 0으로 간주하지 않음"}</small></article><article className="fact-card"><span>계좌 보유 평가손익</span><strong>{money(accountProfitLoss?.unrealized_pnl_krw)}</strong><small>원가 기준 수익률 {percent(accountProfitLoss?.unrealized_return)}</small></article></div></section>
  </>;
}

function InstrumentTable({ instruments, layerByIdentity }: { instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [layer, setLayer] = useState("");
  const [sort, setSort] = useState<SortState>(null);
  const getLayer = (row: JsonObject) => row.layer ?? layerByIdentity?.get(String(row.identity));
  const statusOptions = Object.keys(statusLabel);
  const layerOptions = useMemo(() => uniqueFilterValues(instruments, getLayer), [instruments, layerByIdentity]);
  const rows = useMemo(() => filterAndSortRows({
    rows: instruments,
    query,
    searchText: row => {
      const evidence = (row.evidence ?? {}) as JsonObject;
      const drawdown = (evidence.drawdown ?? {}) as JsonObject;
      return `${String(row.symbol ?? "")} ${String(row.identity ?? "")} ${String(row.market_country ?? "")} ${String(getLayer(row) ?? "")} ${String(row.current ?? "")} ${String(row.target ?? "")} ${String(row.gap ?? "")} ${String(evidence.unrealized_return ?? "")} ${String(drawdown.current ?? "")} ${String(row.status ?? "")} ${String(evidence.state ?? "")} ${String(drawdown.state ?? "")}`;
    },
    filters: [
      ...(status ? [{ value: status, getValue: (row: JsonObject) => row.status }] : []),
      ...(layer ? [{ value: layer, getValue: getLayer }] : []),
    ],
    sort,
    columns: {
      symbol: row => row.symbol ?? row.identity,
      layer: getLayer,
      current: row => row.current,
      target: row => row.target,
      gap: row => row.gap,
      unrealizedReturn: row => ((row.evidence ?? {}) as JsonObject).unrealized_return,
      drawdown: row => (((row.evidence ?? {}) as JsonObject).drawdown as JsonObject | undefined)?.current,
      status: row => row.status,
    },
  }), [instruments, query, status, layer, sort, layerByIdentity]);
  function reset() { setQuery(""); setStatus(""); setLayer(""); setSort(null); }
  return <><TableToolbar query={query} onQueryChange={setQuery} filters={[...(statusOptions.length ? [{ key: "status", label: "상태", value: status, options: statusOptions, onChange: setStatus }] : []), ...(layerOptions.length ? [{ key: "layer", label: "레이어", value: layer, options: layerOptions, onChange: setLayer }] : [])]} onReset={reset} visibleCount={rows.length} totalCount={instruments.length} sortActive={sort !== null} /><div className="table-wrap"><table><thead><tr><SortableHeader label="종목" column="symbol" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="레이어" column="layer" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="현재" column="current" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="목표" column="target" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="갭" column="gap" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="평가손익률" column="unrealizedReturn" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="현재 DD" column="drawdown" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /><SortableHeader label="상태" column="status" sort={sort} onSort={column => setSort(current => toggleSort(current, column))} /></tr></thead><tbody>{rows.length === 0 ? <TableEmptyRow colSpan={8} /> : rows.map(row => { const evidence = (row.evidence ?? {}) as JsonObject; const drawdown = (evidence.drawdown ?? {}) as JsonObject; return <tr key={String(row.identity)}><td><strong>{String(row.symbol ?? row.identity)}</strong><small>{String(row.market_country ?? "")}</small></td><td>{String(getLayer(row) ?? "미분류")}</td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{evidenceValue(evidence.unrealized_return, evidence.state, percent)}</td><td>{evidenceValue(drawdown.current, drawdown.state, percent)}</td><td><Status value={row.status} /></td></tr>; })}</tbody></table></div></>;
}

function AllocationPanel({ cash, layers, instruments, layerByIdentity }: { cash?: JsonObject | null; layers: JsonObject[]; instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">CASH BAND</p><h2>예수금 비중</h2></div><span className="denominator">총계좌 평가금 기준</span></div><GapCard title="현금 리저브" row={cash} /></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">LAYER GAPS</p><h2>레이어별 갭</h2></div><span className="denominator">투자금 평가금 기준</span></div><div className="cards-row">{layers.map(row => <GapCard key={String(row.identity)} title={String(row.identity)} row={row} />)}</div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">INSTRUMENT GAPS</p><h2>종목별 갭</h2></div><span className="denominator">백엔드 평가 순서 유지</span></div><InstrumentTable instruments={instruments} layerByIdentity={layerByIdentity} /></section>
  </>;
}

export default function App() {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [contractSupported, setContractSupported] = useState(false);
  const [performanceRun, setPerformanceRun] = useState<PerformanceRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(initialSidebarCollapsed);

  async function reload() {
    setLoading(true); setError(null); setWarnings([]);
    try {
      const inspectionData = await getJson<InspectionData>("/api/inspection");
      const currentEvaluation = inspectionData.evaluation;
      setEvaluation(currentEvaluation);
      setContractSupported(inspectionData.contract_supported === true);
      const warningMessages: string[] = [];
      const performancePath = currentEvaluation?.performance_run_id
        ? `/api/performance?run_id=${currentEvaluation.performance_run_id}`
        : "";
      if (!performancePath) {
        setPerformanceRun(null);
      } else {
        try {
          const performanceData = await getJson<{ run: PerformanceRun | null }>(performancePath);
          setPerformanceRun(performanceData.run);
        } catch {
          setPerformanceRun(null);
          warningMessages.push("성과 자료를 불러오지 못했습니다.");
        }
      }
      setWarnings(warningMessages);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "화면을 불러오지 못했습니다."); }
    finally { setLoading(false); }
  }
  useEffect(() => { void reload(); }, []);
  useEffect(() => {
    try { window.localStorage.setItem(sidebarStorageKey, String(sidebarCollapsed)); }
    catch { /* Browser storage can be unavailable without affecting the dashboard. */ }
  }, [sidebarCollapsed]);

  const result = evaluation?.result;
  const queue = result?.review_queue ?? [];
  const adjustmentSuggestions = result?.adjustment_suggestions ?? [];
  const layers = result?.layers ?? [];
  const instruments = result?.instruments ?? [];
  const performance = result?.performance as Record<string, unknown> | undefined;
  const account = result?.account as Record<string, unknown> | undefined;
  const source = result?.source as Record<string, unknown> | undefined;
  const sortedInstruments = useMemo(() => instruments, [instruments]);
  const layerByIdentity = useMemo(() => {
    const entries: Array<[string, string]> = instruments.flatMap(instrument => {
      const marketCountry = String(instrument.market_country ?? "").toUpperCase();
      const symbol = String(instrument.symbol ?? "").toUpperCase();
      const layer = instrument.layer;
      return marketCountry && symbol && typeof layer === "string" ? [[`${marketCountry}/${symbol}`, layer]] : [];
    });
    return new Map(entries);
  }, [instruments]);
  const investmentPrincipal = finiteNumber(account?.investment_principal_krw);
  const accountValue = finiteNumber(account?.invested_value_krw);
  const accountProfit = finiteNumber(account?.account_profit_krw);
  const accountReturn = finiteNumber(account?.account_return);
  const sourceState = typeof source?.state === "string" ? source.state : "unknown";
  const activePanel = panelCopy[activeTab];

  return <div className={`shell${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
    <aside className="sidebar">
      <div className="sidebar-header"><div className="brand"><div className="brand-mark">IP</div><div className="brand-copy"><strong>IPS Pilot</strong><small>Toss operating view</small></div></div><button type="button" className="sidebar-toggle" aria-label={sidebarCollapsed ? "사이드바 펼치기" : "사이드바 접기"} title={sidebarCollapsed ? "사이드바 펼치기" : "사이드바 접기"} aria-pressed={sidebarCollapsed} onClick={() => setSidebarCollapsed(value => !value)}>{sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}</button></div>
      <nav aria-label="대시보드 메뉴">{navItems.map(([Icon, label, key]) => <button type="button" className={`nav-item${activeTab === key ? " active" : ""}`} aria-label={label} title={sidebarCollapsed ? label : undefined} aria-current={activeTab === key ? "page" : undefined} key={key} onClick={() => setActiveTab(key)}><Icon size={16} /><span className="nav-label">{label}</span></button>)}</nav>
      <div className="sidebar-note"><CircleCheck size={15} />읽기 전용 검사 화면<br /><span>브로커 사실은 Toss 스냅샷만 사용합니다.</span></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><p className="eyebrow">{activePanel.eyebrow}</p><h1>{activePanel.title}</h1><p className="subhead">{activePanel.description}</p></div><button className="refresh" onClick={() => void reload()} disabled={loading}><RefreshCw size={16} />새로고침</button></header>
      {activeTab === "overview" && <div className="read-only-notice"><ShieldCheck size={17} /><div><strong>읽기 전용 IPS 점검</strong><span>검사 신호만 표시하며 상태와 우선순위는 저장된 백엔드 평가를 그대로 사용합니다.</span></div></div>}
      {error && <div className="banner error"><CircleAlert size={17} />{error}</div>}
      {warnings.map(warning => <div className="banner warning" key={warning}><CircleAlert size={17} />{warning}</div>)}
      {result && sourceState !== "complete" && <div className="banner warning"><CircleAlert size={17} />원천 상태가 {sourceState}입니다. 원천 동기화 근거를 확인하세요.</div>}
      {loading && <div className="empty">평가 결과를 불러오는 중입니다.</div>}
      {!loading && !evaluation && <div className="empty"><CircleAlert size={18} />아직 저장된 Toss 평가가 없습니다. CLI에서 inspection run을 먼저 실행하세요.</div>}
      {!loading && result && <>
        {activeTab === "overview" ? <>
          <section className="facts-grid overview-context">
            <article className="fact-card"><span>투자 원금(매입원가)</span><strong>{money(investmentPrincipal)}</strong><small>현재 보유 종목의 Toss 매입원가</small></article>
            <article className="fact-card"><span>현재 투자자산 평가금</span><strong>{money(accountValue)}</strong><small>{accountProfit === null ? "매입원가 대비 손익 자료 없음" : `매입원가 대비 ${signedMoney(accountProfit)}`}</small></article>
            <article className="fact-card"><span>매입원가 대비 보유 수익률</span><strong>{formatAccountReturn(accountReturn)}</strong><small>투자자산 평가금 - 매입원가 기준</small></article>
            <article className="fact-card"><span>YTD 계좌 수익률</span><strong>{percent(performance?.ytd_twr ?? (String(performance?.measurement ?? "") === "ytd_twr" ? performance?.annual_twr : null))}</strong><small>연간 목표 {percent(performance?.annual_target)} · 전략 평가용</small></article>
          </section>
          <AdjustmentSuggestions items={adjustmentSuggestions} contractSupported={contractSupported} allocationState={result.allocation_state} allocationReason={result.allocation_reason} />
          <AllocationOverview cash={result.cash} layers={layers} instruments={sortedInstruments} layerByIdentity={layerByIdentity} />
        </> : activeTab === "performance" ? <PerformancePanel summary={performance} run={performanceRun} accountProfitLoss={result.account_profit_loss} /> : activeTab === "allocation" ? <AllocationPanel cash={result.cash} layers={layers} instruments={sortedInstruments} layerByIdentity={layerByIdentity} /> : <ReviewQueue items={queue} />}
      </>}
    </main>
  </div>;
}
