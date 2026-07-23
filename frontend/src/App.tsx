import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CircleAlert, CircleCheck, Layers3, PanelLeftClose, PanelLeftOpen, RefreshCw, ShieldCheck, WalletCards, type LucideIcon } from "lucide-react";
import { Evaluation, getJson, type InspectionData, type InspectionItem, type JsonObject, type PerformanceRun } from "./lib/api";
import { evidenceValue, finiteNumber, formatAccountReturn, formatAllocationReason, formatKrw as money, formatPercent as percent, formatSignedKrw as signedMoney, supportedRate } from "./lib/presentation";

const statusLabel: Record<string, string> = { OK: "OK", Watch: "Watch", Review: "Review", Action: "Action" };
const sidebarStorageKey = "ips-pilot.sidebar-collapsed";
type TabKey = "overview" | "performance" | "allocation" | "review" | "profiles" | "source";
type PerformanceView = "ytd" | "trailing_12m" | "cumulative" | "holding";
type NavItem = [LucideIcon, string, TabKey];

const navItems: NavItem[] = [[Activity, "Overview", "overview"], [BarChart3, "Performance", "performance"], [Layers3, "Allocation", "allocation"], [CircleAlert, "Review Queue", "review"], [ShieldCheck, "Profiles & policy", "profiles"], [WalletCards, "Source health", "source"]];
const panelCopy: Record<TabKey, { eyebrow: string; title: string; description: string; source: string }> = {
  overview: { eyebrow: "MONTHLY IPS INSPECTION", title: "이번 달 계좌 운용 점검", description: "현재 상황, 목표 비중, 수익률과 다음 확인 항목을 한 화면에서 확인합니다.", source: "Toss 평가 실행" },
  performance: { eyebrow: "PERFORMANCE HISTORY", title: "성과 이력", description: "YTD·최근 1년·누적 TWR과 원금·평가금 추이를 확인하는 화면입니다.", source: "계좌 성과 이력" },
  allocation: { eyebrow: "ALLOCATION REVIEW", title: "비중 점검", description: "현금과 core·satellite·experiment 레이어의 목표 갭을 확인하는 화면입니다.", source: "레이어·종목 평가" },
  review: { eyebrow: "REVIEW QUEUE", title: "확인할 항목", description: "백엔드 평가 순서를 유지한 전체 읽기 전용 점검 목록입니다.", source: "Toss 평가 Review Queue" },
  profiles: { eyebrow: "IPS PROFILES & POLICY", title: "프로필과 정책", description: "Toss 종목 프로필과 현재 적용된 IPS 정책 버전을 확인하는 화면입니다.", source: "Toss 키 종목 프로필·정책" },
  source: { eyebrow: "TOSS SOURCE HEALTH", title: "원천 상태", description: "계좌 스냅샷의 동기화 상태와 평가 가능 여부를 확인하는 화면입니다.", source: "정규화된 Toss 스냅샷" },
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

function qualityLabel(value: unknown) {
  if (!value || typeof value !== "object") return "—";
  const quality = value as JsonObject;
  const issues = quality.issues;
  if (Array.isArray(issues)) return issues.length ? "검토 필요" : "정상";
  return quality.error ? "검토 필요" : Object.keys(quality).length ? "기록됨" : "—";
}

function Status({ value }: { value?: unknown }) {
  const status = typeof value === "string" && statusLabel[value] ? value : null;
  return status
    ? <span className={`status status-${status.toLowerCase()}`}>{statusLabel[status]}</span>
    : <span className="status status-missing">—</span>;
}

function AdjustmentSuggestions({ items, contractSupported, allocationState, allocationReason }: { items: InspectionItem[]; contractSupported: boolean; allocationState?: unknown; allocationReason?: unknown }) {
  return <section className="adjustment-suggestions" aria-labelledby="adjustment-suggestions-title">
    <div className="section-title"><div><p className="eyebrow">NEXT ALLOCATION REVIEW</p><h2 id="adjustment-suggestions-title">다음 조정 검토</h2></div><span className="denominator">백엔드 우선순위 순</span></div>
    {!contractSupported && <div className="contract-notice"><CircleAlert size={18} /><div><strong>새 조정 계약을 아직 적용하지 않았습니다.</strong><span>현재 저장된 평가는 이전 엔진 결과이므로 우선순위와 조정 제안을 추정하지 않습니다. 새 Toss 평가를 저장하면 이 영역이 활성화됩니다.</span></div></div>}
    {contractSupported && allocationState === "not_evaluable" && <div className="allocation-blocked"><CircleAlert size={18} /><div><strong>비중 조정 판단 보류</strong><span>{formatAllocationReason(allocationReason)} Source health에서 원천·정책 근거를 확인하세요.</span></div></div>}
    {contractSupported && allocationState !== "not_evaluable" && items.length === 0 && <div className="suggestion-empty"><CircleCheck size={18} /><span>현재 우선 검토할 비중 조정 항목이 없습니다. 허용 범위 안에서 관찰을 유지합니다.</span></div>}
    {contractSupported && allocationState !== "not_evaluable" && items.length > 0 && <div className="suggestion-list">{items.slice(0, 3).map((item, index) => {
      const suggestion = item.suggestion;
      return <article className="suggestion-card" key={`${String(item.kind)}-${String(item.identity)}-${index}`}>
        <div className="suggestion-meta"><span className="priority-chip">{String(item.priority ?? "—")} · {String(item.priority_label ?? "검토 시점 미상")}</span><Status value={item.status} /></div>
        <div className="suggestion-title-row"><strong>{String(item.identity ?? "—")}</strong><small>{String(item.kind ?? "allocation")}</small></div>
        <div className="suggestion-band"><span>현재 {percent(item.current)}</span><span>허용 {percent(item.minimum)}–{percent(item.maximum)}</span><span>목표 {percent(item.target)}</span></div>
        <p className="suggestion-label">{String(suggestion?.label ?? "조정 메커니즘 확인 필요")}</p>
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

function AllocationOverview({ cash, layers }: { cash?: InspectionItem | null; layers: InspectionItem[] }) {
  const layerByIdentity = new Map(layers.map(row => [String(row.identity ?? "").toLowerCase(), row]));
  const rows = [
    { label: "현금", row: cash },
    { label: "core", row: layerByIdentity.get("core") },
    { label: "satellite", row: layerByIdentity.get("satellite") },
    { label: "experiment", row: layerByIdentity.get("experiment") },
  ];
  return <section className="allocation-overview">
    <div className="section-title"><div><p className="eyebrow">ALLOCATION BANDS</p><h2>허용 범위와 현재 비중</h2></div><span className="denominator">현금은 총계좌 · 레이어는 투자금 평가금 기준</span></div>
    <div className="allocation-overview-list">{rows.map(({ label, row }) => {
      const target = finiteNumber(row?.target);
      return <article className="allocation-overview-row" key={label}>
        <div className="allocation-row-heading"><div><strong>{label}</strong><small>{denominatorLabel(row?.denominator)}</small></div><Status value={row?.status} /></div>
        <div className="allocation-values"><strong>{percent(row?.current)}</strong><span>허용 {percent(row?.minimum)}–{percent(row?.maximum)} · 목표 {percent(row?.target)}</span></div>
        <div className="allocation-track">
          <span className="allocation-fill" style={{ width: `${visualPercent(row?.current)}%` }} />
          {target !== null && <span className="allocation-target" style={{ left: `${visualPercent(target)}%` }} />}
        </div>
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

function layerUnrealizedPnl(layer: unknown, instruments: JsonObject[], layerByIdentity?: Map<string, string>) {
  let total = 0;
  let supported = false;
  for (const instrument of instruments) {
    const value = finiteNumber(instrument.unrealized_pnl_krw);
    const instrumentLayer = instrument.layer ?? layerByIdentity?.get(String(instrument.identity));
    if (instrumentLayer === layer && value !== null) {
      total += value;
      supported = true;
    }
  }
  return supported ? total : null;
}

function LayerReviewTable({ layers, instruments, layerByIdentity }: { layers: JsonObject[]; instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  return <section className="layer-review-section">
    <div className="section-title"><div><p className="eyebrow">LAYER REVIEW</p><h2>레이어 점검</h2></div><span className="denominator">투자금 평가금 기준</span></div>
    <div className="table-wrap"><table><thead><tr><th>레이어</th><th>현재</th><th>목표</th><th>갭</th><th>지원 손익 합계</th><th>상태</th></tr></thead><tbody>{layers.map(row => <tr key={String(row.identity)}><td><strong>{String(row.identity ?? "—")}</strong><small>{denominatorLabel(row.denominator)}</small></td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{money(layerUnrealizedPnl(row.identity, instruments, layerByIdentity))}</td><td><Status value={row.status} /></td></tr>)}</tbody></table></div>
  </section>;
}

const queueKindLabel: Record<string, string> = { source: "원천", cash: "현금", layer: "레이어", instrument: "종목", performance: "성과", account_risk: "계좌 리스크" };

function ReviewQueue({ items, compact = false }: { items: InspectionItem[]; compact?: boolean }) {
  return <section className={`review-queue-section${compact ? " compact" : " full"}`}>
    <div className="queue-heading"><div><p className="eyebrow">REVIEW QUEUE</p><h2>{compact ? "우선 확인 항목" : "전체 확인 항목"}</h2></div><span className="queue-count">{items.length}</span></div>
    {compact && <p className="review-queue-order">백엔드 순서 · 최대 3개</p>}
    {items.length === 0 && <div className="queue-empty"><CircleCheck size={18} />현재 확인 항목 없음</div>}
    <div className="review-queue-list">{items.map((item, index) => <article className="queue-item" key={`${String(item.kind)}-${String(item.identity)}-${index}`}><div className="queue-top"><div className="queue-axis"><Status value={item.status} />{item.priority && <span className="priority-label">{String(item.priority)} · {String(item.priority_label ?? "검토 시점 미상")}</span>}</div><small>{queueKindLabel[String(item.kind)] ?? String(item.kind ?? "근거")}</small></div><strong>{String(item.identity ?? "—")}</strong>{item.suggestion?.label && <p className="queue-suggestion">{String(item.suggestion.label)}</p>}<p>{String(item.meaning ?? "점검 의미 근거가 필요합니다.")}</p><span>{String(item.verification_task ?? "확인 과제 근거가 필요합니다.")}</span>{item.evidence_refs !== undefined && <details className="evidence-detail"><summary>근거 연결</summary><pre>{String(JSON.stringify(item.evidence_refs as JsonObject, null, 2) ?? "")}</pre></details>}</article>)}</div>
    <div className="queue-footer">이 목록은 읽기 전용 검사 신호이며, 상태와 순서는 백엔드 평가를 그대로 따릅니다.</div>
  </section>;
}

function StagePanel({ tab }: { tab: Exclude<TabKey, "overview"> }) {
  const copy = panelCopy[tab];
  return <section className="stage-panel">
    <p className="eyebrow">{copy.eyebrow}</p>
    <h2>{copy.title}</h2>
    <p>{copy.description}</p>
    <div className="stage-source"><span>연결 예정 데이터</span><strong>{copy.source}</strong></div>
    <small>이 단계에서는 읽기 전용 화면의 구조만 준비되어 있습니다. 현재 확인 가능한 값은 Overview에서 확인하세요.</small>
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
  const points = Array.isArray(run?.points) ? run.points : [];
  const evaluablePoints = points.filter(point => point.evaluation_state === "evaluable");
  const visiblePoints = points.slice(-8).reverse();
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
    <section className="section"><div className="section-title"><div><p className="eyebrow">ACCOUNT PERFORMANCE</p><h2>성과 추이</h2></div><span className="denominator">Toss 스냅샷 기반</span></div><div className="performance-view-selector" role="tablist" aria-label="성과 보기"><span className="performance-view-label">보기</span>{(Object.keys(viewDetails) as PerformanceView[]).map(key => <button key={key} type="button" role="tab" aria-selected={view === key} className={`performance-view-button${view === key ? " active" : ""}`} onClick={() => setView(key)}>{viewDetails[key].label}</button>)}</div><div className="chart-grid"><article className="chart-card"><div className="card-heading"><span>계좌 평가금</span><span>{String(summary?.history_days ?? 0)}일</span></div><TrendChart points={evaluablePoints} valueKey="total_value_krw" label="평가금" /><small>평가 가능 포인트만 연결하며, 비평가 구간은 표에서 상태를 확인합니다.</small></article><article className="chart-card"><div className="card-heading"><span>{selectedView.label}</span><Status value={summary?.status} /></div><div className="chart-number">{selectedView.value === null ? "산출 전" : percent(selectedView.value)}</div><small>{selectedView.description}{view === "ytd" ? ` · 목표 ${percent(summary?.annual_target)}` : ""}</small></article></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">PERFORMANCE POINTS</p><h2>관측 포인트</h2></div><span className="denominator">최근 8개 · 원금·평가금·TWR</span></div><div className="table-wrap"><table><thead><tr><th>시각</th><th>평가금</th><th>투자 원금</th><th>구간 TWR</th><th>상태</th></tr></thead><tbody>{visiblePoints.map((point, index) => <tr key={`${String(point.snapshot_id)}-${index}`}><td>{String(point.point_at ?? "—")}</td><td>{money(point.total_value_krw)}</td><td>{money(point.investment_principal_krw)}</td><td>{percent(point.interval_twr)}</td><td>{String(point.evaluation_state ?? "—")}</td></tr>)}</tbody></table></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">RISK EVIDENCE</p><h2>손익·drawdown 근거</h2></div><span className="denominator">손익만으로 상태를 승격하지 않음</span></div><div className="risk-fact-grid"><article className="fact-card"><span>계좌 현재 drawdown</span><strong>{evidenceValue(drawdown.current, drawdown.state, percent)}</strong><small>최대 {evidenceValue(drawdown.maximum, drawdown.state, percent)} · {String(drawdown.state ?? "자료 없음")}</small></article><article className="fact-card"><span>실현손익</span><strong>{realizedSupported ? money(accountProfitLoss?.actual_realized_pnl_krw) : "자료 없음"}</strong><small>{realizedSupported ? "Toss 체결 근거 지원" : "체결 근거 없음 · 0으로 간주하지 않음"}</small></article><article className="fact-card"><span>계좌 보유 평가손익</span><strong>{money(accountProfitLoss?.unrealized_pnl_krw)}</strong><small>원가 기준 수익률 {percent(accountProfitLoss?.unrealized_return)}</small></article></div></section>
  </>;
}

function InstrumentTable({ instruments, layerByIdentity }: { instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  return <div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>현재</th><th>목표</th><th>갭</th><th>평가손익률</th><th>현재 DD</th><th>factor</th><th>상태</th></tr></thead><tbody>{instruments.map(row => { const evidence = (row.evidence ?? {}) as JsonObject; const drawdown = (evidence.drawdown ?? {}) as JsonObject; const factors = [row.overlap_status, row.management_burden_status, row.holdability_status, row.etf_substitution_status].filter(value => typeof value === "string" && value !== "unknown").join(" · "); return <tr key={String(row.identity)}><td><strong>{String(row.symbol ?? row.identity)}</strong><small>{String(row.market_country ?? "")}</small></td><td>{String(row.layer ?? layerByIdentity?.get(String(row.identity)) ?? "미분류")}</td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{evidenceValue(evidence.unrealized_return, evidence.state, percent)}</td><td>{evidenceValue(drawdown.current, drawdown.state, percent)}</td><td><small className="factor-summary">{factors || "미검토"}</small></td><td><Status value={row.status} /></td></tr>; })}</tbody></table></div>;
}

function AllocationPanel({ cash, layers, instruments, layerByIdentity }: { cash?: JsonObject | null; layers: JsonObject[]; instruments: JsonObject[]; layerByIdentity?: Map<string, string> }) {
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">CASH BAND</p><h2>예수금 비중</h2></div><span className="denominator">총계좌 평가금 기준</span></div><GapCard title="현금 리저브" row={cash} /></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">LAYER GAPS</p><h2>레이어별 갭</h2></div><span className="denominator">투자금 평가금 기준</span></div><div className="cards-row">{layers.map(row => <GapCard key={String(row.identity)} title={String(row.identity)} row={row} />)}</div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">INSTRUMENT GAPS</p><h2>종목별 갭</h2></div><span className="denominator">백엔드 평가 순서 유지</span></div><InstrumentTable instruments={instruments} layerByIdentity={layerByIdentity} /></section>
  </>;
}

function ProfilesPanel({ profiles, policy }: { profiles: JsonObject[]; policy: JsonObject | null }) {
  const policyBody = (policy?.policy ?? {}) as JsonObject;
  const policyInstruments = Array.isArray(policyBody.instruments) ? policyBody.instruments as JsonObject[] : [];
  const targets = new Map(policyInstruments.map(item => [`${String(item.market_country ?? "").toUpperCase()}/${String(item.symbol ?? "").toUpperCase()}`, item]));
  return <section className="section"><div className="section-title"><div><p className="eyebrow">TOSS INSTRUMENT PROFILES</p><h2>프로필과 정책 범위</h2></div><span className="denominator">정책 버전 #{String(policy?.version ?? "—")} · 읽기 전용</span></div><div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>논지 상태</th><th>중복</th><th>부담</th><th>보유 가능성</th><th>ETF 대체</th><th>목표 범위</th><th>메모</th></tr></thead><tbody>{profiles.map(profile => { const identity = `${String(profile.market_country ?? "").toUpperCase()}/${String(profile.symbol ?? "").toUpperCase()}`; const target = targets.get(identity); return <tr key={identity}><td><strong>{String(profile.symbol ?? "—")}</strong><small>{String(profile.market_country ?? "")}</small></td><td>{String(profile.layer ?? "미분류")}</td><td>{String(profile.thesis_status ?? "unknown")}</td><td>{String(profile.overlap_status ?? "unknown")}</td><td>{String(profile.management_burden_status ?? "unknown")}</td><td>{String(profile.holdability_status ?? "unknown")}</td><td>{String(profile.etf_substitution_status ?? "unknown")}</td><td>{percent(target?.minimum)}–{percent(target?.maximum)} · 목표 {percent(target?.target)}</td><td>{String(profile.review_factors_note || profile.thesis_note || "—")}</td></tr>; })}</tbody></table></div></section>;
}

function SourcePanel({ health, snapshots, marketContext, evaluation }: { health: JsonObject | null; snapshots: JsonObject[]; marketContext: JsonObject | null; evaluation: Evaluation | null }) {
  const latest = health?.latest_attempt as JsonObject | null | undefined;
  const verified = health?.last_verified_complete as JsonObject | null | undefined;
  const context = marketContext?.context as JsonObject | undefined;
  const source = evaluation?.result?.source;
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">SOURCE HEALTH</p><h2>원천 상태</h2></div><span className="denominator">Toss 스냅샷만 사용</span></div><div className="source-health-grid"><article className="fact-card"><span>최근 동기화 시도</span><strong>#{String(latest?.id ?? "—")}</strong><small>{String(latest?.state ?? "자료 없음")} · {String(latest?.synced_at ?? "—")}</small></article><article className="fact-card"><span>마지막 검증 완료</span><strong>#{String(verified?.id ?? "—")}</strong><small>{String(verified?.state ?? "자료 없음")} · 현재 평가 가능 {String(verified?.is_current_evaluable ?? false)}</small></article><article className="fact-card"><span>계좌 원천</span><strong>{String(health?.account_alias ?? "toss-brokerage")}</strong><small>시장 데이터와 계좌 사실을 다른 출처로 섞지 않습니다.</small></article></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">MARKET CONTEXT</p><h2>시장 맥락 후보</h2></div><span className="denominator">KR/KOSPI · 활성 정책 변경 없음</span></div><div className="market-context-card"><div><Status value={context?.status} /><strong>{String(context?.candidate_state ?? "observe")}</strong></div><p>{String(context?.verification_task ?? "Toss 일봉 시장 데이터가 아직 없습니다.")}</p><small>현재 목표 {percent(context?.current_target)} · 후보 목표 {percent(context?.proposed_target)} · 이력 {String(context?.history_points ?? 0)}개</small></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">SNAPSHOT HISTORY</p><h2>최근 스냅샷</h2></div><span className="denominator">최신 시도 순</span></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>상태</th><th>평가 가능</th><th>평가금</th><th>동기화</th><th>데이터 품질</th></tr></thead><tbody>{snapshots.map(snapshot => <tr key={String(snapshot.id)}><td>#{String(snapshot.id)}</td><td>{String(snapshot.state ?? "—")}</td><td>{String(snapshot.is_current_evaluable ?? false)}</td><td>{money(snapshot.total_value_krw)}</td><td>{String(snapshot.synced_at ?? "—")}</td><td>{qualityLabel(snapshot.data_quality)}</td></tr>)}</tbody></table></div></section>
    <section className="section"><details className="source-diagnostics"><summary>실행 추적 정보</summary><dl><div><dt>스냅샷</dt><dd>#{String(evaluation?.snapshot_id ?? "—")}</dd></div><div><dt>평가 실행</dt><dd>#{String(evaluation?.id ?? "—")}</dd></div><div><dt>성과 실행</dt><dd>#{String(evaluation?.performance_run_id ?? "—")}</dd></div><div><dt>정책 버전</dt><dd>#{String(evaluation?.policy_version_id ?? "—")}</dd></div><div><dt>동기화</dt><dd>{String(source?.synced_at ?? "—")}</dd></div></dl></details></section>
  </>;
}

export default function App() {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [contractSupported, setContractSupported] = useState(false);
  const [performanceRun, setPerformanceRun] = useState<PerformanceRun | null>(null);
  const [policy, setPolicy] = useState<JsonObject | null>(null);
  const [profiles, setProfiles] = useState<JsonObject[]>([]);
  const [health, setHealth] = useState<JsonObject | null>(null);
  const [snapshots, setSnapshots] = useState<JsonObject[]>([]);
  const [marketContext, setMarketContext] = useState<JsonObject | null>(null);
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
      async function optional<T>(path: string, label: string, fallback: T): Promise<T> {
        try { return await getJson<T>(path); }
        catch { warningMessages.push(`${label} 자료를 불러오지 못했습니다.`); return fallback; }
      }
      const performancePath = currentEvaluation?.performance_run_id
        ? `/api/performance?run_id=${currentEvaluation.performance_run_id}`
        : "";
      const policyPath = currentEvaluation?.policy_version_id
        ? `/api/policy?version_id=${currentEvaluation.policy_version_id}`
        : "";
      const [performanceData, policyData, healthData, snapshotsData, marketContextData] = await Promise.all([
        performancePath
          ? optional<{ run: PerformanceRun | null }>(performancePath, "성과", { run: null })
          : Promise.resolve({ run: null }),
        policyPath
          ? optional<{ policy: JsonObject | null }>(policyPath, "정책", { policy: null })
          : Promise.resolve({ policy: null }),
        optional<JsonObject>("/api/health", "원천 상태", {}),
        optional<{ snapshots: JsonObject[] }>("/api/snapshots?limit=10", "스냅샷 이력", { snapshots: [] }),
        optional<JsonObject>("/api/market-context", "시장 맥락", {}),
      ]);
      setPerformanceRun(performanceData.run);
      setPolicy(policyData.policy);
      setProfiles((currentEvaluation?.profile_snapshot ?? []) as JsonObject[]);
      setHealth(healthData);
      setSnapshots(snapshotsData.snapshots ?? []);
      setMarketContext(Object.keys(marketContextData).length ? marketContextData : null);
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
    const entries: Array<[string, string]> = (evaluation?.profile_snapshot ?? []).flatMap(profile => {
      const marketCountry = String(profile.market_country ?? "").toUpperCase();
      const symbol = String(profile.symbol ?? "").toUpperCase();
      const layer = profile.layer;
      return marketCountry && symbol && typeof layer === "string" ? [[`${marketCountry}/${symbol}`, layer]] : [];
    });
    return new Map(entries);
  }, [evaluation?.profile_snapshot]);
  const investmentPrincipal = finiteNumber(account?.investment_principal_krw);
  const accountValue = finiteNumber(account?.total_value_krw);
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
      {result && sourceState !== "complete" && <div className="banner warning"><CircleAlert size={17} />원천 상태가 {sourceState}입니다. Source health에서 동기화 근거를 확인하세요.</div>}
      {loading && <div className="empty">평가 결과를 불러오는 중입니다.</div>}
      {!loading && !evaluation && <div className="empty"><CircleAlert size={18} />아직 저장된 Toss 평가가 없습니다. CLI에서 inspection run을 먼저 실행하세요.</div>}
      {!loading && result && <>
        {activeTab === "overview" ? <>
          <AdjustmentSuggestions items={adjustmentSuggestions} contractSupported={contractSupported} allocationState={result.allocation_state} allocationReason={result.allocation_reason} />
          <AllocationOverview cash={result.cash} layers={layers} />
          <section className="facts-grid overview-context">
            <article className="fact-card"><span>투자 원금</span><strong>{money(investmentPrincipal)}</strong><small>외부 순입출금을 반영한 계좌 원금</small></article>
            <article className="fact-card"><span>현재 계좌 평가금</span><strong>{money(accountValue)}</strong><small>{accountProfit === null ? "원금 대비 손익 자료 없음" : `원금 대비 ${signedMoney(accountProfit)}`}</small></article>
            <article className="fact-card"><span>원금 대비 계좌 수익률</span><strong>{formatAccountReturn(accountReturn)}</strong><small>평가금 - 투자 원금 기준</small></article>
            <article className="fact-card"><span>YTD 계좌 수익률</span><strong>{percent(performance?.ytd_twr ?? (String(performance?.measurement ?? "") === "ytd_twr" ? performance?.annual_twr : null))}</strong><small>연간 목표 {percent(performance?.annual_target)} · 전략 평가용</small></article>
          </section>
          <div className="overview-review-grid"><LayerReviewTable layers={layers} instruments={sortedInstruments} layerByIdentity={layerByIdentity} /><ReviewQueue items={queue.slice(0, 3)} compact /></div>
        </> : activeTab === "performance" ? <PerformancePanel summary={performance} run={performanceRun} accountProfitLoss={result.account_profit_loss} /> : activeTab === "allocation" ? <AllocationPanel cash={result.cash} layers={layers} instruments={sortedInstruments} layerByIdentity={layerByIdentity} /> : activeTab === "review" ? <ReviewQueue items={queue} /> : activeTab === "profiles" ? <ProfilesPanel profiles={profiles} policy={policy} /> : <SourcePanel health={health} snapshots={snapshots} marketContext={marketContext} evaluation={evaluation} />}
      </>}
    </main>
  </div>;
}
