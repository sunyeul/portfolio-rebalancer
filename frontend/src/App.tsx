import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CircleAlert, CircleCheck, Layers3, RefreshCw, ShieldCheck, WalletCards, type LucideIcon } from "lucide-react";
import { Evaluation, getJson, type JsonObject, type PerformanceRun } from "./lib/api";

const statusLabel: Record<string, string> = { OK: "OK", Watch: "Watch", Review: "Review", Action: "Action" };
type TabKey = "overview" | "performance" | "allocation" | "profiles" | "source";
type NavItem = [LucideIcon, string, TabKey];

const navItems: NavItem[] = [[Activity, "Overview", "overview"], [BarChart3, "Performance", "performance"], [Layers3, "Allocation", "allocation"], [ShieldCheck, "Profiles & policy", "profiles"], [WalletCards, "Source health", "source"]];
const panelCopy: Record<TabKey, { eyebrow: string; title: string; description: string; source: string }> = {
  overview: { eyebrow: "MONTHLY IPS INSPECTION", title: "계좌 운용 루프", description: "현재 상황, 목표 비중, 수익률과 다음 확인 항목을 한 화면에서 확인합니다.", source: "Toss 평가 실행" },
  performance: { eyebrow: "PERFORMANCE HISTORY", title: "성과 이력", description: "누적·연간 TWR과 원금·평가금 추이를 확인하는 화면입니다.", source: "계좌 성과 이력" },
  allocation: { eyebrow: "ALLOCATION REVIEW", title: "비중 점검", description: "현금과 core·satellite·experiment 레이어의 목표 갭을 확인하는 화면입니다.", source: "레이어·종목 평가" },
  profiles: { eyebrow: "IPS PROFILES & POLICY", title: "프로필과 정책", description: "Toss 종목 프로필과 현재 적용된 IPS 정책 버전을 확인하는 화면입니다.", source: "Toss 키 종목 프로필·정책" },
  source: { eyebrow: "TOSS SOURCE HEALTH", title: "원천 상태", description: "계좌 스냅샷의 동기화 상태와 평가 가능 여부를 확인하는 화면입니다.", source: "정규화된 Toss 스냅샷" },
};

function percent(value: unknown) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function money(value: unknown) {
  return typeof value === "number" ? `${value.toLocaleString()} KRW` : "—";
}

function qualityLabel(value: unknown) {
  if (!value || typeof value !== "object") return "—";
  const quality = value as JsonObject;
  const issues = quality.issues;
  if (Array.isArray(issues)) return issues.length ? "검토 필요" : "정상";
  return quality.error ? "검토 필요" : Object.keys(quality).length ? "기록됨" : "—";
}

function Status({ value }: { value?: unknown }) {
  const status = typeof value === "string" ? value : "Review";
  return <span className={`status status-${status.toLowerCase()}`}>{statusLabel[status] ?? status}</span>;
}

function GapCard({ title, row }: { title: string; row?: Record<string, unknown> | null }) {
  return <article className="metric-card">
    <div className="card-heading"><span>{title}</span><Status value={row?.status} /></div>
    <div className="metric-main">{percent(row?.current)}</div>
    <div className="metric-caption">목표 {percent(row?.target)} · 범위 {percent(row?.minimum)}–{percent(row?.maximum)}</div>
    <div className="metric-gap">갭 {percent(row?.gap)} · {String(row?.denominator ?? "자료 확인 필요")}</div>
  </article>;
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

function PerformancePanel({ summary, run }: { summary?: JsonObject; run: PerformanceRun | null }) {
  const points = Array.isArray(run?.points) ? run.points : [];
  const evaluablePoints = points.filter(point => point.evaluation_state === "evaluable");
  const visiblePoints = points.slice(-8).reverse();
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">ACCOUNT PERFORMANCE</p><h2>성과 추이</h2></div><span className="denominator">Toss 스냅샷 기반 · 성과 실행 #{String(run?.id ?? "—")}</span></div><div className="chart-grid"><article className="chart-card"><div className="card-heading"><span>계좌 평가금</span><span>{String(summary?.history_days ?? 0)}일</span></div><TrendChart points={evaluablePoints} valueKey="total_value_krw" label="평가금" /><small>평가 가능 포인트만 연결하며, 비평가 구간은 표에서 상태를 확인합니다.</small></article><article className="chart-card"><div className="card-heading"><span>누적 TWR</span><Status value={summary?.status} /></div><div className="chart-number">{percent(summary?.cumulative_twr)}</div><small>연간 TWR {percent(summary?.annual_twr)} · 목표 {percent(summary?.annual_target)}</small></article></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">PERFORMANCE POINTS</p><h2>관측 포인트</h2></div><span className="denominator">최근 8개 · 원금·평가금·TWR</span></div><div className="table-wrap"><table><thead><tr><th>시각</th><th>평가금</th><th>추적 원금</th><th>구간 TWR</th><th>상태</th></tr></thead><tbody>{visiblePoints.map((point, index) => <tr key={`${String(point.snapshot_id)}-${index}`}><td>{String(point.point_at ?? "—")}</td><td>{money(point.total_value_krw)}</td><td>{money(point.tracking_principal_krw)}</td><td>{percent(point.interval_twr)}</td><td>{String(point.evaluation_state ?? "—")}</td></tr>)}</tbody></table></div></section>
  </>;
}

function InstrumentTable({ instruments }: { instruments: JsonObject[] }) {
  return <div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>현재</th><th>목표</th><th>갭</th><th>손익 근거</th><th>상태</th></tr></thead><tbody>{instruments.map(row => <tr key={String(row.identity)}><td><strong>{String(row.symbol ?? row.identity)}</strong><small>{String(row.market_country ?? "")}</small></td><td>{String(row.layer ?? "미분류")}</td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{typeof row.unrealized_pnl_krw === "number" ? `${row.unrealized_pnl_krw.toLocaleString()} KRW` : "—"}</td><td><Status value={row.status} /></td></tr>)}</tbody></table></div>;
}

function AllocationPanel({ cash, layers, instruments }: { cash?: JsonObject | null; layers: JsonObject[]; instruments: JsonObject[] }) {
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">CASH BAND</p><h2>예수금 비중</h2></div><span className="denominator">총계좌 평가금 기준</span></div><GapCard title="현금 리저브" row={cash} /></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">LAYER GAPS</p><h2>레이어별 갭</h2></div><span className="denominator">투자금 평가금 기준</span></div><div className="cards-row">{layers.map(row => <GapCard key={String(row.identity)} title={String(row.identity)} row={row} />)}</div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">INSTRUMENT GAPS</p><h2>종목별 갭</h2></div><span className="denominator">백엔드 평가 순서 유지</span></div><InstrumentTable instruments={instruments} /></section>
  </>;
}

function ProfilesPanel({ profiles, policy }: { profiles: JsonObject[]; policy: JsonObject | null }) {
  const policyBody = (policy?.policy ?? {}) as JsonObject;
  const policyInstruments = Array.isArray(policyBody.instruments) ? policyBody.instruments as JsonObject[] : [];
  const targets = new Map(policyInstruments.map(item => [`${String(item.market_country ?? "").toUpperCase()}/${String(item.symbol ?? "").toUpperCase()}`, item]));
  return <section className="section"><div className="section-title"><div><p className="eyebrow">TOSS INSTRUMENT PROFILES</p><h2>프로필과 정책 범위</h2></div><span className="denominator">정책 버전 #{String(policy?.version ?? "—")} · 읽기 전용</span></div><div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>논지 상태</th><th>목표 범위</th><th>메모</th></tr></thead><tbody>{profiles.map(profile => { const identity = `${String(profile.market_country ?? "").toUpperCase()}/${String(profile.symbol ?? "").toUpperCase()}`; const target = targets.get(identity); return <tr key={identity}><td><strong>{String(profile.symbol ?? "—")}</strong><small>{String(profile.market_country ?? "")}</small></td><td>{String(profile.layer ?? "미분류")}</td><td>{String(profile.thesis_status ?? "unknown")}</td><td>{percent(target?.minimum)}–{percent(target?.maximum)} · 목표 {percent(target?.target)}</td><td>{String(profile.thesis_note ?? "—")}</td></tr>; })}</tbody></table></div></section>;
}

function SourcePanel({ health, snapshots, marketContext }: { health: JsonObject | null; snapshots: JsonObject[]; marketContext: JsonObject | null }) {
  const latest = health?.latest_attempt as JsonObject | null | undefined;
  const verified = health?.last_verified_complete as JsonObject | null | undefined;
  const context = marketContext?.context as JsonObject | undefined;
  return <>
    <section className="section"><div className="section-title"><div><p className="eyebrow">SOURCE HEALTH</p><h2>원천 상태</h2></div><span className="denominator">Toss 스냅샷만 사용</span></div><div className="source-health-grid"><article className="fact-card"><span>최근 동기화 시도</span><strong>#{String(latest?.id ?? "—")}</strong><small>{String(latest?.state ?? "자료 없음")} · {String(latest?.synced_at ?? "—")}</small></article><article className="fact-card"><span>마지막 검증 완료</span><strong>#{String(verified?.id ?? "—")}</strong><small>{String(verified?.state ?? "자료 없음")} · 현재 평가 가능 {String(verified?.is_current_evaluable ?? false)}</small></article><article className="fact-card"><span>계좌 원천</span><strong>{String(health?.account_alias ?? "toss-brokerage")}</strong><small>시장 데이터와 계좌 사실을 다른 출처로 섞지 않습니다.</small></article></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">MARKET CONTEXT</p><h2>시장 맥락 후보</h2></div><span className="denominator">KR/KOSPI · 활성 정책 변경 없음</span></div><div className="market-context-card"><div><Status value={context?.status} /><strong>{String(context?.candidate_state ?? "observe")}</strong></div><p>{String(context?.verification_task ?? "Toss 일봉 시장 데이터가 아직 없습니다.")}</p><small>현재 목표 {percent(context?.current_target)} · 후보 목표 {percent(context?.proposed_target)} · 이력 {String(context?.history_points ?? 0)}개</small></div></section>
    <section className="section"><div className="section-title"><div><p className="eyebrow">SNAPSHOT HISTORY</p><h2>최근 스냅샷</h2></div><span className="denominator">최신 시도 순</span></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>상태</th><th>평가 가능</th><th>평가금</th><th>동기화</th><th>데이터 품질</th></tr></thead><tbody>{snapshots.map(snapshot => <tr key={String(snapshot.id)}><td>#{String(snapshot.id)}</td><td>{String(snapshot.state ?? "—")}</td><td>{String(snapshot.is_current_evaluable ?? false)}</td><td>{money(snapshot.total_value_krw)}</td><td>{String(snapshot.synced_at ?? "—")}</td><td>{qualityLabel(snapshot.data_quality)}</td></tr>)}</tbody></table></div></section>
  </>;
}

export default function App() {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
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

  async function reload() {
    setLoading(true); setError(null); setWarnings([]);
    try {
      const inspectionData = await getJson<{ evaluation: Evaluation | null }>("/api/inspection");
      const currentEvaluation = inspectionData.evaluation;
      setEvaluation(currentEvaluation);
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

  const result = evaluation?.result;
  const queue = result?.review_queue ?? [];
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
  const activePanel = panelCopy[activeTab];

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">IP</div><div><strong>IPS Pilot</strong><small>Toss operating view</small></div></div>
      <nav aria-label="대시보드 메뉴">{navItems.map(([Icon, label, key]) => <button type="button" className={`nav-item${activeTab === key ? " active" : ""}`} aria-current={activeTab === key ? "page" : undefined} key={key} onClick={() => setActiveTab(key)}><Icon size={16} />{label}</button>)}</nav>
      <div className="sidebar-note"><CircleCheck size={15} />읽기 전용 검사 화면<br /><span>브로커 사실은 Toss 스냅샷만 사용합니다.</span></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><p className="eyebrow">{activePanel.eyebrow}</p><h1>{activePanel.title}</h1><p className="subhead">{activePanel.description}</p></div><button className="refresh" onClick={() => void reload()} disabled={loading}><RefreshCw size={16} />새로고침</button></header>
      {error && <div className="banner error"><CircleAlert size={17} />{error}</div>}
      {warnings.map(warning => <div className="banner warning" key={warning}><CircleAlert size={17} />{warning}</div>)}
      {loading && <div className="empty">평가 결과를 불러오는 중입니다.</div>}
      {!loading && !evaluation && <div className="empty"><CircleAlert size={18} />아직 저장된 Toss 평가가 없습니다. CLI에서 inspection run을 먼저 실행하세요.</div>}
      {!loading && result && <>
        <section className="source-strip"><div><span>원천 상태</span><strong>{String(source?.state ?? "unknown")}</strong></div><div><span>스냅샷</span><strong>#{String(evaluation?.snapshot_id ?? "—")}</strong></div><div><span>평가 실행</span><strong>#{String(evaluation?.id ?? "—")}</strong></div><div><span>성과 실행</span><strong>#{String(evaluation?.performance_run_id ?? "—")}</strong></div><div><span>정책 버전</span><strong>#{String(evaluation?.policy_version_id ?? "—")}</strong></div><div><span>동기화</span><strong>{String(source?.synced_at ?? "—")}</strong></div><div><span>연간 목표</span><strong>{percent(performance?.annual_target)} TWR</strong></div></section>
        <section className="facts-grid"><article className="fact-card"><span>추적 원금</span><strong>{money(account?.tracking_principal_krw)}</strong><small>성과 추적 기준선</small></article><article className="fact-card"><span>계좌 평가금</span><strong>{money(account?.total_value_krw)}</strong><small>투자금 {money(account?.invested_value_krw)}</small></article><article className="fact-card"><span>누적 TWR</span><strong>{percent(performance?.cumulative_twr)}</strong><small>추적 기준 이후 계좌 시간가중수익률</small></article><article className="fact-card"><span>연간 TWR</span><strong>{percent(performance?.annual_twr)}</strong><small>{String(performance?.history_days ?? 0)}일 지원 · 목표 비교는 365일 이후</small></article><article className="fact-card"><span>성과 상태</span><div><Status value={performance?.status} /></div><small>{String(performance?.meaning ?? "")}</small></article></section>
        {activeTab === "overview" ? <>
          <section className="section"><div className="section-title"><div><p className="eyebrow">CASH BAND</p><h2>예수금 비중</h2></div><span className="denominator">총계좌 평가금 기준</span></div><GapCard title="현금 리저브" row={result.cash} /></section>
          <section className="section"><div className="section-title"><div><p className="eyebrow">ALLOCATION GAPS</p><h2>레이어 목표 갭</h2></div><span className="denominator">투자금 평가금 기준</span></div><div className="cards-row">{layers.map(row => <GapCard key={String(row.identity)} title={String(row.identity)} row={row} />)}</div></section>
          <section className="section"><div className="section-title"><div><p className="eyebrow">INSTRUMENTS</p><h2>종목별 관찰</h2></div><span className="denominator">정렬하지 않으면 API 순서 유지</span></div><div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>현재</th><th>목표</th><th>갭</th><th>손익 근거</th><th>상태</th></tr></thead><tbody>{sortedInstruments.map(row => <tr key={String(row.identity)}><td><strong>{String(row.symbol ?? row.identity)}</strong><small>{String(row.market_country ?? "")}</small></td><td>{String(row.layer ?? layerByIdentity.get(String(row.identity)) ?? "미분류")}</td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{typeof row.unrealized_pnl_krw === "number" ? `${row.unrealized_pnl_krw.toLocaleString()} KRW` : "—"}</td><td><Status value={row.status} /></td></tr>)}</tbody></table></div></section>
        </> : activeTab === "performance" ? <PerformancePanel summary={performance} run={performanceRun} /> : activeTab === "allocation" ? <AllocationPanel cash={result.cash} layers={layers} instruments={sortedInstruments} /> : activeTab === "profiles" ? <ProfilesPanel profiles={profiles} policy={policy} /> : <SourcePanel health={health} snapshots={snapshots} marketContext={marketContext} />}
      </>}
    </main>
    <aside className="queue"><div className="queue-heading"><div><p className="eyebrow">REVIEW QUEUE</p><h2>확인할 항목</h2></div><span className="queue-count">{queue.length}</span></div>{queue.length === 0 && <div className="queue-empty"><CircleCheck size={18} />현재 확인 항목 없음</div>}{queue.map(item => <article className="queue-item" key={`${String(item.kind)}-${String(item.identity)}`}><div className="queue-top"><Status value={item.status} /><small>{String(item.kind)}</small></div><strong>{String(item.identity)}</strong><p>{String(item.meaning)}</p><span>{String(item.verification_task)}</span></article>)}<div className="queue-footer">손익·수익률 부족·현금 이탈만으로 거래 의미를 만들지 않습니다.</div></aside>
  </div>;
}
