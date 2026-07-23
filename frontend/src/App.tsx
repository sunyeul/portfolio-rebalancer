import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CircleAlert, CircleCheck, Layers3, RefreshCw, ShieldCheck, WalletCards, type LucideIcon } from "lucide-react";
import { Evaluation, getJson } from "./lib/api";

const statusLabel: Record<string, string> = { OK: "OK", Watch: "Watch", Review: "Review", Action: "Action" };
const navItems: Array<[LucideIcon, string]> = [[Activity, "Overview"], [BarChart3, "Performance"], [Layers3, "Allocation"], [ShieldCheck, "Profiles & policy"], [WalletCards, "Source health"]];

function percent(value: unknown) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function money(value: unknown) {
  return typeof value === "number" ? `${value.toLocaleString()} KRW` : "—";
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

export default function App() {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function reload() {
    setLoading(true); setError(null);
    try {
      const data = await getJson<{ evaluation: Evaluation | null }>("/api/inspection");
      setEvaluation(data.evaluation);
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

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">IP</div><div><strong>IPS Pilot</strong><small>Toss operating view</small></div></div>
      <nav>{navItems.map(([Icon, label]) => <div className="nav-item" key={label}><Icon size={16} />{label}</div>)}</nav>
      <div className="sidebar-note"><CircleCheck size={15} />읽기 전용 검사 화면<br /><span>브로커 사실은 Toss 스냅샷만 사용합니다.</span></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><p className="eyebrow">MONTHLY IPS INSPECTION</p><h1>계좌 운용 루프</h1><p className="subhead">현재 상황, 목표 비중, 수익률과 다음 확인 항목을 한 화면에서 확인합니다.</p></div><button className="refresh" onClick={() => void reload()} disabled={loading}><RefreshCw size={16} />새로고침</button></header>
      {error && <div className="banner error"><CircleAlert size={17} />{error}</div>}
      {loading && <div className="empty">평가 결과를 불러오는 중입니다.</div>}
      {!loading && !evaluation && <div className="empty"><CircleAlert size={18} />아직 저장된 Toss 평가가 없습니다. CLI에서 inspection run을 먼저 실행하세요.</div>}
      {!loading && result && <>
        <section className="source-strip"><div><span>원천 상태</span><strong>{String(source?.state ?? "unknown")}</strong></div><div><span>스냅샷</span><strong>#{String(evaluation?.snapshot_id ?? "—")}</strong></div><div><span>평가 실행</span><strong>#{String(evaluation?.id ?? "—")}</strong></div><div><span>성과 실행</span><strong>#{String(evaluation?.performance_run_id ?? "—")}</strong></div><div><span>정책 버전</span><strong>#{String(evaluation?.policy_version_id ?? "—")}</strong></div><div><span>동기화</span><strong>{String(source?.synced_at ?? "—")}</strong></div><div><span>연간 목표</span><strong>{percent(performance?.annual_target)} TWR</strong></div></section>
        <section className="facts-grid"><article className="fact-card"><span>추적 원금</span><strong>{money(account?.tracking_principal_krw)}</strong><small>성과 추적 기준선</small></article><article className="fact-card"><span>계좌 평가금</span><strong>{money(account?.total_value_krw)}</strong><small>투자금 {money(account?.invested_value_krw)}</small></article><article className="fact-card"><span>누적 TWR</span><strong>{percent(performance?.cumulative_twr)}</strong><small>추적 기준 이후 계좌 시간가중수익률</small></article><article className="fact-card"><span>연간 TWR</span><strong>{percent(performance?.annual_twr)}</strong><small>{String(performance?.history_days ?? 0)}일 지원 · 목표 비교는 365일 이후</small></article><article className="fact-card"><span>성과 상태</span><div><Status value={performance?.status} /></div><small>{String(performance?.meaning ?? "")}</small></article></section>
        <section className="section"><div className="section-title"><div><p className="eyebrow">CASH BAND</p><h2>예수금 비중</h2></div><span className="denominator">총계좌 평가금 기준</span></div><GapCard title="현금 리저브" row={result.cash} /></section>
        <section className="section"><div className="section-title"><div><p className="eyebrow">ALLOCATION GAPS</p><h2>레이어 목표 갭</h2></div><span className="denominator">투자금 평가금 기준</span></div><div className="cards-row">{layers.map(row => <GapCard key={String(row.identity)} title={String(row.identity)} row={row} />)}</div></section>
        <section className="section"><div className="section-title"><div><p className="eyebrow">INSTRUMENTS</p><h2>종목별 관찰</h2></div><span className="denominator">정렬하지 않으면 API 순서 유지</span></div><div className="table-wrap"><table><thead><tr><th>종목</th><th>레이어</th><th>현재</th><th>목표</th><th>갭</th><th>손익 근거</th><th>상태</th></tr></thead><tbody>{sortedInstruments.map(row => <tr key={String(row.identity)}><td><strong>{String(row.symbol ?? row.identity)}</strong><small>{String(row.market_country ?? "")}</small></td><td>{String(row.thesis_status ?? "미분류")}</td><td>{percent(row.current)}</td><td>{percent(row.target)}</td><td>{percent(row.gap)}</td><td>{typeof row.unrealized_pnl_krw === "number" ? `${row.unrealized_pnl_krw.toLocaleString()} KRW` : "—"}</td><td><Status value={row.status} /></td></tr>)}</tbody></table></div></section>
      </>}
    </main>
    <aside className="queue"><div className="queue-heading"><div><p className="eyebrow">REVIEW QUEUE</p><h2>확인할 항목</h2></div><span className="queue-count">{queue.length}</span></div>{queue.length === 0 && <div className="queue-empty"><CircleCheck size={18} />현재 확인 항목 없음</div>}{queue.map(item => <article className="queue-item" key={`${String(item.kind)}-${String(item.identity)}`}><div className="queue-top"><Status value={item.status} /><small>{String(item.kind)}</small></div><strong>{String(item.identity)}</strong><p>{String(item.meaning)}</p><span>{String(item.verification_task)}</span></article>)}<div className="queue-footer">손익·수익률 부족·현금 이탈만으로 거래 의미를 만들지 않습니다.</div></aside>
  </div>;
}
