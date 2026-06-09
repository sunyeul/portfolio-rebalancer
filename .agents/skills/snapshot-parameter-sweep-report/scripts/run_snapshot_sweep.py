from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root() -> Path:
    for parent in SCRIPT_PATH.parents:
        if (parent / "cli.py").exists() and (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repo root containing cli.py and pyproject.toml")


ROOT = find_repo_root()

DEFAULT_SCENARIOS = [
    {
        "id": "baseline_12m_regular",
        "label": "Baseline: 12M, regular review",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "regular_review"],
        "note": "Current default assumptions.",
    },
    {
        "id": "period_6m",
        "label": "Short lookback: 6M",
        "args": ["--period", "6", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "regular_review"],
        "note": "More sensitive to recent market leadership and drawdowns.",
    },
    {
        "id": "period_ytd",
        "label": "YTD lookback",
        "args": ["--period", "YTD", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "regular_review"],
        "note": "Calendar-year framing.",
    },
    {
        "id": "period_max",
        "label": "Max available lookback",
        "args": ["--period", "Max", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "regular_review"],
        "note": "Uses all available price history in the service result.",
    },
    {
        "id": "bench_spy",
        "label": "Benchmark: SPY",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY", "--decision-context", "regular_review"],
        "note": "Compares against a pure S&P 500 benchmark.",
    },
    {
        "id": "bench_qqq",
        "label": "Benchmark: QQQ",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "QQQ", "--decision-context", "regular_review"],
        "note": "Compares against a pure Nasdaq 100 benchmark.",
    },
    {
        "id": "rf_zero",
        "label": "Risk-free rate: 0%",
        "args": ["--period", "12", "--rf", "0", "--bench", "SPY:80,QQQ:20", "--decision-context", "regular_review"],
        "note": "Shows sensitivity of Sharpe and efficiency inputs to the risk-free-rate assumption.",
    },
    {
        "id": "market_correction",
        "label": "Decision context: market correction",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "market_correction"],
        "note": "IPS context for broader market weakness.",
    },
    {
        "id": "sharp_drop",
        "label": "Decision context: sharp drop review",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--decision-context", "sharp_drop_review"],
        "note": "IPS context for abrupt drawdown review.",
    },
    {
        "id": "strict_thresholds",
        "label": "Strict thresholds: rc 1.0, e 0.6",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--rc-threshold", "1.0", "--e-threshold", "0.6", "--decision-context", "regular_review"],
        "note": "Flags risk and weak efficiency more aggressively.",
    },
    {
        "id": "loose_thresholds",
        "label": "Loose thresholds: rc 2.5, e 0.3",
        "args": ["--period", "12", "--rf", "0.025", "--bench", "SPY:80,QQQ:20", "--rc-threshold", "2.5", "--e-threshold", "0.3", "--decision-context", "regular_review"],
        "note": "Requires larger risk excess and tolerates lower efficiency.",
    },
]

SCENARIO_SETS = {"default": DEFAULT_SCENARIOS}


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_text(value: Any, default: str = "-") -> str:
    if value in (None, ""):
        return default
    text = str(value).strip()
    return text if text else default


def listish_text(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    text = str(value).strip()
    if not text or text == "[]":
        return "-"
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed) if parsed else "-"
    return text


def md_cell(value: Any) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", " ")


def action_tickers(rows: list[dict[str, Any]], limit: int = 8) -> str:
    if not rows:
        return "-"
    parts = []
    for row in rows[:limit]:
        trade = row.get("suggested_trade_pct", 0)
        parts.append(f"{row.get('ticker')}({safe_float(trade):+.2f}%)")
    if len(rows) > limit:
        parts.append(f"+{len(rows) - limit} more")
    return ", ".join(parts)


def snapshot_metadata(snapshot_id: int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from storage.database import initialize_database
    from storage.portfolio_store import get_snapshot

    initialize_database()
    snapshot = get_snapshot(snapshot_id)
    if snapshot is None:
        raise RuntimeError(f"snapshot_id={snapshot_id} not found")
    summary = snapshot.get("summary", {})
    return {
        "snapshot_id": snapshot_id,
        "name": summary.get("name") or f"Snapshot {snapshot_id}",
        "portfolio_id": summary.get("portfolio_id"),
        "created_at": summary.get("created_at"),
        "position_count": summary.get("position_count"),
    }


def clean_output(output_dir: Path, snapshot_id: int) -> None:
    targets = [
        output_dir / "artifacts",
        output_dir / "summary.json",
        output_dir / f"snapshot{snapshot_id}_parameter_sweep_report.md",
        output_dir / f"snapshot{snapshot_id}_parameter_sweep_report_ko.md",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def summarize(payload: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    analysis = payload["analysis"]
    agent = payload["agent_summary"]
    recommended = agent["recommended_actions"]
    proposal = payload["evaluation"]["proposal"]
    low_quality = agent["data_quality_warnings"]["low_quality_rows"]
    top_risk = agent["top_risk_contributors"]
    reasons = Counter(row.get("action_reason", "") for row in proposal)
    rec_reasons = Counter(row.get("action_reason", "") for row in recommended)
    group_weights: dict[str, float] = defaultdict(float)
    for row in proposal:
        group_weights[row.get("group", "unknown")] += safe_float(row.get("current_weight_pct"))
    return {
        "id": scenario["id"],
        "label": scenario["label"],
        "note": scenario["note"],
        "ok": payload["ok"],
        "period": payload["input"]["period"],
        "rf": payload["input"]["rf"],
        "bench": payload["input"]["bench"],
        "decision_context": payload["input"]["decision_context"],
        "portfolio_cagr": analysis["portfolio_metrics"].get("cagr"),
        "portfolio_vol": analysis["portfolio_metrics"].get("volatility"),
        "portfolio_sharpe": analysis["portfolio_metrics"].get("sharpe"),
        "portfolio_mdd": analysis["portfolio_metrics"].get("max_drawdown"),
        "benchmark_cagr": (analysis.get("benchmark_metrics") or {}).get("cagr"),
        "benchmark_sharpe": (analysis.get("benchmark_metrics") or {}).get("sharpe"),
        "recommended_count": len(recommended),
        "recommended_tickers": action_tickers(recommended),
        "recommended_names": [row["ticker"] for row in recommended],
        "hold_count": len(agent["hold_actions"]),
        "low_quality": [row["ticker"] for row in low_quality],
        "top_risk": [row["ticker"] for row in top_risk[:5]],
        "top_risk_detail": [
            {
                "ticker": row["ticker"],
                "weight": row.get("weight"),
                "risk_contribution": row.get("risk_contribution"),
                "return_contribution": row.get("return_contribution"),
                "group": row.get("group"),
            }
            for row in top_risk[:5]
        ],
        "reason_counts": dict(reasons),
        "recommended_reason_counts": dict(rec_reasons),
        "group_weights": dict(group_weights),
    }


def run_scenario(snapshot_id: int, output_dir: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = output_dir / "artifacts" / scenario["id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "cli.py"),
        "evaluate",
        "--snapshot-id",
        str(snapshot_id),
        "--output-dir",
        str(artifact_dir),
        *scenario["args"],
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (artifact_dir / "stdout.json").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (artifact_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{scenario['id']} failed with exit {result.returncode}: {result.stdout or result.stderr}")
    payload = json.loads(result.stdout)
    return summarize(payload, scenario)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ko_scenario_name(label: str) -> str:
    names = {
        "Baseline: 12M, regular review": "기본값: 12개월, 정기 점검",
        "Short lookback: 6M": "짧은 관찰기간: 6개월",
        "YTD lookback": "연초 이후(YTD)",
        "Max available lookback": "최대 가용 기간",
        "Benchmark: SPY": "벤치마크: SPY",
        "Benchmark: QQQ": "벤치마크: QQQ",
        "Risk-free rate: 0%": "무위험수익률: 0%",
        "Decision context: market correction": "판단 맥락: 시장 조정",
        "Decision context: sharp drop review": "판단 맥락: 급락 점검",
        "Strict thresholds: rc 1.0, e 0.6": "엄격 임계값: RC 1.0, E 0.6",
        "Loose thresholds: rc 2.5, e 0.3": "느슨한 임계값: RC 2.5, E 0.3",
    }
    return names.get(label, label)


def repeated_names(summaries: list[dict[str, Any]]) -> tuple[list[str], list[str], Counter[str]]:
    counts: Counter[str] = Counter(name for item in summaries for name in item["recommended_names"])
    threshold = max(3, len(summaries) // 2)
    frequent = [name for name, count in counts.most_common() if count >= threshold]
    named_sets = [set(item["recommended_names"]) for item in summaries if item["recommended_names"]]
    stable = sorted(set.intersection(*named_sets)) if named_sets else []
    return frequent, stable, counts


def extra_candidates(source: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    base = set(baseline["recommended_names"])
    return sorted(set(source["recommended_names"]) - base)


def fallback_action_label(row: dict[str, str]) -> str:
    if safe_bool(row.get("data_quality_low")):
        return "행동 보류"
    return clean_text(row.get("action_reason"), "유지·관찰")


def build_baseline_report_rows(output_dir: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, _, counts = repeated_names(summaries)
    baseline_dir = output_dir / "artifacts" / "baseline_12m_regular"
    proposal_rows = read_csv_rows(baseline_dir / "proposal.csv")
    action_rows = {
        row.get("ticker", ""): row
        for row in read_csv_rows(baseline_dir / "ips_actions.csv")
        if row.get("ticker")
    }
    rows = []
    for row in proposal_rows:
        ticker = row["ticker"]
        action = action_rows.get(ticker, {})
        action_label = clean_text(action.get("action_label"), fallback_action_label(row))
        decision_summary = clean_text(action.get("decision_summary"), clean_text(row.get("action_reason")))
        next_step = clean_text(action.get("next_step"), "다음 점검에서 신호 지속 여부를 확인합니다.")
        blocked_reason = clean_text(action.get("blocked_reason"))
        if blocked_reason == "unknown":
            blocked_reason = "-"
        rows.append(
            {
                "ticker": ticker,
                "group": row.get("group", ""),
                "current": safe_float(row.get("current_weight_pct")),
                "target": safe_float(row.get("target_weight_pct")),
                "gap": safe_float(row.get("gap_pct")),
                "should_execute": safe_bool(row.get("should_execute")),
                "suggested_trade_pct": safe_float(row.get("suggested_trade_pct")),
                "efficiency": safe_float(row.get("efficiency_score")),
                "rc_over": safe_float(row.get("rc_over_pct")),
                "frequency": counts.get(ticker, 0),
                "reason": row.get("action_reason", ""),
                "action_label": action_label,
                "decision_summary": decision_summary,
                "next_step": next_step,
                "reason_codes_text": clean_text(action.get("reason_codes_text")),
                "risk_notes": listish_text(action.get("risk_notes")),
                "blocked_reason": blocked_reason,
                "ips_fit_score": safe_float(row.get("ips_fit_score")),
                "ips_fit_band": clean_text(row.get("ips_fit_band")),
                "ips_score_role": safe_float(row.get("ips_score_role")),
                "ips_score_allocation": safe_float(row.get("ips_score_allocation")),
                "ips_score_thesis": safe_float(row.get("ips_score_thesis")),
                "ips_score_risk": safe_float(row.get("ips_score_risk")),
                "ips_score_action": safe_float(row.get("ips_score_action")),
                "ips_score_efficiency": safe_float(row.get("ips_score_efficiency")),
                "ips_score_data_quality": safe_float(row.get("ips_score_data_quality")),
                "efficiency_warning": safe_bool(row.get("efficiency_warning")),
            }
        )
    rows.sort(key=lambda row: (-row["frequency"], row["group"], row["ticker"]))
    return rows


def one_glance_sentence(summaries: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> str:
    frequent, _, _ = repeated_names(summaries)
    increase = [
        row["ticker"]
        for row in baseline_rows
        if row["frequency"] > 0 and row["action_label"] == "정기매수 증액 후보"
    ]
    reduce_or_review = [
        row["ticker"]
        for row in baseline_rows
        if row["frequency"] > 0 and row["action_label"] in {"정기매수 감액/중단 후보", "투자 논리 점검", "예외적 리밸런싱 매도 검토"}
    ]
    data_holds = [
        row["ticker"]
        for row in baseline_rows
        if row["action_label"] == "행동 보류" or "데이터 신뢰도" in row["blocked_reason"]
    ]
    parts = [
        f"이 스냅샷은 성과 지표보다 반복 후보({', '.join(frequent) if frequent else '없음'})와 위험 집중을 우선 점검해야 합니다.",
        f"코어 부족은 {', '.join(increase) if increase else '해당 없음'} 중심의 정기매수 조정으로 다루고, 위성은 {', '.join(reduce_or_review) if reduce_or_review else '해당 없음'} 중심으로 신규 매수 축소와 투자 논리 점검이 우선입니다.",
    ]
    if data_holds:
        parts.append(f"{', '.join(data_holds)}는 데이터 신뢰도 확인 전까지 판단을 보류합니다.")
    parts.append("수치 신호는 즉시 매매 지시가 아니라 정기매수 배분과 리뷰 우선순위를 정하는 입력으로 해석합니다.")
    return " ".join(parts)


def tickers_by_action(
    rows: list[dict[str, Any]],
    labels: set[str],
    *,
    should_execute: bool | None = None,
) -> list[str]:
    return [
        row["ticker"]
        for row in rows
        if row["action_label"] in labels
        and (should_execute is None or row["should_execute"] == should_execute)
    ]


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def write_english_report(output_dir: Path, snapshot: dict[str, Any], summaries: list[dict[str, Any]]) -> Path:
    baseline = summaries[0]
    frequent, stable, _ = repeated_names(summaries)
    path = output_dir / f"snapshot{snapshot['snapshot_id']}_parameter_sweep_report.md"
    lines = [
        f"# Snapshot {snapshot['snapshot_id']} Parameter Sweep Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Snapshot: {snapshot['snapshot_id']} (`{snapshot.get('name') or 'unnamed'}`)",
        f"- Portfolio: {snapshot.get('portfolio_id') or 'n/a'}",
        f"- Scenarios: {len(summaries)}",
        "",
        "## Executive Summary",
        "",
        f"Baseline portfolio metrics are CAGR {pct(baseline['portfolio_cagr'])}, volatility {pct(baseline['portfolio_vol'])}, Sharpe {num(baseline['portfolio_sharpe'])}, and max drawdown {pct(baseline['portfolio_mdd'])}. Benchmark `{baseline['bench']}` has CAGR {pct(baseline['benchmark_cagr'])} and Sharpe {num(baseline['benchmark_sharpe'])}.",
        "",
        f"Repeated candidates: {', '.join(frequent) if frequent else 'none'}. Common to every scenario: {', '.join(stable) if stable else 'none'}.",
        "",
        "The IPS interpretation is conservative: use scheduled allocation changes first, review satellite risk concentration, and treat immediate trades as exceptions.",
        "",
        "## Scenario Comparison",
        "",
        "| Scenario | Period | Benchmark | Context | CAGR | Vol | Sharpe | MDD | Recommended | Low-quality data |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['label']} | {item['period']} | {item['bench']} | {item['decision_context']} | "
            f"{pct(item['portfolio_cagr'])} | {pct(item['portfolio_vol'])} | {num(item['portfolio_sharpe'])} | {pct(item['portfolio_mdd'])} | "
            f"{item['recommended_count']} | {', '.join(item['low_quality']) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_korean_report(output_dir: Path, snapshot: dict[str, Any], summaries: list[dict[str, Any]]) -> Path:
    baseline = summaries[0]
    frequent, stable, _ = repeated_names(summaries)
    baseline_rows = build_baseline_report_rows(output_dir, summaries)
    six_month = next((item for item in summaries if item["id"] == "period_6m"), None)
    ytd = next((item for item in summaries if item["id"] == "period_ytd"), None)
    max_period = next((item for item in summaries if item["id"] == "period_max"), None)
    market = next((item for item in summaries if item["id"] == "market_correction"), None)
    sharp = next((item for item in summaries if item["id"] == "sharp_drop"), None)
    strict = next((item for item in summaries if item["id"] == "strict_thresholds"), None)
    loose = next((item for item in summaries if item["id"] == "loose_thresholds"), None)
    short_extra = sorted(set(extra_candidates(six_month, baseline) if six_month else []) | set(extra_candidates(ytd, baseline) if ytd else []))
    correction_extra = sorted(set(extra_candidates(market, baseline) if market else []) | set(extra_candidates(sharp, baseline) if sharp else []))
    threshold_names = sorted(set((strict or {}).get("recommended_names", [])) | set((loose or {}).get("recommended_names", [])))
    path = output_dir / f"snapshot{snapshot['snapshot_id']}_parameter_sweep_report_ko.md"

    lines = [
        f"# 스냅샷 {snapshot['snapshot_id']} 다중 파라미터 평가 보고서",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 대상 스냅샷: {snapshot['snapshot_id']} (`{snapshot.get('name') or '이름 없음'}`)",
        f"- 포트폴리오: {snapshot.get('portfolio_id') or 'n/a'}",
        f"- 생성일: {snapshot.get('created_at') or 'n/a'}",
        f"- 실행 시나리오: {len(summaries)}개",
        "- 원본 JSON: `artifacts/<scenario>/stdout.json`",
        "- 시나리오별 CSV: `metrics.csv`, `proposal.csv`, `ips_actions.csv`, `group_summary.csv`, `rc_violations.csv`",
        "",
        "## 요약 결론",
        "",
        f"기본 시나리오의 포트폴리오 지표는 CAGR {pct(baseline['portfolio_cagr'])}, 변동성 {pct(baseline['portfolio_vol'])}, Sharpe {num(baseline['portfolio_sharpe'])}, 최대낙폭 {pct(baseline['portfolio_mdd'])}입니다. 같은 조건의 벤치마크(`{baseline['bench']}`)는 CAGR {pct(baseline['benchmark_cagr'])}, Sharpe {num(baseline['benchmark_sharpe'])}입니다.",
        "",
        f"반복 조정 후보는 {', '.join(frequent) if frequent else '없음'}입니다. 모든 시나리오에서 공통으로 잡힌 종목은 {', '.join(stable) if stable else '없음'}입니다.",
        "",
        "이번 평가의 중심 신호는 전면 매매가 아니라 위성 위험 집중 통제와 코어 정기매수 보강입니다. 수치 신호는 즉시 매매 지시가 아니라 정기매수 배분과 리뷰 우선순위를 정하는 입력으로 해석합니다.",
        "",
        "## 시나리오 비교",
        "",
        "| 시나리오 | 기간 | 벤치마크 | 판단 맥락 | CAGR | 변동성 | Sharpe | MDD | 조정 후보 | 저품질 데이터 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {ko_scenario_name(item['label'])} | {item['period']} | {item['bench']} | {item['decision_context']} | "
            f"{pct(item['portfolio_cagr'])} | {pct(item['portfolio_vol'])} | {num(item['portfolio_sharpe'])} | {pct(item['portfolio_mdd'])} | "
            f"{item['recommended_count']} | {', '.join(item['low_quality']) or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 핵심 민감도 해석",
            "",
            "기간 민감도:",
            "",
            f"- 12개월 기본값에서는 {baseline['recommended_tickers']}가 조정 후보입니다.",
            f"- 6개월과 YTD에서 새로 추가되는 후보는 {', '.join(short_extra) if short_extra else '없음'}입니다.",
            f"- Max 구간에서는 포트폴리오 CAGR {pct(max_period['portfolio_cagr']) if max_period else 'n/a'}, Sharpe {num(max_period['portfolio_sharpe']) if max_period else 'n/a'}, 최대낙폭 {pct(max_period['portfolio_mdd']) if max_period else 'n/a'}로 장기 데이터 품질을 함께 봐야 합니다.",
            "",
            "벤치마크·무위험수익률 민감도:",
            "",
            "- 벤치마크와 무위험수익률 변경은 성과 지표와 일부 점수에 영향을 주지만, 반복 후보는 빈도 기준으로 별도 확인합니다.",
            "",
            "판단 맥락·임계값 민감도:",
            "",
            f"- 시장 조정 또는 급락 점검 맥락에서 새로 추가되는 후보는 {', '.join(correction_extra) if correction_extra else '없음'}입니다.",
            f"- 엄격/느슨한 임계값 후보군 합집합은 {', '.join(threshold_names) if threshold_names else '없음'}입니다.",
            "",
            "## 시나리오별 조정 후보",
            "",
            "| 시나리오 | 조정 후보 |",
            "|---|---|",
        ]
    )
    for item in summaries:
        lines.append(f"| {ko_scenario_name(item['label'])} | {item['recommended_tickers']} |")

    lines.extend(
        [
            "",
            "## 위험 집중도",
            "",
            f"기본 12개월 기준 상위 위험기여 종목은 {', '.join(baseline['top_risk'])}입니다. 포트폴리오 비중은 코어 {baseline['group_weights'].get('core', 0):.2f}%, 위성 {baseline['group_weights'].get('satellite', 0):.2f}%로 집계됩니다.",
            "",
            "위성 비중과 위험기여가 높은 종목은 성과가 좋더라도 신규 매수 확대보다 목표 비중, 중복 노출, 장기 보유 가능성을 먼저 점검합니다.",
            "",
            "## 데이터 품질",
            "",
        ]
    )
    all_low_quality = sorted({ticker for item in summaries for ticker in item["low_quality"]})
    lines.append(
        f"저품질 데이터로 표시된 종목은 {', '.join(all_low_quality) if all_low_quality else '없음'}입니다. 이 종목들은 수치만으로 매수·매도 판단을 내리기보다 가격 데이터 기간, 상장 기간, 누락률을 먼저 확인해야 합니다."
    )
    lines.extend(
        [
            "",
            "## 종목별 실행 계획",
            "",
            "| 종목 | 그룹 | 현재/목표 | 갭 | 최종실행 | 최종조정 | 후보 빈도 | 권장 조치 | 판단 요약 | 다음 단계 |",
            "|---|---|---:|---:|---|---:|---:|---|---|---|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {md_cell(row['ticker'])} | {md_cell(row['group'])} | {row['current']:.2f}% / {row['target']:.2f}% | "
            f"{row['gap']:+.2f}% | {'실행' if row['should_execute'] else '보류'} | {row['suggested_trade_pct']:+.2f}% | "
            f"{row['frequency']}/{len(summaries)} | {md_cell(row['action_label'])} | {md_cell(row['decision_summary'])} | {md_cell(row['next_step'])} |"
        )

    lines.extend(
        [
            "",
            "## 점수 구성 요약",
            "",
            "| 종목 | IPS 적합도 | IPS 등급 | 역할 | 비중 | 논리 | 위험 | 실행 | E | 데이터 | 효율 경고 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {md_cell(row['ticker'])} | {row['ips_fit_score']:.1f} | {md_cell(row['ips_fit_band'])} | "
            f"{row['ips_score_role']:.2f} | {row['ips_score_allocation']:.2f} | {row['ips_score_thesis']:.2f} | "
            f"{row['ips_score_risk']:.2f} | {row['ips_score_action']:.2f} | {row['ips_score_efficiency']:.2f} | "
            f"{row['ips_score_data_quality']:.2f} | {'경고' if row['efficiency_warning'] else '정상'} |"
        )

    lines.extend(
        [
            "",
            "## 로직 확인 요약",
            "",
            "| 종목 | 규칙 코드 | 위험 메모 | 차단 사유 |",
            "|---|---|---|---|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {md_cell(row['ticker'])} | {md_cell(row['reason_codes_text'])} | {md_cell(row['risk_notes'])} | {md_cell(row['blocked_reason'])} |"
        )

    core_increase = tickers_by_action(baseline_rows, {"정기매수 증액 후보"}, should_execute=True)
    dca_reduce = tickers_by_action(baseline_rows, {"정기매수 감액/중단 후보"}, should_execute=True)
    thesis_review = tickers_by_action(baseline_rows, {"투자 논리 점검"})
    observe = tickers_by_action(baseline_rows, {"유지·관찰"})
    data_hold = tickers_by_action(baseline_rows, {"행동 보류"})
    exceptional = tickers_by_action(
        baseline_rows,
        {"예외적 즉시매수 검토", "예외적 리밸런싱 매도 검토"},
        should_execute=True,
    )
    lines.extend(
        [
            "",
            "## 액션 우선순위",
            "",
            f"1. 정기매수 증액 우선: {', '.join(core_increase) if core_increase else '해당 없음'}",
            f"2. 정기매수 감액·중단 우선: {', '.join(dca_reduce) if dca_reduce else '해당 없음'}",
            f"3. 투자 논리 점검: {', '.join(thesis_review) if thesis_review else '해당 없음'}",
            f"4. 예외적 즉시매수/매도 검토: {', '.join(exceptional) if exceptional else '해당 없음'}",
            f"5. 관찰 중심: {', '.join(observe) if observe else '해당 없음'}",
            f"6. 데이터 확인 전 보류: {', '.join(data_hold) if data_hold else '해당 없음'}",
            "",
            "## IPS 해석",
            "",
            "- 부족한 코어 비중은 즉시 일괄매수보다 정기매수 증액 또는 배분 변경으로 처리합니다.",
            "- 위성 자산의 초과 위험, 비중 과다, 효율 미달은 신규 매수 축소/중단, 투자 논리 점검, 포트폴리오 단순화 검토로 먼저 번역합니다.",
            "- 즉시 매도는 예외입니다. 투자 논리 손상, 과도한 목표 초과, 단순화 필요, 더 나은 대체 자산이 확인될 때 별도 검토합니다.",
            "- 저품질 데이터 또는 짧은 관측 기간은 액션 강화가 아니라 판단 보류 사유입니다.",
            "",
            "## 한눈에 읽기용 문장",
            "",
            one_glance_sentence(summaries, baseline_rows),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-parameter sweep for a saved portfolio snapshot and write reports.")
    parser.add_argument("--snapshot-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clean", action="store_true", help="Remove generated artifacts and reports in the output directory before running.")
    parser.add_argument("--language", choices=["ko", "en", "both"], default="ko")
    parser.add_argument("--scenario-set", choices=sorted(SCENARIO_SETS), default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or ROOT / "reports" / f"snapshot{args.snapshot_id}_parameter_sweep"
    snapshot = snapshot_metadata(args.snapshot_id)
    if args.clean:
        clean_output(output_dir, args.snapshot_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_scenario(args.snapshot_id, output_dir, scenario) for scenario in SCENARIO_SETS[args.scenario_set]]
    write_summary(output_dir, summaries)
    report_paths: list[Path] = []
    if args.language in ("ko", "both"):
        report_paths.append(write_korean_report(output_dir, snapshot, summaries))
    if args.language in ("en", "both"):
        report_paths.append(write_english_report(output_dir, snapshot, summaries))
    print(json.dumps({
        "ok": True,
        "snapshot_id": args.snapshot_id,
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        "reports": [str(path) for path in report_paths],
        "scenario_count": len(summaries),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
