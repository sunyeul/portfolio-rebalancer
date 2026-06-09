import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "snapshot-parameter-sweep-report"
    / "scripts"
    / "run_snapshot_sweep.py"
)


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_snapshot_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summaries():
    base = {
        "id": "baseline_12m_regular",
        "label": "Baseline: 12M, regular review",
        "note": "Current default assumptions.",
        "ok": True,
        "period": 12,
        "rf": 0.025,
        "bench": "SPY:80,QQQ:20",
        "decision_context": "regular_review",
        "portfolio_cagr": 0.1,
        "portfolio_vol": 0.2,
        "portfolio_sharpe": 0.5,
        "portfolio_mdd": -0.1,
        "benchmark_cagr": 0.08,
        "benchmark_sharpe": 0.4,
        "recommended_count": 1,
        "recommended_tickers": "VOO(+5.00%)",
        "recommended_names": ["VOO"],
        "hold_count": 1,
        "low_quality": [],
        "top_risk": ["QQQ", "VOO"],
        "top_risk_detail": [],
        "reason_counts": {},
        "recommended_reason_counts": {},
        "group_weights": {"core": 70.0, "satellite": 30.0},
    }
    second = {**base, "id": "period_6m", "label": "Short lookback: 6M"}
    third = {**base, "id": "bench_spy", "label": "Benchmark: SPY"}
    return [base, second, third]


def write_baseline_csvs(output_dir: Path) -> None:
    baseline = output_dir / "artifacts" / "baseline_12m_regular"
    write_text(
        baseline / "proposal.csv",
        "\n".join(
            [
                "ticker,current_weight_pct,target_weight_pct,gap_pct,efficiency_score,rc_gap_pct,rc_over_pct,rc_target_pct,return_total_pct,group,dca_enabled,thesis_status,missing_ratio,observation_count,data_quality_low,risk_over,within_hysteresis,below_min_trade,numeric_candidate,should_execute,suggested_trade_pct,reference_trade_pct,ips_score_role,ips_score_allocation,ips_score_thesis,ips_score_risk,ips_score_action,ips_score_efficiency,ips_score_data_quality,ips_fit_score,ips_fit_band,efficiency_warning,action_reason",
                "VOO,40,50,10,0.8,0,0,40,10,core,True,intact,0,100,False,False,False,False,True,True,5,5,1,1,1,1,1,0.8,1,95,high,False,비중 목표 미달",
                "QQQ,60,50,-10,0.3,10,10,50,-5,satellite,True,watch,0,100,False,True,False,False,True,False,0,-10,0.8,0.7,0.6,0.5,0.25,0.3,1,70,medium,True,투자 논리 점검",
            ]
        )
        + "\n",
    )
    write_text(
        baseline / "ips_actions.csv",
        "\n".join(
            [
                "ticker,action_label,decision_summary,next_step,reason_codes_text,risk_notes,blocked_reason",
                "VOO,정기매수 증액 후보,UI 판단 요약,UI 다음 단계,\"ips_fit_high, risk_ok\",[],",
            ]
        )
        + "\n",
    )


def test_korean_report_uses_ui_action_fields_and_score_sections(tmp_path):
    sweep = load_sweep_module()
    write_baseline_csvs(tmp_path)

    path = sweep.write_korean_report(
        tmp_path,
        {"snapshot_id": 99, "name": "fixture", "portfolio_id": 1, "created_at": "2026-06-09"},
        summaries(),
    )

    report = path.read_text(encoding="utf-8")
    assert "## 종목별 실행 계획" in report
    assert "## 점수 구성 요약" in report
    assert "## 로직 확인 요약" in report
    assert "UI 판단 요약" in report
    assert "UI 다음 단계" in report
    assert "ips_fit_high, risk_ok" in report
    assert "| VOO | 95.0 | high | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 정상 |" in report
    assert "1. 정기매수 증액 우선: VOO" in report
    assert "QQQ | satellite" in report
    assert "투자 논리 점검" in report


def test_baseline_report_rows_fall_back_when_ips_action_is_missing(tmp_path):
    sweep = load_sweep_module()
    write_baseline_csvs(tmp_path)

    rows = sweep.build_baseline_report_rows(tmp_path, summaries())
    qqq = next(row for row in rows if row["ticker"] == "QQQ")

    assert qqq["action_label"] == "투자 논리 점검"
    assert qqq["decision_summary"] == "투자 논리 점검"
    assert qqq["next_step"] == "다음 점검에서 신호 지속 여부를 확인합니다."
    assert qqq["frequency"] == 0
