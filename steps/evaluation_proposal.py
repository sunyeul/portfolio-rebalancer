import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.helpers import show_quadrant_explanations


def show_evaluation_proposal(rc_over_thresh_pct: float, e_thresh: float, bench: str):
    """평가 & 실행 계획 제안 단계를 표시하고 처리합니다.

    # AIDEV-NOTE: rc-over-e-quadrant; X축 RC_Over%, Y축 효율점수 E′(보정)로 2×2 사분면 분류

    Args:
        rc_over_thresh_pct: RC_Over 임계값 (%)
        e_thresh: 효율 점수 E′ 임계값
        bench: 벤치마크 티커
    """
    st.subheader("3️⃣ 평가 & 실행 계획 제안")

    if "metrics_df" not in st.session_state or st.session_state.metrics_df is None:
        st.info("먼저 단계 2를 완료해주세요.")
        return

    mdf = st.session_state.metrics_df.copy()

    st.markdown("#### 목표 (선택 사항): 티커별 목표 가중치 입력 (%)")
    init_targets = pd.DataFrame(
        {
            "ticker": mdf.index.tolist(),
            "목표_가중치_pct": (mdf["가중치"] * 100).round(2).values,  # 기본값 = 현재
        }
    )
    user_targets = st.data_editor(init_targets, num_rows="dynamic", width="stretch")

    # 목표 가중치 시리즈 구축
    if "ticker" in user_targets.columns and "목표_가중치_pct" in user_targets.columns:
        tgt = (
            user_targets.set_index("ticker")["목표_가중치_pct"]
            .astype(float)
            .reindex(mdf.index)
            .fillna(0)
            / 100.0
        )
    else:
        tgt = mdf["가중치"]

    # RC 타깃 계산: 목표 가중치에 비례
    # AIDEV-NOTE: rc-target-proportion; RC_Target = 목표 가중치 (직관적, KISS 원칙)
    rc_target = tgt.fillna(0)
    mdf["RC_Target"] = rc_target
    mdf["RC_Over"] = (mdf["위험기여도"] - mdf["RC_Target"]).clip(lower=0)

    # E′ 사용 (보정된 효율 점수); E′ 없으면 E 폴백
    mdf["효율E"] = mdf["E′"] if "E′" in mdf.columns else mdf["E"]

    # 2×2 사분면 분류: RC_Over (X축) vs E′ (Y축)
    rc_over_pct = mdf["RC_Over"] * 100  # 백분율로 표시
    over_thresh = rc_over_pct > rc_over_thresh_pct
    good_eff = mdf["효율E"] >= e_thresh

    mdf["사분면"] = np.select(
        [
            (~over_thresh) & good_eff,  # Q1: 낮은 RC_Over & 높은 E
            over_thresh & good_eff,  # Q2: 높은 RC_Over & 높은 E
            (~over_thresh) & (~good_eff),  # Q3: 낮은 RC_Over & 낮은 E
            over_thresh & (~good_eff),  # Q4: 높은 RC_Over & 낮은 E
        ],
        ["Q1 핵심", "Q2 성장", "Q3 개선", "Q4 위험관리"],
        default="분류 안됨",
    )

    # 산점도 (RC_Over% vs E)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rc_over_pct,
            y=mdf["효율E"],
            mode="markers+text",
            marker=dict(size=10, color="rgba(0, 100, 200, 0.6)"),
            text=mdf.index,
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="<b>%{text}</b><br>RC_Over: %{x:.2f}%<br>E: %{y:.2f}<extra></extra>",
        )
    )

    # 임계값 라인 추가
    fig.add_vline(
        x=rc_over_thresh_pct,
        line_dash="dash",
        line_color="red",
        annotation_text="RC_Over Thresh",
    )
    fig.add_hline(
        y=e_thresh,
        line_dash="dash",
        line_color="red",
        annotation_text="E Thresh",
    )

    # 레이아웃 설정
    fig.update_layout(
        title="사분면 분류 (RC_Over vs 효율E)",
        xaxis_title="RC_Over (%)",
        yaxis_title="효율 점수 E",
        height=500,
        hovermode="closest",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ 사분면 분류 설명 보기", expanded=False):
        st.markdown(
            """
            사분면 분류는 **RC_Over** (x축: 위험 기여도 초과) 와 **효율E** (y축: 효율 점수) 를 기준으로 자산을 4개 카테고리로 분류합니다.
            """
        )
        show_quadrant_explanations()
        st.markdown(
            f"""
            **현재 임계값:**
            - RC_Over: **{rc_over_thresh_pct:.1f}%**
            - 효율 E: **{e_thresh:.2f}**
            
            임계값은 사이드바에서 조정 가능합니다.
            """
        )

    # 갭 분석
    current_w = mdf["가중치"]
    gap = tgt - current_w

    proposal = pd.DataFrame(
        {
            "ticker": mdf.index,
            "사분면": mdf["사분면"].values,
            "현재%": (current_w * 100).round(2).values,
            "목표%": (tgt * 100).round(2).values,
            "갭%": (gap * 100).round(2).values,
            "E′": mdf["효율E"].round(2).values,
            "RC_Over%": rc_over_pct.round(2).values,
            "RC_Target%": (rc_target * 100).round(2).values,
            "return_total%": (mdf["return_total"] * 100).round(2).values,
        }
    )

    # AIDEV-NOTE: trade-filtering-rules; 히스테리시스(±20%), 최소거래(1.0%p) 적용하여 과잉거래 방지

    # 히스테리시스 밴드: 목표 ±20% 내 거래 제외
    # AIDEV-FIXME: hysteresis-definition; 갭의 절대값이 목표의 ±20% 이내면 거래 제외
    max_gap_pct = (tgt * 0.20).clip(lower=0.01)  # 목표의 20% 또는 최소 1%
    within_band = gap.abs() <= max_gap_pct

    # 최소 거래 단위: 1.0%p 이상만 처리
    min_trade_pct = 1.0 / 100.0
    above_min_trade = gap.abs() >= min_trade_pct

    # 거래 대상 필터링
    should_trade = above_min_trade & (~within_band)

    proposal["히스테리시스제외"] = within_band
    proposal["최소거래미만"] = ~above_min_trade
    proposal["실행"] = should_trade

    # 실행 규칙: 우선순위 정의
    sell_list = proposal[
        (proposal["사분면"] == "Q4 위험관리") & proposal["실행"]
    ].copy()
    sell_list = sell_list.sort_values(["현재%", "RC_Over%"], ascending=[False, False])

    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["갭%", "E′"], ascending=[False, False])

    fine_tune = proposal[
        (proposal["사분면"].isin(["Q1 핵심", "Q2 성장"]))
        & (proposal["실행"])
        & (proposal["갭%"].abs() <= 1.0)
    ].copy()

    # AIDEV-NOTE: cash-neutral-scaling; 매도합계 = 매수합계가 되도록 비례 스케일링
    # AIDEV-FIXME: cash-neutral-edge-case; 단순 스케일링으로 RC 상한 위반 가능, 임의 플래그 추가

    total_sell = sell_list["갭%"].abs().sum() if len(sell_list) > 0 else 0
    total_buy_before_scale = buy_list["갭%"].sum() if len(buy_list) > 0 else 0

    if total_buy_before_scale > 0 and total_sell > 0:
        # 매도 규모에 맞춰 매수 스케일 조정
        scale_factor = (
            min(1.0, total_sell / total_buy_before_scale)
            if total_buy_before_scale > 0
            else 1.0
        )
        buy_list["조정갭%"] = (buy_list["갭%"] * scale_factor).round(2)
    else:
        buy_list["조정갭%"] = buy_list["갭%"].round(2)

    if len(sell_list) > 0:
        sell_list["조정갭%"] = sell_list["갭%"].round(2)

    st.markdown("### 📌 실행 계획 (자동 생성)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**(1) 축소 / 청산 — 위험관리 (Q4)**")
        if len(sell_list) > 0:
            st.dataframe(sell_list, width="stretch")
        else:
            st.info("축소 대상 없음")
    with col2:
        st.markdown("**(2) 증가 — 양수 갭 (현금중립 스케일링)**")
        if len(buy_list) > 0:
            st.dataframe(buy_list, width="stretch")
        else:
            st.info("증가 대상 없음")

    st.markdown("**(3) 세부 조정 — Q1/Q2 작은 조정 (|갭| ≤ 1%)**")
    if len(fine_tune) > 0:
        st.dataframe(fine_tune, width="stretch")
    else:
        st.info("미세 조정 대상 없음")

    # RC 상한선 체크 (경고)
    st.markdown("### ⚠️ RC 상한선 체크")
    rc_cap_single = 0.12  # 단일 자산 최대 12%
    rc_cap_target_ratio = 1.5  # RC_Target의 1.5배

    post_trade_rc = mdf["위험기여도"].copy()
    violations = []
    for ticker in mdf.index:
        rc_cap = min(rc_cap_single, rc_target[ticker] * rc_cap_target_ratio)
        if post_trade_rc[ticker] > rc_cap:
            violations.append(
                {
                    "ticker": ticker,
                    "현재RC%": (post_trade_rc[ticker] * 100),
                    "RC상한%": (rc_cap * 100),
                    "상태": "⚠️ 경고: RC 상한 초과 위험",
                }
            )

    if violations:
        violations_df = pd.DataFrame(violations)
        st.warning(
            f"⚠️ {len(violations)}개 자산의 RC가 상한을 초과할 수 있습니다. "
            "현금중립 스케일링 후 재검토해주세요."
        )
        st.dataframe(violations_df, width="stretch")
    else:
        st.success("✅ 모든 자산의 RC가 상한선 내에 있습니다.")

    # 다운로드
    st.markdown("### ⬇️ 출력 다운로드")

    @st.cache_data
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        """데이터프레임을 CSV 바이트로 변환합니다."""
        return df.to_csv(index=False).encode()

    st.download_button(
        "보강된 메트릭 다운로드 (CSV)",
        data=to_csv_bytes(mdf.reset_index().rename(columns={"index": "ticker"})),
        file_name="enriched_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        "실행 계획 다운로드 (CSV)",
        data=to_csv_bytes(proposal),
        file_name="action_plan.csv",
        mime="text/csv",
    )

    # 가정 및 주석
    with st.expander("가정 & 주석"):
        st.markdown(
            f"""
            - **데이터**: `yfinance`를 통한 가격 (조정 종가). `BTC-USD` 같은 암호화폐 티커 지원.
            - **CAGR/변동성/샤프/IR**: 선택된 기간에서 스무딩된 수익률(윈저라이즈 + 3-window MA) 사용.
            - **효율 점수 E**: 0.6·Sharpe_norm + 0.4·IR_norm (z-score→CDF 정규화).
            - **RC_Target**: 목표 가중치에 비례 (RC_Target = target_weight).
            - **RC_Over**: max(0, RC - RC_Target).
            - **사분면**: X축 = RC_Over > {rc_over_thresh_pct:.1f}%, Y축 = E ≥ {e_thresh:.2f}.
            - **히스테리시스**: target ±20% 내 거래 제외.
            - **최소 거래**: 1.0%p 미만은 제외.
            - **현금중립**: 매도합 = 매수합 (비례 스케일링).
            - **RC 상한**: min(12%, 1.5×RC_Target).
            - **벤치마크**: 참고 메트릭 및 IR 계산용 (가중치 제외).
            - **return_total**: 사용자 입력값 우선; 미제공 시 가격 데이터 기준 누적 수익률 계산. Winsorize(2.5/97.5) 전처리.
            - **효율 점수 E′**: E + 0.2×quantile(return_total) (모멘텀 가중치는 사이드바에서 조정 가능).
            """
        )
