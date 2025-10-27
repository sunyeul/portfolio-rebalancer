# Streamlit 프로토타입: 포트폴리오 평가 & 리밸런싱
# 작성자: ChatGPT (GPT-5 Thinking)
# 설명: 3단계 워크플로우용 최소 엔드-투-엔드 프로토타입
# 필수 라이브러리: streamlit, yfinance, pandas, numpy, matplotlib
# 실행 명령:  streamlit run main.py

import streamlit as st

from steps import show_data_analysis, show_evaluation_proposal, show_portfolio_input

# ==============================
# Streamlit 페이지 설정
# ==============================
st.set_page_config(
    page_title="포트폴리오 리밸런서 – 포트폴리오 리밸런싱 프로토타입", layout="wide"
)

st.title("📊 포트폴리오 리밸런서 — 포트폴리오 평가 & 리밸런싱 프로토타입")

with st.expander("ℹ️ 사용 방법", expanded=True):
    st.markdown(
        """
        **단계 1 — 입력**: 티커와 배분을 붙여넣기, CSV 업로드, 또는 표 편집하기.

        **단계 2 — 데이터 조회 & 분석**: 기간, 벤치마크, 무위험 수익률 선택하기.

        **단계 3 — 평가 & 제안**: 사분면, 목표 대비 갭, 실행 계획 보기.
        """
    )

# ==============================
# 사이드바 컨트롤 (글로벌 설정)
# ==============================
st.sidebar.header("⚙️ 설정")

# AIDEV-NOTE: period-selection-ux; 사용자가 직관적으로 기간을 선택할 수 있도록 라디오 버튼 제공 (개월/YTD/Max)
period_mode = st.sidebar.radio(
    "평가 기간 선택", ["개월", "YTD", "Max"], index=0, horizontal=True
)

if period_mode == "개월":
    period = st.sidebar.number_input(
        "평가 기간 (개월)",
        min_value=1,
        max_value=120,
        value=12,
        step=1,
    )
elif period_mode == "YTD":
    period = "YTD"
else:  # Max
    period = "Max"

rf = st.sidebar.number_input("무위험 수익률 (연간, %)", value=0.0, step=0.1) / 100.0
bench = st.sidebar.text_input("벤치마크 티커", value="SPY")

# AIDEV-NOTE: rc-over-e-thresholds; RC_Over(%) 임계값과 효율점수 E 임계값을 사용자 정의 가능하게 설정
rc_over_thresh_pct = st.sidebar.number_input(
    "RC_Over 임계값 (%)", value=1.5, step=0.1, min_value=0.0
)
e_thresh = st.sidebar.number_input(
    "효율 점수 E 임계값", value=0.5, step=0.05, min_value=0.0, max_value=1.0
)

# AIDEV-TODO: smoothing-ui-toggle; 스무딩 토글을 UI에 노출할지 검토 (현재는 자동 적용)
st.sidebar.caption(
    f"임계값: RC_Over > {rc_over_thresh_pct:.1f}% (과노출) & E ≥ {e_thresh:.2f} (효율)으로 사분면 정의."
)

# AIDEV-NOTE: momentum-weight-control; E′ 계산용 모멘텀 가중치(0.1~0.3), 기본값 0.2
momentum_weight = st.sidebar.slider(
    "모멘텀 가중치 (E′ 보정용)",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="높을수록 YTD 수익률이 효율 점수에 더 크게 영향을 줍니다.",
)

# ==============================
# 단계 1: 포트폴리오 입력
# ==============================
show_portfolio_input()

# ==============================
# 단계 2: 데이터 조회 & 보강
# ==============================
show_data_analysis(period, rf, bench, momentum_weight)

# ==============================
# 단계 3: 평가 & 실행 계획 제안
# ==============================
show_evaluation_proposal(rc_over_thresh_pct, e_thresh, bench)
