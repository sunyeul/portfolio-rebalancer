import pandas as pd
import streamlit as st
from pydantic import ValidationError
from core.asset import Asset, parse_text_to_assets
from utils.metrics import normalize_weights


def show_portfolio_input() -> pd.DataFrame | None:
    """포트폴리오 입력 단계를 표시하고 처리합니다.

    Returns:
        정규화된 자산 배분 데이터프레임 또는 None (입력 미완료 시)

    # AIDEV-NOTE: pydantic-integration; Asset 객체는 .model_dump()로 dict로 변환하여 DataFrame 생성
    """
    st.subheader("1️⃣ 포트폴리오 입력")

    sample_text = """TSLA, 13.88
SPY, 18.96
MSFT, 12.0
GOOGL, 10.0
AMZN, 9.0
AAPL, 8.0
BTC-USD, 5.0"""

    input_mode = st.radio(
        "입력 방식",
        ["텍스트 붙여넣기", "CSV 업로드", "수동 편집기"],
        horizontal=True,
    )

    assets: list[Asset] = []

    if input_mode == "텍스트 붙여넣기":
        text = st.text_area(
            "티커와 배분을 붙여넣기 (%, 줄마다 하나)", value=sample_text, height=160
        )
        if st.button("텍스트 파싱"):
            assets = parse_text_to_assets(text)
    elif input_mode == "CSV 업로드":
        st.write("필수 CSV 칼럼: ticker, allocation (선택: return_total)")
        file = st.file_uploader("CSV 업로드", type=["csv"])
        if file:
            df = pd.read_csv(file)
            # AIDEV-NOTE: column-mapping-ko-en; 한글→영문 컬럼 매핑 (가중치→allocation, 누적수익률→return_total)
            col_mapping = {
                "가중치": "allocation",
                "누적수익률": "return_total",
                "전체기간수익률": "return_total",
                "수익률": "return_total",
                "현재수익률": "return_total",
            }
            df.rename(columns=col_mapping, inplace=True)

            if {"ticker", "allocation"}.issubset(df.columns):
                try:
                    asset_list = []
                    for r in df.itertuples(index=False):
                        total = None
                        if hasattr(r, "return_total") and r.return_total:
                            try:
                                total = float(r.return_total) / 100.0  # % → 소수
                            except (ValueError, TypeError):
                                total = None
                        asset_list.append(
                            Asset(
                                ticker=str(r.ticker).upper(),
                                allocation=float(r.allocation),
                                return_total=total,
                            )
                        )
                    assets = asset_list
                except ValidationError as e:
                    st.error(f"❌ CSV 검증 실패:\n{e}")
                    return None
            else:
                st.error("CSV는 'ticker'와 'allocation' 칼럼을 포함해야 합니다")
    else:
        # 수동 편집 모드
        df_edit = pd.DataFrame(
            [
                {"ticker": "SPY", "allocation": 40.0, "return_total": None},
                {"ticker": "AAPL", "allocation": 20.0, "return_total": None},
                {"ticker": "MSFT", "allocation": 20.0, "return_total": None},
                {"ticker": "TSLA", "allocation": 20.0, "return_total": None},
            ]
        )
        edited = st.data_editor(df_edit, num_rows="dynamic", width="stretch")
        if st.button("위 표 사용"):
            try:
                assets = [
                    Asset(
                        ticker=str(r.ticker).upper(),
                        allocation=float(r.allocation),
                        return_total=float(r.return_total) / 100.0
                        if r.return_total
                        else None,
                    )
                    for r in edited.itertuples(index=False)
                ]
            except ValidationError as e:
                st.error(f"❌ 검증 실패:\n{e}")
                return None

    if not assets:
        st.info("입력을 제공하고 해당 버튼을 눌러 파싱하거나 사용합니다.")
        return None

    # AIDEV-NOTE: input-validation; 외부 입력(사용자 텍스트/CSV/편집)의 유효성을 먼저 검증하여 후속 계산 오류 방지
    # Pydantic Asset 객체를 dict로 변환하여 DataFrame 생성
    asset_df = pd.DataFrame([a.model_dump() for a in assets])
    asset_df = asset_df.groupby("ticker", as_index=False).agg(
        {
            "allocation": "sum",
            "return_total": "first",  # 첫 번째 return_total 유지 (중복 제거 시)
        }
    )

    # Pydantic 검증으로 이미 유효성 확인됨, 추가 검사는 최소화
    # (Pydantic의 Field(..., ge=0)로 음수 체크 완료)
    total_allocation = asset_df["allocation"].sum()
    if total_allocation == 0:
        st.warning(
            "⚠️ 포트폴리오 배분의 합이 0입니다. 최소 1개 이상의 배분을 입력해주세요."
        )
        return None

    asset_df["weight"] = normalize_weights(asset_df["allocation"])  # 0-1 범위
    st.dataframe(asset_df, width="stretch")

    # 세션 상태에 저장
    st.session_state.assets = assets
    st.session_state.asset_df = asset_df

    return asset_df
