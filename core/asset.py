from typing import List
from pydantic import BaseModel, field_validator, Field
import streamlit as st
import re


class Asset(BaseModel):
    """자산 클래스: 티커와 배분을 저장합니다.

    # AIDEV-NOTE: pydantic-validation; 티커 형식(A-Z 숫자 하이픈) + allocation >= 0 자동 검증
    """

    ticker: str = Field(..., description="자산 티커 (대문자, 1-8자)")
    allocation: float = Field(..., ge=0, description="배분 비율 (0 이상, 단위 %)")
    return_total: float | None = Field(
        None, description="누적 수익률 (0.1234 = 12.34%, 선택)"
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """티커 형식 검증: 대문자, 1-8자, 알파벳+숫자+하이픈만 허용."""
        if not isinstance(v, str):
            raise ValueError(f"ticker는 문자열이어야 함: {v}")
        v = v.strip().upper()
        if not re.match(r"^[A-Z0-9\-]{1,8}$", v):
            raise ValueError(f"ticker는 1-8자 알파벳/숫자/하이픈만 허용: '{v}'")
        return v

    @field_validator("allocation", mode="before")
    @classmethod
    def validate_allocation(cls, v: float) -> float:
        """배분 검증: 0 이상의 숫자."""
        try:
            val = float(v)
            if val < 0:
                raise ValueError(f"allocation은 0 이상이어야 함: {val}")
            if val == float("inf") or val != val:  # inf 또는 NaN 체크
                raise ValueError(f"allocation은 유효한 숫자여야 함: {val}")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"allocation은 숫자로 변환 불가: {v}") from e

    @field_validator("return_total", mode="before")
    @classmethod
    def validate_return_total(cls, v: float | None) -> float | None:
        """YTD 수익률 검증: -1 ~ 5 범위 (선택 필드)."""
        if v is None:
            return None
        try:
            val = float(v)
            if val < -1.0 or val > 5.0:
                raise ValueError(f"return_total는 -100% ~ 500% 범위여야 함: {val}")
            if val != val:  # NaN 체크
                return None
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"return_total는 숫자로 변환 불가: {v}") from e

    class Config:
        """Pydantic 설정."""

        # 추가 필드 허용 금지 (스트릭트 모드)
        extra = "forbid"


def parse_text_to_assets(text: str) -> List[Asset]:
    """텍스트를 자산 목록으로 파싱합니다.

    지원하는 형식:
    TSLA, 13.88
    SPY, 18.96
    또는 TSLA, 13.88, -5.2 (return_ytd 포함)
    또는 "TSLA 13.88%"

    # AIDEV-NOTE: return-ytd-parsing; 세 번째 숫자는 return_ytd(%로 입력, 소수로 변환)
    # AIDEV-NOTE: pydantic-error-handling; 파싱 실패 시 Pydantic ValidationError를 캐치하여 사용자에게 명확한 메시지 전달
    """
    assets = []
    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # CSV 형식 먼저 시도
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()
        if not parts:
            continue
        # 파트에서 숫자 찾기
        ticker = parts[0].upper()
        allocation = None
        return_total = None

        num_count = 0
        for p in parts[1:]:
            p_clean = p.replace("%", "")
            try:
                num = float(p_clean)
                if num_count == 0:
                    allocation = num
                elif num_count == 1:
                    return_total = num / 100.0  # % → 소수 변환
                num_count += 1
            except ValueError:
                continue

        if allocation is None:
            st.warning(f"줄 {line_num}에서 배분을 찾을 수 없음: '{line}'. 건너뜀.")
            continue
        # Pydantic 검증 실행
        try:
            asset = Asset(
                ticker=ticker, allocation=allocation, return_total=return_total
            )
            assets.append(asset)
        except ValueError as e:
            st.warning(f"줄 {line_num} 검증 실패: {e}. 건너뜀.")
            continue
    return assets
