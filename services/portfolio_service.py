"""포트폴리오 입력 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

from typing import List
import pandas as pd
from pydantic import ValidationError

from core.asset import DEFAULT_GROUP, VALID_GROUPS, Asset, parse_text_to_assets
from storage.config_store import active_codes
from utils.metrics import normalize_weights


class PortfolioInputError(Exception):
    """포트폴리오 입력 처리 중 발생하는 오류."""

    pass


def _get_attr(row, name: str, default=None):
    value = getattr(row, name, default)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return default
    return value


def _normalize_group_input(value, ticker: str, warnings: list[str]) -> str:
    raw = DEFAULT_GROUP if value is None or str(value).strip() == "" else str(value).strip().lower()
    if raw in VALID_GROUPS:
        return raw
    warnings.append(
        f"{ticker}: 지원하지 않는 group '{raw}' 값은 {DEFAULT_GROUP}(으)로 처리합니다."
    )
    return DEFAULT_GROUP


def parse_text_to_assets_service(text: str) -> tuple[List[Asset], List[str]]:
    """텍스트를 자산 목록으로 파싱합니다.

    Args:
        text: 파싱할 텍스트

    Returns:
        (자산 목록, 경고 메시지 목록) 튜플
    """
    assets = parse_text_to_assets(text)
    warnings: List[str] = []
    return assets, warnings


def parse_csv_to_assets(df: pd.DataFrame) -> tuple[List[Asset], List[str]]:
    """CSV 데이터프레임을 자산 목록으로 파싱합니다.

    Args:
        df: CSV 데이터프레임

    Returns:
        (자산 목록, 경고 메시지 목록) 튜플

    Raises:
        PortfolioInputError: 필수 컬럼이 없거나 검증 실패 시
    """
    # AIDEV-NOTE: column-mapping-ko-en; 한글→영문 컬럼 매핑 (가중치→allocation, 누적수익률→return_total)
    col_mapping = {
        "가중치": "allocation",
        "누적수익률": "return_total",
        "전체기간수익률": "return_total",
        "수익률": "return_total",
        "현재수익률": "return_total",
        "그룹": "group",
        "관리그룹": "group",
        "자산군": "group",
        "정기매수": "dca_enabled",
        "정기매수대상": "dca_enabled",
        "투자논리": "thesis_status",
        "논리상태": "thesis_status",
    }
    df = df.rename(columns=col_mapping)

    if not {"ticker", "allocation"}.issubset(df.columns):
        raise PortfolioInputError(
            "CSV는 'ticker'와 'allocation' 칼럼을 포함해야 합니다"
        )

    warnings: List[str] = []
    asset_list: List[Asset] = []

    try:
        for r in df.itertuples(index=False):
            total = None
            raw_return_total = _get_attr(r, "return_total")
            if raw_return_total not in (None, ""):
                try:
                    total = float(raw_return_total) / 100.0  # % → 소수
                except (ValueError, TypeError):
                    total = None

            try:
                ticker = str(r.ticker).upper()
                group = _normalize_group_input(_get_attr(r, "group", DEFAULT_GROUP), ticker, warnings)
                asset = Asset(
                    ticker=ticker,
                    allocation=float(r.allocation),
                    return_total=total,
                    group=group,
                    dca_enabled=_get_attr(r, "dca_enabled", True),
                    thesis_status=_get_attr(r, "thesis_status", "unknown"),
                )
                asset_list.append(asset)
            except ValidationError as e:
                warnings.append(f"행 검증 실패: {e}")

    except Exception as e:
        raise PortfolioInputError(f"CSV 파싱 중 오류 발생: {e}") from e

    return asset_list, warnings


def parse_manual_edit_to_assets(
    edited_data: List[dict[str, float | str | None]],
) -> tuple[List[Asset], List[str]]:
    """수동 편집 데이터를 자산 목록으로 파싱합니다.

    Args:
        edited_data: 편집된 데이터 리스트 (dict 리스트)

    Returns:
        (자산 목록, 경고 메시지 목록) 튜플
    """
    warnings: List[str] = []
    asset_list: List[Asset] = []

    for row in edited_data:
        ticker = str(row.get("ticker", "")).strip()
        allocation = row.get("allocation", "")
        if ticker == "" or allocation is None or str(allocation).strip() == "":
            continue

        try:
            return_total = None
            if row.get("return_total"):
                return_total = float(row["return_total"]) / 100.0

            group = _normalize_group_input(row.get("group", DEFAULT_GROUP), ticker.upper(), warnings)
            asset = Asset(
                ticker=ticker.upper(),
                allocation=float(allocation),
                return_total=return_total,
                group=group,
                dca_enabled=row.get("dca_enabled", True),
                thesis_status=row.get("thesis_status", "unknown"),
            )
            asset_list.append(asset)
        except (ValueError, ValidationError) as e:
            warnings.append(f"행 검증 실패: {e}")

    return asset_list, warnings


def normalize_and_validate_assets(
    assets: List[Asset],
) -> tuple[pd.DataFrame, List[str]]:
    """자산 목록을 정규화하고 검증합니다.

    Args:
        assets: 자산 목록

    Returns:
        (정규화된 데이터프레임, 경고 메시지 목록) 튜플

    Raises:
        PortfolioInputError: 배분 합이 0인 경우
    """
    if not assets:
        raise PortfolioInputError("자산 목록이 비어있습니다")

    warnings: List[str] = []

    # AIDEV-NOTE: pydantic-integration; Asset 객체는 .model_dump()로 dict로 변환하여 DataFrame 생성
    asset_df = pd.DataFrame([a.model_dump() for a in assets])
    valid_thesis = active_codes("thesis_statuses")
    invalid_thesis_mask = ~asset_df["thesis_status"].isin(valid_thesis)
    for ticker, value in asset_df.loc[
        invalid_thesis_mask, ["ticker", "thesis_status"]
    ].itertuples(index=False):
        warnings.append(
            f"{ticker}: 등록되지 않은 thesis_status '{value}' 값은 unknown(으)로 처리합니다."
        )
    asset_df.loc[invalid_thesis_mask, "thesis_status"] = "unknown"

    raw_df = asset_df.copy()

    for ticker, group in raw_df.groupby("ticker"):
        if group["group"].nunique(dropna=True) > 1:
            warnings.append(
                f"{ticker}: group 값이 여러 개입니다. 첫 번째 값을 사용합니다."
            )

    asset_df = asset_df.groupby("ticker", as_index=False).agg(
        {
            "allocation": "sum",
            "return_total": "first",  # 첫 번째 return_total 유지 (중복 제거 시)
            "group": "first",
            "dca_enabled": "first",
            "thesis_status": "first",
        }
    )

    # AIDEV-NOTE: input-validation; 외부 입력(사용자 텍스트/CSV/편집)의 유효성을 먼저 검증하여 후속 계산 오류 방지
    total_allocation = asset_df["allocation"].sum()
    if total_allocation == 0:
        raise PortfolioInputError(
            "포트폴리오 배분의 합이 0입니다. 최소 1개 이상의 배분을 입력해주세요."
        )

    asset_df["weight"] = normalize_weights(asset_df["allocation"])  # 0-1 범위

    return asset_df, warnings
