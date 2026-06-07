import pandas as pd

from services.portfolio_service import (
    normalize_and_validate_assets,
    parse_csv_to_assets,
    parse_manual_edit_to_assets,
)
from core.asset import parse_text_to_assets


def test_parse_csv_keeps_existing_shape_defaults():
    df = pd.DataFrame([{"ticker": "VOO", "allocation": 40}])

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].ticker == "VOO"
    assert assets[0].group == "ungrouped"
    assert assets[0].dca_enabled is True
    assert assets[0].thesis_status == "unknown"


def test_korean_yfinance_tickers_are_accepted_across_input_paths():
    tickers = ["000660.KS", "005930.KS", "069500.KS"]

    text_assets = parse_text_to_assets("\n".join(f"{ticker} 10" for ticker in tickers))
    manual_assets, manual_warnings = parse_manual_edit_to_assets(
        [{"ticker": ticker, "allocation": "10"} for ticker in tickers]
    )
    csv_assets, csv_warnings = parse_csv_to_assets(
        pd.DataFrame({"ticker": tickers, "allocation": [10, 10, 10]})
    )

    assert [asset.ticker for asset in text_assets] == tickers
    assert manual_warnings == []
    assert [asset.ticker for asset in manual_assets] == tickers
    assert csv_warnings == []
    assert [asset.ticker for asset in csv_assets] == tickers


def test_parse_csv_reads_ips_metadata_and_percent_return_total():
    df = pd.DataFrame(
        [
            {
                "ticker": "IONQ",
                "allocation": 2,
                "return_total": -12,
                "group": "satellite_quantum",
                "dca_enabled": False,
                "thesis_status": "watch",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].return_total == -0.12
    assert assets[0].group == "satellite_quantum"
    assert assets[0].dca_enabled is False
    assert assets[0].thesis_status == "watch"


def test_parse_csv_maps_korean_ips_columns():
    df = pd.DataFrame(
        [
            {
                "ticker": "VOO",
                "가중치": 40,
                "그룹": "core",
                "정기매수": "yes",
                "투자논리": "intact",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].group == "core"
    assert assets[0].dca_enabled is True
    assert assets[0].thesis_status == "intact"


def test_normalize_warns_on_duplicate_metadata_conflicts():
    df = pd.DataFrame(
        [
            {"ticker": "VOO", "allocation": 20, "group": "core"},
            {
                "ticker": "VOO",
                "allocation": 20,
                "group": "satellite_space",
            },
        ]
    )
    assets, _ = parse_csv_to_assets(df)

    asset_df, warnings = normalize_and_validate_assets(assets)

    assert asset_df.loc[0, "allocation"] == 40
    assert asset_df.loc[0, "group"] == "core"
    assert any("group 값이 여러 개" in warning for warning in warnings)


def test_parse_manual_edit_ignores_empty_rows():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {"ticker": "", "allocation": ""},
            {"ticker": "   ", "allocation": "20"},
            {"ticker": "VOO", "allocation": "40"},
            {"ticker": "QQQ", "allocation": ""},
        ]
    )

    assert warnings == []
    assert [asset.ticker for asset in assets] == ["VOO"]
    assert assets[0].allocation == 40


def test_parse_manual_edit_reads_checkbox_and_string_dca_values():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {"ticker": "VOO", "allocation": "40", "dca_enabled": True},
            {"ticker": "IONQ", "allocation": "2", "dca_enabled": "false"},
        ]
    )

    assert warnings == []
    assert assets[0].dca_enabled is True
    assert assets[1].dca_enabled is False


def test_parse_manual_edit_preserves_metadata_and_percent_return_total():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {
                "ticker": "ufo",
                "allocation": "3",
                "return_total": "-12",
                "group": "satellite_space",
                "thesis_status": "watch",
            }
        ]
    )

    assert warnings == []
    assert assets[0].ticker == "UFO"
    assert assets[0].return_total == -0.12
    assert assets[0].group == "satellite_space"
    assert assets[0].thesis_status == "watch"
