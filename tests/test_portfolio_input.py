import pandas as pd

from core.asset import parse_text_to_assets
from services.portfolio_service import (
    normalize_and_validate_assets,
    parse_csv_to_assets,
    parse_manual_edit_to_assets,
)


def test_parse_csv_uses_layer_and_thesis_defaults():
    df = pd.DataFrame([{"ticker": "VOO", "allocation": 40}])

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].ticker == "VOO"
    assert assets[0].layer == "core"
    assert assets[0].thesis_status == "valid"
    assert "category" not in assets[0].model_dump()
    assert "dca_enabled" not in assets[0].model_dump()


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


def test_parse_csv_reads_layer_thesis_and_percent_return_total():
    df = pd.DataFrame(
        [
            {
                "ticker": "IONQ",
                "allocation": 2,
                "return_total": -12,
                "layer": "satellite",
                "thesis_status": "watch",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].return_total == -0.12
    assert assets[0].layer == "satellite"
    assert assets[0].thesis_status == "watch"


def test_layer_and_thesis_are_preserved_across_input_paths():
    text_assets = parse_text_to_assets(
        "SMH 8 satellite valid\n"
        "UFO 3 satellite watch\n"
        "BND 5 core valid"
    )
    manual_assets, manual_warnings = parse_manual_edit_to_assets(
        [
            {"ticker": "SMH", "allocation": "8", "layer": "satellite"},
            {
                "ticker": "UFO",
                "allocation": "3",
                "layer": "satellite",
                "thesis_status": "watch",
            },
            {"ticker": "BND", "allocation": "5", "layer": "core"},
        ]
    )
    csv_assets, csv_warnings = parse_csv_to_assets(
        pd.DataFrame(
            [
                {"ticker": "SMH", "allocation": 8, "layer": "satellite"},
                {
                    "ticker": "UFO",
                    "allocation": 3,
                    "layer": "satellite",
                    "thesis_status": "watch",
                },
                {"ticker": "BND", "allocation": 5, "layer": "core"},
            ]
        )
    )

    assert [(asset.layer, asset.thesis_status) for asset in text_assets] == [
        ("satellite", "valid"),
        ("satellite", "watch"),
        ("core", "valid"),
    ]
    assert manual_warnings == []
    assert [(asset.layer, asset.thesis_status) for asset in manual_assets] == [
        ("satellite", "valid"),
        ("satellite", "watch"),
        ("core", "valid"),
    ]
    assert csv_warnings == []
    assert [(asset.layer, asset.thesis_status) for asset in csv_assets] == [
        ("satellite", "valid"),
        ("satellite", "watch"),
        ("core", "valid"),
    ]


def test_parse_csv_maps_korean_current_spec_columns():
    df = pd.DataFrame(
        [
            {
                "ticker": "VOO",
                "가중치": 40,
                "계층": "core",
                "투자논리": "valid",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].layer == "core"
    assert assets[0].thesis_status == "valid"


def test_normalize_warns_on_duplicate_layer_conflicts():
    df = pd.DataFrame(
        [
            {"ticker": "VOO", "allocation": 20, "layer": "core"},
            {"ticker": "VOO", "allocation": 20, "layer": "satellite"},
        ]
    )
    assets, _ = parse_csv_to_assets(df)

    asset_df, warnings = normalize_and_validate_assets(assets)

    assert asset_df.loc[0, "allocation"] == 40
    assert asset_df.loc[0, "layer"] == "core"
    assert "category" not in asset_df.columns
    assert "dca_enabled" not in asset_df.columns
    assert any("layer 값이 여러 개" in warning for warning in warnings)


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


def test_parse_manual_edit_preserves_layer_thesis_and_percent_return_total():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {
                "ticker": "ufo",
                "allocation": "3",
                "return_total": "-12",
                "layer": "satellite",
                "thesis_status": "watch",
            }
        ]
    )

    assert warnings == []
    assert assets[0].ticker == "UFO"
    assert assets[0].return_total == -0.12
    assert assets[0].layer == "satellite"
    assert assets[0].thesis_status == "watch"


def test_normalize_uses_valid_as_standard_thesis_code():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {
                "ticker": "VOO",
                "allocation": "100",
                "layer": "core",
                "thesis_status": "valid",
            }
        ]
    )

    asset_df, validation_warnings = normalize_and_validate_assets(assets)

    assert warnings == []
    assert validation_warnings == []
    assert asset_df.loc[0, "thesis_status"] == "valid"


def test_normalize_accepts_legacy_intact_thesis_as_valid():
    assets, warnings = parse_manual_edit_to_assets(
        [
            {
                "ticker": "VOO",
                "allocation": "100",
                "layer": "core",
                "thesis_status": "intact",
            }
        ]
    )

    asset_df, validation_warnings = normalize_and_validate_assets(assets)

    assert warnings == []
    assert validation_warnings == []
    assert asset_df.loc[0, "thesis_status"] == "valid"
