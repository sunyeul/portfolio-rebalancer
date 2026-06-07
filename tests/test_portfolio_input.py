import pandas as pd

from services.portfolio_service import (
    normalize_and_validate_assets,
    parse_csv_to_assets,
)


def test_parse_csv_keeps_existing_shape_defaults():
    df = pd.DataFrame([{"ticker": "VOO", "allocation": 40}])

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].ticker == "VOO"
    assert assets[0].group == "ungrouped"
    assert assets[0].role == "unknown"
    assert assets[0].dca_enabled is True
    assert assets[0].thesis_status == "unknown"


def test_parse_csv_reads_ips_metadata_and_percent_return_total():
    df = pd.DataFrame(
        [
            {
                "ticker": "IONQ",
                "allocation": 2,
                "return_total": -12,
                "group": "satellite_quantum",
                "role": "individual",
                "dca_enabled": False,
                "thesis_status": "watch",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].return_total == -0.12
    assert assets[0].group == "satellite_quantum"
    assert assets[0].role == "individual"
    assert assets[0].dca_enabled is False
    assert assets[0].thesis_status == "watch"


def test_parse_csv_maps_korean_ips_columns():
    df = pd.DataFrame(
        [
            {
                "ticker": "VOO",
                "가중치": 40,
                "그룹": "core",
                "역할": "broad_etf",
                "정기매수": "yes",
                "투자논리": "intact",
            }
        ]
    )

    assets, warnings = parse_csv_to_assets(df)

    assert warnings == []
    assert assets[0].group == "core"
    assert assets[0].role == "broad_etf"
    assert assets[0].dca_enabled is True
    assert assets[0].thesis_status == "intact"


def test_normalize_warns_on_duplicate_metadata_conflicts():
    df = pd.DataFrame(
        [
            {"ticker": "VOO", "allocation": 20, "group": "core", "role": "broad_etf"},
            {
                "ticker": "VOO",
                "allocation": 20,
                "group": "satellite_space",
                "role": "theme_etf",
            },
        ]
    )
    assets, _ = parse_csv_to_assets(df)

    asset_df, warnings = normalize_and_validate_assets(assets)

    assert asset_df.loc[0, "allocation"] == 40
    assert asset_df.loc[0, "group"] == "core"
    assert any("group 값이 여러 개" in warning for warning in warnings)
    assert any("role 값이 여러 개" in warning for warning in warnings)
