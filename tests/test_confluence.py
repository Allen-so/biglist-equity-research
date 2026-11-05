import pandas as pd

from src.features.confluence import (
    build_confluence_zones,
    classify_level_role,
    collect_structural_evidence,
)


def make_frame(
    *,
    sma20: float,
    sma50: float | None = None,
) -> pd.DataFrame:

    data = {
        "close": [103.0],
        "sma_20": [sma20],
    }

    if sma50 is not None:
        data["sma_50"] = [
            sma50
        ]

    return pd.DataFrame(
        data,
        index=[
            pd.Timestamp(
                "2026-10-01"
            )
        ],
    )


def test_level_role():
    assert (
        classify_level_role(
            level=100.0,
            current_price=103.0,
            current_atr=2.0,
        )
        == "SUPPORT"
    )


def test_duplicate_mas_do_not_create_families():
    daily = make_frame(
        sma20=100.0,
        sma50=100.4,
    )

    zones = pd.DataFrame(
        [
            {
                "zone_low": 99.8,
                "zone_high": 100.5,
                "zone_center": 100.1,
                "role": "SUPPORT",
                "confidence": 85.0,
            }
        ]
    )

    evidence = (
        collect_structural_evidence(
            current_price=103.0,
            current_atr=2.0,
            daily_frame=daily,
            daily_zones=zones,
        )
    )

    result = build_confluence_zones(
        evidence,
        current_price=103.0,
        current_atr=2.0,
    )

    assert not result.empty

    cluster = result.iloc[0]

    #
    # SMA20 + SMA50 = one family.
    # Horizontal zone = second family.
    #
    assert (
        cluster["family_count"]
        == 2
    )


def test_strong_multitimeframe_confluence():
    daily = make_frame(
        sma20=100.2,
    )

    weekly = pd.DataFrame(
        {
            "close": [103.0],
            "sma_20": [100.4],
        },
        index=[
            pd.Timestamp(
                "2026-10-02"
            )
        ],
    )

    zones = pd.DataFrame(
        [
            {
                "zone_low": 99.8,
                "zone_high": 100.5,
                "zone_center": 100.1,
                "role": "SUPPORT",
                "confidence": 90.0,
            }
        ]
    )

    trendlines = pd.DataFrame(
        [
            {
                "kind": "SUPPORT",
                "direction": "RISING",
                "status": "INTACT",
                "line_price_now": 100.3,
                "confidence": 88.0,
            }
        ]
    )

    evidence = (
        collect_structural_evidence(
            current_price=103.0,
            current_atr=2.0,
            daily_frame=daily,
            weekly_frame=weekly,
            daily_zones=zones,
            daily_trendlines=trendlines,
        )
    )

    result = build_confluence_zones(
        evidence,
        current_price=103.0,
        current_atr=2.0,
    )

    assert not result.empty

    cluster = result.iloc[0]

    assert (
        cluster["family_count"]
        == 3
    )

    assert (
        cluster["timeframe_count"]
        == 2
    )

    assert (
        cluster["strength"]
        == "STRONG"
    )


def test_broken_trendline_is_ignored():
    trendlines = pd.DataFrame(
        [
            {
                "kind": "SUPPORT",
                "direction": "RISING",
                "status": "BROKEN_DOWN",
                "line_price_now": 100.0,
                "confidence": 95.0,
            }
        ]
    )

    evidence = (
        collect_structural_evidence(
            current_price=103.0,
            current_atr=2.0,
            daily_trendlines=trendlines,
        )
    )

    assert evidence.empty