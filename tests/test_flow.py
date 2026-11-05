import numpy as np
import pandas as pd

from src.features.flow import (
    add_gap_features,
    add_volume_dryup,
    build_daily_flow_features,
)


def make_base_data(
    periods: int = 60,
) -> pd.DataFrame:

    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.linspace(
        100.0,
        110.0,
        periods,
    )

    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(
                periods,
                1_000_000.0,
            ),
        },
        index=dates,
    )


def test_relative_volume_excludes_today():
    frame = make_base_data()

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "volume"
        ),
    ] = 2_000_000

    result = (
        build_daily_flow_features(
            frame
        )
    )

    assert (
        result[
            "relative_volume_20"
        ].iloc[-1]
        == 2.0
    )


def test_close_location_near_high():
    frame = make_base_data()

    position = -1

    frame.iloc[
        position,
        frame.columns.get_loc("low"),
    ] = 100.0

    frame.iloc[
        position,
        frame.columns.get_loc("high"),
    ] = 110.0

    frame.iloc[
        position,
        frame.columns.get_loc("close"),
    ] = 109.0

    result = (
        build_daily_flow_features(
            frame
        )
    )

    assert (
        result["clv_01"].iloc[-1]
        > 0.85
    )


def test_accumulation_day():
    frame = make_base_data()

    previous_close = float(
        frame["close"].iloc[-2]
    )

    frame.iloc[
        -1,
        frame.columns.get_loc("open"),
    ] = previous_close

    frame.iloc[
        -1,
        frame.columns.get_loc("low"),
    ] = previous_close

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = previous_close * 1.04

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = previous_close * 1.035

    frame.iloc[
        -1,
        frame.columns.get_loc("volume"),
    ] = 1_500_000

    result = (
        build_daily_flow_features(
            frame
        )
    )

    assert bool(
        result[
            "accumulation_day"
        ].iloc[-1]
    )


def test_volume_dryup_uses_prior_baseline():
    frame = make_base_data()

    frame.loc[
        frame.index[-5:],
        "volume",
    ] = 400_000

    result = add_volume_dryup(
        frame,
        recent_days=5,
        baseline_days=20,
    )

    assert (
        result[
            "volume_dryup_ratio"
        ].iloc[-1]
        == 0.4
    )


def test_gap_up_retention():
    frame = make_base_data()

    previous_close = float(
        frame["close"].iloc[-2]
    )

    frame.iloc[
        -1,
        frame.columns.get_loc("open"),
    ] = previous_close * 1.05

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = previous_close * 1.07

    frame.iloc[
        -1,
        frame.columns.get_loc("low"),
    ] = previous_close * 1.04

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = previous_close * 1.06

    result = add_gap_features(
        frame,
        min_abs_gap_pct=0.02,
    )

    assert (
        result[
            "gap_direction"
        ].iloc[-1]
        == "UP"
    )

    assert (
        result[
            "gap_retention"
        ].iloc[-1]
        > 1.0
    )


def test_no_meaningful_gap():
    frame = make_base_data()

    result = add_gap_features(
        frame,
        min_abs_gap_pct=0.02,
    )

    assert (
        result[
            "gap_direction"
        ].iloc[-1]
        == "NONE"
    )

    assert pd.isna(
        result[
            "gap_retention"
        ].iloc[-1]
    )