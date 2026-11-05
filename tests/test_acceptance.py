import numpy as np
import pandas as pd

from src.features.acceptance import (
    add_breakout_features,
    add_breakout_state,
    build_acceptance_features,
    evaluate_zone_interaction,
)


def make_price_data(
    periods: int = 60,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.full(
        periods,
        100.0,
    )

    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": np.full(
                periods,
                101.0,
            ),
            "low": np.full(
                periods,
                99.0,
            ),
            "close": close.copy(),
            "volume": np.full(
                periods,
                1_000_000.0,
            ),
        },
        index=dates,
    )


def test_breakout_probe_without_close_confirmation():
    frame = make_price_data()

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = 105.0

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = 100.5

    result = add_breakout_features(
        frame,
        lookback_days=20,
    )

    assert bool(
        result[
            "breakout_probe"
        ].iloc[-1]
    )

    assert not bool(
        result[
            "breakout_confirmed"
        ].iloc[-1]
    )

    assert bool(
        result[
            "failed_breakout_probe"
        ].iloc[-1]
    )


def test_confirmed_breakout():
    frame = make_price_data()

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = 106.0

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = 105.0

    result = add_breakout_features(
        frame,
        lookback_days=20,
    )

    assert bool(
        result[
            "breakout_confirmed"
        ].iloc[-1]
    )


def test_breakout_event_state_exists():
    frame = make_price_data()

    #
    # Breakout day
    #
    frame.iloc[
        -5,
        frame.columns.get_loc("open"),
    ] = 100.0

    frame.iloc[
        -5,
        frame.columns.get_loc("low"),
    ] = 99.5

    frame.iloc[
        -5,
        frame.columns.get_loc("high"),
    ] = 105.0

    frame.iloc[
        -5,
        frame.columns.get_loc("close"),
    ] = 104.0

    #
    # Follow-through bars
    #
    continuation = [
        {
            "offset": -4,
            "open": 104.0,
            "high": 105.0,
            "low": 103.5,
            "close": 104.5,
        },
        {
            "offset": -3,
            "open": 104.5,
            "high": 105.5,
            "low": 104.0,
            "close": 105.0,
        },
        {
            "offset": -2,
            "open": 105.0,
            "high": 105.2,
            "low": 104.0,
            "close": 104.5,
        },
        {
            "offset": -1,
            "open": 104.5,
            "high": 104.8,
            "low": 103.5,
            "close": 104.0,
        },
    ]

    for bar in continuation:
        position = bar["offset"]

        frame.iloc[
            position,
            frame.columns.get_loc("open"),
        ] = bar["open"]

        frame.iloc[
            position,
            frame.columns.get_loc("high"),
        ] = bar["high"]

        frame.iloc[
            position,
            frame.columns.get_loc("low"),
        ] = bar["low"]

        frame.iloc[
            position,
            frame.columns.get_loc("close"),
        ] = bar["close"]

    result = add_breakout_features(
        frame,
        lookback_days=20,
    )

    result = add_breakout_state(
        result
    )

    assert (
        result[
            "breakout_event_id"
        ].iloc[-1]
        >= 1
    )

    assert (
        result[
            "breakout_state"
        ].iloc[-1]
        in {
            "ACCEPTED",
            "TESTING",
            "FAILED",
        }
    )


def test_build_acceptance_pipeline():
    frame = make_price_data()

    result = build_acceptance_features(
        frame
    )

    expected = {
        "breakout_probe",
        "breakout_confirmed",
        "breakout_state",
        "breakout_retention",
        "progress_retention_10",
    }

    assert expected.issubset(
        result.columns
    )


def test_support_hold():
    frame = make_price_data()

    frame.iloc[
        -1,
        frame.columns.get_loc("open"),
    ] = 100.0

    frame.iloc[
        -1,
        frame.columns.get_loc("low"),
    ] = 98.8

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = 103.0

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = 102.5

    zones = pd.DataFrame(
        [
            {
                "zone_low": 98.5,
                "zone_high": 100.0,
                "zone_center": 99.25,
                "role": "SUPPORT",
                "confidence": 90.0,
                "distance_from_price_atr": 1.0,
            }
        ]
    )

    result = evaluate_zone_interaction(
        frame,
        zones,
        current_atr=2.0,
        min_relative_volume=0.5,
    )

    assert result[
        "support_test"
    ]

    assert result[
        "support_hold"
    ]


def test_resistance_rejection():
    frame = make_price_data()

    frame.iloc[
        -1,
        frame.columns.get_loc("open"),
    ] = 101.0

    frame.iloc[
        -1,
        frame.columns.get_loc("low"),
    ] = 99.0

    frame.iloc[
        -1,
        frame.columns.get_loc("high"),
    ] = 102.5

    frame.iloc[
        -1,
        frame.columns.get_loc("close"),
    ] = 99.2

    zones = pd.DataFrame(
        [
            {
                "zone_low": 101.0,
                "zone_high": 102.0,
                "zone_center": 101.5,
                "role": "RESISTANCE",
                "confidence": 90.0,
                "distance_from_price_atr": 1.0,
            }
        ]
    )

    result = evaluate_zone_interaction(
        frame,
        zones,
        current_atr=2.0,
    )

    assert result[
        "resistance_test"
    ]

    assert result[
        "resistance_rejection"
    ]