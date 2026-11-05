import numpy as np
import pandas as pd

from src.features.setup import (
    calculate_setup_features,
    evaluate_contraction,
    evaluate_pivot,
    evaluate_pullback,
    evaluate_reexpansion,
    evaluate_rs_retention,
    evaluate_volume_dryup,
)


def make_price_data(
    periods: int = 80,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.linspace(
        90.0,
        100.0,
        periods,
    )

    return pd.DataFrame(
        {
            "open":
                close * 0.998,

            "high":
                close * 1.01,

            "low":
                close * 0.99,

            "close":
                close,

            "volume":
                np.full(
                    periods,
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def make_swings(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pivot_time":
                    frame.index[45],

                "confirmed_at":
                    frame.index[47],

                "kind":
                    "LOW",

                "price":
                    94.0,

                "atr":
                    2.0,

                "prominence_atr":
                    1.5,
            },
            {
                "pivot_time":
                    frame.index[55],

                "confirmed_at":
                    frame.index[57],

                "kind":
                    "HIGH",

                "price":
                    102.0,

                "atr":
                    2.0,

                "prominence_atr":
                    1.5,
            },
            {
                "pivot_time":
                    frame.index[60],

                "confirmed_at":
                    frame.index[62],

                "kind":
                    "LOW",

                "price":
                    95.0,

                "atr":
                    2.0,

                "prominence_atr":
                    1.5,
            },
            {
                "pivot_time":
                    frame.index[68],

                "confirmed_at":
                    frame.index[70],

                "kind":
                    "HIGH",

                "price":
                    102.5,

                "atr":
                    2.0,

                "prominence_atr":
                    1.5,
            },
        ]
    )


def test_contraction_detects_tightening():
    frame = make_price_data()

    #
    # Make older bars wide.
    #
    frame.loc[
        frame.index[-20:-5],
        "high",
    ] = (
        frame.loc[
            frame.index[-20:-5],
            "close",
        ]
        + 3.0
    )

    frame.loc[
        frame.index[-20:-5],
        "low",
    ] = (
        frame.loc[
            frame.index[-20:-5],
            "close",
        ]
        - 3.0
    )

    #
    # Recent 5 bars tight.
    #
    frame.loc[
        frame.index[-5:],
        "high",
    ] = (
        frame.loc[
            frame.index[-5:],
            "close",
        ]
        + 0.5
    )

    frame.loc[
        frame.index[-5:],
        "low",
    ] = (
        frame.loc[
            frame.index[-5:],
            "close",
        ]
        - 0.5
    )

    result = evaluate_contraction(
        frame
    )

    assert (
        result[
            "range_ratio"
        ]
        < 0.75
    )

    assert (
        result[
            "range_point"
        ]
        == 1
    )


def test_pullback_rewards_higher_low():
    frame = make_price_data()

    swings = make_swings(
        frame
    )

    #
    # Current close roughly 5% below recent high.
    #
    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = 97.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = 98.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = 96.0

    result = evaluate_pullback(
        frame,
        swings,
    )

    assert (
        result[
            "depth_point"
        ]
        == 1
    )

    assert (
        result[
            "low_relation"
        ]
        == "HL"
    )

    assert (
        result[
            "structure_point"
        ]
        == 1
    )


def test_pivot_requires_recent_confirmed_highs():
    frame = make_price_data()

    swings = make_swings(
        frame
    )

    result = evaluate_pivot(
        frame,
        swings,
        current_atr=2.0,
    )

    assert (
        result[
            "pivot_recency_point"
        ]
        == 1
    )

    assert (
        result[
            "pivot_cluster_count"
        ]
        >= 2
    )

    assert (
        result[
            "pivot_cluster_point"
        ]
        == 1
    )


def test_volume_dryup_detected():
    frame = make_price_data()

    frame.loc[
        frame.index[-25:-5],
        "volume",
    ] = 1_000_000.0

    frame.loc[
        frame.index[-5:],
        "volume",
    ] = 500_000.0

    result = evaluate_volume_dryup(
        frame
    )

    assert np.isclose(
        result[
            "volume_dryup_ratio"
        ],
        0.5,
    )

    assert (
        result[
            "volume_dryup_point"
        ]
        == 1
    )


def test_rs_retention():
    result = evaluate_rs_retention(
        {
            "rs_20d_pct":
                85.0,

            "rs_5d_pct":
                65.0,
        }
    )

    assert (
        result[
            "rs_retention_point"
        ]
        == 1
    )


def test_reexpansion_requires_productive_bar():
    frame = make_price_data()

    previous_close = float(
        frame[
            "close"
        ].iloc[-2]
    )

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "open"
        ),
    ] = previous_close

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = (
        previous_close
        - 0.5
    )

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = (
        previous_close
        + 4.0
    )

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = (
        previous_close
        + 3.5
    )

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "volume"
        ),
    ] = 2_000_000.0

    result = evaluate_reexpansion(
        frame
    )

    assert (
        result[
            "reexpansion_range_point"
        ]
        == 1
    )

    assert (
        result[
            "reexpansion_volume_point"
        ]
        == 1
    )


def test_future_unconfirmed_pivot_is_ignored():
    frame = make_price_data()

    swings = make_swings(
        frame
    )

    future = pd.DataFrame(
        [
            {
                "pivot_time":
                    frame.index[-2],

                #
                # Confirmation occurs after as_of.
                #
                "confirmed_at":
                    frame.index[-1]
                    + pd.Timedelta(
                        days=5
                    ),

                "kind":
                    "HIGH",

                "price":
                    110.0,

                "atr":
                    2.0,

                "prominence_atr":
                    3.0,
            }
        ]
    )

    swings = pd.concat(
        [
            swings,
            future,
        ],
        ignore_index=True,
    )

    result = (
        calculate_setup_features(
            frame,
            as_of=frame.index[-1],
            swings=swings,
            rs_row={
                "rs_20d_pct":
                    90.0,

                "rs_5d_pct":
                    70.0,
            },
        )
    )

    assert (
        result[
            "pivot_price"
        ]
        != 110.0
    )


def test_setup_gate_uses_five_points():
    frame = make_price_data()

    swings = make_swings(
        frame
    )

    #
    # Create clear volume dry-up.
    #
    frame.loc[
        frame.index[-25:-5],
        "volume",
    ] = 1_000_000.0

    frame.loc[
        frame.index[-5:],
        "volume",
    ] = 500_000.0

    result = (
        calculate_setup_features(
            frame,
            as_of=frame.index[-1],
            swings=swings,
            rs_row={
                "rs_20d_pct":
                    90.0,

                "rs_5d_pct":
                    70.0,
            },
        )
    )

    assert (
        0
        <= result[
            "setup_score"
        ]
        <= 10
    )

    assert (
        result[
            "setup_gate_pass"
        ]
        ==
        (
            result[
                "setup_score"
            ]
            >= 5
        )
    )