import numpy as np
import pandas as pd

from src.features.velocity import (
    build_velocity_snapshot,
    calculate_velocity_features,
    directional_efficiency_series,
    time_to_gain_distribution,
)


def make_from_close(
    close: np.ndarray,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2025-01-01",
        periods=len(close),
    )

    close = np.asarray(
        close,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "open":
                close.copy(),

            "high":
                close * 1.01,

            "low":
                close * 0.99,

            "close":
                close,

            "volume":
                np.full(
                    len(close),
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def test_fast_stock_reaches_10pct_sooner():
    fast_close = (
        100.0
        * (
            1.02
            ** np.arange(120)
        )
    )

    slow_close = (
        100.0
        * (
            1.005
            ** np.arange(120)
        )
    )

    fast = time_to_gain_distribution(
        pd.Series(
            fast_close
        ),
        target_return=0.10,
        max_horizon=20,
    )

    slow = time_to_gain_distribution(
        pd.Series(
            slow_close
        ),
        target_return=0.10,
        max_horizon=20,
    )

    assert (
        fast[
            "median_bars"
        ]
        <
        slow[
            "median_bars"
        ]
    )


def test_directional_efficiency_rewards_clean_trend():
    trend = pd.Series(
        np.linspace(
            100.0,
            130.0,
            60,
        )
    )

    choppy = pd.Series(
        np.tile(
            [
                100.0,
                110.0,
                100.0,
                110.0,
            ],
            15,
        )
    )

    trend_efficiency = (
        directional_efficiency_series(
            trend,
            window=20,
        )
        .iloc[-1]
    )

    choppy_efficiency = (
        directional_efficiency_series(
            choppy,
            window=20,
        )
        .iloc[-1]
    )

    assert (
        trend_efficiency
        >
        choppy_efficiency
    )


def test_impulse_frequency_detects_explosive_character():
    close = np.full(
        180,
        100.0,
    )

    #
    # Repeated structural impulse cycles.
    #
    for start in [
        30,
        60,
        90,
        120,
        150,
    ]:
        close[
            start:
            start + 6
        ] = np.linspace(
            100.0,
            115.0,
            6,
        )

        close[
            start + 6:
            start + 12
        ] = np.linspace(
            115.0,
            102.0,
            6,
        )

    frame = make_from_close(
        close
    )

    result = (
        calculate_velocity_features(
            frame,
            as_of=frame.index[-1],
        )
    )

    assert (
        result[
            "impulse_frequency"
        ]
        > 0
    )

    assert (
        result[
            "p90_return_5d"
        ]
        > 0
    )


def test_future_spike_is_not_visible():
    close = np.linspace(
        100.0,
        120.0,
        120,
    )

    frame = make_from_close(
        close
    )

    as_of = frame.index[-2]

    #
    # Massive event AFTER as_of.
    #
    frame.iloc[
        -1,
        frame.columns.get_loc(
            "open"
        ),
    ] = 200.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = 204.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = 198.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = 202.0

    with_future = (
        calculate_velocity_features(
            frame,
            as_of=as_of,
        )
    )

    truncated = (
        calculate_velocity_features(
            frame.iloc[:-1],
            as_of=as_of,
        )
    )

    assert np.isclose(
        with_future[
            "historical_velocity_raw"
        ],
        truncated[
            "historical_velocity_raw"
        ],
        equal_nan=True,
    )


def test_velocity_decay_detects_slowdown():
    #
    # Fast historical phase.
    #
    first = (
        100.0
        * (
            1.012
            ** np.arange(100)
        )
    )

    #
    # Slow recent phase.
    #
    second = (
        first[-1]
        * (
            1.001
            ** np.arange(
                1,
                31,
            )
        )
    )

    close = np.concatenate(
        [
            first,
            second,
        ]
    )

    frame = make_from_close(
        close
    )

    result = (
        calculate_velocity_features(
            frame,
            as_of=frame.index[-1],
            recent_days=20,
        )
    )

    assert (
        result[
            "velocity_decay_ratio"
        ]
        < 1.0
    )


def test_cross_sectional_velocity_rank():
    strong_close = (
        100.0
        * (
            1.015
            ** np.arange(120)
        )
    )

    medium_close = (
        100.0
        * (
            1.007
            ** np.arange(120)
        )
    )

    weak_close = (
        100.0
        * (
            1.001
            ** np.arange(120)
        )
    )

    universe = {
        "STRONG":
            make_from_close(
                strong_close
            ),

        "MEDIUM":
            make_from_close(
                medium_close
            ),

        "WEAK":
            make_from_close(
                weak_close
            ),
    }

    as_of = (
        universe[
            "STRONG"
        ].index[-1]
    )

    result = (
        build_velocity_snapshot(
            universe,
            as_of=as_of,
        )
        .set_index(
            "symbol"
        )
    )

    assert (
        result.loc[
            "STRONG",
            "historical_velocity_pct",
        ]
        >
        result.loc[
            "MEDIUM",
            "historical_velocity_pct",
        ]
    )

    assert (
        result.loc[
            "MEDIUM",
            "historical_velocity_pct",
        ]
        >
        result.loc[
            "WEAK",
            "historical_velocity_pct",
        ]
    )


def test_productive_volatility_exists():
    close = (
        100.0
        * (
            1.01
            ** np.arange(100)
        )
    )

    frame = make_from_close(
        close
    )

    result = (
        calculate_velocity_features(
            frame,
            as_of=frame.index[-1],
        )
    )

    assert (
        result[
            "productive_volatility_raw"
        ]
        > 0
    )

    assert (
        0
        <= result[
            "directional_efficiency_20"
        ]
        <= 1
    )


def test_velocity_snapshot_has_required_columns():
    close = (
        100.0
        * (
            1.01
            ** np.arange(120)
        )
    )

    frame = make_from_close(
        close
    )

    result = (
        build_velocity_snapshot(
            {
                "AAA": frame,
            },
            as_of=frame.index[-1],
        )
    )

    required = {
        "velocity_5d_pct",
        "velocity_10d_pct",
        "velocity_20d_pct",
        "historical_velocity_pct",
        "impulse_frequency_pct",
        "productive_volatility_pct",
        "velocity_decay_ratio",
        "median_bars_to_10pct",
        "median_bars_to_20pct",
    }

    assert required.issubset(
        result.columns
    )