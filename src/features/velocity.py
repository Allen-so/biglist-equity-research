from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)


IMPULSE_WINDOWS = (
    5,
    10,
    20,
)

IMPULSE_WEIGHTS = {
    5: 0.35,
    10: 0.35,
    20: 0.30,
}


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Strict as-of protection.
    """

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return df[
        df.index <= as_of
    ].copy()


def _percentile_rank(
    series: pd.Series,
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = series.dropna()

    if valid.empty:
        return result

    ranked = (
        valid.rank(
            pct=True,
            method="average",
        )
        * 100.0
    )

    result.loc[
        valid.index
    ] = ranked

    return result


def rolling_returns(
    close: pd.Series,
    *,
    window: int,
) -> pd.Series:
    if window < 1:
        raise ValueError(
            "window must be >= 1."
        )

    return close.pct_change(
        periods=window,
        fill_method=None,
    )


def directional_efficiency_series(
    close: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """
    Upside directional efficiency.

    1.0:
        nearly all travelled distance produces
        net upward progress.

    0.0:
        price moved around but produced no
        meaningful upside progress.

    Formula:

        max(
            Close[t] - Close[t-window],
            0
        )
        /
        sum(
            absolute bar-to-bar movement
        )
    """

    if window < 1:
        raise ValueError(
            "window must be >= 1."
        )

    net_progress = (
        close
        - close.shift(window)
    ).clip(
        lower=0.0
    )

    travelled_distance = (
        close.diff()
        .abs()
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    efficiency = (
        net_progress
        /
        travelled_distance.replace(
            0,
            np.nan,
        )
    )

    return efficiency.clip(
        lower=0.0,
        upper=1.0,
    )


def _quantile_or_nan(
    series: pd.Series,
    q: float,
) -> float:
    valid = series.dropna()

    if valid.empty:
        return np.nan

    return float(
        valid.quantile(q)
    )


def _weighted_velocity(
    values: Mapping[int, float],
    *,
    weights: Mapping[int, float],
) -> float:
    components = []

    for window, weight in (
        weights.items()
    ):
        value = values.get(
            window,
            np.nan,
        )

        if pd.isna(value):
            return np.nan

        components.append(
            float(value)
            * float(weight)
        )

    return float(
        sum(components)
    )


def time_to_gain_distribution(
    close: pd.Series,
    *,
    target_return: float,
    max_horizon: int,
    lookback_starts: int = 252,
) -> dict[str, float]:
    """
    Historical speed-to-target.

    For every eligible historical starting point,
    measure the FIRST future bar where the stock
    reached the target return.

    Only starting points with a COMPLETE future
    horizon are used.

    This avoids right-censoring bias near the
    current date.

    Example:
        target_return = 0.10
        max_horizon = 20

    asks:
        historically, how many trading sessions
        did this stock need to gain 10%?
    """

    if target_return <= 0:
        raise ValueError(
            "target_return must be positive."
        )

    if max_horizon < 1:
        raise ValueError(
            "max_horizon must be >= 1."
        )

    values = close.to_numpy(
        dtype=float
    )

    n = len(values)

    if n <= max_horizon:
        return {
            "median_bars": np.nan,
            "best_bars": np.nan,
            "hit_rate": np.nan,
            "sample_size": 0,
        }

    last_valid_start = (
        n - max_horizon - 1
    )

    first_valid_start = max(
        0,
        last_valid_start
        - lookback_starts
        + 1,
    )

    hit_times: list[int] = []

    sample_size = 0

    for start in range(
        first_valid_start,
        last_valid_start + 1,
    ):
        start_price = values[
            start
        ]

        if (
            not np.isfinite(
                start_price
            )
            or start_price <= 0
        ):
            continue

        sample_size += 1

        target_price = (
            start_price
            * (
                1.0
                + target_return
            )
        )

        future = values[
            start + 1:
            start + max_horizon + 1
        ]

        hits = np.flatnonzero(
            future
            >= target_price
        )

        if len(hits) > 0:
            #
            # +1 because future[0]
            # is one bar after start.
            #
            hit_times.append(
                int(
                    hits[0] + 1
                )
            )

    if sample_size == 0:
        return {
            "median_bars": np.nan,
            "best_bars": np.nan,
            "hit_rate": np.nan,
            "sample_size": 0,
        }

    if not hit_times:
        return {
            "median_bars": np.nan,
            "best_bars": np.nan,
            "hit_rate": 0.0,
            "sample_size": sample_size,
        }

    return {
        "median_bars":
            float(
                np.median(
                    hit_times
                )
            ),

        "best_bars":
            float(
                np.min(
                    hit_times
                )
            ),

        "hit_rate":
            float(
                len(hit_times)
                / sample_size
            ),

        "sample_size":
            sample_size,
    }


def calculate_velocity_features(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    lookback_days: int = 252,
    recent_days: int = 20,
    atr_period: int = 14,
) -> dict[str, float]:
    """
    Calculate single-stock velocity characteristics.

    No cross-sectional ranking occurs here.
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    if visible.empty:
        return {}

    #
    # Keep enough extra bars so early rolling
    # observations inside the 252D window have
    # valid return calculations.
    #
    max_window = max(
        IMPULSE_WINDOWS
    )

    working = visible.tail(
        lookback_days
        + max_window
        + 40
    ).copy()

    close = working["close"]

    return_series = {}

    for window in IMPULSE_WINDOWS:
        return_series[
            window
        ] = rolling_returns(
            close,
            window=window,
        )

    history_index = (
        working.index[
            -min(
                lookback_days,
                len(working),
            ):
        ]
    )

    historical_p90 = {}

    for window in IMPULSE_WINDOWS:
        series = (
            return_series[
                window
            ]
            .loc[
                history_index
            ]
        )

        historical_p90[
            window
        ] = _quantile_or_nan(
            series,
            0.90,
        )

    historical_velocity_raw = (
        _weighted_velocity(
            historical_p90,
            weights=IMPULSE_WEIGHTS,
        )
    )

    #
    # Impulse Frequency
    #
    return_5 = (
        return_series[5]
        .loc[
            history_index
        ]
    )

    return_10 = (
        return_series[10]
        .loc[
            history_index
        ]
    )

    impulse_5_valid = (
        return_5.notna()
    )

    impulse_10_valid = (
        return_10.notna()
    )

    impulse_5_frequency = (
        float(
            (
                return_5[
                    impulse_5_valid
                ]
                >= 0.08
            ).mean()
        )
        if impulse_5_valid.any()
        else np.nan
    )

    impulse_10_frequency = (
        float(
            (
                return_10[
                    impulse_10_valid
                ]
                >= 0.12
            ).mean()
        )
        if impulse_10_valid.any()
        else np.nan
    )

    union_valid = (
        impulse_5_valid
        | impulse_10_valid
    )

    if union_valid.any():
        combined_impulse = (
            return_5.fillna(
                -np.inf
            )
            >= 0.08
        ) | (
            return_10.fillna(
                -np.inf
            )
            >= 0.12
        )

        impulse_frequency = float(
            combined_impulse[
                union_valid
            ].mean()
        )

    else:
        impulse_frequency = (
            np.nan
        )

    #
    # Recent vs prior historical velocity.
    #
    # Recent window and historical baseline
    # are intentionally NON-overlapping.
    #
    recent_p90 = {}
    prior_p90 = {}

    recent_start_position = max(
        len(working)
        - recent_days,
        0,
    )

    for window in IMPULSE_WINDOWS:
        series = return_series[
            window
        ]

        recent_series = (
            series.iloc[
                recent_start_position:
            ]
        )

        prior_series = (
            series.iloc[
                :recent_start_position
            ]
            .tail(
                lookback_days
            )
        )

        recent_p90[
            window
        ] = _quantile_or_nan(
            recent_series,
            0.90,
        )

        prior_p90[
            window
        ] = _quantile_or_nan(
            prior_series,
            0.90,
        )

    recent_velocity_raw = (
        _weighted_velocity(
            recent_p90,
            weights=IMPULSE_WEIGHTS,
        )
    )

    prior_velocity_raw = (
        _weighted_velocity(
            prior_p90,
            weights=IMPULSE_WEIGHTS,
        )
    )

    if (
        not pd.isna(
            recent_velocity_raw
        )
        and
        not pd.isna(
            prior_velocity_raw
        )
        and
        prior_velocity_raw > 0
    ):
        velocity_decay_ratio = (
            recent_velocity_raw
            / prior_velocity_raw
        )

    else:
        velocity_decay_ratio = (
            np.nan
        )

    #
    # Time-to-X%.
    #
    speed_10 = (
        time_to_gain_distribution(
            visible["close"],
            target_return=0.10,
            max_horizon=20,
            lookback_starts=(
                lookback_days
            ),
        )
    )

    speed_20 = (
        time_to_gain_distribution(
            visible["close"],
            target_return=0.20,
            max_horizon=40,
            lookback_starts=(
                lookback_days
            ),
        )
    )

    #
    # Directional Efficiency.
    #
    de20_series = (
        directional_efficiency_series(
            visible["close"],
            window=20,
        )
    )

    current_de20 = (
        de20_series.iloc[-1]
        if not de20_series.empty
        else np.nan
    )

    de10_series = (
        directional_efficiency_series(
            visible["close"],
            window=10,
        )
    )

    de10_history = (
        de10_series.tail(
            lookback_days
        )
    )

    historical_de10_p75 = (
        _quantile_or_nan(
            de10_history,
            0.75,
        )
    )

    #
    # ATR provides volatility.
    #
    # Efficiency tells us whether that volatility
    # actually makes upside progress.
    #
    atr_frame = add_atr(
        visible,
        period=atr_period,
    )

    atr_pct = float(
        atr_frame[
            f"atr_pct_{atr_period}"
        ].iloc[-1]
    )

    if (
        pd.isna(current_de20)
        or pd.isna(
            historical_de10_p75
        )
        or pd.isna(
            atr_pct
        )
    ):
        productive_volatility_raw = (
            np.nan
        )

    else:
        efficiency_composite = (
            0.30
            * float(
                current_de20
            )
            +
            0.70
            * float(
                historical_de10_p75
            )
        )

        productive_volatility_raw = (
            atr_pct
            * efficiency_composite
        )

    return {
        "p90_return_5d":
            historical_p90[5],

        "p90_return_10d":
            historical_p90[10],

        "p90_return_20d":
            historical_p90[20],

        "historical_velocity_raw":
            historical_velocity_raw,

        "impulse_frequency_5d":
            impulse_5_frequency,

        "impulse_frequency_10d":
            impulse_10_frequency,

        "impulse_frequency":
            impulse_frequency,

        "recent_velocity_raw":
            recent_velocity_raw,

        "prior_velocity_raw":
            prior_velocity_raw,

        "velocity_decay_ratio":
            velocity_decay_ratio,

        "median_bars_to_10pct":
            speed_10[
                "median_bars"
            ],

        "best_bars_to_10pct":
            speed_10[
                "best_bars"
            ],

        "hit_rate_10pct":
            speed_10[
                "hit_rate"
            ],

        "median_bars_to_20pct":
            speed_20[
                "median_bars"
            ],

        "best_bars_to_20pct":
            speed_20[
                "best_bars"
            ],

        "hit_rate_20pct":
            speed_20[
                "hit_rate"
            ],

        "directional_efficiency_20":
            (
                float(current_de20)
                if not pd.isna(
                    current_de20
                )
                else np.nan
            ),

        "historical_de10_p75":
            historical_de10_p75,

        "atr_pct":
            atr_pct,

        "productive_volatility_raw":
            productive_volatility_raw,
    }


def build_velocity_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build cross-sectional velocity snapshot
    for the entire Universe.
    """

    records = []

    for symbol, frame in (
        universe.items()
    ):
        features = (
            calculate_velocity_features(
                frame,
                as_of=as_of,
            )
        )

        record = {
            "symbol":
                str(symbol),
            **features,
        }

        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
            ]
        )

    result = pd.DataFrame(
        records
    )

    percentile_columns = {
        "p90_return_5d":
            "velocity_5d_pct",

        "p90_return_10d":
            "velocity_10d_pct",

        "p90_return_20d":
            "velocity_20d_pct",

        "impulse_frequency":
            "impulse_frequency_pct",

        "productive_volatility_raw":
            "productive_volatility_pct",
    }

    for source, output in (
        percentile_columns.items()
    ):
        result[
            output
        ] = _percentile_rank(
            result[source]
        )

    #
    # This is the exact raw basis of our
    # Historical Velocity 8-point factor:
    #
    # 35% 5D
    # 35% 10D
    # 30% 20D
    #
    complete = (
        result[
            [
                "velocity_5d_pct",
                "velocity_10d_pct",
                "velocity_20d_pct",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "historical_velocity_pct"
    ] = np.nan

    result.loc[
        complete,
        "historical_velocity_pct",
    ] = (
        0.35
        * result.loc[
            complete,
            "velocity_5d_pct",
        ]
        +
        0.35
        * result.loc[
            complete,
            "velocity_10d_pct",
        ]
        +
        0.30
        * result.loc[
            complete,
            "velocity_20d_pct",
        ]
    )

    result[
        "as_of"
    ] = pd.Timestamp(
        as_of
    )

    return (
        result
        .sort_values(
            "historical_velocity_pct",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )