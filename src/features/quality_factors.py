from __future__ import annotations

import numpy as np
import pandas as pd


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

    result.loc[
        valid.index
    ] = (
        valid.rank(
            method="average",
            pct=True,
            ascending=True,
        )
        * 100.0
    )

    return result


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
) -> None:
    missing = columns.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing quality-factor columns: "
            f"{sorted(missing)}"
        )


def build_quality_factor_snapshot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert underlying evidence into standardized
    0-100 Quality-factor inputs.

    Expected evidence:

        Historical impulse distribution
        Impulse frequency
        Progress retention

        ATR %
        20D median daily range %

        Up/Down Volume
        Volume-weighted CLV
        Accumulation/Distribution balance

    This function does NOT assign final Big List
    weighted points.
    """

    required = {
        "symbol",

        "p90_return_5d",
        "p90_return_10d",
        "p90_return_20d",

        "impulse_frequency",
        "progress_retention_10",

        "atr_pct",
        "range_pct_20",

        "up_down_volume_ratio_20",
        "volume_weighted_clv_20",
        "accumulation_distribution_balance_20",
    }

    _require_columns(
        snapshot,
        required,
    )

    result = snapshot.copy()

    #
    # ------------------------------------------------
    # MONSTER QUALITY
    # ------------------------------------------------
    #
    # Historical Velocity already asks:
    # "How large are typical strong impulses?"
    #
    # Monster impulse strength asks something slightly
    # different:
    #
    # "At its best, how FAST does this stock move?"
    #
    # Therefore returns are normalized by time:
    #
    # 5D return / 5
    # 10D return / 10
    # 20D return / 20
    #
    # and the strongest historical speed is retained.
    #

    impulse_components = pd.DataFrame(
        {
            "speed_5":
                result[
                    "p90_return_5d"
                ] / 5.0,

            "speed_10":
                result[
                    "p90_return_10d"
                ] / 10.0,

            "speed_20":
                result[
                    "p90_return_20d"
                ] / 20.0,
        },
        index=result.index,
    )

    complete_impulse = (
        impulse_components
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "monster_impulse_speed_raw"
    ] = np.nan

    result.loc[
        complete_impulse,
        "monster_impulse_speed_raw",
    ] = (
        impulse_components.loc[
            complete_impulse
        ]
        .max(
            axis=1
        )
    )

    result[
        "monster_impulse_strength_pct"
    ] = _percentile_rank(
        result[
            "monster_impulse_speed_raw"
        ]
    )

    #
    # Impulse Frequency percentile.
    #
    if (
        "impulse_frequency_pct"
        in result.columns
    ):
        result[
            "monster_impulse_frequency_pct"
        ] = result[
            "impulse_frequency_pct"
        ]

    else:
        result[
            "monster_impulse_frequency_pct"
        ] = _percentile_rank(
            result[
                "impulse_frequency"
            ]
        )

    #
    # Retention percentile.
    #
    result[
        "monster_impulse_retention_pct"
    ] = _percentile_rank(
        result[
            "progress_retention_10"
        ]
    )

    monster_complete = (
        result[
            [
                "monster_impulse_strength_pct",
                "monster_impulse_frequency_pct",
                "monster_impulse_retention_pct",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "monster_quality_pct"
    ] = np.nan

    #
    # Original Monster Quality composition:
    #
    # Impulse Strength   3 / 7
    # Impulse Frequency  2 / 7
    # Impulse Retention  2 / 7
    #
    result.loc[
        monster_complete,
        "monster_quality_pct",
    ] = (
        (3.0 / 7.0)
        * result.loc[
            monster_complete,
            "monster_impulse_strength_pct",
        ]
        +
        (2.0 / 7.0)
        * result.loc[
            monster_complete,
            "monster_impulse_frequency_pct",
        ]
        +
        (2.0 / 7.0)
        * result.loc[
            monster_complete,
            "monster_impulse_retention_pct",
        ]
    )

    #
    # ------------------------------------------------
    # VOLATILITY QUALITY INPUT
    # ------------------------------------------------
    #
    # This is only the cross-sectional volatility
    # opportunity level.
    #
    # The ATR multiplier that penalizes:
    #   too slow
    #   or absurdly unstable
    #
    # is applied later by the Scoring Engine.
    #

    result[
        "atr_pct_rank"
    ] = _percentile_rank(
        result[
            "atr_pct"
        ]
    )

    result[
        "range_pct_20_rank"
    ] = _percentile_rank(
        result[
            "range_pct_20"
        ]
    )

    volatility_complete = (
        result[
            [
                "atr_pct_rank",
                "range_pct_20_rank",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "volatility_pct"
    ] = np.nan

    result.loc[
        volatility_complete,
        "volatility_pct",
    ] = (
        0.60
        * result.loc[
            volatility_complete,
            "atr_pct_rank",
        ]
        +
        0.40
        * result.loc[
            volatility_complete,
            "range_pct_20_rank",
        ]
    )

    #
    # ------------------------------------------------
    # SUPPLY / DEMAND
    # ------------------------------------------------
    #
    # 40% Up / Down Volume
    # 35% Volume-weighted Close Location
    # 25% Accumulation / Distribution balance
    #

    result[
        "up_down_volume_pct"
    ] = _percentile_rank(
        result[
            "up_down_volume_ratio_20"
        ]
    )

    result[
        "volume_weighted_clv_pct"
    ] = _percentile_rank(
        result[
            "volume_weighted_clv_20"
        ]
    )

    result[
        "accumulation_balance_pct"
    ] = _percentile_rank(
        result[
            "accumulation_distribution_balance_20"
        ]
    )

    supply_complete = (
        result[
            [
                "up_down_volume_pct",
                "volume_weighted_clv_pct",
                "accumulation_balance_pct",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "supply_demand_pct"
    ] = np.nan

    result.loc[
        supply_complete,
        "supply_demand_pct",
    ] = (
        0.40
        * result.loc[
            supply_complete,
            "up_down_volume_pct",
        ]
        +
        0.35
        * result.loc[
            supply_complete,
            "volume_weighted_clv_pct",
        ]
        +
        0.25
        * result.loc[
            supply_complete,
            "accumulation_balance_pct",
        ]
    )

    return (
        result
        .sort_values(
            [
                "monster_quality_pct",
                "supply_demand_pct",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )