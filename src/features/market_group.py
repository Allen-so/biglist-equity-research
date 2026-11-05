from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.features.basic import normalize_ohlcv
from src.features.relative_strength import (
    relative_total_return,
    return_over_sessions,
)


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


def _visible_close(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.Series:
    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return (
        df.loc[
            df.index <= as_of,
            "close",
        ]
        .astype(float)
        .copy()
    )


def stress_relative_strength(
    stock_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    lookback_sessions: int = 60,
    min_stress_sessions: int = 5,
) -> dict[str, float]:
    """
    Measure stock behaviour specifically on benchmark
    DOWN days.

    Stress Relative Strength asks:

        When SPY is under pressure,
        does this stock lose less,
        hold flat,
        or even rise?

    Only aligned trading dates are used.
    Future data is discarded.
    """

    stock_close = _visible_close(
        stock_frame,
        as_of=as_of,
    )

    benchmark_close = _visible_close(
        benchmark_frame,
        as_of=as_of,
    )

    aligned = pd.concat(
        {
            "stock": stock_close,
            "benchmark": benchmark_close,
        },
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        return {
            "stress_stock_return": np.nan,
            "stress_benchmark_return": np.nan,
            "stress_excess_return": np.nan,
            "stress_day_count": 0,
        }

    returns = aligned.pct_change(
        fill_method=None
    ).dropna()

    returns = returns.tail(
        lookback_sessions
    )

    stress = returns[
        returns["benchmark"] < 0
    ]

    stress_day_count = len(
        stress
    )

    if (
        stress_day_count
        < min_stress_sessions
    ):
        return {
            "stress_stock_return": np.nan,
            "stress_benchmark_return": np.nan,
            "stress_excess_return": np.nan,
            "stress_day_count":
                stress_day_count,
        }

    stock_growth = float(
        (
            1.0
            + stress["stock"]
        ).prod()
    )

    benchmark_growth = float(
        (
            1.0
            + stress["benchmark"]
        ).prod()
    )

    stress_stock_return = (
        stock_growth - 1.0
    )

    stress_benchmark_return = (
        benchmark_growth - 1.0
    )

    stress_excess_return = (
        relative_total_return(
            stress_stock_return,
            stress_benchmark_return,
        )
    )

    return {
        "stress_stock_return":
            stress_stock_return,

        "stress_benchmark_return":
            stress_benchmark_return,

        "stress_excess_return":
            stress_excess_return,

        "stress_day_count":
            stress_day_count,
    }


def build_market_fit_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    benchmark_frame: pd.DataFrame,
    as_of: pd.Timestamp,
    rs_snapshot: pd.DataFrame | None = None,
    stress_lookback_sessions: int = 60,
    min_stress_sessions: int = 5,
    stress_weight: float = 0.60,
    excess_20d_weight: float = 0.40,
) -> pd.DataFrame:
    """
    Build cross-sectional Market Fit.

    Components:

        60% Stress Relative Strength
        40% 20D Excess Return vs SPY

    Output remains 0-100.

    This is not yet converted to the final
    Big List 4-point factor.
    """

    if not np.isclose(
        stress_weight
        + excess_20d_weight,
        1.0,
    ):
        raise ValueError(
            "Market Fit weights must total 1.0."
        )

    benchmark_return_20 = (
        return_over_sessions(
            benchmark_frame,
            as_of=as_of,
            sessions=20,
        )
    )

    records = []

    for symbol, frame in (
        universe.items()
    ):
        stress = (
            stress_relative_strength(
                frame,
                benchmark_frame,
                as_of=as_of,
                lookback_sessions=(
                    stress_lookback_sessions
                ),
                min_stress_sessions=(
                    min_stress_sessions
                ),
            )
        )

        stock_return_20 = (
            return_over_sessions(
                frame,
                as_of=as_of,
                sessions=20,
            )
        )

        excess_20 = (
            relative_total_return(
                stock_return_20,
                benchmark_return_20,
            )
        )

        records.append(
            {
                "symbol":
                    str(symbol),

                **stress,

                "relative_vs_spy_20d":
                    excess_20,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
            ]
        )

    result = pd.DataFrame(
        records
    )

    #
    # If a previously calculated RS snapshot already
    # contains SPY-relative 20D return, prefer it.
    #
    if (
        rs_snapshot is not None
        and not rs_snapshot.empty
        and "relative_vs_spy_20d"
        in rs_snapshot.columns
    ):
        replacement = (
            rs_snapshot[
                [
                    "symbol",
                    "relative_vs_spy_20d",
                ]
            ]
            .rename(
                columns={
                    "relative_vs_spy_20d":
                        "_rs_excess_20d"
                }
            )
        )

        result = result.merge(
            replacement,
            on="symbol",
            how="left",
        )

        result[
            "relative_vs_spy_20d"
        ] = result[
            "_rs_excess_20d"
        ].combine_first(
            result[
                "relative_vs_spy_20d"
            ]
        )

        result = result.drop(
            columns=[
                "_rs_excess_20d",
            ]
        )

    result[
        "stress_rs_pct"
    ] = _percentile_rank(
        result[
            "stress_excess_return"
        ]
    )

    result[
        "excess_20d_pct"
    ] = _percentile_rank(
        result[
            "relative_vs_spy_20d"
        ]
    )

    complete = (
        result[
            [
                "stress_rs_pct",
                "excess_20d_pct",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "market_fit_pct"
    ] = np.nan

    result.loc[
        complete,
        "market_fit_pct",
    ] = (
        stress_weight
        * result.loc[
            complete,
            "stress_rs_pct",
        ]
        +
        excess_20d_weight
        * result.loc[
            complete,
            "excess_20d_pct",
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
            "market_fit_pct",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def _prepare_metadata(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "symbol",
        "industry",
    }

    missing = required.difference(
        metadata.columns
    )

    if missing:
        raise ValueError(
            "Metadata missing columns: "
            f"{sorted(missing)}"
        )

    result = metadata[
        [
            "symbol",
            "industry",
        ]
    ].copy()

    result["symbol"] = (
        result["symbol"]
        .astype(str)
    )

    result["industry"] = (
        result["industry"]
        .astype("string")
        .str.strip()
    )

    result.loc[
        result["industry"] == "",
        "industry",
    ] = pd.NA

    return (
        result
        .drop_duplicates(
            subset=[
                "symbol",
            ],
            keep="last",
        )
    )


def build_industry_momentum(
    rs_snapshot: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    minimum_group_size: int = 2,
) -> pd.DataFrame:
    """
    Objective Group / Industry momentum.

    For every industry:

        20% median 5D return
        40% median 20D return
        40% median 60D return

    Then rank industries cross-sectionally.
    """

    required_rs = {
        "symbol",
        "return_5d",
        "return_20d",
        "return_60d",
    }

    missing = required_rs.difference(
        rs_snapshot.columns
    )

    if missing:
        raise ValueError(
            "RS snapshot missing columns: "
            f"{sorted(missing)}"
        )

    meta = _prepare_metadata(
        metadata
    )

    merged = rs_snapshot.merge(
        meta,
        on="symbol",
        how="left",
    )

    usable = merged[
        merged["industry"].notna()
    ].copy()

    if usable.empty:
        return pd.DataFrame(
            columns=[
                "industry",
                "group_size",
                "industry_return_5d",
                "industry_return_20d",
                "industry_return_60d",
                "industry_momentum_raw",
                "industry_momentum_pct",
            ]
        )

    grouped = (
        usable
        .groupby(
            "industry",
            dropna=True,
        )
        .agg(
            group_size=(
                "symbol",
                "nunique",
            ),

            industry_return_5d=(
                "return_5d",
                "median",
            ),

            industry_return_20d=(
                "return_20d",
                "median",
            ),

            industry_return_60d=(
                "return_60d",
                "median",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped[
            "group_size"
        ]
        >= minimum_group_size
    ].copy()

    if grouped.empty:
        grouped[
            "industry_momentum_raw"
        ] = np.nan

        grouped[
            "industry_momentum_pct"
        ] = np.nan

        return grouped

    grouped[
        "industry_momentum_raw"
    ] = (
        0.20
        * grouped[
            "industry_return_5d"
        ]
        +
        0.40
        * grouped[
            "industry_return_20d"
        ]
        +
        0.40
        * grouped[
            "industry_return_60d"
        ]
    )

    grouped[
        "industry_momentum_pct"
    ] = _percentile_rank(
        grouped[
            "industry_momentum_raw"
        ]
    )

    return (
        grouped
        .sort_values(
            "industry_momentum_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def add_group_fit(
    rs_snapshot: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    minimum_group_size: int = 2,
) -> pd.DataFrame:
    """
    Attach objective Industry Momentum
    back to every stock.
    """

    meta = _prepare_metadata(
        metadata
    )

    groups = build_industry_momentum(
        rs_snapshot,
        metadata,
        minimum_group_size=(
            minimum_group_size
        ),
    )

    result = rs_snapshot.merge(
        meta,
        on="symbol",
        how="left",
    )

    result = result.merge(
        groups,
        on="industry",
        how="left",
    )

    result[
        "group_fit_pct"
    ] = result[
        "industry_momentum_pct"
    ]

    return result


def build_within_group_leadership(
    rs_snapshot: pd.DataFrame,
    velocity_snapshot: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    minimum_group_size: int = 2,
) -> pd.DataFrame:
    """
    Determine which stocks are the real leaders
    INSIDE their own Industry.

    Leadership raw composite:

        40% global RS20 percentile
        35% global RS60 percentile
        25% Historical Velocity percentile

    The resulting composite is then ranked again
    within each Industry.
    """

    required_rs = {
        "symbol",
        "rs_20d_pct",
        "rs_60d_pct",
    }

    missing_rs = (
        required_rs.difference(
            rs_snapshot.columns
        )
    )

    if missing_rs:
        raise ValueError(
            "RS snapshot missing columns: "
            f"{sorted(missing_rs)}"
        )

    required_velocity = {
        "symbol",
        "historical_velocity_pct",
    }

    missing_velocity = (
        required_velocity.difference(
            velocity_snapshot.columns
        )
    )

    if missing_velocity:
        raise ValueError(
            "Velocity snapshot missing columns: "
            f"{sorted(missing_velocity)}"
        )

    meta = _prepare_metadata(
        metadata
    )

    result = (
        rs_snapshot[
            [
                "symbol",
                "rs_20d_pct",
                "rs_60d_pct",
            ]
        ]
        .merge(
            velocity_snapshot[
                [
                    "symbol",
                    "historical_velocity_pct",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            meta,
            on="symbol",
            how="left",
        )
    )

    complete = (
        result[
            [
                "rs_20d_pct",
                "rs_60d_pct",
                "historical_velocity_pct",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    result[
        "leadership_raw"
    ] = np.nan

    result.loc[
        complete,
        "leadership_raw",
    ] = (
        0.40
        * result.loc[
            complete,
            "rs_20d_pct",
        ]
        +
        0.35
        * result.loc[
            complete,
            "rs_60d_pct",
        ]
        +
        0.25
        * result.loc[
            complete,
            "historical_velocity_pct",
        ]
    )

    result[
        "within_group_leadership_pct"
    ] = np.nan

    for industry, group in (
        result.groupby(
            "industry",
            dropna=True,
        )
    ):
        if (
            group["symbol"].nunique()
            < minimum_group_size
        ):
            continue

        ranks = _percentile_rank(
            group[
                "leadership_raw"
            ]
        )

        result.loc[
            group.index,
            "within_group_leadership_pct",
        ] = ranks

    return (
        result
        .sort_values(
            [
                "industry",
                "within_group_leadership_pct",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def build_market_group_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    benchmark_frame: pd.DataFrame,
    as_of: pd.Timestamp,
    rs_snapshot: pd.DataFrame,
    velocity_snapshot: pd.DataFrame,
    metadata: pd.DataFrame,
    minimum_group_size: int = 2,
) -> pd.DataFrame:
    """
    Unified Phase 4C snapshot.

    Contains:

        Market Fit
        Group Fit
        Within-Group Leadership
    """

    market = (
        build_market_fit_snapshot(
            universe,
            benchmark_frame=(
                benchmark_frame
            ),
            as_of=as_of,
            rs_snapshot=(
                rs_snapshot
            ),
        )
    )

    group_fit = add_group_fit(
        rs_snapshot,
        metadata,
        minimum_group_size=(
            minimum_group_size
        ),
    )

    leadership = (
        build_within_group_leadership(
            rs_snapshot,
            velocity_snapshot,
            metadata,
            minimum_group_size=(
                minimum_group_size
            ),
        )
    )

    group_columns = [
        "symbol",
        "industry",
        "group_size",
        "industry_momentum_raw",
        "industry_momentum_pct",
        "group_fit_pct",
    ]

    leadership_columns = [
        "symbol",
        "leadership_raw",
        "within_group_leadership_pct",
    ]

    result = market.merge(
        group_fit[
            group_columns
        ],
        on="symbol",
        how="left",
    )

    result = result.merge(
        leadership[
            leadership_columns
        ],
        on="symbol",
        how="left",
    )

    return (
        result
        .sort_values(
            [
                "market_fit_pct",
                "group_fit_pct",
                "within_group_leadership_pct",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )