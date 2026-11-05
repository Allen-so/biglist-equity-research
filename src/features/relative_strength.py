from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.basic import normalize_ohlcv


DEFAULT_RS_WINDOWS = (
    5,
    10,
    20,
    60,
)

DEFAULT_PERSISTENT_WEIGHTS = {
    5: 0.15,
    10: 0.20,
    20: 0.30,
    60: 0.35,
}


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return only information available at `as_of`.

    Future bars are always discarded.
    """

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return df[
        df.index <= as_of
    ].copy()


def return_over_sessions(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    sessions: int,
) -> float:
    """
    Point-to-point return over an exact number
    of trading observations.

    Example:
        sessions = 20

    compares:
        current close
        vs
        close 20 completed observations ago.
    """

    if sessions < 1:
        raise ValueError(
            "sessions must be >= 1."
        )

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    required_rows = sessions + 1

    if len(visible) < required_rows:
        return np.nan

    current_close = float(
        visible["close"].iloc[-1]
    )

    previous_close = float(
        visible[
            "close"
        ].iloc[-required_rows]
    )

    if previous_close <= 0:
        return np.nan

    return (
        current_close
        / previous_close
        - 1.0
    )


def _percentile_rank(
    series: pd.Series,
) -> pd.Series:
    """
    Cross-sectional percentile rank.

    Highest valid value receives 100.

    Missing observations remain missing.
    """

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
            method="average",
            pct=True,
            ascending=True,
        )
        * 100.0
    )

    result.loc[
        valid.index
    ] = ranked

    return result


def relative_total_return(
    stock_return: float,
    benchmark_return: float,
) -> float:
    """
    True multiplicative relative return.

    This is preferable to simply:

        stock_return - benchmark_return

    Example:
        stock +20%
        market +10%

        relative =
        1.20 / 1.10 - 1
    """

    if (
        pd.isna(stock_return)
        or pd.isna(benchmark_return)
    ):
        return np.nan

    denominator = (
        1.0 + benchmark_return
    )

    if denominator <= 0:
        return np.nan

    return (
        (1.0 + stock_return)
        / denominator
        - 1.0
    )


def build_raw_return_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
    windows: Sequence[int] = (
        DEFAULT_RS_WINDOWS
    ),
) -> pd.DataFrame:
    """
    Build raw stock returns for the entire Universe
    at one single as-of timestamp.

    No ranking occurs until all stocks have been
    evaluated.
    """

    records: list[dict] = []

    for symbol, frame in universe.items():
        record: dict[str, float | str] = {
            "symbol": str(symbol),
        }

        for window in windows:
            record[
                f"return_{window}d"
            ] = return_over_sessions(
                frame,
                as_of=as_of,
                sessions=window,
            )

        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=["symbol"]
        )

    return (
        pd.DataFrame(records)
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def _benchmark_returns(
    benchmarks: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
    windows: Sequence[int],
) -> dict[
    str,
    dict[int, float],
]:
    results: dict[
        str,
        dict[int, float],
    ] = {}

    for benchmark_name, frame in (
        benchmarks.items()
    ):
        results[
            benchmark_name
        ] = {}

        for window in windows:
            results[
                benchmark_name
            ][window] = (
                return_over_sessions(
                    frame,
                    as_of=as_of,
                    sessions=window,
                )
            )

    return results


def build_rs_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
    benchmarks: Mapping[
        str,
        pd.DataFrame,
    ] | None = None,
    windows: Sequence[int] = (
        DEFAULT_RS_WINDOWS
    ),
    persistent_weights: Mapping[
        int,
        float,
    ] | None = None,
    strong_percentile: float = 75.0,
    elite_percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Build one complete cross-sectional RS snapshot.

    RS percentile is based on stock performance
    relative to the OTHER STOCKS in the Universe.

    Benchmark-relative performance is stored
    separately.
    """

    snapshot = (
        build_raw_return_snapshot(
            universe,
            as_of=as_of,
            windows=windows,
        )
    )

    if snapshot.empty:
        return snapshot

    weights = dict(
        persistent_weights
        or DEFAULT_PERSISTENT_WEIGHTS
    )

    total_weight = sum(
        weights.get(
            window,
            0.0,
        )
        for window in windows
    )

    if not np.isclose(
        total_weight,
        1.0,
    ):
        raise ValueError(
            "Persistent RS weights across selected "
            f"windows must total 1.0. "
            f"Current total: {total_weight}"
        )

    #
    # Cross-sectional ranking.
    #
    for window in windows:
        return_column = (
            f"return_{window}d"
        )

        rs_column = (
            f"rs_{window}d_pct"
        )

        snapshot[
            rs_column
        ] = _percentile_rank(
            snapshot[
                return_column
            ]
        )

    #
    # Persistent RS requires ALL configured
    # RS horizons to be available.
    #
    persistent = pd.Series(
        0.0,
        index=snapshot.index,
        dtype=float,
    )

    complete = pd.Series(
        True,
        index=snapshot.index,
        dtype=bool,
    )

    for window in windows:
        rs_column = (
            f"rs_{window}d_pct"
        )

        complete &= (
            snapshot[
                rs_column
            ].notna()
        )

        persistent += (
            snapshot[
                rs_column
            ].fillna(0.0)
            * weights[window]
        )

    persistent.loc[
        ~complete
    ] = np.nan

    snapshot[
        "persistent_rs"
    ] = persistent

    rs_columns = [
        f"rs_{window}d_pct"
        for window in windows
    ]

    snapshot[
        "strong_rs_window_count"
    ] = (
        snapshot[
            rs_columns
        ]
        .ge(
            strong_percentile
        )
        .sum(
            axis=1
        )
    )

    snapshot[
        "elite_rs_window_count"
    ] = (
        snapshot[
            rs_columns
        ]
        .ge(
            elite_percentile
        )
        .sum(
            axis=1
        )
    )

    #
    # Market-relative performance.
    #
    # These are NOT cross-sectional RS ranks.
    #
    if benchmarks:
        benchmark_data = (
            _benchmark_returns(
                benchmarks,
                as_of=as_of,
                windows=windows,
            )
        )

        for benchmark_name, returns in (
            benchmark_data.items()
        ):
            safe_name = (
                str(
                    benchmark_name
                )
                .strip()
                .lower()
            )

            for window in windows:
                benchmark_return = (
                    returns[window]
                )

                stock_column = (
                    f"return_{window}d"
                )

                output_column = (
                    f"relative_"
                    f"vs_{safe_name}_"
                    f"{window}d"
                )

                snapshot[
                    output_column
                ] = snapshot[
                    stock_column
                ].apply(
                    lambda stock_return:
                        relative_total_return(
                            stock_return,
                            benchmark_return,
                        )
                )

    snapshot[
        "as_of"
    ] = pd.Timestamp(as_of)

    return (
        snapshot
        .sort_values(
            [
                "persistent_rs",
                "rs_20d_pct",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def add_rank_delta(
    current_snapshot: pd.DataFrame,
    past_snapshot: pd.DataFrame,
    *,
    windows: Sequence[int] = (
        20,
        60,
    ),
) -> pd.DataFrame:
    """
    Compare today's CROSS-SECTIONAL rank with a
    previous cross-sectional snapshot.

    This is the correct way to calculate Rank Delta.

    Do NOT attempt to calculate rank delta from
    one stock in isolation.
    """

    if current_snapshot.empty:
        return current_snapshot.copy()

    current = (
        current_snapshot.copy()
    )

    past = (
        past_snapshot.copy()
    )

    if "symbol" not in current.columns:
        raise ValueError(
            "Current snapshot missing symbol."
        )

    if "symbol" not in past.columns:
        raise ValueError(
            "Past snapshot missing symbol."
        )

    columns = [
        "symbol",
    ]

    for window in windows:
        column = (
            f"rs_{window}d_pct"
        )

        if column not in past.columns:
            raise ValueError(
                f"Past snapshot missing {column}."
            )

        columns.append(column)

    past_subset = (
        past[
            columns
        ]
        .copy()
    )

    rename_map = {
        f"rs_{window}d_pct":
            f"past_rs_{window}d_pct"
        for window in windows
    }

    past_subset = (
        past_subset.rename(
            columns=rename_map
        )
    )

    result = current.merge(
        past_subset,
        on="symbol",
        how="left",
    )

    for window in windows:
        current_column = (
            f"rs_{window}d_pct"
        )

        past_column = (
            f"past_rs_{window}d_pct"
        )

        result[
            f"rs_{window}d_rank_delta"
        ] = (
            result[
                current_column
            ]
            -
            result[
                past_column
            ]
        )

    return result


def rank_level_score(
    snapshot: pd.DataFrame,
) -> pd.Series:
    """
    Original Big List definition:

        40% RS20
        60% RS60

    Output scale:
        0–100
    """

    required = {
        "rs_20d_pct",
        "rs_60d_pct",
    }

    missing = required.difference(
        snapshot.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return (
        0.40
        * snapshot[
            "rs_20d_pct"
        ]
        +
        0.60
        * snapshot[
            "rs_60d_pct"
        ]
    )