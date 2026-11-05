from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.basic import (
    add_atr,
    add_true_range,
    normalize_ohlcv,
)
from src.features.flow import (
    add_close_location,
    add_relative_volume,
    add_volume_dryup,
)


SETUP_COMPONENT_COLUMNS = [
    "contraction_atr",
    "contraction_range",
    "pullback_depth",
    "pullback_structure",
    "pivot_recency",
    "pivot_cluster",
    "volume_dryup",
    "rs_retention",
    "reexpansion_range",
    "reexpansion_volume",
]


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return float(
        numerator / denominator
    )


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return df[
        df.index <= as_of
    ].copy()


def _confirmed_swings(
    swings: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp,
    visible_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Only pivots that were already confirmed by as_of
    are allowed to influence Setup.
    """

    if swings is None or swings.empty:
        return pd.DataFrame()

    required = {
        "pivot_time",
        "confirmed_at",
        "kind",
        "price",
        "atr",
    }

    missing = required.difference(
        swings.columns
    )

    if missing:
        raise ValueError(
            "Swings missing columns: "
            f"{sorted(missing)}"
        )

    result = swings.copy()

    result["pivot_time"] = pd.to_datetime(
        result["pivot_time"]
    )

    result["confirmed_at"] = pd.to_datetime(
        result["confirmed_at"]
    )

    as_of = pd.Timestamp(as_of)

    result = result[
        (
            result["confirmed_at"]
            <= as_of
        )
        &
        (
            result["pivot_time"]
            <= as_of
        )
        &
        (
            result["pivot_time"]
            .isin(
                visible_index
            )
        )
    ].copy()

    return (
        result
        .sort_values(
            "pivot_time"
        )
        .reset_index(
            drop=True
        )
    )


def _bar_position(
    index: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
) -> int | None:
    location = index.get_indexer(
        [
            pd.Timestamp(
                timestamp
            )
        ]
    )[0]

    if location < 0:
        return None

    return int(location)


def evaluate_contraction(
    frame: pd.DataFrame,
    *,
    atr_period: int = 14,
    recent_days: int = 5,
    baseline_days: int = 20,
    atr_ratio_max: float = 0.75,
    range_ratio_max: float = 0.75,
) -> dict[str, Any]:
    """
    Contraction = 2 points.

    1 point:
        recent ATR regime contracted.

    1 point:
        recent raw bar range contracted.

    Both ratios intentionally compare recent behaviour
    with the broader current baseline.
    """

    df = frame.copy()

    atr_column = f"atr_{atr_period}"

    if atr_column not in df.columns:
        df = add_atr(
            df,
            period=atr_period,
        )

    raw_range = (
        df["high"]
        - df["low"]
    )

    if len(df) < baseline_days:
        return {
            "atr_ratio":
                np.nan,

            "range_ratio":
                np.nan,

            "atr_point":
                0,

            "range_point":
                0,
        }

    recent_atr = float(
        df[
            atr_column
        ]
        .tail(
            recent_days
        )
        .median()
    )

    baseline_atr = float(
        df[
            atr_column
        ]
        .tail(
            baseline_days
        )
        .median()
    )

    recent_range = float(
        raw_range
        .tail(
            recent_days
        )
        .median()
    )

    baseline_range = float(
        raw_range
        .tail(
            baseline_days
        )
        .median()
    )

    atr_ratio = _safe_ratio(
        recent_atr,
        baseline_atr,
    )

    range_ratio = _safe_ratio(
        recent_range,
        baseline_range,
    )

    atr_point = int(
        not pd.isna(
            atr_ratio
        )
        and
        atr_ratio
        <= atr_ratio_max
    )

    range_point = int(
        not pd.isna(
            range_ratio
        )
        and
        range_ratio
        <= range_ratio_max
    )

    return {
        "atr_ratio":
            atr_ratio,

        "range_ratio":
            range_ratio,

        "atr_point":
            atr_point,

        "range_point":
            range_point,
    }


def evaluate_pullback(
    frame: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    lookback_days: int = 20,
    min_depth_pct: float = 0.03,
    max_depth_pct: float = 0.15,
    equality_tolerance_atr: float = 0.15,
) -> dict[str, Any]:
    """
    Base / Pullback = 2 points.

    1 point:
        current pullback depth is 3%-15%
        from recent peak.

    1 point:
        latest confirmed swing low has not materially
        undercut previous confirmed swing low.
    """

    if frame.empty:
        return {
            "pullback_depth":
                np.nan,

            "depth_point":
                0,

            "low_relation":
                None,

            "structure_point":
                0,
        }

    recent = frame.tail(
        lookback_days
    )

    recent_peak = float(
        recent["high"].max()
    )

    current_close = float(
        frame["close"].iloc[-1]
    )

    if recent_peak <= 0:
        pullback_depth = np.nan
    else:
        pullback_depth = (
            recent_peak
            - current_close
        ) / recent_peak

    depth_point = int(
        not pd.isna(
            pullback_depth
        )
        and
        min_depth_pct
        <= pullback_depth
        <= max_depth_pct
    )

    lows = (
        swings[
            swings["kind"]
            == "LOW"
        ]
        .tail(2)
    )

    low_relation = None
    structure_point = 0

    if len(lows) >= 2:
        previous = lows.iloc[-2]
        latest = lows.iloc[-1]

        previous_price = float(
            previous["price"]
        )

        latest_price = float(
            latest["price"]
        )

        reference_atr = float(
            latest["atr"]
        )

        tolerance = (
            equality_tolerance_atr
            * reference_atr
        )

        difference = (
            latest_price
            - previous_price
        )

        if difference > tolerance:
            low_relation = "HL"
            structure_point = 1

        elif difference < -tolerance:
            low_relation = "LL"
            structure_point = 0

        else:
            low_relation = "EL"

            #
            # Essentially equal low is acceptable
            # base structure.
            #
            structure_point = 1

    return {
        "pullback_depth":
            pullback_depth,

        "depth_point":
            depth_point,

        "low_relation":
            low_relation,

        "structure_point":
            structure_point,
    }


def evaluate_pivot(
    frame: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    current_atr: float,
    min_age_bars: int = 5,
    max_age_bars: int = 30,
    cluster_lookback_bars: int = 30,
    cluster_distance_atr: float = 1.0,
    min_cluster_highs: int = 2,
) -> dict[str, Any]:
    """
    Pivot = 2 points.

    1 point:
        latest confirmed swing high is recent enough
        to be actionable, but not today's noise.

    1 point:
        multiple confirmed highs exist in the same
        ATR-normalized resistance area.
    """

    highs = (
        swings[
            swings["kind"]
            == "HIGH"
        ]
        .copy()
    )

    if highs.empty:
        return {
            "pivot_price":
                np.nan,

            "pivot_age_bars":
                np.nan,

            "pivot_recency_point":
                0,

            "pivot_cluster_count":
                0,

            "pivot_dispersion_atr":
                np.nan,

            "pivot_cluster_point":
                0,
        }

    latest = highs.iloc[-1]

    latest_position = _bar_position(
        frame.index,
        latest["pivot_time"],
    )

    if latest_position is None:
        return {
            "pivot_price":
                np.nan,

            "pivot_age_bars":
                np.nan,

            "pivot_recency_point":
                0,

            "pivot_cluster_count":
                0,

            "pivot_dispersion_atr":
                np.nan,

            "pivot_cluster_point":
                0,
        }

    current_position = (
        len(frame) - 1
    )

    pivot_age = (
        current_position
        - latest_position
    )

    recency_point = int(
        min_age_bars
        <= pivot_age
        <= max_age_bars
    )

    earliest_position = max(
        0,
        current_position
        - cluster_lookback_bars
    )

    eligible_records = []

    for _, row in highs.iterrows():
        position = _bar_position(
            frame.index,
            row["pivot_time"],
        )

        if position is None:
            continue

        if position < earliest_position:
            continue

        eligible_records.append(
            row
        )

    if not eligible_records:
        cluster_count = 0
        dispersion_atr = np.nan

    else:
        recent_highs = pd.DataFrame(
            eligible_records
        )

        latest_price = float(
            latest["price"]
        )

        near_latest = recent_highs[
            (
                recent_highs["price"]
                - latest_price
            )
            .abs()
            <= (
                cluster_distance_atr
                * current_atr
            )
        ]

        cluster_count = len(
            near_latest
        )

        if cluster_count >= 2:
            dispersion = (
                float(
                    near_latest[
                        "price"
                    ].max()
                )
                -
                float(
                    near_latest[
                        "price"
                    ].min()
                )
            )

            dispersion_atr = (
                dispersion
                / current_atr
            )

        else:
            dispersion_atr = (
                np.nan
            )

    cluster_point = int(
        cluster_count
        >= min_cluster_highs
        and
        (
            pd.isna(
                dispersion_atr
            )
            or
            dispersion_atr
            <= cluster_distance_atr
        )
    )

    return {
        "pivot_price":
            float(
                latest["price"]
            ),

        "pivot_age_bars":
            pivot_age,

        "pivot_recency_point":
            recency_point,

        "pivot_cluster_count":
            cluster_count,

        "pivot_dispersion_atr":
            dispersion_atr,

        "pivot_cluster_point":
            cluster_point,
    }


def evaluate_volume_dryup(
    frame: pd.DataFrame,
    *,
    recent_days: int = 5,
    baseline_days: int = 20,
    max_ratio: float = 0.80,
) -> dict[str, Any]:
    """
    Volume Dry-up = 1 point.

    Uses NON-overlapping windows:

        recent 5D
        vs
        preceding 20D
    """

    df = add_volume_dryup(
        frame,
        recent_days=recent_days,
        baseline_days=baseline_days,
    )

    ratio = df[
        "volume_dryup_ratio"
    ].iloc[-1]

    point = int(
        not pd.isna(
            ratio
        )
        and
        ratio
        <= max_ratio
    )

    return {
        "volume_dryup_ratio":
            (
                float(ratio)
                if not pd.isna(
                    ratio
                )
                else np.nan
            ),

        "volume_dryup_point":
            point,
    }


def evaluate_rs_retention(
    rs_row: Mapping[str, Any] | pd.Series | None,
    *,
    rs20_min: float = 75.0,
    rs5_min: float = 50.0,
) -> dict[str, Any]:
    """
    RS Retention = 1 point.

    Setup should not be awarded merely because price
    compressed. Relative strength must remain intact.
    """

    if rs_row is None:
        return {
            "rs_20d_pct":
                np.nan,

            "rs_5d_pct":
                np.nan,

            "rs_retention_point":
                0,
        }

    rs20 = rs_row.get(
        "rs_20d_pct",
        np.nan,
    )

    rs5 = rs_row.get(
        "rs_5d_pct",
        np.nan,
    )

    point = int(
        not pd.isna(
            rs20
        )
        and
        not pd.isna(
            rs5
        )
        and
        float(rs20)
        >= rs20_min
        and
        float(rs5)
        >= rs5_min
    )

    return {
        "rs_20d_pct":
            (
                float(rs20)
                if not pd.isna(
                    rs20
                )
                else np.nan
            ),

        "rs_5d_pct":
            (
                float(rs5)
                if not pd.isna(
                    rs5
                )
                else np.nan
            ),

        "rs_retention_point":
            point,
    }


def evaluate_reexpansion(
    frame: pd.DataFrame,
    *,
    range_ratio_min: float = 1.30,
    volume_ratio_min: float = 1.50,
    close_location_min: float = 0.75,
) -> dict[str, Any]:
    """
    Re-expansion = 2 points.

    Point 1:
        True Range expands AND bar closes strongly.

    Point 2:
        Volume expands AND close advances.

    This distinguishes meaningful expansion from
    high-volume rejection.
    """

    df = frame.copy()

    if "true_range" not in df.columns:
        df = add_true_range(
            df
        )

    if "clv_01" not in df.columns:
        df = add_close_location(
            df
        )

    if "relative_volume_20" not in df.columns:
        df = add_relative_volume(
            df,
            baseline_days=20,
        )

    prior_range_median = (
        df["true_range"]
        .shift(1)
        .rolling(
            window=20,
            min_periods=20,
        )
        .median()
    )

    range_ratio = (
        df["true_range"]
        /
        prior_range_median.replace(
            0,
            np.nan,
        )
    )

    latest_range_ratio = (
        range_ratio.iloc[-1]
    )

    latest_rvol = (
        df[
            "relative_volume_20"
        ].iloc[-1]
    )

    latest_clv = (
        df[
            "clv_01"
        ].iloc[-1]
    )

    previous_close = (
        df["close"].iloc[-2]
        if len(df) >= 2
        else np.nan
    )

    current_close = (
        df["close"].iloc[-1]
    )

    range_point = int(
        not pd.isna(
            latest_range_ratio
        )
        and
        not pd.isna(
            latest_clv
        )
        and
        latest_range_ratio
        >= range_ratio_min
        and
        latest_clv
        >= close_location_min
    )

    volume_point = int(
        not pd.isna(
            latest_rvol
        )
        and
        not pd.isna(
            previous_close
        )
        and
        latest_rvol
        >= volume_ratio_min
        and
        current_close
        > previous_close
    )

    return {
        "true_range_ratio_20":
            (
                float(
                    latest_range_ratio
                )
                if not pd.isna(
                    latest_range_ratio
                )
                else np.nan
            ),

        "relative_volume_20":
            (
                float(
                    latest_rvol
                )
                if not pd.isna(
                    latest_rvol
                )
                else np.nan
            ),

        "close_location":
            (
                float(
                    latest_clv
                )
                if not pd.isna(
                    latest_clv
                )
                else np.nan
            ),

        "reexpansion_range_point":
            range_point,

        "reexpansion_volume_point":
            volume_point,
    }


def calculate_setup_features(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    swings: pd.DataFrame | None = None,
    rs_row: Mapping[str, Any] | pd.Series | None = None,
    min_setup_score: int = 5,
) -> dict[str, Any]:
    """
    Complete Phase 5A Setup Quality engine.

    Maximum score = 10.

    No Extension / Stop / RR logic occurs here.
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    if visible.empty:
        return {}

    visible = add_atr(
        visible,
        period=14,
    )

    current_atr = visible[
        "atr_14"
    ].iloc[-1]

    confirmed = _confirmed_swings(
        swings,
        as_of=as_of,
        visible_index=visible.index,
    )

    contraction = (
        evaluate_contraction(
            visible,
        )
    )

    pullback = (
        evaluate_pullback(
            visible,
            confirmed,
        )
    )

    if (
        pd.isna(
            current_atr
        )
        or current_atr <= 0
    ):
        pivot = {
            "pivot_price":
                np.nan,

            "pivot_age_bars":
                np.nan,

            "pivot_recency_point":
                0,

            "pivot_cluster_count":
                0,

            "pivot_dispersion_atr":
                np.nan,

            "pivot_cluster_point":
                0,
        }

    else:
        pivot = evaluate_pivot(
            visible,
            confirmed,
            current_atr=float(
                current_atr
            ),
        )

    dryup = (
        evaluate_volume_dryup(
            visible,
        )
    )

    rs = evaluate_rs_retention(
        rs_row
    )

    reexpansion = (
        evaluate_reexpansion(
            visible
        )
    )

    component_points = {
        "contraction_atr":
            contraction[
                "atr_point"
            ],

        "contraction_range":
            contraction[
                "range_point"
            ],

        "pullback_depth":
            pullback[
                "depth_point"
            ],

        "pullback_structure":
            pullback[
                "structure_point"
            ],

        "pivot_recency":
            pivot[
                "pivot_recency_point"
            ],

        "pivot_cluster":
            pivot[
                "pivot_cluster_point"
            ],

        "volume_dryup":
            dryup[
                "volume_dryup_point"
            ],

        "rs_retention":
            rs[
                "rs_retention_point"
            ],

        "reexpansion_range":
            reexpansion[
                "reexpansion_range_point"
            ],

        "reexpansion_volume":
            reexpansion[
                "reexpansion_volume_point"
            ],
    }

    setup_score = int(
        sum(
            component_points.values()
        )
    )

    setup_gate_pass = (
        setup_score
        >= min_setup_score
    )

    if setup_score >= 8:
        setup_state = (
            "HIGH_QUALITY"
        )

    elif setup_score >= min_setup_score:
        setup_state = "VALID"

    elif setup_score >= 3:
        setup_state = "FORMING"

    else:
        setup_state = "NO_SETUP"

    return {
        "setup_score":
            setup_score,

        "setup_gate_pass":
            bool(
                setup_gate_pass
            ),

        "setup_state":
            setup_state,

        **component_points,

        "atr_contraction_ratio":
            contraction[
                "atr_ratio"
            ],

        "range_contraction_ratio":
            contraction[
                "range_ratio"
            ],

        "pullback_depth_pct":
            pullback[
                "pullback_depth"
            ],

        "latest_low_relation":
            pullback[
                "low_relation"
            ],

        "pivot_price":
            pivot[
                "pivot_price"
            ],

        "pivot_age_bars":
            pivot[
                "pivot_age_bars"
            ],

        "pivot_cluster_count":
            pivot[
                "pivot_cluster_count"
            ],

        "pivot_dispersion_atr":
            pivot[
                "pivot_dispersion_atr"
            ],

        "volume_dryup_ratio":
            dryup[
                "volume_dryup_ratio"
            ],

        "rs_20d_pct":
            rs[
                "rs_20d_pct"
            ],

        "rs_5d_pct":
            rs[
                "rs_5d_pct"
            ],

        "reexpansion_range_ratio":
            reexpansion[
                "true_range_ratio_20"
            ],

        "reexpansion_relative_volume":
            reexpansion[
                "relative_volume_20"
            ],

        "reexpansion_close_location":
            reexpansion[
                "close_location"
            ],
    }


def build_setup_snapshot(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
    swings_by_symbol: Mapping[
        str,
        pd.DataFrame,
    ] | None = None,
    rs_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate Setup Quality for an entire candidate set.

    Important:
        This should normally run AFTER broader
        Quality / Structure filtering, rather than
        wasting deep structural calculations on every
        junk stock in the full Universe.
    """

    rs_lookup: dict[
        str,
        pd.Series,
    ] = {}

    if (
        rs_snapshot is not None
        and not rs_snapshot.empty
    ):
        rs_lookup = {
            str(row["symbol"]):
                row
            for _, row in (
                rs_snapshot.iterrows()
            )
        }

    records = []

    for symbol, frame in (
        universe.items()
    ):
        symbol = str(symbol)

        swings = None

        if swings_by_symbol:
            swings = (
                swings_by_symbol.get(
                    symbol
                )
            )

        features = (
            calculate_setup_features(
                frame,
                as_of=as_of,
                swings=swings,
                rs_row=(
                    rs_lookup.get(
                        symbol
                    )
                ),
            )
        )

        records.append(
            {
                "symbol":
                    symbol,

                **features,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
            ]
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            [
                "setup_gate_pass",
                "setup_score",
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