from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.basic import (
    add_true_range,
    normalize_ohlcv,
)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


def add_relative_volume(
    frame: pd.DataFrame,
    *,
    baseline_days: int = 20,
) -> pd.DataFrame:
    """
    Daily relative volume.

    Crucially, today's volume is NOT included in its own
    historical baseline.
    """

    df = frame.copy()

    baseline = (
        df["volume"]
        .shift(1)
        .rolling(
            window=baseline_days,
            min_periods=baseline_days,
        )
        .median()
    )

    df[
        f"median_volume_prior_{baseline_days}"
    ] = baseline

    df[
        f"relative_volume_{baseline_days}"
    ] = _safe_divide(
        df["volume"],
        baseline,
    )

    return df


def add_volume_zscore(
    frame: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compare today's volume with the PRIOR historical
    distribution, excluding today's observation.
    """

    df = frame.copy()

    prior_volume = df["volume"].shift(1)

    mean = (
        prior_volume
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    std = (
        prior_volume
        .rolling(
            window=window,
            min_periods=window,
        )
        .std(ddof=0)
        .replace(0, np.nan)
    )

    df[f"volume_zscore_{window}"] = (
        df["volume"] - mean
    ) / std

    return df


def add_close_location(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Close Location Value.

    clv_01:
        0.0 = close at low
        0.5 = close in middle
        1.0 = close at high

    clv_signed:
        -1.0 = close at low
         0.0 = middle
        +1.0 = close at high
    """

    df = frame.copy()

    bar_range = (
        df["high"] - df["low"]
    )

    clv = _safe_divide(
        df["close"] - df["low"],
        bar_range,
    )

    # Zero-range bars have no meaningful close-location.
    clv = clv.clip(
        lower=0.0,
        upper=1.0,
    )

    df["clv_01"] = clv

    df["clv_signed"] = (
        2.0 * clv - 1.0
    )

    return df


def add_volume_weighted_close_location(
    frame: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    Measure whether volume tends to occur on bars closing
    near their highs or near their lows.
    """

    df = frame.copy()

    if "clv_signed" not in df.columns:
        df = add_close_location(df)

    weighted = (
        df["clv_signed"]
        * df["volume"]
    )

    # A zero-volume bar contributes nothing to a
    # volume-weighted measure, even when its CLV is
    # undefined because high == low.
    zero_volume_undefined = (
        df["volume"].eq(0)
        & weighted.isna()
    )

    weighted.loc[
        zero_volume_undefined
    ] = 0.0

    numerator = (
        weighted
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    denominator = (
        df["volume"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    df[
        f"volume_weighted_clv_{window}"
    ] = _safe_divide(
        numerator,
        denominator,
    )

    return df


def add_up_down_volume(
    frame: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    Rolling Up-Volume / Down-Volume relationship.
    """

    df = frame.copy()

    daily_return = (
        df["close"].pct_change()
    )

    up_volume = df["volume"].where(
        daily_return > 0,
        0.0,
    )

    down_volume = df["volume"].where(
        daily_return < 0,
        0.0,
    )

    rolling_up = (
        up_volume
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    rolling_down = (
        down_volume
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    df[f"up_volume_{window}"] = (
        rolling_up
    )

    df[f"down_volume_{window}"] = (
        rolling_down
    )

    df[
        f"up_down_volume_ratio_{window}"
    ] = _safe_divide(
        rolling_up,
        rolling_down,
    )

    return df


def add_accumulation_distribution(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    accumulation_return: float = 0.01,
    distribution_return: float = -0.01,
    min_relative_volume: float = 1.20,
    accumulation_clv: float = 0.65,
    distribution_clv: float = 0.35,
) -> pd.DataFrame:
    """
    Mechanical accumulation/distribution day detection.

    No chart judgement is involved.
    """

    df = frame.copy()

    if "relative_volume_20" not in df.columns:
        df = add_relative_volume(
            df,
            baseline_days=20,
        )

    if "clv_01" not in df.columns:
        df = add_close_location(df)

    daily_return = (
        df["close"].pct_change()
    )

    df["accumulation_day"] = (
        (daily_return >= accumulation_return)
        & (
            df["relative_volume_20"]
            >= min_relative_volume
        )
        & (
            df["clv_01"]
            >= accumulation_clv
        )
    )

    df["distribution_day"] = (
        (daily_return <= distribution_return)
        & (
            df["relative_volume_20"]
            >= min_relative_volume
        )
        & (
            df["clv_01"]
            <= distribution_clv
        )
    )

    df[
        f"accumulation_count_{window}"
    ] = (
        df["accumulation_day"]
        .astype(int)
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    df[
        f"distribution_count_{window}"
    ] = (
        df["distribution_day"]
        .astype(int)
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    df[
        f"accumulation_distribution_balance_{window}"
    ] = (
        df[f"accumulation_count_{window}"]
        -
        df[f"distribution_count_{window}"]
    ) / window

    return df


def add_volume_dryup(
    frame: pd.DataFrame,
    *,
    recent_days: int = 5,
    baseline_days: int = 20,
) -> pd.DataFrame:
    """
    Compare recent volume with the PRECEDING baseline.

    The baseline excludes the recent window itself.

    Example:
        recent = last 5 days
        baseline = 20 days before those 5 days
    """

    df = frame.copy()

    recent_median = (
        df["volume"]
        .rolling(
            window=recent_days,
            min_periods=recent_days,
        )
        .median()
    )

    baseline_median = (
        df["volume"]
        .shift(recent_days)
        .rolling(
            window=baseline_days,
            min_periods=baseline_days,
        )
        .median()
    )

    df[
        f"median_volume_recent_{recent_days}"
    ] = recent_median

    df[
        f"median_volume_baseline_{baseline_days}"
    ] = baseline_median

    df["volume_dryup_ratio"] = (
        _safe_divide(
            recent_median,
            baseline_median,
        )
    )

    return df


def add_range_expansion(
    frame: pd.DataFrame,
    *,
    baseline_days: int = 20,
) -> pd.DataFrame:
    """
    Today's True Range relative to prior historical range.
    """

    df = frame.copy()

    if "true_range" not in df.columns:
        df = add_true_range(df)

    prior_median_range = (
        df["true_range"]
        .shift(1)
        .rolling(
            window=baseline_days,
            min_periods=baseline_days,
        )
        .median()
    )

    df[
        f"median_true_range_prior_{baseline_days}"
    ] = prior_median_range

    df[
        f"true_range_ratio_{baseline_days}"
    ] = _safe_divide(
        df["true_range"],
        prior_median_range,
    )

    return df


def add_gap_features(
    frame: pd.DataFrame,
    *,
    min_abs_gap_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Gap direction and retention.

    Gap retention interpretation:

        0.0 = gap fully filled by close
        1.0 = close remains at opening gap level
        >1  = price extended further in gap direction

    Works symmetrically for gap-up and gap-down.
    """

    df = frame.copy()

    previous_close = (
        df["close"].shift(1)
    )

    gap_pct = _safe_divide(
        df["open"] - previous_close,
        previous_close,
    )

    df["gap_pct"] = gap_pct

    meaningful_gap = (
        gap_pct.abs()
        >= min_abs_gap_pct
    )

    gap_up = meaningful_gap & (
        gap_pct > 0
    )

    gap_down = meaningful_gap & (
        gap_pct < 0
    )

    df["gap_direction"] = "NONE"

    df.loc[
        gap_up,
        "gap_direction",
    ] = "UP"

    df.loc[
        gap_down,
        "gap_direction",
    ] = "DOWN"

    retention = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    gap_up_size = (
        df["open"] - previous_close
    )

    gap_down_size = (
        previous_close - df["open"]
    )

    retention.loc[gap_up] = (
        (
            df.loc[gap_up, "close"]
            - previous_close.loc[gap_up]
        )
        /
        gap_up_size.loc[gap_up]
    )

    retention.loc[gap_down] = (
        (
            previous_close.loc[gap_down]
            - df.loc[gap_down, "close"]
        )
        /
        gap_down_size.loc[gap_down]
    )

    df["gap_retention"] = retention

    return df


def add_flow_flags(
    frame: pd.DataFrame,
    *,
    expansion_relative_volume: float = 1.50,
    expansion_range_ratio: float = 1.30,
    dryup_ratio_max: float = 0.80,
) -> pd.DataFrame:
    """
    Simple Boolean primitives.

    These are NOT final Big List gates.
    """

    df = frame.copy()

    required = {
        "relative_volume_20",
        "true_range_ratio_20",
        "volume_dryup_ratio",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Flow flags require missing columns: "
            f"{sorted(missing)}"
        )

    df["volume_expansion"] = (
        df["relative_volume_20"]
        >= expansion_relative_volume
    )

    df["range_expansion"] = (
        df["true_range_ratio_20"]
        >= expansion_range_ratio
    )

    df["volume_dryup"] = (
        df["volume_dryup_ratio"]
        <= dryup_ratio_max
    )

    df["price_volume_expansion"] = (
        df["volume_expansion"]
        & df["range_expansion"]
    )

    return df


def build_daily_flow_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Phase 3A daily capital-flow feature pipeline.

    No final scoring.
    No subjective interpretation.
    """

    df = normalize_ohlcv(frame)

    df = add_relative_volume(
        df,
        baseline_days=20,
    )

    df = add_volume_zscore(
        df,
        window=20,
    )

    df = add_close_location(df)

    df = (
        add_volume_weighted_close_location(
            df,
            window=20,
        )
    )

    df = add_up_down_volume(
        df,
        window=20,
    )

    df = add_accumulation_distribution(
        df,
        window=20,
    )

    df = add_volume_dryup(
        df,
        recent_days=5,
        baseline_days=20,
    )

    df = add_range_expansion(
        df,
        baseline_days=20,
    )

    df = add_gap_features(
        df,
        min_abs_gap_pct=0.02,
    )

    df = add_flow_flags(df)

    return df