import numpy as np
import pandas as pd

from src.core.constants import (
    DAILY_MA_WINDOWS,
    DAILY_RETURN_WINDOWS,
    REQUIRED_OHLCV_COLUMNS,
)


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize one symbol's OHLCV dataframe.

    Required index:
        DatetimeIndex

    Required columns:
        open, high, low, close, volume
    """

    if frame.empty:
        raise ValueError("OHLCV dataframe is empty.")

    df = frame.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    missing = [
        column
        for column in REQUIRED_OHLCV_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "OHLCV dataframe index must be a DatetimeIndex."
        )

    df = df.sort_index()

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    for column in REQUIRED_OHLCV_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=list(REQUIRED_OHLCV_COLUMNS)
    )

    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["volume"] < 0)
        | (df["close"] <= 0)
    )

    if invalid.any():
        bad_rows = df.index[invalid].tolist()

        raise ValueError(
            "Invalid OHLCV relationships detected at: "
            f"{bad_rows[:5]}"
        )

    return df


def add_true_range(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()

    previous_close = df["close"].shift(1)

    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    df["true_range"] = tr_components.max(axis=1)

    return df


def add_atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Wilder ATR.
    """

    df = add_true_range(frame)

    df[f"atr_{period}"] = (
        df["true_range"]
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    df[f"atr_pct_{period}"] = (
        df[f"atr_{period}"]
        / df["close"]
    )

    return df


def add_moving_averages(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()

    for window in DAILY_MA_WINDOWS:
        df[f"sma_{window}"] = (
            df["close"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    return df


def add_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()

    for window in DAILY_RETURN_WINDOWS:
        df[f"return_{window}d"] = (
            df["close"].pct_change(window)
        )

    return df


def add_liquidity_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()

    df["dollar_volume"] = (
        df["close"] * df["volume"]
    )

    df["median_dollar_volume_20d"] = (
        df["dollar_volume"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .median()
    )

    return df


def add_range_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()

    safe_close = df["close"].replace(
        0,
        np.nan,
    )

    df["daily_range_pct"] = (
        (df["high"] - df["low"])
        / safe_close
    )

    rolling_high = (
        df["high"]
        .rolling(
            window=252,
            min_periods=60,
        )
        .max()
    )

    rolling_low = (
        df["low"]
        .rolling(
            window=252,
            min_periods=60,
        )
        .min()
    )

    denominator = (
        rolling_high - rolling_low
    ).replace(0, np.nan)

    df["range_position_252"] = (
        (df["close"] - rolling_low)
        / denominator
    )

    return df


def build_basic_daily_features(
    frame: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Main Phase-1 daily feature pipeline.
    """

    df = normalize_ohlcv(frame)

    df = add_atr(
        df,
        period=atr_period,
    )

    df = add_moving_averages(df)

    df = add_returns(df)

    df = add_liquidity_features(df)

    df = add_range_features(df)

    return df