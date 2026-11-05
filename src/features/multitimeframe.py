from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)
from src.features.timeframes import (
    daily_to_weekly,
)


DEFAULT_WEEKLY_MA_WINDOWS = (
    10,
    20,
    40,
)


def build_weekly_features(
    daily_frame: pd.DataFrame,
    *,
    atr_period: int = 14,
    ma_windows: Iterable[int] = (
        DEFAULT_WEEKLY_MA_WINDOWS
    ),
) -> pd.DataFrame:
    """
    Build standardized weekly structural features
    from the SAME daily OHLCV source.

    Weekly bars are never fetched independently.
    """

    weekly = daily_to_weekly(
        daily_frame
    )

    weekly = normalize_ohlcv(
        weekly
    )

    weekly = add_atr(
        weekly,
        period=atr_period,
    )

    for window in ma_windows:
        weekly[f"sma_{window}"] = (
            weekly["close"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    return weekly