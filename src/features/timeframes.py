import pandas as pd

from src.features.basic import normalize_ohlcv


def daily_to_weekly(
    daily_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resample standardized daily OHLCV data
    into weekly OHLCV ending on Friday.
    """

    daily = normalize_ohlcv(daily_frame)

    weekly = daily.resample(
        "W-FRI",
        label="right",
        closed="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    weekly = weekly.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return weekly