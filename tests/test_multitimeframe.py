import numpy as np
import pandas as pd

from src.features.multitimeframe import (
    build_weekly_features,
)


def make_daily_data(
    periods: int = 120,
) -> pd.DataFrame:

    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.linspace(
        50.0,
        80.0,
        periods,
    )

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(
                periods,
                1_000_000,
            ),
        },
        index=dates,
    )


def test_build_weekly_features():
    daily = make_daily_data()

    weekly = build_weekly_features(
        daily
    )

    assert not weekly.empty

    assert "atr_14" in weekly.columns
    assert "sma_10" in weekly.columns
    assert "sma_20" in weekly.columns
    assert "sma_40" in weekly.columns


def test_weekly_volume_is_aggregated():
    daily = make_daily_data()

    weekly = build_weekly_features(
        daily
    )

    assert (
        weekly["volume"] > 0
    ).all()