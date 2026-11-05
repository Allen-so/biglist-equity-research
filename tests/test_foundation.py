import numpy as np
import pandas as pd

from src.core.config import (
    load_thresholds,
    load_weights,
    validate_weights,
)
from src.features.basic import (
    build_basic_daily_features,
)
from src.features.timeframes import (
    daily_to_weekly,
)


def make_test_data(
    periods: int = 260,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2025-01-01",
        periods=periods,
    )

    close = np.linspace(
        20.0,
        40.0,
        periods,
    )

    frame = pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": np.full(
                periods,
                1_000_000,
            ),
        },
        index=dates,
    )

    return frame


def test_configuration_loads():
    thresholds = load_thresholds()
    weights = load_weights()

    assert thresholds["model"]["name"] == (
        "Big List Engine"
    )

    validate_weights(weights)


def test_basic_daily_features():
    frame = make_test_data()

    result = build_basic_daily_features(frame)

    expected_columns = {
        "atr_14",
        "atr_pct_14",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "return_5d",
        "return_20d",
        "dollar_volume",
        "median_dollar_volume_20d",
        "range_position_252",
    }

    assert expected_columns.issubset(
        result.columns
    )

    assert not pd.isna(
        result["atr_14"].iloc[-1]
    )


def test_daily_to_weekly():
    daily = make_test_data()

    weekly = daily_to_weekly(daily)

    assert len(weekly) < len(daily)
    assert weekly["close"].notna().all()
    assert weekly["volume"].notna().all()