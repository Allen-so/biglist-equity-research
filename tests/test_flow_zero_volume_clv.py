import numpy as np
import pandas as pd

from src.features.flow import (
    add_volume_weighted_close_location,
)


def make_frame(*, zero_bar_volume):
    index = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="B",
    )

    frame = pd.DataFrame(
        {
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "volume": [1000.0] * 20,
        },
        index=index,
    )

    # One zero-range bar inside the rolling window.
    frame.loc[
        index[10],
        ["open", "high", "low", "close"],
    ] = [100.0, 100.0, 100.0, 100.0]

    frame.loc[
        index[10],
        "volume",
    ] = zero_bar_volume

    return frame


def test_zero_range_zero_volume_does_not_poison_vw_clv():
    out = add_volume_weighted_close_location(
        make_frame(zero_bar_volume=0.0),
        window=20,
    )

    value = out[
        "volume_weighted_clv_20"
    ].iloc[-1]

    assert np.isfinite(value)
    assert np.isclose(value, 0.5)


def test_zero_range_positive_volume_remains_undefined():
    out = add_volume_weighted_close_location(
        make_frame(zero_bar_volume=1000.0),
        window=20,
    )

    value = out[
        "volume_weighted_clv_20"
    ].iloc[-1]

    assert np.isnan(value)
