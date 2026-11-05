import numpy as np
import pandas as pd

from src.features.trendlines import (
    find_trendlines,
)


def make_price_data(
    periods: int = 70,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.full(
        periods,
        100.0,
    )

    frame = pd.DataFrame(
        {
            "open": close,
            "high": np.full(
                periods,
                102.0,
            ),
            "low": np.full(
                periods,
                98.0,
            ),
            "close": close,
            "volume": np.full(
                periods,
                1_000_000,
            ),
        },
        index=dates,
    )

    return frame


def make_resistance_swings(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    positions = [
        10,
        25,
        40,
        55,
    ]

    prices = [
        120.0,
        115.0,
        110.0,
        105.0,
    ]

    records = []

    for position, price in zip(
        positions,
        prices,
    ):
        records.append(
            {
                "pivot_time":
                    frame.index[position],
                "confirmed_at":
                    frame.index[position + 2],
                "kind": "HIGH",
                "price": price,
                "atr": 4.0,
                "prominence_atr": 2.0,
            }
        )

    return pd.DataFrame(records)


def make_support_swings(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    positions = [
        10,
        25,
        40,
        55,
    ]

    prices = [
        80.0,
        85.0,
        90.0,
        95.0,
    ]

    records = []

    for position, price in zip(
        positions,
        prices,
    ):
        records.append(
            {
                "pivot_time":
                    frame.index[position],
                "confirmed_at":
                    frame.index[position + 2],
                "kind": "LOW",
                "price": price,
                "atr": 4.0,
                "prominence_atr": 2.0,
            }
        )

    return pd.DataFrame(records)


def test_detects_falling_resistance():
    frame = make_price_data()

    swings = make_resistance_swings(
        frame
    )

    result = find_trendlines(
        frame,
        swings,
        as_of=frame.index[-1],
        kind="HIGH",
    )

    assert not result.empty

    line = result.iloc[0]

    assert line["kind"] == "RESISTANCE"
    assert line["direction"] == "FALLING"
    assert line["anchor_count"] >= 3


def test_detects_rising_support():
    frame = make_price_data()

    swings = make_support_swings(
        frame
    )

    result = find_trendlines(
        frame,
        swings,
        as_of=frame.index[-1],
        kind="LOW",
    )

    assert not result.empty

    line = result.iloc[0]

    assert line["kind"] == "SUPPORT"
    assert line["direction"] == "RISING"
    assert line["anchor_count"] >= 3


def test_two_anchors_are_not_enough():
    frame = make_price_data()

    swings = (
        make_resistance_swings(
            frame
        )
        .iloc[:2]
        .copy()
    )

    result = find_trendlines(
        frame,
        swings,
        as_of=frame.index[-1],
        kind="HIGH",
    )

    assert result.empty


def test_future_unconfirmed_anchor_is_ignored():
    frame = make_price_data()

    swings = make_resistance_swings(
        frame
    )

    early_as_of = frame.index[35]

    result = find_trendlines(
        frame,
        swings,
        as_of=early_as_of,
        kind="HIGH",
    )

    # Only the first two pivots have been confirmed
    # by this point.
    assert result.empty


def test_flat_pivots_are_not_trendline():
    frame = make_price_data()

    records = []

    for position in [10, 25, 40]:
        records.append(
            {
                "pivot_time":
                    frame.index[position],
                "confirmed_at":
                    frame.index[position + 2],
                "kind": "HIGH",
                "price": 110.0,
                "atr": 4.0,
                "prominence_atr": 2.0,
            }
        )

    swings = pd.DataFrame(records)

    result = find_trendlines(
        frame,
        swings,
        as_of=frame.index[-1],
        kind="HIGH",
    )

    assert result.empty

def test_synthetic_regression_ignores_future_price_changes():
    # Synthetic test data - no real market prices
    frame = make_price_data()
    for kind, builder in [('HIGH', make_resistance_swings), ('LOW', make_support_swings)]:
        swings = builder(frame)
        as_of = frame.index[60]
        expected = find_trendlines(frame.iloc[:61], swings, as_of=as_of, kind=kind)
        assert not expected.empty
        frame.loc[frame.index[61]:, ['open', 'high', 'low', 'close']] = 9999.0
        actual = find_trendlines(frame, swings, as_of=as_of, kind=kind)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
