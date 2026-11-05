import numpy as np
import pandas as pd

from src.features.structure import (
    classify_swing_relation,
    confirmed_swings_as_of,
    detect_swings,
)


def make_structure_data(
    periods: int = 50,
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
                101.0,
            ),
            "low": np.full(
                periods,
                99.0,
            ),
            "close": close,
            "volume": np.full(
                periods,
                1_000_000,
            ),
        },
        index=dates,
    )

    # Significant structural highs
    frame.iloc[20, frame.columns.get_loc("high")] = 110.0
    frame.iloc[30, frame.columns.get_loc("high")] = 115.0

    # Significant structural lows
    frame.iloc[25, frame.columns.get_loc("low")] = 94.0
    frame.iloc[35, frame.columns.get_loc("low")] = 97.0

    return frame


def test_detects_structural_swings():
    frame = make_structure_data()

    swings = detect_swings(frame)

    highs = swings[
        swings["kind"] == "HIGH"
    ]

    lows = swings[
        swings["kind"] == "LOW"
    ]

    assert 110.0 in highs["price"].values
    assert 115.0 in highs["price"].values

    assert 94.0 in lows["price"].values
    assert 97.0 in lows["price"].values


def test_confirmation_prevents_lookahead():
    frame = make_structure_data()

    swings = detect_swings(frame)

    pivot = swings[
        swings["price"] == 110.0
    ].iloc[0]

    before_confirmation = (
        pivot["confirmed_at"]
        - pd.Timedelta(days=1)
    )

    visible_before = confirmed_swings_as_of(
        swings,
        before_confirmation,
    )

    visible_after = confirmed_swings_as_of(
        swings,
        pivot["confirmed_at"],
    )

    assert 110.0 not in (
        visible_before["price"].values
    )

    assert 110.0 in (
        visible_after["price"].values
    )


def test_high_relation():
    result = classify_swing_relation(
        previous_price=100.0,
        latest_price=110.0,
        reference_atr=5.0,
        kind="HIGH",
    )

    assert result == "HH"


def test_low_relation():
    result = classify_swing_relation(
        previous_price=90.0,
        latest_price=96.0,
        reference_atr=4.0,
        kind="LOW",
    )

    assert result == "HL"


def test_small_difference_is_equal_structure():
    result = classify_swing_relation(
        previous_price=100.0,
        latest_price=100.2,
        reference_atr=4.0,
        kind="HIGH",
        equality_tolerance_atr=0.15,
    )

    assert result == "EH"