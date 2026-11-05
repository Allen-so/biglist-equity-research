import numpy as np
import pandas as pd

from src.features.vwap import (
    anchored_vwap_series,
    build_avwap_anchors,
    classify_avwap_side,
    evaluate_avwap_anchors,
)


def make_price_data(
    periods: int = 60,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.full(
        periods,
        100.0,
    )

    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": np.full(
                periods,
                101.0,
            ),
            "low": np.full(
                periods,
                99.0,
            ),
            "close": close.copy(),
            "volume": np.full(
                periods,
                1_000_000.0,
            ),
        },
        index=dates,
    )


def test_anchored_vwap_weighting():
    dates = pd.bdate_range(
        "2026-01-01",
        periods=3,
    )

    frame = pd.DataFrame(
        {
            "open": [
                10.0,
                20.0,
                30.0,
            ],
            "high": [
                10.0,
                20.0,
                30.0,
            ],
            "low": [
                10.0,
                20.0,
                30.0,
            ],
            "close": [
                10.0,
                20.0,
                30.0,
            ],
            "volume": [
                1.0,
                2.0,
                1.0,
            ],
        },
        index=dates,
    )

    result = anchored_vwap_series(
        frame,
        anchor_time=dates[0],
    )

    #
    # (10*1 + 20*2 + 30*1) / 4
    # = 20
    #
    assert result.iloc[-1] == 20.0


def test_values_before_anchor_are_nan():
    frame = make_price_data()

    anchor = frame.index[20]

    result = anchored_vwap_series(
        frame,
        anchor_time=anchor,
    )

    assert result.iloc[:20].isna().all()

    assert result.iloc[
        20:
    ].notna().all()


def test_future_unconfirmed_swing_is_ignored():
    frame = make_price_data()

    swings = pd.DataFrame(
        [
            {
                "pivot_time":
                    frame.index[20],
                "confirmed_at":
                    frame.index[22],
                "kind":
                    "LOW",
                "price":
                    95.0,
                "atr":
                    2.0,
                "prominence_atr":
                    2.0,
            }
        ]
    )

    anchors_before = (
        build_avwap_anchors(
            frame,
            as_of=frame.index[21],
            swings=swings,
            max_recent_breakouts=0,
        )
    )

    assert anchors_before.empty

    anchors_after = (
        build_avwap_anchors(
            frame,
            as_of=frame.index[22],
            swings=swings,
            max_recent_breakouts=0,
        )
    )

    assert len(
        anchors_after
    ) == 1

    assert (
        anchors_after[
            "anchor_type"
        ].iloc[0]
        == "SWING_LOW"
    )


def test_anchor_count_is_capped():
    frame = make_price_data()

    records = []

    for position in [
        10,
        15,
        20,
        25,
    ]:
        records.append(
            {
                "pivot_time":
                    frame.index[position],
                "confirmed_at":
                    frame.index[
                        position + 2
                    ],
                "kind":
                    "LOW",
                "price":
                    95.0,
                "atr":
                    2.0,
                "prominence_atr":
                    2.0,
            }
        )

    swings = pd.DataFrame(
        records
    )

    anchors = build_avwap_anchors(
        frame,
        as_of=frame.index[-1],
        swings=swings,
        max_recent_swing_lows=2,
        max_recent_swing_highs=0,
        max_recent_breakouts=0,
    )

    assert len(anchors) == 2

    assert (
        anchors[
            "anchor_time"
        ].min()
        == frame.index[20]
    )


def test_classify_avwap_side():
    assert (
        classify_avwap_side(
            close=103.0,
            avwap=100.0,
            atr=2.0,
        )
        == "ABOVE"
    )

    assert (
        classify_avwap_side(
            close=97.0,
            avwap=100.0,
            atr=2.0,
        )
        == "BELOW"
    )

    assert (
        classify_avwap_side(
            close=100.1,
            avwap=100.0,
            atr=2.0,
        )
        == "AT"
    )


def test_avwap_reclaim():
    frame = make_price_data()

    anchor_time = frame.index[20]

    #
    # Make anchored history centered near 100.
    #
    frame.iloc[
        -2,
        frame.columns.get_loc(
            "open"
        ),
    ] = 99.0

    frame.iloc[
        -2,
        frame.columns.get_loc(
            "high"
        ),
    ] = 100.0

    frame.iloc[
        -2,
        frame.columns.get_loc(
            "low"
        ),
    ] = 98.0

    frame.iloc[
        -2,
        frame.columns.get_loc(
            "close"
        ),
    ] = 99.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "open"
        ),
    ] = 99.5

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = 103.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = 99.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = 102.5

    anchors = pd.DataFrame(
        [
            {
                "anchor_type":
                    "SWING_LOW",
                "anchor_time":
                    anchor_time,
                "available_at":
                    frame.index[22],
                "reference_price":
                    95.0,
                "prominence_atr":
                    2.0,
            }
        ]
    )

    result = evaluate_avwap_anchors(
        frame,
        anchors,
        as_of=frame.index[-1],
        reclaim_buffer_atr=0.05,
    )

    assert not result.empty

    assert bool(
        result[
            "reclaim"
        ].iloc[0]
    )