import numpy as np
import pandas as pd

from src.features.tradability import (
    calculate_tradability_features,
    classify_pivot_proximity,
    collect_resistance_candidates,
    collect_support_candidates,
    evaluate_reward_risk,
    select_structural_stop,
)


def make_price_data(
    periods: int = 80,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.full(
        periods,
        102.0,
    )

    return pd.DataFrame(
        {
            "open":
                close.copy(),

            "high":
                np.full(
                    periods,
                    103.0,
                ),

            "low":
                np.full(
                    periods,
                    101.0,
                ),

            "close":
                close.copy(),

            "volume":
                np.full(
                    periods,
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def make_support_zone(
    *,
    low: float = 99.0,
    high: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone_low":
                    low,

                "zone_high":
                    high,

                "zone_center":
                    (
                        low + high
                    ) / 2.0,

                "role":
                    "SUPPORT",

                "confidence":
                    90.0,

                "distance_from_price_atr":
                    1.0,
            }
        ]
    )


def make_resistance_zone(
    *,
    low: float = 110.0,
    high: float = 111.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone_low":
                    low,

                "zone_high":
                    high,

                "zone_center":
                    (
                        low + high
                    ) / 2.0,

                "role":
                    "RESISTANCE",

                "confidence":
                    90.0,

                "distance_from_price_atr":
                    4.0,
            }
        ]
    )


def test_pivot_proximity_states():
    ready = classify_pivot_proximity(
        current_price=102.0,
        pivot_price=102.5,
        current_atr=2.0,
    )

    developing = (
        classify_pivot_proximity(
            current_price=102.0,
            pivot_price=105.0,
            current_atr=2.0,
        )
    )

    watch = classify_pivot_proximity(
        current_price=102.0,
        pivot_price=106.4,
        current_atr=2.0,
    )

    assert (
        ready[
            "entry_proximity_state"
        ]
        == "READY"
    )

    assert (
        developing[
            "entry_proximity_state"
        ]
        == "DEVELOPING"
    )

    assert (
        watch[
            "entry_proximity_state"
        ]
        == "WATCH"
    )


def test_structural_stop_below_zone():
    frame = make_price_data()

    supports = (
        collect_support_candidates(
            frame,
            current_price=102.0,
            current_atr=2.0,
            zones=(
                make_support_zone()
            ),
        )
    )

    stop = select_structural_stop(
        supports,
        current_price=102.0,
        current_atr=2.0,
        buffer_atr=0.25,
    )

    #
    # Zone low = 99
    # Buffer = 0.5
    #
    assert np.isclose(
        stop[
            "stop_price"
        ],
        98.5,
    )

    assert np.isclose(
        stop[
            "stop_distance_atr"
        ],
        1.75,
    )


def test_reward_risk_uses_nearest_resistance():
    resistance = (
        collect_resistance_candidates(
            current_price=102.0,
            current_atr=2.0,
            zones=(
                make_resistance_zone(
                    low=110.0,
                    high=111.0,
                )
            ),
        )
    )

    result = evaluate_reward_risk(
        resistance,
        current_price=102.0,
        risk_per_share=3.5,
        current_atr=2.0,
    )

    assert np.isclose(
        result[
            "resistance_price"
        ],
        110.0,
    )

    assert np.isclose(
        result[
            "reward_per_share"
        ],
        8.0,
    )

    assert (
        result[
            "reward_risk"
        ]
        > 2.0
    )


def test_blue_sky_passes_without_fake_target():
    empty = pd.DataFrame()

    result = evaluate_reward_risk(
        empty,
        current_price=102.0,
        risk_per_share=3.5,
        current_atr=2.0,
    )

    assert result[
        "blue_sky"
    ]

    assert (
        result[
            "reward_risk_state"
        ]
        == "BLUE_SKY"
    )

    assert pd.isna(
        result[
            "reward_risk"
        ]
    )


def test_extension_over_three_atr_hard_fails():
    frame = make_price_data()

    support = make_support_zone(
        low=92.0,
        high=94.0,
    )

    result = (
        calculate_tradability_features(
            frame,
            as_of=frame.index[-1],
            setup_row={
                "pivot_price":
                    102.0,
            },
            zones=support,
        )
    )

    assert (
        result[
            "extension_atr"
        ]
        > 3.0
    )

    assert not result[
        "tradability_gate_pass"
    ]

    assert (
        result[
            "classification"
        ]
        == "EXTENDED"
    )


def test_missing_support_rejects_trade():
    frame = make_price_data()

    result = (
        calculate_tradability_features(
            frame,
            as_of=frame.index[-1],
            setup_row={
                "pivot_price":
                    102.0,
            },
        )
    )

    #
    # SMA20/SMA50 are exactly 102,
    # therefore not valid support BELOW price.
    #
    assert not result[
        "tradability_gate_pass"
    ]

    assert (
        "NO_STRUCTURAL_STOP"
        in result[
            "rejection_reason"
        ]
    )

    assert (
        result[
            "classification"
        ]
        == "REJECTED"
    )


def test_ready_classification():
    frame = make_price_data()

    zones = pd.concat(
        [
            make_support_zone(),
            make_resistance_zone(),
        ],
        ignore_index=True,
    )

    result = (
        calculate_tradability_features(
            frame,
            as_of=frame.index[-1],
            setup_row={
                "pivot_price":
                    102.4,
            },
            zones=zones,
        )
    )

    assert result[
        "tradability_gate_pass"
    ]

    assert (
        result[
            "classification"
        ]
        == "READY"
    )

    assert (
        result[
            "tradability_score"
        ]
        > 0
    )


def test_developing_and_watch_classification():
    frame = make_price_data()

    zones = pd.concat(
        [
            make_support_zone(),
            make_resistance_zone(),
        ],
        ignore_index=True,
    )

    developing = (
        calculate_tradability_features(
            frame,
            as_of=frame.index[-1],
            setup_row={
                "pivot_price":
                    105.0,
            },
            zones=zones,
        )
    )

    watch = (
        calculate_tradability_features(
            frame,
            as_of=frame.index[-1],
            setup_row={
                "pivot_price":
                    106.4,
            },
            zones=zones,
        )
    )

    assert (
        developing[
            "classification"
        ]
        == "DEVELOPING"
    )

    assert (
        watch[
            "classification"
        ]
        == "WATCH"
    )


def test_future_bar_is_ignored():
    frame = make_price_data()

    as_of = frame.index[-2]

    #
    # Massive future move after as_of.
    #
    frame.iloc[
        -1,
        frame.columns.get_loc(
            "open"
        ),
    ] = 119.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = 122.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = 118.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = 120.0

    zones = pd.concat(
        [
            make_support_zone(),
            make_resistance_zone(),
        ],
        ignore_index=True,
    )

    full = (
        calculate_tradability_features(
            frame,
            as_of=as_of,
            setup_row={
                "pivot_price":
                    102.4,
            },
            zones=zones,
        )
    )

    truncated = (
        calculate_tradability_features(
            frame.iloc[:-1],
            as_of=as_of,
            setup_row={
                "pivot_price":
                    102.4,
            },
            zones=zones,
        )
    )

    assert np.isclose(
        full[
            "current_price"
        ],
        truncated[
            "current_price"
        ],
    )

    assert np.isclose(
        full[
            "extension_atr"
        ],
        truncated[
            "extension_atr"
        ],
    )

    assert (
        full[
            "classification"
        ]
        ==
        truncated[
            "classification"
        ]
    )