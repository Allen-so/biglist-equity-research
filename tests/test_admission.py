import numpy as np
import pandas as pd

from src.core.admission import (
    calculate_structure_evaluation,
    evaluate_admission_channel,
    evaluate_candidate_admission,
    evaluate_hard_gates,
)


def make_trending_stock(
    periods: int = 260,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2025-01-01",
        periods=periods,
    )

    close = np.linspace(
        50.0,
        120.0,
        periods,
    )

    return pd.DataFrame(
        {
            "open":
                close * 0.998,

            "high":
                close * 1.01,

            "low":
                close * 0.99,

            "close":
                close,

            "volume":
                np.full(
                    periods,
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def make_good_swings(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pivot_time":
                    frame.index[180],

                "confirmed_at":
                    frame.index[182],

                "kind":
                    "LOW",

                "price":
                    90.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[195],

                "confirmed_at":
                    frame.index[197],

                "kind":
                    "HIGH",

                "price":
                    105.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[210],

                "confirmed_at":
                    frame.index[212],

                "kind":
                    "LOW",

                "price":
                    95.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[230],

                "confirmed_at":
                    frame.index[232],

                "kind":
                    "HIGH",

                "price":
                    112.0,

                "atr":
                    3.0,
            },
        ]
    )


def test_structure_score_is_eight_for_strong_structure():
    frame = make_trending_stock()

    result = (
        calculate_structure_evaluation(
            frame,
            as_of=frame.index[-1],
            swings=make_good_swings(
                frame
            ),
        )
    )

    assert np.isclose(
        result[
            "structure_score"
        ],
        8.0,
    )


def test_broken_structure_detected():
    frame = make_trending_stock()

    #
    # Destroy final part of structure.
    #
    frame.loc[
        frame.index[-70:],
        "close",
    ] = np.linspace(
        100.0,
        55.0,
        70,
    )

    frame.loc[
        frame.index[-70:],
        "open",
    ] = (
        frame.loc[
            frame.index[-70:],
            "close",
        ]
    )

    frame.loc[
        frame.index[-70:],
        "high",
    ] = (
        frame.loc[
            frame.index[-70:],
            "close",
        ]
        + 1.0
    )

    frame.loc[
        frame.index[-70:],
        "low",
    ] = (
        frame.loc[
            frame.index[-70:],
            "close",
        ]
        - 1.0
    )

    swings = pd.DataFrame(
        [
            {
                "pivot_time":
                    frame.index[-50],

                "confirmed_at":
                    frame.index[-48],

                "kind":
                    "HIGH",

                "price":
                    90.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[-45],

                "confirmed_at":
                    frame.index[-43],

                "kind":
                    "LOW",

                "price":
                    80.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[-25],

                "confirmed_at":
                    frame.index[-23],

                "kind":
                    "HIGH",

                "price":
                    70.0,

                "atr":
                    3.0,
            },
            {
                "pivot_time":
                    frame.index[-15],

                "confirmed_at":
                    frame.index[-13],

                "kind":
                    "LOW",

                "price":
                    60.0,

                "atr":
                    3.0,
            },
        ]
    )

    result = (
        calculate_structure_evaluation(
            frame,
            as_of=frame.index[-1],
            swings=swings,
        )
    )

    assert (
        result[
            "broken_condition_count"
        ]
        >= 3
    )

    assert result[
        "broken_structure_core"
    ]


def test_existing_leader_primary_channel():
    result = evaluate_admission_channel(
        {
            "rs_20d_pct":
                80.0,

            "rs_60d_pct":
                82.0,

            "historical_velocity_pct":
                75.0,

            "structure_score":
                6.0,
        }
    )

    assert result[
        "existing_primary"
    ]

    assert result[
        "admission_channel_pass"
    ]


def test_existing_leader_alternate_channel():
    result = evaluate_admission_channel(
        {
            "rs_20d_pct":
                90.0,

            "rs_60d_pct":
                65.0,

            "historical_velocity_pct":
                20.0,

            "structure_score":
                2.0,
        }
    )

    assert result[
        "existing_alternate"
    ]

    assert (
        result[
            "admission_channel"
        ]
        == "EXISTING_LEADER"
    )


def test_emerging_leader_channel():
    result = evaluate_admission_channel(
        {
            "rs_20d_pct":
                60.0,

            "rs_60d_pct":
                50.0,

            "historical_velocity_pct":
                50.0,

            "structure_score":
                3.0,

            "return_5d":
                0.09,

            "return_10d":
                0.12,

            "volume_zscore_20":
                2.0,

            "rs_20d_rank_delta":
                22.0,

            "close":
                105.0,

            "sma_20":
                100.0,

            "progress_retention_10":
                0.70,
        }
    )

    assert result[
        "emerging_leader"
    ]

    assert (
        result[
            "admission_channel"
        ]
        == "EMERGING_LEADER"
    )


def test_no_channel_rejected():
    result = evaluate_admission_channel(
        {
            "rs_20d_pct":
                40.0,

            "rs_60d_pct":
                40.0,

            "historical_velocity_pct":
                40.0,

            "structure_score":
                3.0,

            "return_5d":
                0.02,

            "return_10d":
                0.03,

            "volume_zscore_20":
                0.5,

            "rs_20d_rank_delta":
                2.0,

            "close":
                100.0,

            "sma_20":
                100.0,

            "progress_retention_10":
                0.20,
        }
    )

    assert not result[
        "admission_channel_pass"
    ]

    assert (
        result[
            "admission_channel"
        ]
        == "NONE"
    )


def test_near_52w_low_fails_non_emerging():
    admission = {
        "admission_channel_pass":
            True,

        "emerging_leader":
            False,
    }

    structure = {
        "broken_structure_core":
            False,

        "near_52w_low":
            True,
    }

    gates = evaluate_hard_gates(
        admission=admission,
        structure=structure,
        setup_row={
            "setup_score":
                7,
        },
        tradability_row={
            "tradability_gate_pass":
                True,
        },
    )

    assert gates[
        "structure_hard_fail"
    ]

    assert not gates[
        "admitted"
    ]


def test_emerging_leader_gets_near_low_exemption():
    admission = {
        "admission_channel_pass":
            True,

        "emerging_leader":
            True,
    }

    structure = {
        "broken_structure_core":
            False,

        "near_52w_low":
            True,
    }

    gates = evaluate_hard_gates(
        admission=admission,
        structure=structure,
        setup_row={
            "setup_score":
                7,
        },
        tradability_row={
            "tradability_gate_pass":
                True,
        },
    )

    assert not gates[
        "structure_hard_fail"
    ]

    assert gates[
        "admitted"
    ]


def test_setup_and_tradability_are_real_hard_gates():
    admission = {
        "admission_channel_pass":
            True,

        "emerging_leader":
            False,
    }

    structure = {
        "broken_structure_core":
            False,

        "near_52w_low":
            False,
    }

    setup_fail = evaluate_hard_gates(
        admission=admission,
        structure=structure,
        setup_row={
            "setup_score":
                4,
        },
        tradability_row={
            "tradability_gate_pass":
                True,
        },
    )

    assert not setup_fail[
        "admitted"
    ]

    assert (
        "SETUP_LT_5"
        in setup_fail[
            "rejection_reasons"
        ]
    )

    trade_fail = evaluate_hard_gates(
        admission=admission,
        structure=structure,
        setup_row={
            "setup_score":
                8,
        },
        tradability_row={
            "tradability_gate_pass":
                False,
        },
    )

    assert not trade_fail[
        "admitted"
    ]

    assert (
        "TRADABILITY_FAIL"
        in trade_fail[
            "rejection_reasons"
        ]
    )


def test_future_crash_does_not_change_as_of_result():
    frame = make_trending_stock()

    as_of = frame.index[-2]

    #
    # Future crash after as_of.
    #
    frame.iloc[
        -1,
        frame.columns.get_loc(
            "open"
        ),
    ] = 20.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "high"
        ),
    ] = 21.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "low"
        ),
    ] = 19.0

    frame.iloc[
        -1,
        frame.columns.get_loc(
            "close"
        ),
    ] = 20.0

    full = (
        calculate_structure_evaluation(
            frame,
            as_of=as_of,
            swings=make_good_swings(
                frame
            ),
        )
    )

    truncated = (
        calculate_structure_evaluation(
            frame.iloc[:-1],
            as_of=as_of,
            swings=make_good_swings(
                frame
            ),
        )
    )

    assert np.isclose(
        full[
            "structure_score"
        ],
        truncated[
            "structure_score"
        ],
    )

    assert (
        full[
            "broken_structure_core"
        ]
        ==
        truncated[
            "broken_structure_core"
        ]
    )


def test_forming_watch_hard_gate_eligibility_is_separate_from_admission():
    # Preserve the near-low Emerging Leader exception of the existing gate.
    for setup, channel, broken, near_low, emerging, trade, expected in [
        (4, True, False, False, False, True, True),
        (4, True, False, True, True, True, True),
        (4, True, False, True, False, True, False),
        (4, True, True, False, True, True, False),
        (4, False, False, False, False, True, False),
        (4, True, False, False, False, False, False),
        (3, True, False, False, False, True, False),
        (5, True, False, False, False, True, False),
        (float("nan"), True, False, False, False, True, False),
    ]:
        gates = evaluate_hard_gates(
            admission={"admission_channel_pass": channel, "emerging_leader": emerging},
            structure={"broken_structure_core": broken, "near_52w_low": near_low},
            setup_row={"setup_score": setup},
            tradability_row={"tradability_gate_pass": trade},
        )
        assert gates["forming_watch_eligible"] is expected
        assert gates["admitted"] is (setup == 5)
        if setup == 4:
            assert gates["setup_hard_fail"]
            assert "SETUP_LT_5" in gates["rejection_reasons"]
