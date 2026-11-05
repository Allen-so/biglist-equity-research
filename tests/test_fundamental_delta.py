import numpy as np
import pandas as pd

from src.features.fundamental_delta import (
    calculate_earnings_reaction,
    calculate_eps_surprise,
    calculate_fundamental_delta,
    calculate_income_statement_delta,
    score_earnings_reaction,
)


def make_income_statement():
    columns = pd.to_datetime(
        [
            "2026-06-30",
            "2026-03-31",
            "2025-12-31",
            "2025-09-30",
            "2025-06-30",
            "2025-03-31",
        ]
    )

    return pd.DataFrame(
        {
            columns[0]: [
                150.0,
                30.0,
            ],
            columns[1]: [
                130.0,
                22.0,
            ],
            columns[2]: [
                125.0,
                20.0,
            ],
            columns[3]: [
                120.0,
                18.0,
            ],
            columns[4]: [
                100.0,
                10.0,
            ],
            columns[5]: [
                100.0,
                10.0,
            ],
        },
        index=[
            "Total Revenue",
            "Operating Income",
        ],
    )


def make_earnings_history():
    dates = pd.to_datetime(
        [
            "2026-07-30",
            "2026-04-30",
        ]
    )

    return pd.DataFrame(
        {
            "epsEstimate": [
                1.00,
                0.90,
            ],

            "epsActual": [
                1.20,
                0.95,
            ],
        },
        index=dates,
    )


def make_daily():
    dates = pd.bdate_range(
        "2026-06-01",
        "2026-08-10",
    )

    close = np.full(
        len(dates),
        100.0,
    )

    volume = np.full(
        len(dates),
        1_000_000.0,
    )

    frame = pd.DataFrame(
        {
            "open":
                close.copy(),

            "high":
                close + 1.0,

            "low":
                close - 1.0,

            "close":
                close.copy(),

            "volume":
                volume,
        },
        index=dates,
    )

    reaction_candidates = (
        frame[
            frame.index
            > pd.Timestamp(
                "2026-07-30"
            )
        ]
    )

    reaction_date = (
        reaction_candidates.index[0]
    )

    location = (
        frame.index.get_loc(
            reaction_date
        )
    )

    frame.iloc[
        location,
        frame.columns.get_loc(
            "open"
        ),
    ] = 105.0

    frame.iloc[
        location,
        frame.columns.get_loc(
            "high"
        ),
    ] = 112.0

    frame.iloc[
        location,
        frame.columns.get_loc(
            "low"
        ),
    ] = 104.0

    frame.iloc[
        location,
        frame.columns.get_loc(
            "close"
        ),
    ] = 110.0

    frame.iloc[
        location,
        frame.columns.get_loc(
            "volume"
        ),
    ] = 2_000_000.0

    return frame


def test_revenue_yoy():
    result = (
        calculate_income_statement_delta(
            make_income_statement()
        )
    )

    assert np.isclose(
        result[
            "revenue_yoy"
        ],
        0.50,
    )


def test_revenue_acceleration():
    result = (
        calculate_income_statement_delta(
            make_income_statement()
        )
    )

    #
    # Latest:
    # 150 / 100 - 1 = 50%
    #
    # Previous:
    # 130 / 100 - 1 = 30%
    #
    # Acceleration = +20pp
    #
    assert np.isclose(
        result[
            "revenue_acceleration"
        ],
        0.20,
    )


def test_operating_margin_delta():
    result = (
        calculate_income_statement_delta(
            make_income_statement()
        )
    )

    #
    # Latest margin = 30 / 150 = 20%
    # Year ago      = 10 / 100 = 10%
    #
    assert np.isclose(
        result[
            "operating_margin_delta"
        ],
        0.10,
    )


def test_eps_surprise():
    result = (
        calculate_eps_surprise(
            make_earnings_history()
        )
    )

    assert np.isclose(
        result[
            "eps_surprise"
        ],
        0.20,
    )


def test_positive_earnings_reaction():
    frame = make_daily()

    result = (
        calculate_earnings_reaction(
            frame,
            earnings_date=(
                pd.Timestamp(
                    "2026-07-30"
                )
            ),
        )
    )

    assert np.isclose(
        result[
            "earnings_reaction_return"
        ],
        0.10,
    )

    assert np.isclose(
        result[
            "earnings_reaction_rvol"
        ],
        2.0,
    )


def test_positive_high_volume_reaction_scores_high():
    score = score_earnings_reaction(
        reaction_return=0.10,
        reaction_rvol=2.0,
    )

    assert np.isclose(
        score,
        100.0,
    )


def test_negative_high_volume_reaction_scores_low():
    score = score_earnings_reaction(
        reaction_return=-0.10,
        reaction_rvol=2.0,
    )

    assert np.isclose(
        score,
        0.0,
    )


def test_complete_fundamental_delta():
    result = (
        calculate_fundamental_delta(
            quarterly_income=(
                make_income_statement()
            ),
            earnings_history=(
                make_earnings_history()
            ),
            daily_frame=(
                make_daily()
            ),
        )
    )

    assert (
        result[
            "fundamental_delta_status"
        ]
        == "VALUE"
    )

    assert np.isclose(
        result[
            "fundamental_coverage_weight"
        ],
        1.0,
    )

    assert (
        0
        <= result[
            "fundamental_delta_pct"
        ]
        <= 100
    )


def test_missing_data_is_not_silently_imputed():
    result = (
        calculate_fundamental_delta(
            quarterly_income=(
                pd.DataFrame()
            ),
            earnings_history=(
                pd.DataFrame()
            ),
            daily_frame=(
                pd.DataFrame()
            ),
        )
    )

    assert (
        result[
            "fundamental_delta_status"
        ]
        == "MISSING"
    )

    assert pd.isna(
        result[
            "fundamental_delta_pct"
        ]
    )

    assert np.isclose(
        result[
            "fundamental_coverage_weight"
        ],
        0.0,
    )


def test_missing_acceleration_can_still_be_value_with_85pct_coverage():
    statement = (
        make_income_statement()
        .iloc[
            :,
            :5,
        ]
    )

    result = (
        calculate_fundamental_delta(
            quarterly_income=statement,
            earnings_history=(
                make_earnings_history()
            ),
            daily_frame=(
                make_daily()
            ),
        )
    )

    assert pd.isna(
        result[
            "revenue_acceleration"
        ]
    )

    assert np.isclose(
        result[
            "fundamental_coverage_weight"
        ],
        0.85,
    )

    assert (
        result[
            "fundamental_delta_status"
        ]
        == "VALUE"
    )