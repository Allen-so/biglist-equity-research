from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


COMPONENT_WEIGHTS = {
    "revenue_yoy": 0.25,
    "revenue_acceleration": 0.15,
    "eps_surprise": 0.20,
    "operating_margin_delta": 0.20,
    "earnings_reaction": 0.20,
}


REVENUE_ALIASES = (
    "Total Revenue",
    "TotalRevenue",
    "Operating Revenue",
    "OperatingRevenue",
    "Revenue",
)


OPERATING_INCOME_ALIASES = (
    "Operating Income",
    "OperatingIncome",
    "Total Operating Income As Reported",
    "TotalOperatingIncomeAsReported",
)


def _linear_score(
    value: float,
    *,
    zero_at: float,
    full_at: float,
) -> float:

    if pd.isna(value):
        return np.nan

    if full_at <= zero_at:
        raise ValueError(
            "full_at must be greater than zero_at."
        )

    fraction = (
        float(value)
        - zero_at
    ) / (
        full_at
        - zero_at
    )

    fraction = min(
        max(
            fraction,
            0.0,
        ),
        1.0,
    )

    return float(
        fraction * 100.0
    )


def _find_statement_row(
    statement: pd.DataFrame,
    aliases: tuple[str, ...],
) -> pd.Series | None:

    if (
        statement is None
        or statement.empty
    ):
        return None

    normalized = {
        str(index).strip().lower():
            index
        for index in statement.index
    }

    for alias in aliases:
        key = (
            alias
            .strip()
            .lower()
        )

        if key in normalized:
            row = statement.loc[
                normalized[key]
            ]

            return pd.to_numeric(
                row,
                errors="coerce",
            )

    return None


def calculate_income_statement_delta(
    statement: pd.DataFrame,
) -> dict[str, Any]:
    """
    Financial-statement Delta.

    Requires quarterly columns ordered newest → oldest.

    Revenue YoY:
        Q0 / Q4 - 1

    Revenue acceleration:
        current YoY growth
        minus
        previous-quarter YoY growth

        Q1 / Q5 - 1

    Operating margin delta:
        latest margin
        minus
        same quarter one year ago
    """

    revenue = _find_statement_row(
        statement,
        REVENUE_ALIASES,
    )

    operating_income = (
        _find_statement_row(
            statement,
            OPERATING_INCOME_ALIASES,
        )
    )

    result = {
        "revenue_yoy":
            np.nan,

        "previous_revenue_yoy":
            np.nan,

        "revenue_acceleration":
            np.nan,

        "operating_margin":
            np.nan,

        "operating_margin_yoy":
            np.nan,

        "operating_margin_delta":
            np.nan,
    }

    if revenue is None:
        return result

    revenue = revenue.dropna()

    if len(revenue) >= 5:
        current = float(
            revenue.iloc[0]
        )

        year_ago = float(
            revenue.iloc[4]
        )

        if year_ago != 0:
            result[
                "revenue_yoy"
            ] = (
                current
                / year_ago
                - 1.0
            )

    if len(revenue) >= 6:
        previous = float(
            revenue.iloc[1]
        )

        previous_year = float(
            revenue.iloc[5]
        )

        if previous_year != 0:
            previous_yoy = (
                previous
                / previous_year
                - 1.0
            )

            result[
                "previous_revenue_yoy"
            ] = previous_yoy

            if not pd.isna(
                result[
                    "revenue_yoy"
                ]
            ):
                result[
                    "revenue_acceleration"
                ] = (
                    result[
                        "revenue_yoy"
                    ]
                    - previous_yoy
                )

    if (
        operating_income
        is not None
    ):
        operating_income = (
            operating_income.dropna()
        )

        if (
            len(revenue) >= 5
            and
            len(
                operating_income
            ) >= 5
        ):
            current_revenue = float(
                revenue.iloc[0]
            )

            old_revenue = float(
                revenue.iloc[4]
            )

            current_operating = float(
                operating_income.iloc[0]
            )

            old_operating = float(
                operating_income.iloc[4]
            )

            if (
                current_revenue != 0
                and
                old_revenue != 0
            ):
                current_margin = (
                    current_operating
                    / current_revenue
                )

                old_margin = (
                    old_operating
                    / old_revenue
                )

                result[
                    "operating_margin"
                ] = current_margin

                result[
                    "operating_margin_yoy"
                ] = old_margin

                result[
                    "operating_margin_delta"
                ] = (
                    current_margin
                    - old_margin
                )

    return result


def calculate_eps_surprise(
    earnings_history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Prefer direct EPS Actual vs Estimate:

        (Actual - Estimate)
        / abs(Estimate)

    This avoids relying on provider-specific
    percentage formatting.
    """

    if (
        earnings_history is None
        or earnings_history.empty
    ):
        return {
            "latest_earnings_date":
                pd.NaT,

            "eps_estimate":
                np.nan,

            "eps_actual":
                np.nan,

            "eps_surprise":
                np.nan,
        }

    history = earnings_history.copy()

    history.index = pd.to_datetime(
        history.index
    )

    history = history.sort_index(
        ascending=False
    )

    # Yahoo earnings_history is indexed by fiscal quarter, not release date.
    quarter_index = str(history.index.name).lower() == "quarter"

    column_map = {
        str(column).lower():
            column
        for column in history.columns
    }

    estimate_column = None
    actual_column = None

    for candidate in (
        "epsestimate",
        "eps estimate",
    ):
        if candidate in column_map:
            estimate_column = (
                column_map[
                    candidate
                ]
            )

            break

    for candidate in (
        "epsactual",
        "reported eps",
        "eps actual",
    ):
        if candidate in column_map:
            actual_column = (
                column_map[
                    candidate
                ]
            )

            break

    if (
        estimate_column is None
        or actual_column is None
    ):
        return {
            "latest_earnings_date":
                pd.NaT if quarter_index else history.index[0],

            "eps_estimate":
                np.nan,

            "eps_actual":
                np.nan,

            "eps_surprise":
                np.nan,
        }

    for date, row in history.iterrows():
        estimate = pd.to_numeric(
            row[
                estimate_column
            ],
            errors="coerce",
        )

        actual = pd.to_numeric(
            row[
                actual_column
            ],
            errors="coerce",
        )

        if (
            pd.isna(estimate)
            or pd.isna(actual)
        ):
            continue

        estimate = float(
            estimate
        )

        actual = float(
            actual
        )

        if estimate == 0:
            surprise = np.nan

        else:
            surprise = (
                actual
                - estimate
            ) / abs(
                estimate
            )

        return {
            "latest_earnings_date":
                pd.NaT if quarter_index else pd.Timestamp(date),

            "eps_estimate":
                estimate,

            "eps_actual":
                actual,

            "eps_surprise":
                surprise,
        }

    return {
        "latest_earnings_date":
            pd.NaT,

        "eps_estimate":
            np.nan,

        "eps_actual":
            np.nan,

        "eps_surprise":
            np.nan,
    }


def calculate_earnings_reaction(
    daily_frame: pd.DataFrame,
    *,
    earnings_date: pd.Timestamp | None,
) -> dict[str, Any]:
    """
    Conservative V1 earnings reaction.

    Use the first COMPLETE regular trading session
    strictly AFTER the earnings calendar date.

    This prevents accidentally using pre-event price
    action when the exact BMO / AMC timestamp is
    unavailable.

    Reaction:
        session close vs previous session close

    Volume:
        reaction-day volume
        /
        prior 20-session median volume
    """

    if (
        earnings_date is None
        or pd.isna(
            earnings_date
        )
        or daily_frame is None
        or daily_frame.empty
    ):
        return {
            "earnings_reaction_date":
                pd.NaT,

            "earnings_reaction_return":
                np.nan,

            "earnings_reaction_rvol":
                np.nan,
        }

    frame = daily_frame.copy()

    frame.index = pd.to_datetime(
        frame.index
    )

    event_date = pd.Timestamp(
        earnings_date
    ).tz_localize(
        None
    ) if (
        pd.Timestamp(
            earnings_date
        ).tzinfo
        is not None
    ) else pd.Timestamp(
        earnings_date
    )

    event_date = (
        event_date.normalize()
    )

    sessions = frame[
        frame.index.normalize()
        > event_date
    ]

    if sessions.empty:
        return {
            "earnings_reaction_date":
                pd.NaT,

            "earnings_reaction_return":
                np.nan,

            "earnings_reaction_rvol":
                np.nan,
        }

    reaction_date = (
        sessions.index[0]
    )

    location = (
        frame.index.get_loc(
            reaction_date
        )
    )

    if (
        not isinstance(
            location,
            (int, np.integer),
        )
        or location < 1
    ):
        return {
            "earnings_reaction_date":
                reaction_date,

            "earnings_reaction_return":
                np.nan,

            "earnings_reaction_rvol":
                np.nan,
        }

    previous_close = float(
        frame[
            "close"
        ].iloc[
            location - 1
        ]
    )

    reaction_close = float(
        frame[
            "close"
        ].iloc[
            location
        ]
    )

    if previous_close <= 0:
        reaction_return = np.nan

    else:
        reaction_return = (
            reaction_close
            / previous_close
            - 1.0
        )

    prior_volume = (
        frame[
            "volume"
        ]
        .iloc[
            max(
                0,
                location - 20,
            ):
            location
        ]
    )

    if prior_volume.empty:
        rvol = np.nan

    else:
        baseline = float(
            prior_volume.median()
        )

        if baseline <= 0:
            rvol = np.nan

        else:
            rvol = (
                float(
                    frame[
                        "volume"
                    ].iloc[
                        location
                    ]
                )
                / baseline
            )

    return {
        "earnings_reaction_date":
            pd.Timestamp(
                reaction_date
            ),

        "earnings_reaction_return":
            reaction_return,

        "earnings_reaction_rvol":
            rvol,
    }


def score_earnings_reaction(
    *,
    reaction_return: float,
    reaction_rvol: float,
) -> float:

    if pd.isna(
        reaction_return
    ):
        return np.nan

    return_score = _linear_score(
        reaction_return,
        zero_at=-0.10,
        full_at=0.10,
    )

    if pd.isna(
        reaction_rvol
    ):
        return return_score

    volume_strength = (
        _linear_score(
            reaction_rvol,
            zero_at=0.50,
            full_at=2.00,
        )
    )

    if reaction_return >= 0:
        volume_confirmation = (
            volume_strength
        )

    else:
        #
        # Heavy volume on a negative reaction is
        # bearish confirmation.
        #
        volume_confirmation = (
            100.0
            - volume_strength
        )

    return (
        0.70
        * return_score
        +
        0.30
        * volume_confirmation
    )


def calculate_fundamental_delta(
    *,
    quarterly_income: pd.DataFrame,
    earnings_history: pd.DataFrame,
    daily_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Complete Fundamental Delta 0-100.

    Component weights:

        Revenue YoY          25%
        Revenue Acceleration 15%
        EPS Surprise         20%
        Operating Margin Δ   20%
        Earnings Reaction    20%

    Missing data is explicitly tracked.

    Fundamental status becomes VALUE only when
    >= 80% of component weight is available.
    """

    income = (
        calculate_income_statement_delta(
            quarterly_income
        )
    )

    eps = calculate_eps_surprise(
        earnings_history
    )

    reaction = (
        calculate_earnings_reaction(
            daily_frame,
            earnings_date=(
                eps[
                    "latest_earnings_date"
                ]
            ),
        )
    )

    component_values = {
        "revenue_yoy":
            income[
                "revenue_yoy"
            ],

        "revenue_acceleration":
            income[
                "revenue_acceleration"
            ],

        "eps_surprise":
            eps[
                "eps_surprise"
            ],

        "operating_margin_delta":
            income[
                "operating_margin_delta"
            ],
    }

    component_scores = {
        "revenue_yoy":
            _linear_score(
                component_values[
                    "revenue_yoy"
                ],
                zero_at=-0.10,
                full_at=0.30,
            ),

        "revenue_acceleration":
            _linear_score(
                component_values[
                    "revenue_acceleration"
                ],
                zero_at=-0.10,
                full_at=0.15,
            ),

        "eps_surprise":
            _linear_score(
                component_values[
                    "eps_surprise"
                ],
                zero_at=-0.20,
                full_at=0.20,
            ),

        "operating_margin_delta":
            _linear_score(
                component_values[
                    "operating_margin_delta"
                ],
                zero_at=-0.05,
                full_at=0.05,
            ),

        "earnings_reaction":
            score_earnings_reaction(
                reaction_return=(
                    reaction[
                        "earnings_reaction_return"
                    ]
                ),
                reaction_rvol=(
                    reaction[
                        "earnings_reaction_rvol"
                    ]
                ),
            ),
    }

    available = {
        name:
            score
        for name, score in (
            component_scores.items()
        )
        if not pd.isna(score)
    }

    available_weight = float(
        sum(
            COMPONENT_WEIGHTS[
                name
            ]
            for name in available
        )
    )

    weighted_score = float(
        sum(
            available[
                name
            ]
            * COMPONENT_WEIGHTS[
                name
            ]
            for name in available
        )
    )

    if available_weight > 0:
        normalized_score = (
            weighted_score
            / available_weight
        )

    else:
        normalized_score = np.nan

    if (
        available_weight
        >= 0.80
        and
        not pd.isna(
            normalized_score
        )
    ):
        status = "VALUE"

    else:
        status = "MISSING"

    missing_components = [
        name
        for name in (
            COMPONENT_WEIGHTS
        )
        if name not in available
    ]

    return {
        **income,
        **eps,
        **reaction,

        "revenue_yoy_score":
            component_scores[
                "revenue_yoy"
            ],

        "revenue_acceleration_score":
            component_scores[
                "revenue_acceleration"
            ],

        "eps_surprise_score":
            component_scores[
                "eps_surprise"
            ],

        "operating_margin_delta_score":
            component_scores[
                "operating_margin_delta"
            ],

        "earnings_reaction_score":
            component_scores[
                "earnings_reaction"
            ],

        "fundamental_coverage_weight":
            available_weight,

        "fundamental_missing_components":
            missing_components,

        "fundamental_delta_status":
            status,

        "fundamental_delta_pct":
            (
                float(
                    normalized_score
                )
                if status
                == "VALUE"
                else np.nan
            ),
    }

def build_fundamental_snapshot(
    symbols: list[str],
    *,
    provider,
    daily_data: dict[
        str,
        pd.DataFrame,
    ],
    refresh: bool = False,
) -> pd.DataFrame:

    records = []

    for symbol in symbols:

        try:
            metadata = (
                provider.get_company_metadata(
                    symbol,
                    refresh=refresh,
                )
            )

            income = (
                provider.get_quarterly_income_statement(
                    symbol,
                    refresh=refresh,
                )
            )

            earnings = (
                provider.get_earnings_history(
                    symbol,
                    refresh=refresh,
                )
            )

            # Event rows may contain future estimates. Only completed-session data
            # belongs to this live scan; no historical point-in-time claim is made.
            daily_frame = daily_data.get(symbol, pd.DataFrame())
            if not earnings.empty and not daily_frame.empty:
                cutoff = pd.Timestamp(daily_frame.index.max()).normalize() + pd.Timedelta(hours=16)
                earnings = earnings.copy()
                earnings.index = pd.to_datetime(earnings.index)
                if earnings.index.tz is not None:
                    earnings.index = earnings.index.tz_convert("America/New_York").tz_localize(None)
                earnings = earnings.loc[earnings.index <= cutoff]

            fundamental = (
                calculate_fundamental_delta(
                    quarterly_income=income,
                    earnings_history=earnings,
                    daily_frame=(
                        daily_data.get(
                            symbol,
                            pd.DataFrame(),
                        )
                    ),
                )
            )

        except Exception as exc:
            metadata = {
                "symbol":
                    symbol,

                "sector":
                    None,

                "industry":
                    None,
            }

            fundamental = {
                "fundamental_delta_status":
                    "MISSING",

                "fundamental_delta_pct":
                    np.nan,

                "fundamental_coverage_weight":
                    0.0,

                "fundamental_missing_components":
                    list(
                        COMPONENT_WEIGHTS.keys()
                    ),

                "fundamental_error":
                    str(exc),
            }

        records.append(
            {
                "symbol":
                    symbol,

                "sector":
                    metadata.get(
                        "sector"
                    ),

                "industry":
                    metadata.get(
                        "industry"
                    ),

                **fundamental,
            }
        )

    return pd.DataFrame(
        records
    )