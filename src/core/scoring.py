from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


FACTOR_WEIGHTS = {
    "market_fit": 4.0,
    "group_fit": 3.0,
    "structure": 8.0,
    "monster_quality": 7.0,
    "volatility": 7.0,
    "historical_velocity": 8.0,
    "productive_volatility": 8.0,
    "supply_demand": 7.0,
    "progress_retention": 7.0,
    "persistent_rs": 10.0,
    "rank_level_delta": 6.0,
    "group_leadership": 5.0,
    "fundamental_delta": 5.0,
    "setup_quality": 10.0,
    "tradability": 5.0,
}


REGIME_MINIMUM_SCORE = {
    "STRONG_RISK_ON": 75.0,
    "CONSTRUCTIVE": 77.0,
    "SELECTIVE": 80.0,
    "DEFENSIVE": 83.0,
}


# Factors whose absence reflects external / contextual data coverage
# rather than an unusable trading setup. Missing these factors may
# reduce the available denominator, but does not by itself invalidate
# the entire score.
OPTIONAL_DATA_FACTORS = frozenset({
    "group_fit",
    "group_leadership",
    "fundamental_delta",
})

# Never normalize a score built from too little of the 100-point model.
MIN_AVAILABLE_MAXIMUM = 85.0


def validate_factor_weights() -> None:
    total = sum(
        FACTOR_WEIGHTS.values()
    )

    if not np.isclose(
        total,
        100.0,
    ):
        raise ValueError(
            "Final factor weights must total 100. "
            f"Current total: {total}"
        )


def _clamp(
    value: float,
    *,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return float(
        min(
            max(
                value,
                minimum,
            ),
            maximum,
        )
    )


def percentile_points(
    value: float,
    *,
    maximum_points: float,
) -> float:
    """
    Convert a 0-100 normalized factor
    into weighted Big List points.
    """

    if pd.isna(value):
        return np.nan

    normalized = _clamp(
        float(value)
    )

    return (
        normalized
        / 100.0
        * maximum_points
    )


def score_progress_retention(
    retention: float,
) -> float:
    """
    Progress Retention = 7 points.

        < 35%       = 0
        35-50%      = 2
        50-65%      = 4
        65-80%      = 5.5
        >= 80%      = 7
    """

    if pd.isna(retention):
        return np.nan

    value = float(
        retention
    )

    if value < 0.35:
        return 0.0

    if value < 0.50:
        return 2.0

    if value < 0.65:
        return 4.0

    if value < 0.80:
        return 5.5

    return 7.0


def volatility_multiplier(
    atr_pct: float,
) -> float:
    """
    ATR% preference:

        < 2%       -> 0.40
        2-3%       -> 0.70
        3-8%       -> 1.00
        8-12%      -> 0.85
        > 12%      -> 0.60
    """

    if pd.isna(atr_pct):
        return np.nan

    value = float(
        atr_pct
    )

    if value < 0.02:
        return 0.40

    if value < 0.03:
        return 0.70

    if value <= 0.08:
        return 1.00

    if value <= 0.12:
        return 0.85

    return 0.60


def score_volatility(
    *,
    volatility_pct: float,
    atr_pct: float,
) -> float:
    if (
        pd.isna(
            volatility_pct
        )
        or
        pd.isna(
            atr_pct
        )
    ):
        return np.nan

    base = percentile_points(
        volatility_pct,
        maximum_points=7.0,
    )

    multiplier = (
        volatility_multiplier(
            atr_pct
        )
    )

    return float(
        base * multiplier
    )


def score_rank_level_delta(
    *,
    rs20: float,
    rs60: float,
    rs20_delta: float,
    rs60_delta: float,
    full_improvement: float = 20.0,
) -> dict[str, float]:
    """
    Rank Level + Rank Delta = 6 points.

    Level = 4 points:
        40% RS20
        60% RS60

    Delta = 2 points:
        average positive improvement in
        RS20 / RS60 rank.

    +20 percentile points of average improvement
    receives the full 2 delta points.
    """

    values = (
        rs20,
        rs60,
        rs20_delta,
        rs60_delta,
    )

    if any(
        pd.isna(value)
        for value in values
    ):
        return {
            "rank_level_pct":
                np.nan,

            "rank_level_points":
                np.nan,

            "rank_delta_positive":
                np.nan,

            "rank_delta_points":
                np.nan,

            "rank_level_delta_points":
                np.nan,
        }

    level_pct = (
        0.40
        * float(rs20)
        +
        0.60
        * float(rs60)
    )

    level_points = (
        _clamp(
            level_pct
        )
        / 100.0
        * 4.0
    )

    positive_delta = (
        max(
            float(
                rs20_delta
            ),
            0.0,
        )
        +
        max(
            float(
                rs60_delta
            ),
            0.0,
        )
    ) / 2.0

    delta_ratio = min(
        positive_delta
        / full_improvement,
        1.0,
    )

    delta_points = (
        2.0
        * delta_ratio
    )

    return {
        "rank_level_pct":
            float(
                level_pct
            ),

        "rank_level_points":
            float(
                level_points
            ),

        "rank_delta_positive":
            float(
                positive_delta
            ),

        "rank_delta_points":
            float(
                delta_points
            ),

        "rank_level_delta_points":
            float(
                level_points
                + delta_points
            ),
    }


def _direct_points(
    value: float,
    *,
    maximum: float,
) -> float:
    """
    Factors already calculated directly on
    their final point scale.
    """

    if pd.isna(value):
        return np.nan

    return float(
        min(
            max(
                float(value),
                0.0,
            ),
            maximum,
        )
    )


def _fundamental_points(
    row: Mapping[
        str,
        Any,
    ],
) -> tuple[
    float,
    str,
]:
    """
    Fundamental data is never silently imputed.

    Accepted states:

        VALUE
        MISSING
        NOT_APPLICABLE
    """

    status = str(
        row.get(
            "fundamental_delta_status",
            "MISSING",
        )
    ).upper()

    value = row.get(
        "fundamental_delta_pct",
        np.nan,
    )

    if status == "VALUE":
        if pd.isna(value):
            return (
                np.nan,
                "MISSING",
            )

        return (
            percentile_points(
                float(value),
                maximum_points=5.0,
            ),
            "VALUE",
        )

    if status in {
        "MISSING",
        "NOT_APPLICABLE",
    }:
        return (
            np.nan,
            status,
        )

    raise ValueError(
        "fundamental_delta_status must be "
        "VALUE, MISSING or NOT_APPLICABLE."
    )


def score_candidate(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """
    Calculate frozen Big List 100-point score.

    Missing factor:
        total_score = NaN
        score_status = INCOMPLETE

    No silent neutral-value imputation.
    """

    validate_factor_weights()

    market_fit = percentile_points(
        row.get(
            "market_fit_pct",
            np.nan,
        ),
        maximum_points=4.0,
    )

    group_fit = percentile_points(
        row.get(
            "group_fit_pct",
            np.nan,
        ),
        maximum_points=3.0,
    )

    structure = _direct_points(
        row.get(
            "structure_score",
            np.nan,
        ),
        maximum=8.0,
    )

    monster = percentile_points(
        row.get(
            "monster_quality_pct",
            np.nan,
        ),
        maximum_points=7.0,
    )

    volatility = score_volatility(
        volatility_pct=row.get(
            "volatility_pct",
            np.nan,
        ),
        atr_pct=row.get(
            "atr_pct",
            np.nan,
        ),
    )

    historical_velocity = (
        percentile_points(
            row.get(
                "historical_velocity_pct",
                np.nan,
            ),
            maximum_points=8.0,
        )
    )

    productive_volatility = (
        percentile_points(
            row.get(
                "productive_volatility_pct",
                np.nan,
            ),
            maximum_points=8.0,
        )
    )

    supply_demand = (
        percentile_points(
            row.get(
                "supply_demand_pct",
                np.nan,
            ),
            maximum_points=7.0,
        )
    )

    progress_value = row.get(
        "progress_retention_ratio",
        row.get(
            "progress_retention_10",
            np.nan,
        ),
    )

    progress_retention = (
        score_progress_retention(
            progress_value
        )
    )

    persistent_rs = percentile_points(
        row.get(
            "persistent_rs",
            np.nan,
        ),
        maximum_points=10.0,
    )

    rank_data = (
        score_rank_level_delta(
            rs20=row.get(
                "rs_20d_pct",
                np.nan,
            ),
            rs60=row.get(
                "rs_60d_pct",
                np.nan,
            ),
            rs20_delta=row.get(
                "rs_20d_rank_delta",
                np.nan,
            ),
            rs60_delta=row.get(
                "rs_60d_rank_delta",
                np.nan,
            ),
        )
    )

    rank_level_delta = (
        rank_data[
            "rank_level_delta_points"
        ]
    )

    group_leadership = (
        percentile_points(
            row.get(
                "within_group_leadership_pct",
                np.nan,
            ),
            maximum_points=5.0,
        )
    )

    (
        fundamental_delta,
        fundamental_status,
    ) = _fundamental_points(
        row
    )

    setup_quality = _direct_points(
        row.get(
            "setup_score",
            np.nan,
        ),
        maximum=10.0,
    )

    tradability = _direct_points(
        row.get(
            "tradability_score",
            np.nan,
        ),
        maximum=5.0,
    )

    factor_points = {
        "market_fit":
            market_fit,

        "group_fit":
            group_fit,

        "structure":
            structure,

        "monster_quality":
            monster,

        "volatility":
            volatility,

        "historical_velocity":
            historical_velocity,

        "productive_volatility":
            productive_volatility,

        "supply_demand":
            supply_demand,

        "progress_retention":
            progress_retention,

        "persistent_rs":
            persistent_rs,

        "rank_level_delta":
            rank_level_delta,

        "group_leadership":
            group_leadership,

        "fundamental_delta":
            fundamental_delta,

        "setup_quality":
            setup_quality,

        "tradability":
            tradability,
    }

    missing_factors = [
        name
        for name, value in (
            factor_points.items()
        )
        if pd.isna(value)
    ]

    available_points = float(
        sum(
            value
            for value in (
                factor_points.values()
            )
            if not pd.isna(value)
        )
    )

    available_maximum = float(
        sum(
            FACTOR_WEIGHTS[
                name
            ]
            for name, value in (
                factor_points.items()
            )
            if not pd.isna(value)
        )
    )

    if available_maximum > 0:
        normalized_available_score = (
            available_points
            / available_maximum
            * 100.0
        )

    else:
        normalized_available_score = (
            np.nan
        )

    optional_missing_factors = [
        name
        for name in missing_factors
        if name in OPTIONAL_DATA_FACTORS
    ]

    blocking_missing_factors = [
        name
        for name in missing_factors
        if name not in OPTIONAL_DATA_FACTORS
    ]

    fundamental_error = row.get(
        "fundamental_error",
        None,
    )

    has_fundamental_provider_error = (
        fundamental_error is not None
        and not pd.isna(fundamental_error)
        and str(fundamental_error).strip() != ""
    )

    scoring_complete = (
        len(blocking_missing_factors) == 0
        and not has_fundamental_provider_error
        and available_maximum >= MIN_AVAILABLE_MAXIMUM
        and not pd.isna(normalized_available_score)
    )

    if scoring_complete:
        # When optional factors are unavailable, compare candidates on
        # the same 0-100 scale using only the available model weight.
        total_score = float(
            normalized_available_score
        )

        score_status = (
            "COMPLETE"
        )

    else:
        total_score = np.nan
        score_status = "INCOMPLETE"

    return {
        "market_fit_points":
            market_fit,

        "group_fit_points":
            group_fit,

        "structure_points":
            structure,

        "monster_quality_points":
            monster,

        "volatility_points":
            volatility,

        "historical_velocity_points":
            historical_velocity,

        "productive_volatility_points":
            productive_volatility,

        "supply_demand_points":
            supply_demand,

        "progress_retention_points":
            progress_retention,

        "persistent_rs_points":
            persistent_rs,

        "rank_level_delta_points":
            rank_level_delta,

        "group_leadership_points":
            group_leadership,

        "fundamental_delta_points":
            fundamental_delta,

        "setup_quality_points":
            setup_quality,

        "tradability_points":
            tradability,

        **rank_data,

        "fundamental_delta_status":
            fundamental_status,

        "available_score_points":
            round(
                available_points,
                4,
            ),

        "available_maximum_points":
            available_maximum,

        "normalized_available_score":
            (
                round(
                    normalized_available_score,
                    4,
                )
                if not pd.isna(
                    normalized_available_score
                )
                else np.nan
            ),

        "missing_factors":
            missing_factors,

        "missing_factor_count":
            len(
                missing_factors
            ),

        "scoring_complete":
            bool(
                scoring_complete
            ),

        "score_status":
            score_status,

        "total_score":
            (
                round(
                    total_score,
                    4,
                )
                if not pd.isna(
                    total_score
                )
                else np.nan
            ),
    }


def regime_minimum_score(
    regime: str,
) -> float:
    normalized = (
        str(regime)
        .strip()
        .upper()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )

    if (
        normalized
        not in REGIME_MINIMUM_SCORE
    ):
        raise ValueError(
            "Unknown market regime: "
            f"{regime}"
        )

    return (
        REGIME_MINIMUM_SCORE[
            normalized
        ]
    )


def build_score_snapshot(
    candidates: pd.DataFrame,
    *,
    regime: str,
) -> pd.DataFrame:
    """
    Score candidate universe.

    Final score eligibility requires:

        Hard Gates passed
        Score complete
        Score >= regime threshold
    """

    if candidates.empty:
        return candidates.copy()

    minimum_score = (
        regime_minimum_score(
            regime
        )
    )

    records = []

    for _, row in (
        candidates.iterrows()
    ):
        score = score_candidate(
            row
        )

        record = {
            **row.to_dict(),
            **score,

            "regime":
                str(regime),

            "regime_minimum_score":
                minimum_score,
        }

        admitted = bool(
            row.get(
                "admitted",
                False,
            )
        )

        score_pass = (
            score[
                "scoring_complete"
            ]
            and
            score[
                "total_score"
            ]
            >= minimum_score
        )

        record[
            "score_gate_pass"
        ] = bool(
            score_pass
        )

        record[
            "final_score_eligible"
        ] = bool(
            admitted
            and
            score_pass
        )

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    return (
        result
        .sort_values(
            [
                "final_score_eligible",
                "total_score",
                "normalized_available_score",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )