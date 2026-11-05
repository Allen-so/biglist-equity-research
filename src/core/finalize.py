from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CAPACITY_RANGES = {
    "STRONG_RISK_ON": {
        "score_low": 75.0,
        "score_high": 100.0,
        "capacity_min": 45,
        "capacity_max": 70,
    },

    "CONSTRUCTIVE": {
        "score_low": 60.0,
        "score_high": 75.0,
        "capacity_min": 35,
        "capacity_max": 55,
    },

    "SELECTIVE": {
        "score_low": 45.0,
        "score_high": 60.0,
        "capacity_min": 20,
        "capacity_max": 40,
    },

    "DEFENSIVE": {
        "score_low": 0.0,
        "score_high": 45.0,
        "capacity_min": 10,
        "capacity_max": 25,
    },
}


CLASSIFICATION_PRIORITY = {
    "READY": 0,
    "DEVELOPING": 1,
    "WATCH": 2,
    "EXTENDED": 3,
    "REJECTED": 4,
    "INCOMPLETE": 5,
}


def normalize_regime(
    regime: str,
) -> str:
    value = (
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

    if value not in CAPACITY_RANGES:
        raise ValueError(
            f"Unknown market regime: {regime}"
        )

    return value


def regime_from_score(
    regime_score: float,
) -> str:
    """
    Convert the future Market Regime 0-100 score
    into its frozen regime classification.
    """

    if pd.isna(regime_score):
        raise ValueError(
            "regime_score cannot be NaN."
        )

    score = float(
        regime_score
    )

    if not 0 <= score <= 100:
        raise ValueError(
            "regime_score must be between 0 and 100."
        )

    if score >= 75:
        return "STRONG_RISK_ON"

    if score >= 60:
        return "CONSTRUCTIVE"

    if score >= 45:
        return "SELECTIVE"

    return "DEFENSIVE"


def dynamic_capacity(
    *,
    regime: str,
    regime_score: float | None = None,
) -> int:
    """
    Dynamic Capacity is a CEILING, never a quota.

    Example:

        Strong Risk-on:
            score 75  -> capacity 45
            score 100 -> capacity 70

    If the precise regime score is unavailable,
    use the conservative minimum capacity for the
    regime.

    This avoids inventing false confidence.
    """

    normalized = normalize_regime(
        regime
    )

    config = CAPACITY_RANGES[
        normalized
    ]

    minimum = int(
        config[
            "capacity_min"
        ]
    )

    maximum = int(
        config[
            "capacity_max"
        ]
    )

    if regime_score is None:
        return minimum

    score = float(
        regime_score
    )

    inferred = regime_from_score(
        score
    )

    if inferred != normalized:
        raise ValueError(
            "regime and regime_score are inconsistent: "
            f"{normalized} vs score={score} "
            f"({inferred})"
        )

    score_low = float(
        config[
            "score_low"
        ]
    )

    score_high = float(
        config[
            "score_high"
        ]
    )

    if score_high <= score_low:
        return minimum

    fraction = (
        score
        - score_low
    ) / (
        score_high
        - score_low
    )

    fraction = min(
        max(
            fraction,
            0.0,
        ),
        1.0,
    )

    raw_capacity = (
        minimum
        +
        fraction
        * (
            maximum
            - minimum
        )
    )

    #
    # Conventional half-up rounding.
    #
    return int(
        np.floor(
            raw_capacity
            + 0.5
        )
    )


def _tradability_state(
    row: pd.Series | dict[str, Any],
) -> str:
    """
    Prefer an explicitly renamed tradability field.

    Fall back to the original Phase-5B
    `classification` column.
    """

    value = row.get(
        "tradability_classification",
        row.get(
            "classification",
            None,
        ),
    )

    if value is None:
        return "INCOMPLETE"

    return str(
        value
    ).upper()


def _hourly_confirmation(
    row: pd.Series | dict[str, Any],
) -> str:
    """
    1H Flow is intentionally NOT a hard gate.

    Missing 1H confirmation behaves as NEUTRAL.
    """

    value = row.get(
        "hourly_confirmation",
        "NEUTRAL",
    )

    if (
        value is None
        or pd.isna(value)
    ):
        return "NEUTRAL"

    value = str(
        value
    ).upper()

    if value not in {
        "SUPPORTIVE",
        "NEUTRAL",
        "NEGATIVE",
    }:
        return "NEUTRAL"

    return value


def classify_final_candidate(
    row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    """
    Final classification hierarchy.

    Priority:

        Hard Gates
        ↓
        Score completeness
        ↓
        Regime score gate
        ↓
        Tradability location
        ↓
        1H timing confirmation

    1H Flow may downgrade timing by ONE level.

    It may NOT rescue bad Tradability,
    and may NOT hard-reject a good Daily setup.
    """

    admitted = bool(
        row.get(
            "admitted",
            False,
        )
    )

    scoring_complete = bool(
        row.get(
            "scoring_complete",
            False,
        )
    )

    score_gate_pass = bool(
        row.get(
            "score_gate_pass",
            False,
        )
    )

    tradability = (
        _tradability_state(
            row
        )
    )

    hourly = (
        _hourly_confirmation(
            row
        )
    )

    #
    # Hard Gates dominate everything.
    #
    if not admitted:
        # A separate Watch-only path never changes formal admission. Missing
        # eligibility (including legacy rows) fails closed.
        forming_watch_eligible = row.get("forming_watch_eligible", False)
        if pd.notna(forming_watch_eligible) and bool(forming_watch_eligible):
            if not scoring_complete:
                return {
                    "final_classification": "REJECTED",
                    "classification_reason": "SCORE_INCOMPLETE",
                }
            if not score_gate_pass:
                return {
                    "final_classification": "REJECTED",
                    "classification_reason": "REGIME_SCORE_FAIL",
                }
            if tradability in {"READY", "DEVELOPING", "WATCH"}:
                return {
                    "final_classification": "WATCH",
                    "classification_reason": "FORMING_SETUP_WATCH",
                }
            return {
                "final_classification": "REJECTED",
                "classification_reason": "FORMING_TRADABILITY_FAIL",
            }

        # An invalid setup remains rejected even at an extended location.
        if tradability == "EXTENDED" and not bool(row.get("setup_hard_fail", False)):
            return {
                "final_classification":
                    "EXTENDED",

                "classification_reason":
                    "HARD_GATE_FAIL_EXTENDED",
            }

        return {
            "final_classification":
                "REJECTED",

            "classification_reason":
                "HARD_GATE_FAIL",
        }

    #
    # Missing factors are a DATA state,
    # not a trading rejection.
    #
    if not scoring_complete:
        return {
            "final_classification":
                "INCOMPLETE",

            "classification_reason":
                "SCORE_INCOMPLETE",
        }

    #
    # Complete score, but does not clear the
    # market-regime threshold.
    #
    if not score_gate_pass:
        return {
            "final_classification":
                "REJECTED",

            "classification_reason":
                "REGIME_SCORE_FAIL",
        }

    #
    # Tradability itself may already indicate
    # an invalid location.
    #
    if tradability == "EXTENDED":
        return {
            "final_classification":
                "EXTENDED",

            "classification_reason":
                "EXTENSION_HARD_FAIL",
        }

    if tradability == "REJECTED":
        return {
            "final_classification":
                "REJECTED",

            "classification_reason":
                "TRADABILITY_REJECTED",
        }

    if tradability == "INCOMPLETE":
        return {
            "final_classification":
                "INCOMPLETE",

            "classification_reason":
                "TRADABILITY_INCOMPLETE",
        }

    #
    # READY:
    #
    # Negative 1H flow downgrades timing
    # one level only.
    #
    if tradability == "READY":
        if hourly == "NEGATIVE":
            return {
                "final_classification":
                    "DEVELOPING",

                "classification_reason":
                    "HOURLY_NEGATIVE_DOWNGRADE",
            }

        return {
            "final_classification":
                "READY",

            "classification_reason":
                (
                    "HOURLY_SUPPORTIVE"
                    if hourly
                    == "SUPPORTIVE"
                    else
                    "DAILY_READY"
                ),
        }

    #
    # DEVELOPING:
    #
    # Negative 1H flow moves it to WATCH.
    #
    # Supportive 1H does NOT upgrade it to READY,
    # because its Daily entry location is still
    # objectively not Ready.
    #
    if tradability == "DEVELOPING":
        if hourly == "NEGATIVE":
            return {
                "final_classification":
                    "WATCH",

                "classification_reason":
                    "HOURLY_NEGATIVE_DOWNGRADE",
            }

        return {
            "final_classification":
                "DEVELOPING",

            "classification_reason":
                (
                    "HOURLY_SUPPORTIVE_BUT_LOCATION_DEVELOPING"
                    if hourly
                    == "SUPPORTIVE"
                    else
                    "DAILY_DEVELOPING"
                ),
        }

    if tradability == "WATCH":
        return {
            "final_classification":
                "WATCH",

            "classification_reason":
                "DAILY_WATCH",
        }

    return {
        "final_classification":
            "INCOMPLETE",

        "classification_reason":
            "UNKNOWN_TRADABILITY_STATE",
    }


def apply_final_classification(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply final classification to every candidate.
    """

    if candidates.empty:
        return candidates.copy()

    records = []

    for _, row in (
        candidates.iterrows()
    ):
        result = (
            classify_final_candidate(
                row
            )
        )

        record = {
            **row.to_dict(),
            **result,
        }

        record[
            "hourly_confirmation"
        ] = _hourly_confirmation(
            row
        )

        record[
            "tradability_classification"
        ] = _tradability_state(
            row
        )

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    # Data completeness is independent of classification and selection.
    incomplete = result.get(
        "scoring_complete", pd.Series(True, index=result.index)
    ).eq(False).fillna(False)
    result["data_status"] = incomplete.map(
        {True: "INCOMPLETE", False: "COMPLETE"}
    )

    result[
        "_class_priority"
    ] = (
        result[
            "final_classification"
        ]
        .map(
            CLASSIFICATION_PRIORITY
        )
        .fillna(
            99
        )
    )

    return (
        result
        .sort_values(
            [
                "_class_priority",
                "total_score",
                "normalized_available_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_class_priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_final_lists(
    candidates: pd.DataFrame,
    *,
    regime: str,
    regime_score: float | None = None,
) -> dict[str, Any]:
    """
    Build final user-facing Big List.

    BIG LIST =
        READY
        +
        DEVELOPING

    WATCH is deliberately separate.

    Capacity is an upper bound only.
    Weak names are NEVER added just to fill it.
    """

    classified = (
        apply_final_classification(
            candidates
        )
    )

    capacity = dynamic_capacity(
        regime=regime,
        regime_score=regime_score,
    )

    if classified.empty:
        empty = classified.copy()

        return {
            "regime":
                normalize_regime(
                    regime
                ),

            "regime_score":
                regime_score,

            "capacity":
                capacity,

            "capacity_used":
                0,

            "biglist":
                empty,

            "ready":
                empty,

            "developing":
                empty,

            "watch":
                empty,

            "extended":
                empty,

            "rejected":
                empty,

            "incomplete":
                empty,

            "full":
                empty,
        }

    classified[
        "selected_biglist"
    ] = False

    classified[
        "selection_reason"
    ] = None

    eligible = classified[
        classified[
            "final_classification"
        ].isin(
            [
                "READY",
                "DEVELOPING",
            ]
        )
    ].copy()

    eligible[
        "_selection_priority"
    ] = (
        eligible[
            "final_classification"
        ]
        .map(
            {
                "READY": 0,
                "DEVELOPING": 1,
            }
        )
    )

    #
    # READY receives first priority.
    #
    # Within the same class:
    # highest Final 100-point score wins.
    #
    eligible = (
        eligible
        .sort_values(
            [
                "_selection_priority",
                "total_score",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
    )

    selected_indices = (
        eligible
        .head(
            capacity
        )
        .index
    )

    classified.loc[
        selected_indices,
        "selected_biglist",
    ] = True

    classified.loc[
        selected_indices,
        "selection_reason",
    ] = "SELECTED"

    overflow_indices = (
        eligible.index.difference(
            selected_indices
        )
    )

    classified.loc[
        overflow_indices,
        "selection_reason",
    ] = "CAPACITY_CUTOFF"

    biglist = classified[
        classified[
            "selected_biglist"
        ]
    ].copy()

    #
    # Restore deterministic READY-first ordering.
    #
    biglist[
        "_priority"
    ] = (
        biglist[
            "final_classification"
        ]
        .map(
            {
                "READY": 0,
                "DEVELOPING": 1,
            }
        )
    )

    biglist = (
        biglist
        .sort_values(
            [
                "_priority",
                "total_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns=[
                "_priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ready = biglist[
        biglist[
            "final_classification"
        ]
        == "READY"
    ].copy()

    developing = biglist[
        biglist[
            "final_classification"
        ]
        == "DEVELOPING"
    ].copy()

    watch = classified[
        classified[
            "final_classification"
        ]
        == "WATCH"
    ].copy()

    extended = classified[
        classified[
            "final_classification"
        ]
        == "EXTENDED"
    ].copy()

    rejected = classified[
        classified[
            "final_classification"
        ]
        == "REJECTED"
    ].copy()

    incomplete = classified[
        classified[
            "final_classification"
        ]
        == "INCOMPLETE"
    ].copy()

    return {
        "regime":
            normalize_regime(
                regime
            ),

        "regime_score":
            regime_score,

        "capacity":
            capacity,

        "capacity_used":
            len(
                biglist
            ),

        "biglist":
            biglist,

        "ready":
            ready,

        "developing":
            developing,

        "watch":
            watch,

        "extended":
            extended,

        "rejected":
            rejected,

        "incomplete":
            incomplete,

        "full":
            classified,
    }