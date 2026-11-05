from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Strict as-of protection.
    """

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return df[
        df.index <= as_of
    ].copy()


def classify_pivot_proximity(
    *,
    current_price: float,
    pivot_price: float,
    current_atr: float,
    ready_max: float = 0.80,
    developing_max: float = 1.80,
    watch_max: float = 2.50,
) -> dict[str, Any]:
    """
    Classify current distance from objective pivot.

    Signed distance:
        positive = price above pivot
        negative = price below pivot

    Absolute distance is used for proximity state.
    """

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    if pd.isna(pivot_price):
        return {
            "pivot_distance_atr":
                np.nan,

            "pivot_distance_atr_signed":
                np.nan,

            "entry_proximity_state":
                "NO_PIVOT",
        }

    signed_distance = (
        current_price
        - pivot_price
    ) / current_atr

    distance = abs(
        signed_distance
    )

    if distance <= ready_max:
        state = "READY"

    elif distance <= developing_max:
        state = "DEVELOPING"

    elif distance <= watch_max:
        state = "WATCH"

    else:
        state = "TOO_FAR"

    return {
        "pivot_distance_atr":
            float(distance),

        "pivot_distance_atr_signed":
            float(
                signed_distance
            ),

        "entry_proximity_state":
            state,
    }


def collect_support_candidates(
    frame: pd.DataFrame,
    *,
    current_price: float,
    current_atr: float,
    zones: pd.DataFrame | None = None,
    confluence: pd.DataFrame | None = None,
    trendlines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build objective structural support candidates.

    Sources:
        Structural Confluence
        Horizontal Support Zone
        Intact Support Trendline
        Daily SMA20
        Daily SMA50

    Resistance and support are not manually selected.
    """

    records: list[
        dict[str, Any]
    ] = []

    #
    # Structural confluence.
    #
    if (
        confluence is not None
        and not confluence.empty
    ):
        supports = confluence[
            confluence["role"]
            == "SUPPORT"
        ]

        for _, row in supports.iterrows():
            low = float(
                row["cluster_low"]
            )

            high = float(
                row["cluster_high"]
            )

            if high >= current_price:
                continue

            records.append(
                {
                    "source":
                        "CONFLUENCE",

                    "lower_bound":
                        low,

                    "upper_bound":
                        high,

                    "reference_level":
                        high,

                    "confidence":
                        float(
                            row.get(
                                "score",
                                np.nan,
                            )
                        ),
                }
            )

    #
    # Horizontal zones.
    #
    if (
        zones is not None
        and not zones.empty
    ):
        supports = zones[
            zones["role"]
            == "SUPPORT"
        ]

        for _, row in supports.iterrows():
            low = float(
                row["zone_low"]
            )

            high = float(
                row["zone_high"]
            )

            if high >= current_price:
                continue

            records.append(
                {
                    "source":
                        "HORIZONTAL_ZONE",

                    "lower_bound":
                        low,

                    "upper_bound":
                        high,

                    "reference_level":
                        high,

                    "confidence":
                        float(
                            row.get(
                                "confidence",
                                np.nan,
                            )
                        ),
                }
            )

    #
    # Support trendlines.
    #
    if (
        trendlines is not None
        and not trendlines.empty
    ):
        valid = trendlines[
            ~trendlines[
                "status"
            ].isin(
                [
                    "BROKEN_UP",
                    "BROKEN_DOWN",
                ]
            )
        ]

        valid = valid[
            valid["kind"]
            == "SUPPORT"
        ]

        for _, row in valid.iterrows():
            level = float(
                row[
                    "line_price_now"
                ]
            )

            if level >= current_price:
                continue

            records.append(
                {
                    "source":
                        "TRENDLINE",

                    "lower_bound":
                        level,

                    "upper_bound":
                        level,

                    "reference_level":
                        level,

                    "confidence":
                        float(
                            row.get(
                                "confidence",
                                np.nan,
                            )
                        ),
                }
            )

    #
    # Dynamic daily support.
    #
    for window in (
        20,
        50,
    ):
        if len(frame) < window:
            continue

        level = float(
            frame["close"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .iloc[-1]
        )

        if (
            pd.isna(level)
            or level >= current_price
        ):
            continue

        records.append(
            {
                "source":
                    f"SMA{window}",

                "lower_bound":
                    level,

                "upper_bound":
                    level,

                "reference_level":
                    level,

                "confidence":
                    50.0,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "source",
                "lower_bound",
                "upper_bound",
                "reference_level",
                "confidence",
                "distance_atr",
            ]
        )

    result = pd.DataFrame(
        records
    )

    result[
        "distance_atr"
    ] = (
        current_price
        - result[
            "reference_level"
        ]
    ) / current_atr

    result = result[
        result[
            "distance_atr"
        ] >= 0
    ]

    return (
        result
        .sort_values(
            [
                "distance_atr",
                "confidence",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def select_structural_stop(
    support_candidates: pd.DataFrame,
    *,
    current_price: float,
    current_atr: float,
    buffer_atr: float = 0.25,
) -> dict[str, Any]:
    """
    Structural stop:

        nearest valid support
        minus
        0.25 ATR

    For a support ZONE, stop goes below the LOWER
    boundary, not inside the zone.
    """

    if support_candidates.empty:
        return {
            "support_source":
                None,

            "support_reference":
                np.nan,

            "stop_price":
                np.nan,

            "risk_per_share":
                np.nan,

            "stop_distance_atr":
                np.nan,
        }

    support = (
        support_candidates.iloc[0]
    )

    lower_bound = float(
        support[
            "lower_bound"
        ]
    )

    reference = float(
        support[
            "reference_level"
        ]
    )

    stop_price = (
        lower_bound
        - buffer_atr
        * current_atr
    )

    risk = (
        current_price
        - stop_price
    )

    if risk <= 0:
        risk_atr = np.nan

    else:
        risk_atr = (
            risk
            / current_atr
        )

    return {
        "support_source":
            str(
                support[
                    "source"
                ]
            ),

        "support_reference":
            reference,

        "stop_price":
            float(
                stop_price
            ),

        "risk_per_share":
            float(risk),

        "stop_distance_atr":
            (
                float(risk_atr)
                if not pd.isna(
                    risk_atr
                )
                else np.nan
            ),
    }


def collect_resistance_candidates(
    *,
    current_price: float,
    current_atr: float,
    zones: pd.DataFrame | None = None,
    confluence: pd.DataFrame | None = None,
    trendlines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build objective overhead supply candidates.

    Deliberately excludes moving averages from RR
    targeting. Reward targets should represent actual
    structural supply / resistance.
    """

    records: list[
        dict[str, Any]
    ] = []

    if (
        confluence is not None
        and not confluence.empty
    ):
        resistance = confluence[
            confluence["role"]
            == "RESISTANCE"
        ]

        for _, row in resistance.iterrows():
            low = float(
                row["cluster_low"]
            )

            high = float(
                row["cluster_high"]
            )

            if low <= current_price:
                continue

            records.append(
                {
                    "source":
                        "CONFLUENCE",

                    "lower_bound":
                        low,

                    "upper_bound":
                        high,

                    "target_level":
                        low,

                    "confidence":
                        float(
                            row.get(
                                "score",
                                np.nan,
                            )
                        ),
                }
            )

    if (
        zones is not None
        and not zones.empty
    ):
        resistance = zones[
            zones["role"]
            == "RESISTANCE"
        ]

        for _, row in resistance.iterrows():
            low = float(
                row["zone_low"]
            )

            high = float(
                row["zone_high"]
            )

            if low <= current_price:
                continue

            records.append(
                {
                    "source":
                        "HORIZONTAL_ZONE",

                    "lower_bound":
                        low,

                    "upper_bound":
                        high,

                    "target_level":
                        low,

                    "confidence":
                        float(
                            row.get(
                                "confidence",
                                np.nan,
                            )
                        ),
                }
            )

    if (
        trendlines is not None
        and not trendlines.empty
    ):
        valid = trendlines[
            ~trendlines[
                "status"
            ].isin(
                [
                    "BROKEN_UP",
                    "BROKEN_DOWN",
                ]
            )
        ]

        valid = valid[
            valid["kind"]
            == "RESISTANCE"
        ]

        for _, row in valid.iterrows():
            level = float(
                row[
                    "line_price_now"
                ]
            )

            if level <= current_price:
                continue

            records.append(
                {
                    "source":
                        "TRENDLINE",

                    "lower_bound":
                        level,

                    "upper_bound":
                        level,

                    "target_level":
                        level,

                    "confidence":
                        float(
                            row.get(
                                "confidence",
                                np.nan,
                            )
                        ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "source",
                "lower_bound",
                "upper_bound",
                "target_level",
                "confidence",
                "distance_atr",
            ]
        )

    result = pd.DataFrame(
        records
    )

    result[
        "distance_atr"
    ] = (
        result[
            "target_level"
        ]
        - current_price
    ) / current_atr

    return (
        result
        .sort_values(
            [
                "distance_atr",
                "confidence",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def evaluate_reward_risk(
    resistance_candidates: pd.DataFrame,
    *,
    current_price: float,
    risk_per_share: float,
    current_atr: float,
) -> dict[str, Any]:
    """
    Conservative reward uses the FIRST boundary
    of the nearest resistance zone.

    If no valid overhead resistance exists:
        BLUE_SKY

    BLUE_SKY passes the RR hard gate, but no arbitrary
    price target is invented.
    """

    if (
        pd.isna(
            risk_per_share
        )
        or risk_per_share <= 0
    ):
        return {
            "resistance_source":
                None,

            "resistance_price":
                np.nan,

            "overhead_room_atr":
                np.nan,

            "reward_per_share":
                np.nan,

            "reward_risk":
                np.nan,

            "reward_risk_state":
                "NO_VALID_RISK",

            "blue_sky":
                False,
        }

    if resistance_candidates.empty:
        return {
            "resistance_source":
                None,

            "resistance_price":
                np.nan,

            "overhead_room_atr":
                np.nan,

            "reward_per_share":
                np.nan,

            "reward_risk":
                np.nan,

            "reward_risk_state":
                "BLUE_SKY",

            "blue_sky":
                True,
        }

    resistance = (
        resistance_candidates.iloc[0]
    )

    target = float(
        resistance[
            "target_level"
        ]
    )

    reward = (
        target
        - current_price
    )

    overhead_room_atr = (
        reward
        / current_atr
    )

    reward_risk = (
        reward
        / risk_per_share
    )

    return {
        "resistance_source":
            str(
                resistance[
                    "source"
                ]
            ),

        "resistance_price":
            target,

        "overhead_room_atr":
            float(
                overhead_room_atr
            ),

        "reward_per_share":
            float(reward),

        "reward_risk":
            float(
                reward_risk
            ),

        "reward_risk_state":
            "DEFINED",

        "blue_sky":
            False,
    }


def score_tradability(
    *,
    extension_atr: float,
    pivot_distance_atr: float,
    stop_distance_atr: float,
    overhead_room_atr: float,
    reward_risk: float,
    blue_sky: bool,
) -> dict[str, float]:
    """
    Tradability score = 5 points.

        Extension          1.50
        Pivot Proximity    1.00
        Stop Efficiency    1.00
        Overhead Room      0.75
        Reward / Risk      0.75
    """

    #
    # Extension — 1.5
    #
    if pd.isna(
        extension_atr
    ):
        extension_score = 0.0

    elif extension_atr <= 1.0:
        extension_score = 1.50

    elif extension_atr <= 2.0:
        extension_score = 1.00

    elif extension_atr <= 2.5:
        extension_score = 0.50

    elif extension_atr <= 3.0:
        extension_score = 0.25

    else:
        extension_score = 0.0

    #
    # Pivot proximity — 1.0
    #
    if pd.isna(
        pivot_distance_atr
    ):
        pivot_score = 0.0

    elif pivot_distance_atr <= 0.80:
        pivot_score = 1.00

    elif pivot_distance_atr <= 1.80:
        pivot_score = 0.70

    elif pivot_distance_atr <= 2.50:
        pivot_score = 0.30

    else:
        pivot_score = 0.0

    #
    # Stop efficiency — 1.0
    #
    if pd.isna(
        stop_distance_atr
    ):
        stop_score = 0.0

    elif stop_distance_atr <= 1.0:
        stop_score = 1.00

    elif stop_distance_atr <= 2.0:
        stop_score = 0.75

    elif stop_distance_atr <= 3.0:
        stop_score = 0.40

    else:
        stop_score = 0.0

    #
    # Overhead room — 0.75
    #
    if blue_sky:
        room_score = 0.75

    elif pd.isna(
        overhead_room_atr
    ):
        room_score = 0.0

    elif overhead_room_atr >= 4.0:
        room_score = 0.75

    elif overhead_room_atr >= 2.0:
        room_score = 0.50

    elif overhead_room_atr >= 1.0:
        room_score = 0.25

    else:
        room_score = 0.0

    #
    # Reward / Risk — 0.75
    #
    if blue_sky:
        rr_score = 0.75

    elif pd.isna(
        reward_risk
    ):
        rr_score = 0.0

    elif reward_risk >= 3.0:
        rr_score = 0.75

    elif reward_risk >= 2.0:
        rr_score = 0.50

    else:
        rr_score = 0.0

    total = (
        extension_score
        + pivot_score
        + stop_score
        + room_score
        + rr_score
    )

    return {
        "extension_score":
            extension_score,

        "pivot_proximity_score":
            pivot_score,

        "stop_efficiency_score":
            stop_score,

        "overhead_room_score":
            room_score,

        "reward_risk_score":
            rr_score,

        "tradability_score":
            round(
                float(total),
                2,
            ),
    }


def calculate_tradability_features(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    setup_row: Mapping[
        str,
        Any,
    ] | pd.Series | None = None,
    zones: pd.DataFrame | None = None,
    confluence: pd.DataFrame | None = None,
    trendlines: pd.DataFrame | None = None,
    extension_hard_fail: float = 3.0,
    stop_hard_fail: float = 3.0,
    min_reward_risk: float = 2.0,
    stop_buffer_atr: float = 0.25,
) -> dict[str, Any]:
    """
    Complete Phase 5B Tradability Engine.

    Hard failures:

        No structural stop
        Extension > 3 ATR
        Stop distance > 3 ATR
        Defined RR < 2

    BLUE_SKY is explicitly exempt from the RR gate.
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    if visible.empty:
        return {}

    visible = add_atr(
        visible,
        period=14,
    )

    current_price = float(
        visible[
            "close"
        ].iloc[-1]
    )

    current_atr = visible[
        "atr_14"
    ].iloc[-1]

    if (
        pd.isna(
            current_atr
        )
        or current_atr <= 0
    ):
        return {
            "tradability_gate_pass":
                False,

            "classification":
                "REJECTED",

            "rejection_reason":
                "ATR_NOT_AVAILABLE",
        }

    current_atr = float(
        current_atr
    )

    pivot_price = np.nan

    if setup_row is not None:
        value = setup_row.get(
            "pivot_price",
            np.nan,
        )

        if not pd.isna(value):
            pivot_price = float(
                value
            )

    proximity = (
        classify_pivot_proximity(
            current_price=current_price,
            pivot_price=pivot_price,
            current_atr=current_atr,
        )
    )

    support_candidates = (
        collect_support_candidates(
            visible,
            current_price=current_price,
            current_atr=current_atr,
            zones=zones,
            confluence=confluence,
            trendlines=trendlines,
        )
    )

    stop = select_structural_stop(
        support_candidates,
        current_price=current_price,
        current_atr=current_atr,
        buffer_atr=(
            stop_buffer_atr
        ),
    )

    if not support_candidates.empty:
        nearest_support_level = float(
            support_candidates.iloc[0][
                "reference_level"
            ]
        )

        extension_atr = (
            current_price
            - nearest_support_level
        ) / current_atr

    else:
        extension_atr = np.nan

    resistance_candidates = (
        collect_resistance_candidates(
            current_price=current_price,
            current_atr=current_atr,
            zones=zones,
            confluence=confluence,
            trendlines=trendlines,
        )
    )

    reward_risk = (
        evaluate_reward_risk(
            resistance_candidates,
            current_price=current_price,
            risk_per_share=(
                stop[
                    "risk_per_share"
                ]
            ),
            current_atr=current_atr,
        )
    )

    rejection_reasons = []

    #
    # Hard Gate 1:
    # must have a definable structural stop.
    #
    if pd.isna(
        stop[
            "stop_distance_atr"
        ]
    ):
        rejection_reasons.append(
            "NO_STRUCTURAL_STOP"
        )

    #
    # Hard Gate 2:
    # extension.
    #
    if (
        not pd.isna(
            extension_atr
        )
        and
        extension_atr
        > extension_hard_fail
    ):
        rejection_reasons.append(
            "EXTENSION_GT_3ATR"
        )

    #
    # Hard Gate 3:
    # stop distance.
    #
    if (
        not pd.isna(
            stop[
                "stop_distance_atr"
            ]
        )
        and
        stop[
            "stop_distance_atr"
        ]
        > stop_hard_fail
    ):
        rejection_reasons.append(
            "STOP_GT_3ATR"
        )

    #
    # Hard Gate 4:
    # reward / risk.
    #
    if (
        not reward_risk[
            "blue_sky"
        ]
        and
        reward_risk[
            "reward_risk_state"
        ]
        == "DEFINED"
        and
        reward_risk[
            "reward_risk"
        ]
        < min_reward_risk
    ):
        rejection_reasons.append(
            "RR_LT_2"
        )

    tradability_gate_pass = (
        len(
            rejection_reasons
        )
        == 0
    )

    scores = score_tradability(
        extension_atr=(
            extension_atr
        ),
        pivot_distance_atr=(
            proximity[
                "pivot_distance_atr"
            ]
        ),
        stop_distance_atr=(
            stop[
                "stop_distance_atr"
            ]
        ),
        overhead_room_atr=(
            reward_risk[
                "overhead_room_atr"
            ]
        ),
        reward_risk=(
            reward_risk[
                "reward_risk"
            ]
        ),
        blue_sky=(
            reward_risk[
                "blue_sky"
            ]
        ),
    )

    #
    # Final Phase-5B location classification.
    #
    # 1H confirmation will be applied later in the
    # Final Classifier, not here.
    #
    if (
        "EXTENSION_GT_3ATR"
        in rejection_reasons
    ):
        classification = (
            "EXTENDED"
        )

    elif not tradability_gate_pass:
        classification = (
            "REJECTED"
        )

    else:
        proximity_state = (
            proximity[
                "entry_proximity_state"
            ]
        )

        if proximity_state == "READY":
            classification = "READY"

        elif (
            proximity_state
            == "DEVELOPING"
        ):
            classification = (
                "DEVELOPING"
            )

        else:
            #
            # WATCH includes:
            # WATCH,
            # TOO_FAR,
            # NO_PIVOT.
            #
            classification = "WATCH"

    return {
        "current_price":
            current_price,

        "current_atr":
            current_atr,

        "pivot_price":
            pivot_price,

        **proximity,

        "support_source":
            stop[
                "support_source"
            ],

        "support_reference":
            stop[
                "support_reference"
            ],

        "extension_atr":
            (
                float(
                    extension_atr
                )
                if not pd.isna(
                    extension_atr
                )
                else np.nan
            ),

        "stop_price":
            stop[
                "stop_price"
            ],

        "risk_per_share":
            stop[
                "risk_per_share"
            ],

        "stop_distance_atr":
            stop[
                "stop_distance_atr"
            ],

        **reward_risk,

        **scores,

        "tradability_gate_pass":
            bool(
                tradability_gate_pass
            ),

        "classification":
            classification,

        "rejection_reason":
            (
                "|".join(
                    rejection_reasons
                )
                if rejection_reasons
                else None
            ),
    }