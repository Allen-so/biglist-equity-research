from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.basic import (
    add_moving_averages,
    normalize_ohlcv,
)


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Strict as-of protection.

    Only bars available on or before `as_of`
    may influence the result.
    """

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    return df[
        df.index <= as_of
    ].copy()


def _safe_slope(
    series: pd.Series,
    *,
    lookback: int,
) -> float:
    """
    Percentage slope:

        current_value / value_N_bars_ago - 1
    """

    if len(series) <= lookback:
        return np.nan

    current = series.iloc[-1]

    previous = series.iloc[
        -(lookback + 1)
    ]

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous == 0
    ):
        return np.nan

    return float(
        current / previous - 1.0
    )


def _confirmed_swings(
    swings: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return only pivots that were already confirmed
    by the requested as-of timestamp.
    """

    if swings is None or swings.empty:
        return pd.DataFrame(columns=["pivot_time", "confirmed_at", "kind", "price", "atr"])

    required = {
        "pivot_time",
        "confirmed_at",
        "kind",
        "price",
        "atr",
    }

    missing = required.difference(
        swings.columns
    )

    if missing:
        raise ValueError(
            "Swings missing columns: "
            f"{sorted(missing)}"
        )

    result = swings.copy()

    result["pivot_time"] = pd.to_datetime(
        result["pivot_time"]
    )

    result["confirmed_at"] = pd.to_datetime(
        result["confirmed_at"]
    )

    as_of = pd.Timestamp(as_of)

    result = result[
        (
            result["confirmed_at"]
            <= as_of
        )
        &
        (
            result["pivot_time"]
            <= as_of
        )
    ].copy()

    return (
        result
        .sort_values(
            "pivot_time"
        )
        .reset_index(
            drop=True
        )
    )


def _swing_relation(
    swings: pd.DataFrame,
    *,
    kind: str,
    tolerance_atr: float = 0.15,
) -> str | None:
    """
    Compare the latest two confirmed pivots.

    HIGH:
        HH = Higher High
        LH = Lower High
        EH = Essentially Equal High

    LOW:
        HL = Higher Low
        LL = Lower Low
        EL = Essentially Equal Low
    """

    subset = (
        swings[
            swings["kind"]
            == kind
        ]
        .tail(2)
    )

    if len(subset) < 2:
        return None

    previous = subset.iloc[-2]
    latest = subset.iloc[-1]

    previous_price = float(
        previous["price"]
    )

    latest_price = float(
        latest["price"]
    )

    reference_atr = float(
        latest["atr"]
    )

    tolerance = (
        tolerance_atr
        * reference_atr
    )

    difference = (
        latest_price
        - previous_price
    )

    if kind == "HIGH":
        if difference > tolerance:
            return "HH"

        if difference < -tolerance:
            return "LH"

        return "EH"

    if kind == "LOW":
        if difference > tolerance:
            return "HL"

        if difference < -tolerance:
            return "LL"

        return "EL"

    raise ValueError(
        "kind must be HIGH or LOW."
    )


def calculate_structure_evaluation(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    swings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Big List Structure factor.

    Maximum = 8 points.

    Score:
        Close >= SMA20                 1.5
        SMA20 >= SMA50                 1.5
        SMA20 10D slope > 0            1.0
        SMA50 20D slope > 0            1.0
        52W range position >= 60%      1.0
        Latest Swing Low = HL          1.0
        Latest Swing High = HH/EH      1.0

    Also calculates Broken Structure conditions.

    Emerging-Leader exemptions are NOT applied here.
    They are applied later inside the hard-gate layer.
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    if visible.empty:
        return {}

    #
    # IMPORTANT:
    #
    # add_moving_averages() already uses the
    # project's standard Daily MA windows:
    #
    # 10 / 20 / 50 / 200
    #
    # Therefore no `windows=` keyword is passed.
    #
    visible = add_moving_averages(
        visible
    )

    close = float(
        visible["close"].iloc[-1]
    )

    sma20 = visible[
        "sma_20"
    ].iloc[-1]

    sma50 = visible[
        "sma_50"
    ].iloc[-1]

    sma20_slope_10 = _safe_slope(
        visible["sma_20"],
        lookback=10,
    )

    sma50_slope_20 = _safe_slope(
        visible["sma_50"],
        lookback=20,
    )

    #
    # 52-week / 252-session range.
    #
    range_window = visible.tail(
        252
    )

    if len(range_window) >= 60:
        range_high = float(
            range_window[
                "high"
            ].max()
        )

        range_low = float(
            range_window[
                "low"
            ].min()
        )

        denominator = (
            range_high
            - range_low
        )

        if denominator > 0:
            range_position = (
                close
                - range_low
            ) / denominator

        else:
            range_position = np.nan

        if range_low > 0:
            distance_from_52w_low = (
                close
                - range_low
            ) / range_low

        else:
            distance_from_52w_low = (
                np.nan
            )

    else:
        range_position = np.nan
        distance_from_52w_low = np.nan

    confirmed = _confirmed_swings(
        swings,
        as_of=as_of,
    )

    high_relation = _swing_relation(
        confirmed,
        kind="HIGH",
    )

    low_relation = _swing_relation(
        confirmed,
        kind="LOW",
    )

    #
    # Structure score.
    #
    score = 0.0

    close_above_20 = (
        not pd.isna(sma20)
        and
        close >= sma20
    )

    ma_alignment = (
        not pd.isna(sma20)
        and
        not pd.isna(sma50)
        and
        sma20 >= sma50
    )

    positive_sma20_slope = (
        not pd.isna(
            sma20_slope_10
        )
        and
        sma20_slope_10 > 0
    )

    positive_sma50_slope = (
        not pd.isna(
            sma50_slope_20
        )
        and
        sma50_slope_20 > 0
    )

    good_range_position = (
        not pd.isna(
            range_position
        )
        and
        range_position >= 0.60
    )

    higher_low = (
        low_relation == "HL"
    )

    non_declining_high = (
        high_relation
        in {
            "HH",
            "EH",
        }
    )

    if close_above_20:
        score += 1.5

    if ma_alignment:
        score += 1.5

    if positive_sma20_slope:
        score += 1.0

    if positive_sma50_slope:
        score += 1.0

    if good_range_position:
        score += 1.0

    if higher_low:
        score += 1.0

    if non_declining_high:
        score += 1.0

    #
    # Broken Structure Core
    #
    # Hard failure candidate if >= 3 of:
    #
    # 1. Close < SMA50
    # 2. SMA20 < SMA50
    # 3. SMA50 20D slope <= -3%
    # 4. Latest highs declining AND latest lows declining
    #
    below_sma50 = (
        not pd.isna(sma50)
        and
        close < sma50
    )

    bearish_ma_alignment = (
        not pd.isna(sma20)
        and
        not pd.isna(sma50)
        and
        sma20 < sma50
    )

    strongly_negative_sma50 = (
        not pd.isna(
            sma50_slope_20
        )
        and
        sma50_slope_20 <= -0.03
    )

    declining_swings = (
        high_relation == "LH"
        and
        low_relation == "LL"
    )

    broken_conditions = {
        "below_sma50":
            bool(
                below_sma50
            ),

        "sma20_below_sma50":
            bool(
                bearish_ma_alignment
            ),

        "sma50_slope_le_minus_3pct":
            bool(
                strongly_negative_sma50
            ),

        "declining_highs_and_lows":
            bool(
                declining_swings
            ),
    }

    broken_condition_count = int(
        sum(
            broken_conditions.values()
        )
    )

    broken_structure_core = (
        broken_condition_count >= 3
    )

    #
    # Near 52-week low condition.
    #
    near_52w_low = (
        not pd.isna(
            distance_from_52w_low
        )
        and
        distance_from_52w_low
        <= 0.10
    )

    return {
        "structure_score":
            round(
                float(score),
                2,
            ),

        "close":
            close,

        "sma_20":
            (
                float(sma20)
                if not pd.isna(
                    sma20
                )
                else np.nan
            ),

        "sma_50":
            (
                float(sma50)
                if not pd.isna(
                    sma50
                )
                else np.nan
            ),

        "sma20_slope_10d":
            sma20_slope_10,

        "sma50_slope_20d":
            sma50_slope_20,

        "range_position_252":
            (
                float(
                    range_position
                )
                if not pd.isna(
                    range_position
                )
                else np.nan
            ),

        "distance_from_52w_low":
            (
                float(
                    distance_from_52w_low
                )
                if not pd.isna(
                    distance_from_52w_low
                )
                else np.nan
            ),

        "latest_high_relation":
            high_relation,

        "latest_low_relation":
            low_relation,

        "broken_condition_count":
            broken_condition_count,

        "broken_structure_core":
            bool(
                broken_structure_core
            ),

        "near_52w_low":
            bool(
                near_52w_low
            ),

        **broken_conditions,
    }


def evaluate_admission_channel(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """
    Two-channel admission.

    Channel A:
        Existing Leader / Mature Monster

    Channel B:
        Emerging Leader / New Monster
    """

    def value(
        name: str,
    ) -> float:
        result = row.get(
            name,
            np.nan,
        )

        if pd.isna(result):
            return np.nan

        return float(result)

    rs20 = value(
        "rs_20d_pct"
    )

    rs60 = value(
        "rs_60d_pct"
    )

    velocity = value(
        "historical_velocity_pct"
    )

    structure_score = value(
        "structure_score"
    )

    return_5d = value(
        "return_5d"
    )

    return_10d = value(
        "return_10d"
    )

    volume_zscore = value(
        "volume_zscore_20"
    )

    rs20_delta = value(
        "rs_20d_rank_delta"
    )

    close = value(
        "close"
    )

    sma20 = value(
        "sma_20"
    )

    progress_retention = value(
        "progress_retention_10"
    )

    #
    # Existing Leader — Primary Route
    #
    # RS20 >= 70
    # RS60 >= 70
    # Historical Velocity >= 60
    # Structure >= 4 / 8
    #
    existing_primary = (
        not pd.isna(rs20)
        and
        rs20 >= 70
        and
        not pd.isna(rs60)
        and
        rs60 >= 70
        and
        not pd.isna(velocity)
        and
        velocity >= 60
        and
        not pd.isna(
            structure_score
        )
        and
        structure_score >= 4
    )

    #
    # Existing Leader — Alternate Route
    #
    # Very strong RS20 can compensate for slightly
    # lower RS60 / velocity maturity.
    #
    existing_alternate = (
        not pd.isna(rs20)
        and
        rs20 >= 80
        and
        not pd.isna(rs60)
        and
        rs60 >= 60
    )

    existing_leader = (
        existing_primary
        or
        existing_alternate
    )

    #
    # Emerging Leader
    #
    # Fast recent move:
    #
    # 5D >= +7%
    # OR
    # 10D >= +10%
    #
    fast_move = (
        (
            not pd.isna(
                return_5d
            )
            and
            return_5d >= 0.07
        )
        or
        (
            not pd.isna(
                return_10d
            )
            and
            return_10d >= 0.10
        )
    )

    #
    # Participation / attention.
    #
    high_volume = (
        not pd.isna(
            volume_zscore
        )
        and
        volume_zscore >= 1.50
    )

    #
    # Rapid cross-sectional improvement.
    #
    rapid_rank_improvement = (
        not pd.isna(
            rs20_delta
        )
        and
        rs20_delta >= 15
    )

    #
    # Must still be structurally above SMA20.
    #
    above_sma20 = (
        not pd.isna(close)
        and
        not pd.isna(sma20)
        and
        close >= sma20
    )

    #
    # Recent progress must still be retained.
    #
    retained_progress = (
        not pd.isna(
            progress_retention
        )
        and
        progress_retention >= 0.50
    )

    emerging_leader = (
        fast_move
        and
        high_volume
        and
        rapid_rank_improvement
        and
        above_sma20
        and
        retained_progress
    )

    if (
        existing_leader
        and
        emerging_leader
    ):
        channel = "DUAL"

    elif existing_leader:
        channel = (
            "EXISTING_LEADER"
        )

    elif emerging_leader:
        channel = (
            "EMERGING_LEADER"
        )

    else:
        channel = "NONE"

    return {
        "existing_primary":
            bool(
                existing_primary
            ),

        "existing_alternate":
            bool(
                existing_alternate
            ),

        "existing_leader":
            bool(
                existing_leader
            ),

        "emerging_fast_move":
            bool(
                fast_move
            ),

        "emerging_high_volume":
            bool(
                high_volume
            ),

        "emerging_rank_improvement":
            bool(
                rapid_rank_improvement
            ),

        "emerging_above_sma20":
            bool(
                above_sma20
            ),

        "emerging_progress_retained":
            bool(
                retained_progress
            ),

        "emerging_leader":
            bool(
                emerging_leader
            ),

        "admission_channel":
            channel,

        "admission_channel_pass":
            bool(
                existing_leader
                or
                emerging_leader
            ),
    }


def evaluate_hard_gates(
    *,
    admission: Mapping[
        str,
        Any,
    ],
    structure: Mapping[
        str,
        Any,
    ],
    setup_row: Mapping[
        str,
        Any,
    ] | pd.Series | None,
    tradability_row: Mapping[
        str,
        Any,
    ] | pd.Series | None,
) -> dict[str, Any]:
    """
    Hard Gates remain completely separate
    from final score.

    A high future 100-point score may NOT override:

        No admission channel
        Broken structure
        Setup < 5
        Tradability hard failure
    """

    reasons: list[str] = []

    #
    # Admission Channel Gate.
    #
    if not bool(
        admission.get(
            "admission_channel_pass",
            False,
        )
    ):
        reasons.append(
            "NO_ADMISSION_CHANNEL"
        )

    emerging = bool(
        admission.get(
            "emerging_leader",
            False,
        )
    )

    broken_core = bool(
        structure.get(
            "broken_structure_core",
            False,
        )
    )

    near_low = bool(
        structure.get(
            "near_52w_low",
            False,
        )
    )

    #
    # Structure Hard Gate:
    #
    # broken core always fails.
    #
    # Near 52W low fails UNLESS this is an
    # objectively qualified Emerging Leader.
    #
    structure_fail = (
        broken_core
        or
        (
            near_low
            and
            not emerging
        )
    )

    if structure_fail:
        reasons.append(
            "STRUCTURE_FAIL"
        )

    #
    # Setup Gate.
    #
    setup_score = np.nan

    if setup_row is not None:
        setup_score = (
            setup_row.get(
                "setup_score",
                np.nan,
            )
        )

    if (
        pd.isna(
            setup_score
        )
        or
        float(
            setup_score
        ) < 5
    ):
        reasons.append(
            "SETUP_LT_5"
        )

    #
    # Tradability Gate.
    #
    tradability_pass = False

    if tradability_row is not None:
        tradability_pass = bool(
            tradability_row.get(
                "tradability_gate_pass",
                False,
            )
        )

    if not tradability_pass:
        reasons.append(
            "TRADABILITY_FAIL"
        )

    admitted = (
        len(reasons) == 0
    )

    # Watch-only hard-gate eligibility; finalize must still require complete
    # scoring, the regime score gate, and a valid tradability location.
    # Formal Big List admission and its rejection reasons remain unchanged.
    forming_watch_eligible = (
        not pd.isna(setup_score)
        and float(setup_score) == 4
        and bool(admission.get("admission_channel_pass", False))
        and not structure_fail
        and tradability_pass
    )

    return {
        "forming_watch_eligible": bool(forming_watch_eligible),

        "structure_hard_fail":
            bool(
                structure_fail
            ),

        "setup_hard_fail":
            bool(
                "SETUP_LT_5"
                in reasons
            ),

        "tradability_hard_fail":
            bool(
                "TRADABILITY_FAIL"
                in reasons
            ),

        "admission_channel_hard_fail":
            bool(
                "NO_ADMISSION_CHANNEL"
                in reasons
            ),

        "admitted":
            bool(
                admitted
            ),

        "rejection_reasons":
            reasons,

        "rejection_reason":
            (
                "|".join(
                    reasons
                )
                if reasons
                else None
            ),
    }


def evaluate_candidate_admission(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    candidate_row: Mapping[
        str,
        Any,
    ],
    swings: pd.DataFrame | None = None,
    setup_row: Mapping[
        str,
        Any,
    ] | pd.Series | None = None,
    tradability_row: Mapping[
        str,
        Any,
    ] | pd.Series | None = None,
) -> dict[str, Any]:
    """
    Complete Phase 6A orchestration
    for a single stock.

    Pipeline:

        Structure
            ↓
        Two-Channel Admission
            ↓
        Structure Hard Gate
            ↓
        Setup Hard Gate
            ↓
        Tradability Hard Gate
            ↓
        ADMITTED / REJECTED
    """

    structure = (
        calculate_structure_evaluation(
            frame,
            as_of=as_of,
            swings=swings,
        )
    )

    combined = dict(
        candidate_row
    )

    #
    # Current structure-derived values are
    # authoritative for this as-of run.
    #
    combined.update(
        {
            "structure_score":
                structure.get(
                    "structure_score",
                    np.nan,
                ),

            "close":
                structure.get(
                    "close",
                    np.nan,
                ),

            "sma_20":
                structure.get(
                    "sma_20",
                    np.nan,
                ),
        }
    )

    admission = (
        evaluate_admission_channel(
            combined
        )
    )

    gates = evaluate_hard_gates(
        admission=admission,
        structure=structure,
        setup_row=setup_row,
        tradability_row=(
            tradability_row
        ),
    )

    return {
        **structure,
        **admission,
        **gates,
    }