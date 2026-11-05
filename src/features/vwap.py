from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.acceptance import (
    build_acceptance_features,
)
from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)
from src.features.flow import (
    add_close_location,
)


ANCHOR_COLUMNS = [
    "anchor_type",
    "anchor_time",
    "available_at",
    "reference_price",
    "prominence_atr",
]


AVWAP_EVALUATION_COLUMNS = [
    "anchor_type",
    "anchor_time",
    "available_at",
    "avwap",
    "distance_atr",
    "side",
    "touched",
    "hold",
    "rejection",
    "reclaim",
    "loss",
]


def _empty_anchors() -> pd.DataFrame:
    return pd.DataFrame(
        columns=ANCHOR_COLUMNS
    )


def _empty_avwap_evaluation() -> pd.DataFrame:
    return pd.DataFrame(
        columns=AVWAP_EVALUATION_COLUMNS
    )


def anchored_vwap_series(
    frame: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
) -> pd.Series:
    """
    Daily Anchored VWAP using HLC3 as the
    representative price for each daily bar.

    AVWAP =
        cumulative(
            typical_price * volume
        )
        /
        cumulative(volume)

    Bars before the anchor remain NaN.
    """

    df = normalize_ohlcv(frame)

    anchor_time = pd.Timestamp(
        anchor_time
    )

    if anchor_time not in df.index:
        raise ValueError(
            "anchor_time must exist in OHLCV index."
        )

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
        name="anchored_vwap",
    )

    anchored = df.loc[
        anchor_time:
    ].copy()

    anchored_typical = (
        typical_price.loc[
            anchor_time:
        ]
    )

    cumulative_dollar = (
        anchored_typical
        * anchored["volume"]
    ).cumsum()

    cumulative_volume = (
        anchored["volume"]
        .cumsum()
        .replace(0, np.nan)
    )

    result.loc[
        anchor_time:
    ] = (
        cumulative_dollar
        / cumulative_volume
    )

    return result


def _prepare_confirmed_swings(
    swings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    min_prominence_atr: float,
) -> pd.DataFrame:
    if swings is None or swings.empty:
        return pd.DataFrame()

    required = {
        "pivot_time",
        "confirmed_at",
        "kind",
        "price",
        "prominence_atr",
    }

    missing = required.difference(
        swings.columns
    )

    if missing:
        raise ValueError(
            "Swings dataframe missing columns: "
            f"{sorted(missing)}"
        )

    df = swings.copy()

    df["pivot_time"] = pd.to_datetime(
        df["pivot_time"]
    )

    df["confirmed_at"] = pd.to_datetime(
        df["confirmed_at"]
    )

    as_of = pd.Timestamp(as_of)

    df = df[
        (df["confirmed_at"] <= as_of)
        & (
            df["prominence_atr"]
            >= min_prominence_atr
        )
    ].copy()

    return (
        df.sort_values(
            "pivot_time"
        )
        .reset_index(drop=True)
    )


def _add_swing_anchors(
    records: list[dict[str, Any]],
    *,
    swings: pd.DataFrame,
    kind: str,
    max_recent: int,
) -> None:
    if swings.empty:
        return

    subset = (
        swings[
            swings["kind"] == kind
        ]
        .sort_values(
            "pivot_time"
        )
        .tail(max_recent)
    )

    anchor_type = (
        "SWING_HIGH"
        if kind == "HIGH"
        else "SWING_LOW"
    )

    for _, row in subset.iterrows():
        records.append(
            {
                "anchor_type":
                    anchor_type,
                "anchor_time":
                    pd.Timestamp(
                        row["pivot_time"]
                    ),
                "available_at":
                    pd.Timestamp(
                        row["confirmed_at"]
                    ),
                "reference_price":
                    float(row["price"]),
                "prominence_atr":
                    float(
                        row["prominence_atr"]
                    ),
            }
        )


def _add_breakout_anchors(
    records: list[dict[str, Any]],
    *,
    acceptance: pd.DataFrame,
    as_of: pd.Timestamp,
    max_recent: int,
) -> None:
    if acceptance.empty:
        return

    if "breakout_event" not in acceptance.columns:
        raise ValueError(
            "Acceptance dataframe must contain "
            "'breakout_event'."
        )

    as_of = pd.Timestamp(as_of)

    visible = acceptance[
        acceptance.index <= as_of
    ]

    events = visible[
        visible["breakout_event"]
    ].tail(
        max_recent
    )

    for timestamp, row in events.iterrows():
        reference_price = row.get(
            "active_breakout_level",
            row.get(
                "breakout_threshold",
                row["close"],
            ),
        )

        records.append(
            {
                "anchor_type":
                    "BREAKOUT",
                "anchor_time":
                    pd.Timestamp(timestamp),
                "available_at":
                    pd.Timestamp(timestamp),
                "reference_price":
                    float(reference_price),
                "prominence_atr":
                    np.nan,
            }
        )


def build_avwap_anchors(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    swings: pd.DataFrame | None = None,
    acceptance_frame: pd.DataFrame | None = None,
    min_swing_prominence_atr: float = 0.75,
    max_recent_swing_highs: int = 2,
    max_recent_swing_lows: int = 2,
    max_recent_breakouts: int = 2,
) -> pd.DataFrame:
    """
    Build objective AVWAP anchors.

    Allowed V1 anchor families:

        Confirmed Swing High
        Confirmed Swing Low
        Confirmed Breakout Event

    No manually selected anchor is allowed.
    """

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    visible = df[
        df.index <= as_of
    ].copy()

    if visible.empty:
        return _empty_anchors()

    records: list[
        dict[str, Any]
    ] = []

    if swings is not None:
        confirmed_swings = (
            _prepare_confirmed_swings(
                swings,
                as_of=as_of,
                min_prominence_atr=(
                    min_swing_prominence_atr
                ),
            )
        )

        _add_swing_anchors(
            records,
            swings=confirmed_swings,
            kind="HIGH",
            max_recent=(
                max_recent_swing_highs
            ),
        )

        _add_swing_anchors(
            records,
            swings=confirmed_swings,
            kind="LOW",
            max_recent=(
                max_recent_swing_lows
            ),
        )

    if acceptance_frame is None:
        acceptance = (
            build_acceptance_features(
                visible
            )
        )
    else:
        acceptance = (
            acceptance_frame[
                acceptance_frame.index
                <= as_of
            ].copy()
        )

    _add_breakout_anchors(
        records,
        acceptance=acceptance,
        as_of=as_of,
        max_recent=(
            max_recent_breakouts
        ),
    )

    if not records:
        return _empty_anchors()

    result = pd.DataFrame(
        records,
        columns=ANCHOR_COLUMNS,
    )

    #
    # Defensive as-of protection.
    #
    result = result[
        result["available_at"]
        <= as_of
    ]

    #
    # Ensure anchor itself exists in price history.
    #
    result = result[
        result["anchor_time"].isin(
            visible.index
        )
    ]

    return (
        result
        .sort_values(
            [
                "available_at",
                "anchor_time",
            ]
        )
        .reset_index(drop=True)
    )


def classify_avwap_side(
    *,
    close: float,
    avwap: float,
    atr: float,
    tolerance_atr: float = 0.15,
) -> str:
    if atr <= 0:
        raise ValueError(
            "atr must be positive."
        )

    distance = (
        close - avwap
    ) / atr

    if distance > tolerance_atr:
        return "ABOVE"

    if distance < -tolerance_atr:
        return "BELOW"

    return "AT"


def evaluate_avwap_anchors(
    frame: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    atr_period: int = 14,
    touch_buffer_atr: float = 0.20,
    side_tolerance_atr: float = 0.15,
    reclaim_buffer_atr: float = 0.10,
    strong_close_location: float = 0.65,
) -> pd.DataFrame:
    """
    Evaluate current price interaction with every
    valid AVWAP anchor.

    Produces mechanical:
        ABOVE / BELOW / AT
        touch
        hold
        rejection
        reclaim
        loss
    """

    if anchors is None or anchors.empty:
        return _empty_avwap_evaluation()

    df = normalize_ohlcv(frame)

    as_of = pd.Timestamp(as_of)

    df = df[
        df.index <= as_of
    ].copy()

    if len(df) < 2:
        return _empty_avwap_evaluation()

    df = add_atr(
        df,
        period=atr_period,
    )

    df = add_close_location(
        df
    )

    atr_column = f"atr_{atr_period}"

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    current_atr = float(
        latest[atr_column]
    )

    if (
        pd.isna(current_atr)
        or current_atr <= 0
    ):
        return _empty_avwap_evaluation()

    current_close = float(
        latest["close"]
    )

    current_high = float(
        latest["high"]
    )

    current_low = float(
        latest["low"]
    )

    current_clv = float(
        latest["clv_01"]
    )

    records: list[
        dict[str, Any]
    ] = []

    valid_anchors = anchors[
        anchors["available_at"]
        <= as_of
    ]

    for _, anchor in (
        valid_anchors.iterrows()
    ):
        anchor_time = pd.Timestamp(
            anchor["anchor_time"]
        )

        if anchor_time not in df.index:
            continue

        series = anchored_vwap_series(
            df,
            anchor_time=anchor_time,
        )

        current_avwap = series.iloc[-1]
        previous_avwap = series.iloc[-2]

        if pd.isna(current_avwap):
            continue

        distance_atr = (
            current_close
            - current_avwap
        ) / current_atr

        side = classify_avwap_side(
            close=current_close,
            avwap=float(
                current_avwap
            ),
            atr=current_atr,
            tolerance_atr=(
                side_tolerance_atr
            ),
        )

        touched = (
            current_low
            <= (
                current_avwap
                + touch_buffer_atr
                * current_atr
            )
            and
            current_high
            >= (
                current_avwap
                - touch_buffer_atr
                * current_atr
            )
        )

        hold = (
            touched
            and current_close
            > current_avwap
            and current_clv
            >= strong_close_location
        )

        rejection = (
            touched
            and current_close
            < current_avwap
            and current_clv
            <= (
                1.0
                - strong_close_location
            )
        )

        reclaim = False
        loss = False

        if not pd.isna(
            previous_avwap
        ):
            previous_close = float(
                previous["close"]
            )

            reclaim = (
                previous_close
                <= previous_avwap
                and current_close
                > (
                    current_avwap
                    + reclaim_buffer_atr
                    * current_atr
                )
            )

            loss = (
                previous_close
                >= previous_avwap
                and current_close
                < (
                    current_avwap
                    - reclaim_buffer_atr
                    * current_atr
                )
            )

        records.append(
            {
                "anchor_type":
                    anchor["anchor_type"],
                "anchor_time":
                    anchor_time,
                "available_at":
                    anchor["available_at"],
                "avwap":
                    float(
                        current_avwap
                    ),
                "distance_atr":
                    float(
                        distance_atr
                    ),
                "side":
                    side,
                "touched":
                    bool(touched),
                "hold":
                    bool(hold),
                "rejection":
                    bool(rejection),
                "reclaim":
                    bool(reclaim),
                "loss":
                    bool(loss),
            }
        )

    if not records:
        return _empty_avwap_evaluation()

    return (
        pd.DataFrame(
            records,
            columns=(
                AVWAP_EVALUATION_COLUMNS
            ),
        )
        .sort_values(
            "distance_atr",
            key=lambda series:
                series.abs(),
        )
        .reset_index(drop=True)
    )


def build_avwap_features(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    swings: pd.DataFrame | None = None,
    acceptance_frame: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Phase 3C AVWAP pipeline.

    Returns:
        anchors
        evaluation
    """

    anchors = build_avwap_anchors(
        frame,
        as_of=as_of,
        swings=swings,
        acceptance_frame=(
            acceptance_frame
        ),
    )

    evaluation = (
        evaluate_avwap_anchors(
            frame,
            anchors,
            as_of=as_of,
        )
    )

    return (
        anchors,
        evaluation,
    )