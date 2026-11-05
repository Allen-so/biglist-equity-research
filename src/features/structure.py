from __future__ import annotations

import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)


SWING_COLUMNS = [
    "pivot_time",
    "confirmed_at",
    "kind",
    "price",
    "atr",
    "prominence",
    "prominence_atr",
    "pivot_position",
    "confirmed_position",
]


def _empty_swings() -> pd.DataFrame:
    return pd.DataFrame(columns=SWING_COLUMNS)


def _filter_spacing(
    swings: pd.DataFrame,
    min_spacing_bars: int,
) -> pd.DataFrame:
    """
    Prevent multiple almost-identical pivots of the same type
    from being counted as separate structural pivots.

    For competing swing highs, keep the higher high.
    For competing swing lows, keep the lower low.

    If price is equal, keep the pivot with larger prominence.
    """

    if swings.empty:
        return swings.copy()

    result: list[dict] = []

    for kind in ("HIGH", "LOW"):
        subset = (
            swings[swings["kind"] == kind]
            .sort_values("pivot_position")
            .to_dict("records")
        )

        accepted: list[dict] = []

        for candidate in subset:
            if not accepted:
                accepted.append(candidate)
                continue

            previous = accepted[-1]

            distance = (
                candidate["pivot_position"]
                - previous["pivot_position"]
            )

            if distance >= min_spacing_bars:
                accepted.append(candidate)
                continue

            if kind == "HIGH":
                candidate_is_better = (
                    candidate["price"] > previous["price"]
                    or (
                        candidate["price"] == previous["price"]
                        and candidate["prominence_atr"]
                        > previous["prominence_atr"]
                    )
                )
            else:
                candidate_is_better = (
                    candidate["price"] < previous["price"]
                    or (
                        candidate["price"] == previous["price"]
                        and candidate["prominence_atr"]
                        > previous["prominence_atr"]
                    )
                )

            if candidate_is_better:
                accepted[-1] = candidate

        result.extend(accepted)

    if not result:
        return _empty_swings()

    return (
        pd.DataFrame(result)
        .sort_values(
            ["pivot_position", "kind"]
        )
        .reset_index(drop=True)
    )


def detect_swings(
    frame: pd.DataFrame,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    min_prominence_atr: float = 0.50,
    min_spacing_bars: int = 3,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Detect confirmed structural swing highs and swing lows.

    Important:
    A pivot at bar t is NOT known at bar t.
    It becomes usable only after `right_bars` future bars
    have completed.

    Therefore every pivot contains:
        pivot_time
        confirmed_at

    This prevents future-data leakage in historical testing.
    """

    if left_bars < 1:
        raise ValueError("left_bars must be >= 1.")

    if right_bars < 1:
        raise ValueError("right_bars must be >= 1.")

    if min_spacing_bars < 1:
        raise ValueError(
            "min_spacing_bars must be >= 1."
        )

    df = normalize_ohlcv(frame)

    atr_column = f"atr_{atr_period}"

    if atr_column not in df.columns:
        df = add_atr(
            df,
            period=atr_period,
        )

    candidates: list[dict] = []

    start = left_bars
    stop = len(df) - right_bars

    for position in range(start, stop):
        atr = df[atr_column].iloc[position]

        if pd.isna(atr) or atr <= 0:
            continue

        current_high = df["high"].iloc[position]
        current_low = df["low"].iloc[position]

        left_highs = df["high"].iloc[
            position - left_bars : position
        ]

        right_highs = df["high"].iloc[
            position + 1 : position + right_bars + 1
        ]

        left_lows = df["low"].iloc[
            position - left_bars : position
        ]

        right_lows = df["low"].iloc[
            position + 1 : position + right_bars + 1
        ]

        # -------------------------
        # Confirmed Swing High
        # -------------------------

        is_high = (
            current_high > left_highs.max()
            and current_high > right_highs.max()
        )

        if is_high:
            left_reference = left_lows.min()
            right_reference = right_lows.min()

            # Conservative prominence:
            # use the higher of the two surrounding troughs.
            reference = max(
                left_reference,
                right_reference,
            )

            prominence = (
                current_high - reference
            )

            prominence_atr = (
                prominence / atr
            )

            if (
                prominence_atr
                >= min_prominence_atr
            ):
                candidates.append(
                    {
                        "pivot_time":
                            df.index[position],
                        "confirmed_at":
                            df.index[
                                position + right_bars
                            ],
                        "kind": "HIGH",
                        "price":
                            float(current_high),
                        "atr":
                            float(atr),
                        "prominence":
                            float(prominence),
                        "prominence_atr":
                            float(prominence_atr),
                        "pivot_position":
                            position,
                        "confirmed_position":
                            position + right_bars,
                    }
                )

        # -------------------------
        # Confirmed Swing Low
        # -------------------------

        is_low = (
            current_low < left_lows.min()
            and current_low < right_lows.min()
        )

        if is_low:
            left_reference = left_highs.max()
            right_reference = right_highs.max()

            # Conservative prominence:
            # use the lower of surrounding peaks.
            reference = min(
                left_reference,
                right_reference,
            )

            prominence = (
                reference - current_low
            )

            prominence_atr = (
                prominence / atr
            )

            if (
                prominence_atr
                >= min_prominence_atr
            ):
                candidates.append(
                    {
                        "pivot_time":
                            df.index[position],
                        "confirmed_at":
                            df.index[
                                position + right_bars
                            ],
                        "kind": "LOW",
                        "price":
                            float(current_low),
                        "atr":
                            float(atr),
                        "prominence":
                            float(prominence),
                        "prominence_atr":
                            float(prominence_atr),
                        "pivot_position":
                            position,
                        "confirmed_position":
                            position + right_bars,
                    }
                )

    if not candidates:
        return _empty_swings()

    swings = pd.DataFrame(
        candidates,
        columns=SWING_COLUMNS,
    )

    return _filter_spacing(
        swings,
        min_spacing_bars=min_spacing_bars,
    )


def confirmed_swings_as_of(
    swings: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return only pivots that were actually known by `as_of`.

    Never filter on pivot_time alone.
    """

    if swings.empty:
        return swings.copy()

    as_of = pd.Timestamp(as_of)

    return (
        swings[
            swings["confirmed_at"] <= as_of
        ]
        .copy()
        .reset_index(drop=True)
    )


def classify_swing_relation(
    previous_price: float,
    latest_price: float,
    reference_atr: float,
    *,
    kind: str,
    equality_tolerance_atr: float = 0.15,
) -> str:
    """
    Classify two structural highs/lows while ignoring
    insignificant differences smaller than the ATR tolerance.
    """

    if reference_atr <= 0:
        raise ValueError(
            "reference_atr must be positive."
        )

    normalized_difference = (
        latest_price - previous_price
    ) / reference_atr

    tolerance = equality_tolerance_atr

    if kind == "HIGH":
        if normalized_difference > tolerance:
            return "HH"

        if normalized_difference < -tolerance:
            return "LH"

        return "EH"

    if kind == "LOW":
        if normalized_difference > tolerance:
            return "HL"

        if normalized_difference < -tolerance:
            return "LL"

        return "EL"

    raise ValueError(
        "kind must be either 'HIGH' or 'LOW'."
    )


def latest_structure_state(
    swings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    equality_tolerance_atr: float = 0.15,
) -> dict[str, str | None]:
    """
    Summarize the latest confirmed swing structure.

    Returns examples such as:

        high_relation = HH
        low_relation = HL

    or None if insufficient confirmed pivots exist.
    """

    confirmed = confirmed_swings_as_of(
        swings,
        as_of,
    )

    highs = (
        confirmed[
            confirmed["kind"] == "HIGH"
        ]
        .sort_values("pivot_time")
    )

    lows = (
        confirmed[
            confirmed["kind"] == "LOW"
        ]
        .sort_values("pivot_time")
    )

    high_relation = None
    low_relation = None

    if len(highs) >= 2:
        previous = highs.iloc[-2]
        latest = highs.iloc[-1]

        reference_atr = (
            previous["atr"]
            + latest["atr"]
        ) / 2

        high_relation = classify_swing_relation(
            previous["price"],
            latest["price"],
            reference_atr,
            kind="HIGH",
            equality_tolerance_atr=(
                equality_tolerance_atr
            ),
        )

    if len(lows) >= 2:
        previous = lows.iloc[-2]
        latest = lows.iloc[-1]

        reference_atr = (
            previous["atr"]
            + latest["atr"]
        ) / 2

        low_relation = classify_swing_relation(
            previous["price"],
            latest["price"],
            reference_atr,
            kind="LOW",
            equality_tolerance_atr=(
                equality_tolerance_atr
            ),
        )

    return {
        "high_relation": high_relation,
        "low_relation": low_relation,
    }