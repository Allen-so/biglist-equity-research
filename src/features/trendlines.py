from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)


TRENDLINE_COLUMNS = [
    "kind",
    "direction",
    "status",
    "slope",
    "intercept",
    "slope_atr_per_20bars",
    "anchor_count",
    "first_anchor",
    "last_anchor",
    "span_bars",
    "mean_touch_error_atr",
    "mean_reaction_atr",
    "violation_episodes",
    "line_price_now",
    "distance_from_price_atr",
    "confidence",
]


def _empty_trendlines() -> pd.DataFrame:
    return pd.DataFrame(
        columns=TRENDLINE_COLUMNS
    )


def _prepare_swings(
    swings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    kind: str,
) -> pd.DataFrame:
    if kind not in {"HIGH", "LOW"}:
        raise ValueError(
            "kind must be HIGH or LOW."
        )

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
            f"Missing swing columns: {sorted(missing)}"
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
        & (df["kind"] == kind)
    ].copy()

    return (
        df.sort_values("pivot_time")
        .reset_index(drop=True)
    )


def _map_pivot_positions(
    frame: pd.DataFrame,
    pivots: pd.DataFrame,
) -> pd.DataFrame:
    result = pivots.copy()

    positions = []

    for pivot_time in result["pivot_time"]:
        location = frame.index.get_indexer(
            [pivot_time]
        )[0]

        if location < 0:
            positions.append(np.nan)
        else:
            positions.append(location)

    result["bar_position"] = positions

    result = result.dropna(
        subset=["bar_position"]
    )

    result["bar_position"] = (
        result["bar_position"].astype(int)
    )

    return result


def _line_value(
    slope: float,
    intercept: float,
    x: int | float,
) -> float:
    return slope * x + intercept


def _count_true_episodes(
    values: pd.Series,
) -> int:
    """
    Count separate True episodes rather than bars.

    Example:
        False, True, True, True, False, True

    = 2 violation episodes, not 4.
    """

    if values.empty:
        return 0

    boolean = values.fillna(False).astype(bool)

    starts = (
        boolean
        & ~boolean.shift(
            1,
            fill_value=False,
        )
    )

    return int(starts.sum())


def _reaction_atr(
    frame: pd.DataFrame,
    *,
    position: int,
    pivot_price: float,
    pivot_atr: float,
    kind: str,
    lookahead_bars: int,
    as_of_position: int,
) -> float:
    if pivot_atr <= 0:
        return 0.0

    start = position + 1

    stop = min(
        position + 1 + lookahead_bars,
        as_of_position + 1,
    )

    if start >= stop:
        return 0.0

    future = frame.iloc[start:stop]

    if future.empty:
        return 0.0

    if kind == "HIGH":
        reaction = (
            pivot_price
            - float(future["low"].min())
        )

    else:
        reaction = (
            float(future["high"].max())
            - pivot_price
        )

    return max(
        reaction / pivot_atr,
        0.0,
    )


def _direction_from_slope(
    slope_atr_per_20bars: float,
    threshold: float,
) -> str:
    if slope_atr_per_20bars > threshold:
        return "RISING"

    if slope_atr_per_20bars < -threshold:
        return "FALLING"

    return "FLAT"


def _current_status(
    *,
    kind: str,
    current_close: float,
    line_price: float,
    current_atr: float,
    violation_buffer_atr: float,
    testing_distance_atr: float,
) -> str:
    normalized_distance = (
        current_close - line_price
    ) / current_atr

    if kind == "HIGH":
        if (
            normalized_distance
            > violation_buffer_atr
        ):
            return "BROKEN_UP"

    else:
        if (
            normalized_distance
            < -violation_buffer_atr
        ):
            return "BROKEN_DOWN"

    if (
        abs(normalized_distance)
        <= testing_distance_atr
    ):
        return "TESTING"

    return "INTACT"


def _trendline_confidence(
    *,
    anchor_count: int,
    span_bars: int,
    mean_touch_error_atr: float,
    mean_reaction_atr: float,
    bars_since_last_anchor: int,
    violation_episodes: int,
    anchor_count_full: int,
    span_full_bars: int,
    recency_full_bars: int,
    recency_zero_bars: int,
    max_touch_distance_atr: float,
    reaction_full_atr: float,
) -> float:

    anchor_score = min(
        anchor_count / anchor_count_full,
        1.0,
    )

    span_score = min(
        span_bars / span_full_bars,
        1.0,
    )

    if max_touch_distance_atr <= 0:
        touch_score = 0.0
    else:
        touch_score = max(
            0.0,
            1.0
            - (
                mean_touch_error_atr
                / max_touch_distance_atr
            ),
        )

    reaction_score = min(
        mean_reaction_atr
        / reaction_full_atr,
        1.0,
    )

    if bars_since_last_anchor <= recency_full_bars:
        recency_score = 1.0

    elif bars_since_last_anchor >= recency_zero_bars:
        recency_score = 0.0

    else:
        recency_score = (
            recency_zero_bars
            - bars_since_last_anchor
        ) / (
            recency_zero_bars
            - recency_full_bars
        )

    violation_score = (
        1.0
        if violation_episodes == 0
        else 0.5
    )

    confidence = (
        30.0 * anchor_score
        + 20.0 * span_score
        + 20.0 * touch_score
        + 15.0 * reaction_score
        + 10.0 * recency_score
        + 5.0 * violation_score
    )

    return round(
        min(
            max(confidence, 0.0),
            100.0,
        ),
        2,
    )


def find_trendlines(
    frame: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    kind: str,
    atr_period: int = 14,
    min_anchors: int = 3,
    min_span_bars: int = 15,
    min_anchor_spacing_bars: int = 3,
    max_touch_distance_atr: float = 0.50,
    testing_distance_atr: float = 0.25,
    violation_buffer_atr: float = 0.35,
    max_major_violation_episodes: int = 1,
    min_abs_slope_atr_per_20bars: float = 0.15,
    reaction_lookahead_bars: int = 5,
    reaction_full_atr: float = 2.0,
    anchor_count_full: int = 5,
    span_full_bars: int = 60,
    recency_full_bars: int = 10,
    recency_zero_bars: int = 100,
    max_lines: int = 3,
) -> pd.DataFrame:
    """
    Detect structural trendlines using confirmed pivots only.

    Resistance lines:
        confirmed Swing Highs

    Support lines:
        confirmed Swing Lows

    Candidate lines are defined by pairs of real pivots.
    A line is accepted only if at least `min_anchors`
    confirmed pivots subsequently validate it.
    """

    df = normalize_ohlcv(frame)

    atr_column = f"atr_{atr_period}"

    if atr_column not in df.columns:
        df = add_atr(
            df,
            period=atr_period,
        )

    as_of = pd.Timestamp(as_of)

    visible = df[
        df.index <= as_of
    ]

    if visible.empty:
        return _empty_trendlines()

    as_of_position = len(visible) - 1

    current_close = float(
        visible["close"].iloc[-1]
    )

    current_atr = float(
        visible[atr_column].iloc[-1]
    )

    if (
        pd.isna(current_atr)
        or current_atr <= 0
    ):
        return _empty_trendlines()

    pivots = _prepare_swings(
        swings,
        as_of=as_of,
        kind=kind,
    )

    pivots = _map_pivot_positions(
        df,
        pivots,
    )

    pivots = pivots[
        pivots["bar_position"]
        <= as_of_position
    ].copy()

    if len(pivots) < min_anchors:
        return _empty_trendlines()

    candidates = []
    # Pivot values and their median ATR do not change between candidate lines.
    # Reuse plain records to avoid constructing a pandas Series in the inner loop.
    pivot_records = pivots.to_dict("records")
    median_atr = float(pivots["atr"].median())
    reaction_cache = {}
    pivot_positions = pivots["bar_position"].to_numpy(dtype=int)
    pivot_prices = pivots["price"].to_numpy(dtype=float)
    pivot_atrs = pivots["atr"].to_numpy(dtype=float)

    for first_idx, second_idx in combinations(
        range(len(pivots)),
        2,
    ):
        first = pivot_records[first_idx]
        second = pivot_records[second_idx]

        x1 = int(first["bar_position"])
        x2 = int(second["bar_position"])

        if (
            x2 - x1
            < min_anchor_spacing_bars
        ):
            continue

        y1 = float(first["price"])
        y2 = float(second["price"])

        slope = (
            y2 - y1
        ) / (
            x2 - x1
        )

        intercept = (
            y1 - slope * x1
        )

        if median_atr <= 0:
            continue

        slope_atr_20 = (
            slope * 20
        ) / median_atr

        direction = _direction_from_slope(
            slope_atr_20,
            min_abs_slope_atr_per_20bars,
        )

        if direction == "FLAT":
            continue

        # Same per-pivot distance formula, evaluated together over immutable arrays.
        expected = slope * pivot_positions + intercept
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.abs(pivot_prices - expected) / pivot_atrs
        touching = np.flatnonzero((pivot_atrs > 0) & (errors <= max_touch_distance_atr))
        touch_records = [
            {"position": int(pivot_positions[i]),
             "pivot_time": pivot_records[i]["pivot_time"],
             "price": float(pivot_prices[i]), "atr": float(pivot_atrs[i]),
             "error_atr": float(errors[i])}
            for i in touching
        ]

        if len(touch_records) < min_anchors:
            continue

        touches = pd.DataFrame(
            touch_records
        ).sort_values("position")

        first_anchor_position = int(
            touches["position"].iloc[0]
        )

        last_anchor_position = int(
            touches["position"].iloc[-1]
        )

        span_bars = (
            last_anchor_position
            - first_anchor_position
        )

        if span_bars < min_span_bars:
            continue

        validation = visible.iloc[
            first_anchor_position:
            as_of_position + 1
        ].copy()

        x_values = np.arange(
            first_anchor_position,
            as_of_position + 1,
        )

        validation["trendline"] = (
            slope * x_values
            + intercept
        )

        if kind == "HIGH":
            violations = (
                validation["close"]
                >
                validation["trendline"]
                + violation_buffer_atr
                * validation[atr_column]
            )
        else:
            violations = (
                validation["close"]
                <
                validation["trendline"]
                - violation_buffer_atr
                * validation[atr_column]
            )

        violation_episodes = (
            _count_true_episodes(
                violations
            )
        )

        if (
            violation_episodes
            > max_major_violation_episodes
        ):
            continue

        reactions = []

        for _, touch in touches.iterrows():
            position = int(touch["position"])
            if position not in reaction_cache:
                reaction = _reaction_atr(
                    df,
                    position=int(
                        touch["position"]
                    ),
                    pivot_price=float(
                        touch["price"]
                    ),
                    pivot_atr=float(
                        touch["atr"]
                    ),
                    kind=kind,
                    lookahead_bars=(
                        reaction_lookahead_bars
                    ),
                    as_of_position=(
                        as_of_position
                    ),
                )
                reaction_cache[position] = reaction
            reaction = reaction_cache[position]

            reactions.append(reaction)

        mean_reaction_atr = float(
            np.mean(reactions)
        )

        mean_touch_error_atr = float(
            touches["error_atr"].mean()
        )

        line_price_now = _line_value(
            slope,
            intercept,
            as_of_position,
        )

        distance_atr = (
            current_close
            - line_price_now
        ) / current_atr

        status = _current_status(
            kind=kind,
            current_close=current_close,
            line_price=line_price_now,
            current_atr=current_atr,
            violation_buffer_atr=(
                violation_buffer_atr
            ),
            testing_distance_atr=(
                testing_distance_atr
            ),
        )

        bars_since_last_anchor = (
            as_of_position
            - last_anchor_position
        )

        confidence = (
            _trendline_confidence(
                anchor_count=len(touches),
                span_bars=span_bars,
                mean_touch_error_atr=(
                    mean_touch_error_atr
                ),
                mean_reaction_atr=(
                    mean_reaction_atr
                ),
                bars_since_last_anchor=(
                    bars_since_last_anchor
                ),
                violation_episodes=(
                    violation_episodes
                ),
                anchor_count_full=(
                    anchor_count_full
                ),
                span_full_bars=(
                    span_full_bars
                ),
                recency_full_bars=(
                    recency_full_bars
                ),
                recency_zero_bars=(
                    recency_zero_bars
                ),
                max_touch_distance_atr=(
                    max_touch_distance_atr
                ),
                reaction_full_atr=(
                    reaction_full_atr
                ),
            )
        )

        candidates.append(
            {
                "kind":
                    (
                        "RESISTANCE"
                        if kind == "HIGH"
                        else "SUPPORT"
                    ),
                "direction":
                    direction,
                "status":
                    status,
                "slope":
                    slope,
                "intercept":
                    intercept,
                "slope_atr_per_20bars":
                    slope_atr_20,
                "anchor_count":
                    len(touches),
                "first_anchor":
                    touches[
                        "pivot_time"
                    ].iloc[0],
                "last_anchor":
                    touches[
                        "pivot_time"
                    ].iloc[-1],
                "span_bars":
                    span_bars,
                "mean_touch_error_atr":
                    mean_touch_error_atr,
                "mean_reaction_atr":
                    mean_reaction_atr,
                "violation_episodes":
                    violation_episodes,
                "line_price_now":
                    line_price_now,
                "distance_from_price_atr":
                    distance_atr,
                "confidence":
                    confidence,
            }
        )

    if not candidates:
        return _empty_trendlines()

    result = pd.DataFrame(
        candidates,
        columns=TRENDLINE_COLUMNS,
    )

    # Pair-generated candidates frequently represent
    # effectively the same structural line.
    # Round structural characteristics to deduplicate.
    result["_slope_key"] = (
        result["slope_atr_per_20bars"]
        .round(2)
    )

    result["_price_key"] = (
        (
            result["line_price_now"]
            / current_atr
        )
        .round(1)
    )

    result = (
        result
        .sort_values(
            [
                "confidence",
                "anchor_count",
                "span_bars",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "kind",
                "direction",
                "_slope_key",
                "_price_key",
            ],
            keep="first",
        )
        .head(max_lines)
        .drop(
            columns=[
                "_slope_key",
                "_price_key",
            ]
        )
        .reset_index(drop=True)
    )

    return result