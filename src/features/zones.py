from __future__ import annotations

from typing import Any

import pandas as pd


ZONE_COLUMNS = [
    "zone_low",
    "zone_high",
    "zone_center",
    "pivot_count",
    "high_pivot_count",
    "low_pivot_count",
    "first_touch",
    "last_touch",
    "mean_prominence_atr",
    "median_pivot_atr",
    "zone_width_atr",
    "distance_from_price_atr",
    "role",
    "confidence",
]


def _empty_zones() -> pd.DataFrame:
    return pd.DataFrame(columns=ZONE_COLUMNS)


def _validate_swings(swings: pd.DataFrame) -> None:
    required = {
        "pivot_time",
        "confirmed_at",
        "kind",
        "price",
        "atr",
        "prominence_atr",
    }

    missing = required.difference(swings.columns)

    if missing:
        raise ValueError(
            "Swings dataframe is missing required columns: "
            f"{sorted(missing)}"
        )


def _confirmed_swings(
    swings: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Use only pivots that were actually known at `as_of`.

    This preserves look-ahead safety.
    """

    if swings.empty:
        return swings.copy()

    _validate_swings(swings)

    as_of = pd.Timestamp(as_of)

    confirmed = swings[
        pd.to_datetime(swings["confirmed_at"]) <= as_of
    ].copy()

    if confirmed.empty:
        return confirmed

    confirmed["pivot_time"] = pd.to_datetime(
        confirmed["pivot_time"]
    )

    confirmed["confirmed_at"] = pd.to_datetime(
        confirmed["confirmed_at"]
    )

    return confirmed.reset_index(drop=True)


def _prospective_zone_width_atr(
    cluster: pd.DataFrame,
) -> float:
    """
    Measure the entire cluster width using the median ATR
    of the pivots inside that cluster.

    This prevents chain-link clustering from creating
    unrealistically wide zones.
    """

    if cluster.empty:
        return 0.0

    median_atr = float(cluster["atr"].median())

    if median_atr <= 0:
        raise ValueError(
            "Pivot ATR must be positive."
        )

    width = (
        float(cluster["price"].max())
        - float(cluster["price"].min())
    )

    return width / median_atr


def _cluster_pivots(
    swings: pd.DataFrame,
    *,
    cluster_distance_atr: float,
    min_pivots: int,
) -> list[pd.DataFrame]:
    """
    Deterministically group nearby structural pivots.

    Pivots are sorted by price.

    A candidate can join the current cluster only if the
    ENTIRE prospective zone remains within the configured
    ATR-normalized width.

    High and Low pivots are intentionally allowed in the
    same cluster so that role reversal can be detected.
    """

    if cluster_distance_atr <= 0:
        raise ValueError(
            "cluster_distance_atr must be positive."
        )

    if min_pivots < 2:
        raise ValueError(
            "min_pivots must be >= 2."
        )

    if swings.empty:
        return []

    ordered = (
        swings
        .sort_values(
            ["price", "pivot_time"]
        )
        .reset_index(drop=True)
    )

    clusters: list[pd.DataFrame] = []

    current_indices: list[int] = []

    for index in range(len(ordered)):
        if not current_indices:
            current_indices = [index]
            continue

        prospective_indices = (
            current_indices + [index]
        )

        prospective = ordered.iloc[
            prospective_indices
        ]

        width_atr = (
            _prospective_zone_width_atr(
                prospective
            )
        )

        if width_atr <= cluster_distance_atr:
            current_indices.append(index)
        else:
            if len(current_indices) >= min_pivots:
                clusters.append(
                    ordered.iloc[
                        current_indices
                    ].copy()
                )

            current_indices = [index]

    if len(current_indices) >= min_pivots:
        clusters.append(
            ordered.iloc[
                current_indices
            ].copy()
        )

    return clusters


def classify_zone_role(
    *,
    zone_low: float,
    zone_high: float,
    current_price: float,
) -> str:
    """
    Mechanical classification relative to current price.
    """

    if current_price > zone_high:
        return "SUPPORT"

    if current_price < zone_low:
        return "RESISTANCE"

    return "CURRENT"


def _distance_from_zone_atr(
    *,
    zone_low: float,
    zone_high: float,
    current_price: float,
    current_atr: float,
) -> float:
    """
    Distance from current price to the nearest boundary
    of the zone, normalized by current ATR.
    """

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    if current_price > zone_high:
        distance = current_price - zone_high

    elif current_price < zone_low:
        distance = zone_low - current_price

    else:
        distance = 0.0

    return distance / current_atr


def _linear_recency_score(
    *,
    days_since_last_touch: int,
    full_days: int,
    zero_days: int,
) -> float:
    """
    1.0 when very recent.
    Linearly decays toward 0.0 as the zone becomes stale.
    """

    if zero_days <= full_days:
        raise ValueError(
            "recency_zero_days must be greater "
            "than recency_full_days."
        )

    if days_since_last_touch <= full_days:
        return 1.0

    if days_since_last_touch >= zero_days:
        return 0.0

    remaining = (
        zero_days - days_since_last_touch
    )

    span = zero_days - full_days

    return remaining / span


def zone_confidence(
    cluster: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    pivot_count_full: int = 5,
    prominence_full_atr: float = 2.0,
    recency_full_days: int = 14,
    recency_zero_days: int = 126,
    time_span_full_days: int = 90,
) -> float:
    """
    Phase-2 structural confidence only.

    Current weights:

        Pivot count       35%
        Prominence        25%
        Recency           20%
        Time span         10%
        Role diversity    10%

    Volume confirmation will be added later in Flow Engine,
    not here.
    """

    if cluster.empty:
        return 0.0

    if pivot_count_full <= 0:
        raise ValueError(
            "pivot_count_full must be positive."
        )

    if prominence_full_atr <= 0:
        raise ValueError(
            "prominence_full_atr must be positive."
        )

    if time_span_full_days <= 0:
        raise ValueError(
            "time_span_full_days must be positive."
        )

    as_of = pd.Timestamp(as_of)

    pivot_count = len(cluster)

    pivot_count_score = min(
        pivot_count / pivot_count_full,
        1.0,
    )

    mean_prominence_atr = float(
        cluster["prominence_atr"].mean()
    )

    prominence_score = min(
        mean_prominence_atr
        / prominence_full_atr,
        1.0,
    )

    last_touch = pd.Timestamp(
        cluster["pivot_time"].max()
    )

    days_since_last_touch = max(
        (as_of - last_touch).days,
        0,
    )

    recency_score = _linear_recency_score(
        days_since_last_touch=(
            days_since_last_touch
        ),
        full_days=recency_full_days,
        zero_days=recency_zero_days,
    )

    first_touch = pd.Timestamp(
        cluster["pivot_time"].min()
    )

    span_days = max(
        (last_touch - first_touch).days,
        0,
    )

    time_span_score = min(
        span_days / time_span_full_days,
        1.0,
    )

    has_high = (
        cluster["kind"] == "HIGH"
    ).any()

    has_low = (
        cluster["kind"] == "LOW"
    ).any()

    if has_high and has_low:
        role_diversity_score = 1.0
    else:
        # Pure support or pure resistance can still be
        # highly valid. Role reversal receives extra credit.
        role_diversity_score = 0.5

    confidence = (
        35.0 * pivot_count_score
        + 25.0 * prominence_score
        + 20.0 * recency_score
        + 10.0 * time_span_score
        + 10.0 * role_diversity_score
    )

    return round(
        max(
            0.0,
            min(confidence, 100.0),
        ),
        2,
    )


def build_zones(
    swings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    current_price: float,
    current_atr: float,
    cluster_distance_atr: float = 0.75,
    min_pivots: int = 2,
    confidence_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build horizontal structural support/resistance zones
    from confirmed swing pivots.

    No AI judgement.
    No manually drawn lines.
    No future pivots.
    """

    if current_price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    confirmed = _confirmed_swings(
        swings,
        as_of,
    )

    if confirmed.empty:
        return _empty_zones()

    clusters = _cluster_pivots(
        confirmed,
        cluster_distance_atr=(
            cluster_distance_atr
        ),
        min_pivots=min_pivots,
    )

    if not clusters:
        return _empty_zones()

    confidence_config = (
        confidence_config or {}
    )

    records: list[dict[str, Any]] = []

    for cluster in clusters:
        zone_low = float(
            cluster["price"].min()
        )

        zone_high = float(
            cluster["price"].max()
        )

        zone_center = float(
            cluster["price"].median()
        )

        median_pivot_atr = float(
            cluster["atr"].median()
        )

        zone_width_atr = (
            zone_high - zone_low
        ) / median_pivot_atr

        high_pivot_count = int(
            (
                cluster["kind"] == "HIGH"
            ).sum()
        )

        low_pivot_count = int(
            (
                cluster["kind"] == "LOW"
            ).sum()
        )

        first_touch = pd.Timestamp(
            cluster["pivot_time"].min()
        )

        last_touch = pd.Timestamp(
            cluster["pivot_time"].max()
        )

        mean_prominence_atr = float(
            cluster[
                "prominence_atr"
            ].mean()
        )

        role = classify_zone_role(
            zone_low=zone_low,
            zone_high=zone_high,
            current_price=current_price,
        )

        distance_atr = (
            _distance_from_zone_atr(
                zone_low=zone_low,
                zone_high=zone_high,
                current_price=current_price,
                current_atr=current_atr,
            )
        )

        confidence = zone_confidence(
            cluster,
            as_of=as_of,
            **confidence_config,
        )

        records.append(
            {
                "zone_low": zone_low,
                "zone_high": zone_high,
                "zone_center": zone_center,
                "pivot_count": len(cluster),
                "high_pivot_count":
                    high_pivot_count,
                "low_pivot_count":
                    low_pivot_count,
                "first_touch": first_touch,
                "last_touch": last_touch,
                "mean_prominence_atr":
                    mean_prominence_atr,
                "median_pivot_atr":
                    median_pivot_atr,
                "zone_width_atr":
                    zone_width_atr,
                "distance_from_price_atr":
                    distance_atr,
                "role": role,
                "confidence": confidence,
            }
        )

    result = pd.DataFrame(
        records,
        columns=ZONE_COLUMNS,
    )

    return (
        result
        .sort_values(
            [
                "confidence",
                "distance_from_price_atr",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def nearest_zone(
    zones: pd.DataFrame,
    *,
    role: str,
) -> pd.Series | None:
    """
    Return the nearest SUPPORT or RESISTANCE zone.
    """

    if zones.empty:
        return None

    if role not in {
        "SUPPORT",
        "RESISTANCE",
    }:
        raise ValueError(
            "role must be SUPPORT or RESISTANCE."
        )

    candidates = zones[
        zones["role"] == role
    ]

    if candidates.empty:
        return None

    index = (
        candidates[
            "distance_from_price_atr"
        ]
        .astype(float)
        .idxmin()
    )

    return candidates.loc[index]