from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = [
    "timeframe",
    "family",
    "name",
    "level_low",
    "level_high",
    "level_center",
    "role",
    "confidence",
]


CONFLUENCE_COLUMNS = [
    "cluster_low",
    "cluster_high",
    "cluster_center",
    "role",
    "evidence_count",
    "family_count",
    "timeframe_count",
    "families",
    "timeframes",
    "distance_atr",
    "mean_confidence",
    "score",
    "strength",
]


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=EVIDENCE_COLUMNS
    )


def _empty_confluence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=CONFLUENCE_COLUMNS
    )


def classify_level_role(
    *,
    level: float,
    current_price: float,
    current_atr: float,
    tolerance_atr: float = 0.20,
) -> str:
    """
    Mechanical structural role.

    Levels very close to current price are CURRENT.
    """

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    distance_atr = (
        current_price - level
    ) / current_atr

    if abs(distance_atr) <= tolerance_atr:
        return "CURRENT"

    if current_price > level:
        return "SUPPORT"

    return "RESISTANCE"


def _add_ma_evidence(
    records: list[dict[str, Any]],
    *,
    frame: pd.DataFrame | None,
    timeframe: str,
    windows: tuple[int, ...],
    current_price: float,
    current_atr: float,
    tolerance_atr: float,
) -> None:

    if frame is None or frame.empty:
        return

    latest = frame.iloc[-1]

    for window in windows:
        column = f"sma_{window}"

        if column not in frame.columns:
            continue

        value = latest[column]

        if pd.isna(value):
            continue

        level = float(value)

        role = classify_level_role(
            level=level,
            current_price=current_price,
            current_atr=current_atr,
            tolerance_atr=tolerance_atr,
        )

        records.append(
            {
                "timeframe": timeframe,
                "family": "MOVING_AVERAGE",
                "name": f"{timeframe}_SMA{window}",
                "level_low": level,
                "level_high": level,
                "level_center": level,
                "role": role,

                # MA reliability scoring will be built
                # separately later.
                # 50 = neutral structural evidence.
                "confidence": 50.0,
            }
        )


def _add_zone_evidence(
    records: list[dict[str, Any]],
    *,
    zones: pd.DataFrame | None,
    timeframe: str,
) -> None:

    if zones is None or zones.empty:
        return

    required = {
        "zone_low",
        "zone_high",
        "zone_center",
        "role",
        "confidence",
    }

    if not required.issubset(
        zones.columns
    ):
        raise ValueError(
            "Zone dataframe missing required columns."
        )

    for _, zone in zones.iterrows():
        records.append(
            {
                "timeframe": timeframe,
                "family": "HORIZONTAL_ZONE",
                "name": (
                    f"{timeframe}_ZONE"
                ),
                "level_low":
                    float(zone["zone_low"]),
                "level_high":
                    float(zone["zone_high"]),
                "level_center":
                    float(zone["zone_center"]),
                "role":
                    str(zone["role"]),
                "confidence":
                    float(zone["confidence"]),
            }
        )


def _add_trendline_evidence(
    records: list[dict[str, Any]],
    *,
    trendlines: pd.DataFrame | None,
    timeframe: str,
    current_price: float,
    current_atr: float,
    tolerance_atr: float,
) -> None:

    if (
        trendlines is None
        or trendlines.empty
    ):
        return

    for _, line in trendlines.iterrows():
        status = str(
            line["status"]
        )

        # Broken structural lines no longer provide
        # valid support/resistance evidence.
        if status in {
            "BROKEN_UP",
            "BROKEN_DOWN",
        }:
            continue

        level = float(
            line["line_price_now"]
        )

        role = classify_level_role(
            level=level,
            current_price=current_price,
            current_atr=current_atr,
            tolerance_atr=tolerance_atr,
        )

        records.append(
            {
                "timeframe": timeframe,
                "family": "TRENDLINE",
                "name": (
                    f"{timeframe}_"
                    f"{line['kind']}_"
                    f"{line['direction']}"
                ),
                "level_low": level,
                "level_high": level,
                "level_center": level,
                "role": role,
                "confidence":
                    float(line["confidence"]),
            }
        )


def collect_structural_evidence(
    *,
    current_price: float,
    current_atr: float,
    daily_frame: pd.DataFrame | None = None,
    weekly_frame: pd.DataFrame | None = None,
    daily_zones: pd.DataFrame | None = None,
    weekly_zones: pd.DataFrame | None = None,
    daily_trendlines: pd.DataFrame | None = None,
    weekly_trendlines: pd.DataFrame | None = None,
    current_tolerance_atr: float = 0.20,
) -> pd.DataFrame:
    """
    Convert all structural tools into one standardized
    evidence table.

    No scoring decisions happen here.
    """

    if current_price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    records: list[dict[str, Any]] = []

    _add_ma_evidence(
        records,
        frame=daily_frame,
        timeframe="DAILY",
        windows=(10, 20, 50, 200),
        current_price=current_price,
        current_atr=current_atr,
        tolerance_atr=current_tolerance_atr,
    )

    _add_ma_evidence(
        records,
        frame=weekly_frame,
        timeframe="WEEKLY",
        windows=(10, 20, 40),
        current_price=current_price,
        current_atr=current_atr,
        tolerance_atr=current_tolerance_atr,
    )

    _add_zone_evidence(
        records,
        zones=daily_zones,
        timeframe="DAILY",
    )

    _add_zone_evidence(
        records,
        zones=weekly_zones,
        timeframe="WEEKLY",
    )

    _add_trendline_evidence(
        records,
        trendlines=daily_trendlines,
        timeframe="DAILY",
        current_price=current_price,
        current_atr=current_atr,
        tolerance_atr=current_tolerance_atr,
    )

    _add_trendline_evidence(
        records,
        trendlines=weekly_trendlines,
        timeframe="WEEKLY",
        current_price=current_price,
        current_atr=current_atr,
        tolerance_atr=current_tolerance_atr,
    )

    if not records:
        return _empty_evidence()

    return (
        pd.DataFrame(
            records,
            columns=EVIDENCE_COLUMNS,
        )
        .sort_values(
            "level_center"
        )
        .reset_index(drop=True)
    )


def _roles_are_compatible(
    roles: set[str],
) -> bool:
    """
    CURRENT can coexist with support OR resistance.

    A single cluster may not contain both genuine
    SUPPORT and genuine RESISTANCE evidence.
    """

    return not (
        "SUPPORT" in roles
        and "RESISTANCE" in roles
    )


def _cluster_role(
    roles: set[str],
) -> str:

    if "SUPPORT" in roles:
        return "SUPPORT"

    if "RESISTANCE" in roles:
        return "RESISTANCE"

    return "CURRENT"


def _distance_to_interval_atr(
    *,
    current_price: float,
    low: float,
    high: float,
    current_atr: float,
) -> float:

    if low <= current_price <= high:
        return 0.0

    if current_price < low:
        return (
            low - current_price
        ) / current_atr

    return (
        current_price - high
    ) / current_atr


def _calculate_confluence_score(
    *,
    family_count: int,
    timeframe_count: int,
    mean_confidence: float,
    distance_atr: float,
) -> float:
    """
    Structural confluence score.

    This is NOT yet part of the final Big List
    100-point score.

    Independent evidence breadth matters more than
    raw indicator count.
    """

    family_score = min(
        family_count / 3.0,
        1.0,
    )

    timeframe_score = min(
        timeframe_count / 2.0,
        1.0,
    )

    confidence_score = min(
        max(
            mean_confidence / 100.0,
            0.0,
        ),
        1.0,
    )

    proximity_score = max(
        0.0,
        1.0 - distance_atr / 2.0,
    )

    score = (
        50.0 * family_score
        + 20.0 * timeframe_score
        + 20.0 * confidence_score
        + 10.0 * proximity_score
    )

    return round(
        min(
            max(score, 0.0),
            100.0,
        ),
        2,
    )


def build_confluence_zones(
    evidence: pd.DataFrame,
    *,
    current_price: float,
    current_atr: float,
    cluster_distance_atr: float = 0.75,
    min_evidence_families: int = 2,
    strong_evidence_families: int = 3,
    strong_timeframes: int = 2,
) -> pd.DataFrame:
    """
    Cluster independent structural evidence.

    Crucially:
    multiple moving averages do NOT count as
    multiple evidence families.
    """

    if evidence.empty:
        return _empty_confluence()

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    ordered = (
        evidence
        .sort_values("level_center")
        .reset_index(drop=True)
    )

    max_distance = (
        cluster_distance_atr
        * current_atr
    )

    raw_clusters: list[pd.DataFrame] = []

    indices: list[int] = []

    for index in range(len(ordered)):
        if not indices:
            indices = [index]
            continue

        prospective_indices = (
            indices + [index]
        )

        prospective = ordered.iloc[
            prospective_indices
        ]

        center_range = (
            prospective[
                "level_center"
            ].max()
            -
            prospective[
                "level_center"
            ].min()
        )

        roles = set(
            prospective["role"]
            .astype(str)
        )

        if (
            center_range <= max_distance
            and _roles_are_compatible(roles)
        ):
            indices.append(index)

        else:
            raw_clusters.append(
                ordered.iloc[
                    indices
                ].copy()
            )

            indices = [index]

    if indices:
        raw_clusters.append(
            ordered.iloc[
                indices
            ].copy()
        )

    records = []

    for cluster in raw_clusters:
        families = sorted(
            set(
                cluster["family"]
                .astype(str)
            )
        )

        timeframes = sorted(
            set(
                cluster["timeframe"]
                .astype(str)
            )
        )

        family_count = len(
            families
        )

        if (
            family_count
            < min_evidence_families
        ):
            continue

        timeframe_count = len(
            timeframes
        )

        cluster_low = float(
            cluster[
                "level_low"
            ].min()
        )

        cluster_high = float(
            cluster[
                "level_high"
            ].max()
        )

        cluster_center = float(
            cluster[
                "level_center"
            ].median()
        )

        roles = set(
            cluster["role"]
            .astype(str)
        )

        role = _cluster_role(
            roles
        )

        distance_atr = (
            _distance_to_interval_atr(
                current_price=current_price,
                low=cluster_low,
                high=cluster_high,
                current_atr=current_atr,
            )
        )

        #
        # Prevent multiple indicators from one family
        # from dominating confidence.
        #
        # For each timeframe/family combination,
        # retain only its strongest confidence.
        #
        independent_confidence = (
            cluster.groupby(
                [
                    "timeframe",
                    "family",
                ]
            )["confidence"]
            .max()
        )

        mean_confidence = float(
            independent_confidence.mean()
        )

        score = (
            _calculate_confluence_score(
                family_count=family_count,
                timeframe_count=(
                    timeframe_count
                ),
                mean_confidence=(
                    mean_confidence
                ),
                distance_atr=distance_atr,
            )
        )

        if (
            family_count
            >= strong_evidence_families
            and timeframe_count
            >= strong_timeframes
        ):
            strength = "STRONG"
        else:
            strength = "VALID"

        records.append(
            {
                "cluster_low":
                    cluster_low,
                "cluster_high":
                    cluster_high,
                "cluster_center":
                    cluster_center,
                "role":
                    role,
                "evidence_count":
                    len(cluster),
                "family_count":
                    family_count,
                "timeframe_count":
                    timeframe_count,
                "families":
                    ",".join(families),
                "timeframes":
                    ",".join(timeframes),
                "distance_atr":
                    distance_atr,
                "mean_confidence":
                    mean_confidence,
                "score":
                    score,
                "strength":
                    strength,
            }
        )

    if not records:
        return _empty_confluence()

    return (
        pd.DataFrame(
            records,
            columns=CONFLUENCE_COLUMNS,
        )
        .sort_values(
            [
                "score",
                "distance_atr",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )