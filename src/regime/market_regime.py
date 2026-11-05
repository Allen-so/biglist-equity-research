from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.basic import (
    add_moving_averages,
    normalize_ohlcv,
)
from src.features.relative_strength import (
    relative_total_return,
    return_over_sessions,
)


def _visible_frame(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Strict as-of protection.
    """

    df = normalize_ohlcv(
        frame
    )

    as_of = pd.Timestamp(
        as_of
    )

    return df[
        df.index <= as_of
    ].copy()


def _safe_slope(
    series: pd.Series,
    *,
    lookback: int,
) -> float:
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


def _linear_points(
    value: float,
    *,
    zero_at: float,
    full_at: float,
    max_points: float,
) -> float:
    """
    Linear scoring with hard clipping.

    value <= zero_at:
        0 points

    value >= full_at:
        full points
    """

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
        fraction
        * max_points
    )


def classify_regime_score(
    score: float,
) -> str:
    """
    Frozen market regime thresholds.

        >= 75  STRONG_RISK_ON
        >= 60  CONSTRUCTIVE
        >= 45  SELECTIVE
        <  45  DEFENSIVE
    """

    if pd.isna(score):
        return "INCOMPLETE"

    value = float(
        score
    )

    if not 0 <= value <= 100:
        raise ValueError(
            "Regime score must be between 0 and 100."
        )

    if value >= 75:
        return "STRONG_RISK_ON"

    if value >= 60:
        return "CONSTRUCTIVE"

    if value >= 45:
        return "SELECTIVE"

    return "DEFENSIVE"


def _calculate_single_index_trend(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Each major index contributes max 15 points.

    Conditions:

        Close >= SMA20      2.5
        Close >= SMA50      2.5
        Close >= SMA200     2.5
        SMA20 >= SMA50      2.5
        SMA20 slope > 0     2.5
        SMA50 slope > 0     2.5
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    if visible.empty:
        return {
            "points": np.nan,
        }

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

    sma200 = visible[
        "sma_200"
    ].iloc[-1]

    slope20 = _safe_slope(
        visible["sma_20"],
        lookback=10,
    )

    slope50 = _safe_slope(
        visible["sma_50"],
        lookback=20,
    )

    required = (
        sma20,
        sma50,
        sma200,
        slope20,
        slope50,
    )

    if any(
        pd.isna(value)
        for value in required
    ):
        return {
            "points": np.nan,
        }

    conditions = {
        "close_above_sma20":
            close >= sma20,

        "close_above_sma50":
            close >= sma50,

        "close_above_sma200":
            close >= sma200,

        "sma20_above_sma50":
            sma20 >= sma50,

        "sma20_slope_positive":
            slope20 > 0,

        "sma50_slope_positive":
            slope50 > 0,
    }

    points = (
        sum(
            bool(value)
            for value in conditions.values()
        )
        * 2.5
    )

    return {
        "points":
            float(points),

        "close":
            close,

        "sma20":
            float(sma20),

        "sma50":
            float(sma50),

        "sma200":
            float(sma200),

        "sma20_slope_10d":
            float(slope20),

        "sma50_slope_20d":
            float(slope50),

        **conditions,
    }


def calculate_index_trend(
    spy_frame: pd.DataFrame,
    qqq_frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Index Trend = 30 points.

        SPY 15
        QQQ 15
    """

    spy = _calculate_single_index_trend(
        spy_frame,
        as_of=as_of,
    )

    qqq = _calculate_single_index_trend(
        qqq_frame,
        as_of=as_of,
    )

    if (
        pd.isna(
            spy.get(
                "points",
                np.nan,
            )
        )
        or
        pd.isna(
            qqq.get(
                "points",
                np.nan,
            )
        )
    ):
        total = np.nan

    else:
        total = (
            float(
                spy["points"]
            )
            +
            float(
                qqq["points"]
            )
        )

    result = {
        "index_trend_points":
            total,
    }

    for key, value in (
        spy.items()
    ):
        result[
            f"spy_{key}"
        ] = value

    for key, value in (
        qqq.items()
    ):
        result[
            f"qqq_{key}"
        ] = value

    return result


def calculate_market_breadth(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Breadth = 30 points.

        % above SMA20 -> 15
        % above SMA50 -> 15

    Scoring uses absolute breadth thresholds,
    not cross-sectional percentiles.
    """

    valid20 = 0
    above20 = 0

    valid50 = 0
    above50 = 0

    for frame in universe.values():
        visible = _visible_frame(
            frame,
            as_of=as_of,
        )

        if visible.empty:
            continue

        visible = (
            add_moving_averages(
                visible
            )
        )

        close = visible[
            "close"
        ].iloc[-1]

        sma20 = visible[
            "sma_20"
        ].iloc[-1]

        sma50 = visible[
            "sma_50"
        ].iloc[-1]

        if not pd.isna(
            sma20
        ):
            valid20 += 1

            if close >= sma20:
                above20 += 1

        if not pd.isna(
            sma50
        ):
            valid50 += 1

            if close >= sma50:
                above50 += 1

    fraction20 = (
        above20 / valid20
        if valid20 > 0
        else np.nan
    )

    fraction50 = (
        above50 / valid50
        if valid50 > 0
        else np.nan
    )

    points20 = _linear_points(
        fraction20,
        zero_at=0.25,
        full_at=0.75,
        max_points=15.0,
    )

    points50 = _linear_points(
        fraction50,
        zero_at=0.20,
        full_at=0.70,
        max_points=15.0,
    )

    if (
        pd.isna(points20)
        or pd.isna(points50)
    ):
        total = np.nan

    else:
        total = (
            points20
            + points50
        )

    return {
        "breadth_points":
            total,

        "above_sma20_fraction":
            fraction20,

        "above_sma50_fraction":
            fraction50,

        "above_sma20_count":
            above20,

        "above_sma50_count":
            above50,

        "breadth_valid_sma20":
            valid20,

        "breadth_valid_sma50":
            valid50,

        "breadth_sma20_points":
            points20,

        "breadth_sma50_points":
            points50,
    }


def calculate_leadership_breadth(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    rs_snapshot: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Leadership Breadth = 20 points.

    A leader must have:

        RS20 percentile >= 80
        AND
        Close >= SMA20

    Since RS >= 80 is inherently approximately
    the top 20% of the valid universe, leadership
    breadth is normalized against 2%-18%.
    """

    required = {
        "symbol",
        "rs_20d_pct",
    }

    missing = required.difference(
        rs_snapshot.columns
    )

    if missing:
        raise ValueError(
            "RS snapshot missing columns: "
            f"{sorted(missing)}"
        )

    rs_lookup = {
        str(row["symbol"]):
            row["rs_20d_pct"]
        for _, row in (
            rs_snapshot.iterrows()
        )
    }

    valid = 0
    leaders = 0

    for symbol, frame in (
        universe.items()
    ):
        symbol = str(
            symbol
        )

        rs20 = rs_lookup.get(
            symbol,
            np.nan,
        )

        if pd.isna(rs20):
            continue

        visible = _visible_frame(
            frame,
            as_of=as_of,
        )

        if visible.empty:
            continue

        visible = add_moving_averages(
            visible
        )

        sma20 = visible[
            "sma_20"
        ].iloc[-1]

        if pd.isna(sma20):
            continue

        close = float(
            visible[
                "close"
            ].iloc[-1]
        )

        valid += 1

        if (
            float(rs20) >= 80.0
            and
            close >= sma20
        ):
            leaders += 1

    fraction = (
        leaders / valid
        if valid > 0
        else np.nan
    )

    points = _linear_points(
        fraction,
        zero_at=0.02,
        full_at=0.18,
        max_points=20.0,
    )

    return {
        "leadership_breadth_points":
            points,

        "leadership_fraction":
            fraction,

        "leadership_count":
            leaders,

        "leadership_valid_count":
            valid,
    }


def _high_low_balance_60d(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Responsive market participation measure.

    Near 60D high:
        Close >= 98% of 60D high

    Near 60D low:
        Close <= 102% of 60D low
    """

    valid = 0
    near_high = 0
    near_low = 0

    for frame in universe.values():
        visible = _visible_frame(
            frame,
            as_of=as_of,
        )

        if len(visible) < 60:
            continue

        recent = visible.tail(
            60
        )

        current = float(
            recent[
                "close"
            ].iloc[-1]
        )

        high60 = float(
            recent[
                "high"
            ].max()
        )

        low60 = float(
            recent[
                "low"
            ].min()
        )

        valid += 1

        if current >= (
            high60 * 0.98
        ):
            near_high += 1

        if current <= (
            low60 * 1.02
        ):
            near_low += 1

    if valid == 0:
        return {
            "high_low_balance_60d":
                np.nan,

            "near_60d_high_fraction":
                np.nan,

            "near_60d_low_fraction":
                np.nan,

            "high_low_valid_count":
                0,
        }

    high_fraction = (
        near_high / valid
    )

    low_fraction = (
        near_low / valid
    )

    balance = (
        high_fraction
        - low_fraction
    )

    return {
        "high_low_balance_60d":
            float(balance),

        "near_60d_high_fraction":
            float(
                high_fraction
            ),

        "near_60d_low_fraction":
            float(
                low_fraction
            ),

        "high_low_valid_count":
            valid,
    }


def calculate_risk_participation(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    spy_frame: pd.DataFrame,
    iwm_frame: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Risk Participation = 20 points.

        IWM vs SPY 20D       8
        Median Stock 20D     6
        60D High/Low Balance 6
    """

    spy_return = (
        return_over_sessions(
            spy_frame,
            as_of=as_of,
            sessions=20,
        )
    )

    iwm_return = (
        return_over_sessions(
            iwm_frame,
            as_of=as_of,
            sessions=20,
        )
    )

    iwm_vs_spy = (
        relative_total_return(
            iwm_return,
            spy_return,
        )
    )

    iwm_points = _linear_points(
        iwm_vs_spy,
        zero_at=-0.05,
        full_at=0.05,
        max_points=8.0,
    )

    universe_returns = []

    for frame in universe.values():
        value = return_over_sessions(
            frame,
            as_of=as_of,
            sessions=20,
        )

        if not pd.isna(value):
            universe_returns.append(
                float(value)
            )

    if universe_returns:
        median_return = float(
            np.median(
                universe_returns
            )
        )

    else:
        median_return = np.nan

    median_points = _linear_points(
        median_return,
        zero_at=-0.10,
        full_at=0.10,
        max_points=6.0,
    )

    high_low = (
        _high_low_balance_60d(
            universe,
            as_of=as_of,
        )
    )

    high_low_points = (
        _linear_points(
            high_low[
                "high_low_balance_60d"
            ],
            zero_at=-0.20,
            full_at=0.20,
            max_points=6.0,
        )
    )

    components = (
        iwm_points,
        median_points,
        high_low_points,
    )

    if any(
        pd.isna(value)
        for value in components
    ):
        total = np.nan

    else:
        total = float(
            sum(
                components
            )
        )

    return {
        "risk_participation_points":
            total,

        "iwm_vs_spy_20d":
            iwm_vs_spy,

        "iwm_vs_spy_points":
            iwm_points,

        "median_stock_return_20d":
            median_return,

        "median_return_points":
            median_points,

        "high_low_points":
            high_low_points,

        **high_low,
    }


def calculate_bollinger_context(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window: int = 20,
    std_multiplier: float = 2.0,
    percentile_lookback: int = 120,
) -> dict[str, Any]:
    """
    Bollinger is MARKET CONTEXT ONLY.

    It never enters individual-stock scoring.

    Outputs:
        %B
        Bandwidth
        Bandwidth percentile
        Squeeze
        Expansion
    """

    visible = _visible_frame(
        frame,
        as_of=as_of,
    )

    close = visible[
        "close"
    ].astype(float)

    if len(close) < window:
        return {
            "bb_percent_b":
                np.nan,

            "bb_bandwidth":
                np.nan,

            "bb_bandwidth_pct":
                np.nan,

            "bb_squeeze":
                False,

            "bb_expansion":
                False,
        }

    middle = (
        close
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    std = (
        close
        .rolling(
            window=window,
            min_periods=window,
        )
        .std(
            ddof=0
        )
    )

    upper = (
        middle
        + std_multiplier
        * std
    )

    lower = (
        middle
        - std_multiplier
        * std
    )

    width = (
        upper
        - lower
    )

    percent_b = (
        (
            close
            - lower
        )
        /
        width.replace(
            0,
            np.nan,
        )
    )

    bandwidth = (
        width
        /
        middle.replace(
            0,
            np.nan,
        )
    )

    current_bandwidth = (
        bandwidth.iloc[-1]
    )

    current_percent_b = (
        percent_b.iloc[-1]
    )

    history = (
        bandwidth
        .dropna()
        .tail(
            percentile_lookback
        )
    )

    if (
        history.empty
        or pd.isna(
            current_bandwidth
        )
    ):
        bandwidth_pct = np.nan

    else:
        bandwidth_pct = float(
            (
                history
                <= current_bandwidth
            ).mean()
            * 100.0
        )

    previous_bandwidth = (
        bandwidth.iloc[-2]
        if len(
            bandwidth
        ) >= 2
        else np.nan
    )

    squeeze = bool(
        not pd.isna(
            bandwidth_pct
        )
        and
        bandwidth_pct <= 20.0
    )

    expansion = bool(
        not pd.isna(
            bandwidth_pct
        )
        and
        bandwidth_pct >= 80.0
        and
        not pd.isna(
            previous_bandwidth
        )
        and
        current_bandwidth
        > previous_bandwidth
    )

    return {
        "bb_percent_b":
            (
                float(
                    current_percent_b
                )
                if not pd.isna(
                    current_percent_b
                )
                else np.nan
            ),

        "bb_bandwidth":
            (
                float(
                    current_bandwidth
                )
                if not pd.isna(
                    current_bandwidth
                )
                else np.nan
            ),

        "bb_bandwidth_pct":
            bandwidth_pct,

        "bb_squeeze":
            squeeze,

        "bb_expansion":
            expansion,
    }


def calculate_market_regime(
    universe: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    spy_frame: pd.DataFrame,
    qqq_frame: pd.DataFrame,
    iwm_frame: pd.DataFrame,
    rs_snapshot: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """
    Complete Market Regime Engine.

        Index Trend         30
        Breadth             30
        Leadership Breadth  20
        Risk Participation  20
        ----------------------
        Total              100
    """

    index_trend = (
        calculate_index_trend(
            spy_frame,
            qqq_frame,
            as_of=as_of,
        )
    )

    breadth = (
        calculate_market_breadth(
            universe,
            as_of=as_of,
        )
    )

    leadership = (
        calculate_leadership_breadth(
            universe,
            rs_snapshot,
            as_of=as_of,
        )
    )

    risk = (
        calculate_risk_participation(
            universe,
            spy_frame=spy_frame,
            iwm_frame=iwm_frame,
            as_of=as_of,
        )
    )

    major_components = {
        "index_trend":
            index_trend[
                "index_trend_points"
            ],

        "breadth":
            breadth[
                "breadth_points"
            ],

        "leadership_breadth":
            leadership[
                "leadership_breadth_points"
            ],

        "risk_participation":
            risk[
                "risk_participation_points"
            ],
    }

    missing_components = [
        name
        for name, value in (
            major_components.items()
        )
        if pd.isna(value)
    ]

    if missing_components:
        regime_score = np.nan
        regime = "INCOMPLETE"
        status = "INCOMPLETE"

    else:
        regime_score = float(
            sum(
                major_components.values()
            )
        )

        regime_score = min(
            max(
                regime_score,
                0.0,
            ),
            100.0,
        )

        regime = (
            classify_regime_score(
                regime_score
            )
        )

        status = "COMPLETE"

    spy_bb = (
        calculate_bollinger_context(
            spy_frame,
            as_of=as_of,
        )
    )

    qqq_bb = (
        calculate_bollinger_context(
            qqq_frame,
            as_of=as_of,
        )
    )

    iwm_bb = (
        calculate_bollinger_context(
            iwm_frame,
            as_of=as_of,
        )
    )

    result = {
        "as_of":
            pd.Timestamp(
                as_of
            ),

        "regime_score":
            (
                round(
                    regime_score,
                    4,
                )
                if not pd.isna(
                    regime_score
                )
                else np.nan
            ),

        "regime":
            regime,

        "regime_status":
            status,

        "missing_regime_components":
            missing_components,

        **index_trend,
        **breadth,
        **leadership,
        **risk,
    }

    for key, value in (
        spy_bb.items()
    ):
        result[
            f"spy_{key}"
        ] = value

    for key, value in (
        qqq_bb.items()
    ):
        result[
            f"qqq_{key}"
        ] = value

    for key, value in (
        iwm_bb.items()
    ):
        result[
            f"iwm_{key}"
        ] = value

    return result