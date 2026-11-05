from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.basic import (
    add_atr,
    normalize_ohlcv,
)
from src.features.flow import (
    build_daily_flow_features,
)


def add_breakout_features(
    frame: pd.DataFrame,
    *,
    lookback_days: int = 20,
    atr_period: int = 14,
    breakout_buffer_atr: float = 0.10,
) -> pd.DataFrame:
    """
    Detect objective breakout attempts and confirmed
    breakout closes.

    Prior resistance excludes today's own bar.
    """

    df = normalize_ohlcv(frame)

    atr_column = f"atr_{atr_period}"

    if atr_column not in df.columns:
        df = add_atr(
            df,
            period=atr_period,
        )

    prior_high = (
        df["high"]
        .shift(1)
        .rolling(
            window=lookback_days,
            min_periods=lookback_days,
        )
        .max()
    )

    df[
        f"prior_high_{lookback_days}"
    ] = prior_high

    breakout_threshold = (
        prior_high
        +
        breakout_buffer_atr
        * df[atr_column]
    )

    df["breakout_threshold"] = (
        breakout_threshold
    )

    #
    # Intraday price exceeded the structural threshold.
    #
    df["breakout_probe"] = (
        df["high"]
        > breakout_threshold
    )

    #
    # More important:
    # did price actually CLOSE above it?
    #
    df["breakout_confirmed"] = (
        df["close"]
        > breakout_threshold
    )

    #
    # Price went through resistance intraday,
    # but failed to retain the breakout by close.
    #
    df["failed_breakout_probe"] = (
        df["breakout_probe"]
        & ~df["breakout_confirmed"]
    )

    df["breakout_margin_atr"] = (
        (
            df["close"]
            - prior_high
        )
        /
        df[atr_column]
    )

    return df


def add_breakout_state(
    frame: pd.DataFrame,
    *,
    atr_period: int = 14,
    failure_buffer_atr: float = 0.35,
    accepted_min_holding_ratio: float = 0.67,
    accepted_min_retention: float = 0.50,
) -> pd.DataFrame:
    """
    Track what happens AFTER a confirmed breakout.

    Important:
    each row only uses information available up to
    that row.

    No future bars are used.
    """

    df = frame.copy()

    required = {
        "breakout_confirmed",
        "breakout_threshold",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Breakout state requires columns: "
            f"{sorted(missing)}"
        )

    atr_column = f"atr_{atr_period}"

    if atr_column not in df.columns:
        df = add_atr(
            df,
            period=atr_period,
        )

    #
    # Avoid treating several consecutive breakout bars
    # as several separate breakout events.
    #
    previous_confirmed = (
        df["breakout_confirmed"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    df["breakout_event"] = (
        df["breakout_confirmed"]
        & ~previous_confirmed
    )

    event_id = (
        df["breakout_event"]
        .astype(int)
        .cumsum()
    )

    df["breakout_event_id"] = (
        event_id
    )

    #
    # Store the breakout level from the start
    # of each event.
    #
    breakout_level = (
        df["breakout_threshold"]
        .where(
            df["breakout_event"]
        )
        .ffill()
    )

    df["active_breakout_level"] = (
        breakout_level
    )

    has_event = (
        event_id > 0
    )

    #
    # Number of bars since current breakout event began.
    #
    days_since = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if has_event.any():
        grouped_count = (
            df.loc[
                has_event
            ]
            .groupby(
                event_id[has_event]
            )
            .cumcount()
        )

        days_since.loc[
            has_event
        ] = grouped_count.astype(float)

    df["bars_since_breakout"] = (
        days_since
    )

    #
    # Highest close achieved since current breakout.
    #
    peak_close = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if has_event.any():
        peak_close.loc[
            has_event
        ] = (
            df.loc[
                has_event,
                "close",
            ]
            .groupby(
                event_id[has_event]
            )
            .cummax()
        )

    df["peak_close_since_breakout"] = (
        peak_close
    )

    #
    # Is price still above the breakout level?
    #
    holding = (
        df["close"]
        >= breakout_level
    )

    holding = (
        holding
        & has_event
    )

    df["breakout_holding"] = holding

    #
    # Percentage of bars since breakout that have
    # remained above the breakout level.
    #
    holding_ratio = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if has_event.any():
        values = (
            holding.loc[
                has_event
            ]
            .astype(float)
        )

        ratio = (
            values
            .groupby(
                event_id[has_event]
            )
            .expanding()
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        holding_ratio.loc[
            has_event
        ] = ratio

    df[
        "breakout_holding_ratio"
    ] = holding_ratio

    #
    # Progress retention.
    #
    # Example:
    #
    # breakout level = 100
    # peak close = 120
    # current close = 115
    #
    # retention = 75%
    #
    denominator = (
        peak_close
        - breakout_level
    )

    retention = (
        df["close"]
        - breakout_level
    ) / denominator.replace(
        0,
        np.nan,
    )

    #
    # On the breakout bar itself denominator can
    # be zero or extremely small.
    #
    retention = retention.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df["breakout_retention"] = (
        retention
    )

    failure_threshold = (
        breakout_level
        -
        failure_buffer_atr
        * df[atr_column]
    )

    df["breakout_failure_level"] = (
        failure_threshold
    )

    failed = (
        has_event
        & (
            df["close"]
            < failure_threshold
        )
    )

    df["breakout_failed"] = (
        failed
    )

    state = pd.Series(
        "NO_EVENT",
        index=df.index,
        dtype="object",
    )

    state.loc[
        has_event
    ] = "TESTING"

    state.loc[
        failed
    ] = "FAILED"

    accepted = (
        has_event
        & ~failed
        & (
            holding_ratio
            >= accepted_min_holding_ratio
        )
        & (
            retention.fillna(1.0)
            >= accepted_min_retention
        )
        & holding
    )

    state.loc[
        accepted
    ] = "ACCEPTED"

    df["breakout_state"] = state

    return df


def add_progress_retention(
    frame: pd.DataFrame,
    *,
    impulse_window: int = 10,
) -> pd.DataFrame:
    """
    Rolling price-progress retention.

    Measures how much of a recent upside impulse is
    still retained.

    No future information is used.
    """

    df = frame.copy()

    start_price = (
        df["close"]
        .shift(impulse_window)
    )

    rolling_peak = (
        df["close"]
        .rolling(
            window=impulse_window + 1,
            min_periods=impulse_window + 1,
        )
        .max()
    )

    impulse_gain = (
        rolling_peak
        - start_price
    )

    retained_gain = (
        df["close"]
        - start_price
    )

    # Distinguish true data insufficiency from a valid
    # zero-impulse observation.
    #
    # - insufficient history -> NaN
    # - full history but no positive impulse -> 0.0
    # - positive impulse -> retained fraction
    valid_history = (
        start_price.notna()
        & rolling_peak.notna()
    )

    positive_impulse = (
        valid_history
        & impulse_gain.gt(0)
    )

    zero_impulse = (
        valid_history
        & impulse_gain.eq(0)
    )

    retention = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    retention.loc[
        zero_impulse
    ] = 0.0

    retention.loc[
        positive_impulse
    ] = (
        retained_gain.loc[
            positive_impulse
        ]
        /
        impulse_gain.loc[
            positive_impulse
        ]
    )

    df[
        f"progress_retention_{impulse_window}"
    ] = retention

    return df


def evaluate_zone_interaction(
    frame: pd.DataFrame,
    zones: pd.DataFrame,
    *,
    current_atr: float,
    touch_buffer_atr: float = 0.25,
    strong_close_location: float = 0.65,
    min_relative_volume: float = 1.00,
) -> dict:
    """
    Evaluate CURRENT bar interaction with nearest
    support/resistance zones.

    This is intended for the current screener run.

    Zone generation itself must already be as-of safe.
    """

    if frame.empty:
        raise ValueError(
            "Price frame is empty."
        )

    if current_atr <= 0:
        raise ValueError(
            "current_atr must be positive."
        )

    if zones is None or zones.empty:
        return {
            "support_test": False,
            "support_hold": False,
            "resistance_test": False,
            "resistance_rejection": False,
            "zone_breakout": False,
        }

    df = frame.copy()

    if (
        "relative_volume_20"
        not in df.columns
        or "clv_01"
        not in df.columns
    ):
        df = build_daily_flow_features(
            df
        )

    latest = df.iloc[-1]

    current_low = float(
        latest["low"]
    )

    current_high = float(
        latest["high"]
    )

    current_close = float(
        latest["close"]
    )

    previous_close = float(
        df["close"].iloc[-2]
    )

    rvol = latest[
        "relative_volume_20"
    ]

    clv = latest[
        "clv_01"
    ]

    result = {
        "support_test": False,
        "support_hold": False,
        "resistance_test": False,
        "resistance_rejection": False,
        "zone_breakout": False,
    }

    support_zones = (
        zones[
            zones["role"]
            == "SUPPORT"
        ]
    )

    if not support_zones.empty:
        support = support_zones.sort_values(
            "distance_from_price_atr"
        ).iloc[0]

        zone_low = float(
            support["zone_low"]
        )

        zone_high = float(
            support["zone_high"]
        )

        touched = (
            current_low
            <= (
                zone_high
                + touch_buffer_atr
                * current_atr
            )
            and current_high >= zone_low
        )

        result[
            "support_test"
        ] = bool(touched)

        support_hold = (
            touched
            and current_close > zone_high
            and clv >= strong_close_location
            and (
                pd.isna(rvol)
                or rvol
                >= min_relative_volume
            )
        )

        result[
            "support_hold"
        ] = bool(
            support_hold
        )

    resistance_zones = (
        zones[
            zones["role"]
            == "RESISTANCE"
        ]
    )

    if not resistance_zones.empty:
        resistance = (
            resistance_zones
            .sort_values(
                "distance_from_price_atr"
            )
            .iloc[0]
        )

        zone_low = float(
            resistance["zone_low"]
        )

        zone_high = float(
            resistance["zone_high"]
        )

        touched = (
            current_high
            >= (
                zone_low
                - touch_buffer_atr
                * current_atr
            )
            and current_low <= zone_high
        )

        result[
            "resistance_test"
        ] = bool(touched)

        rejection = (
            touched
            and current_close < zone_low
            and clv
            <= (
                1.0
                - strong_close_location
            )
        )

        result[
            "resistance_rejection"
        ] = bool(
            rejection
        )

        breakout = (
            previous_close <= zone_high
            and current_close
            > (
                zone_high
                + touch_buffer_atr
                * current_atr
            )
            and clv
            >= strong_close_location
            and (
                pd.isna(rvol)
                or rvol
                >= min_relative_volume
            )
        )

        result[
            "zone_breakout"
        ] = bool(
            breakout
        )

    return result


def build_acceptance_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Phase 3B daily acceptance pipeline.
    """

    df = build_daily_flow_features(
        frame
    )

    df = add_atr(
        df,
        period=14,
    )

    df = add_breakout_features(
        df,
        lookback_days=20,
        atr_period=14,
        breakout_buffer_atr=0.10,
    )

    df = add_breakout_state(
        df,
        atr_period=14,
        failure_buffer_atr=0.35,
        accepted_min_holding_ratio=0.67,
        accepted_min_retention=0.50,
    )

    df = add_progress_retention(
        df,
        impulse_window=10,
    )

    return df