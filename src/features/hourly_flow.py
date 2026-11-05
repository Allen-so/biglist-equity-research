from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.basic import (
    normalize_ohlcv,
)
from src.features.flow import (
    add_close_location,
)


def add_session_metadata(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deterministic intraday session metadata.

    Assumption:
        DataProvider has already removed
        pre-market and after-hours bars.

    Each calendar trading date is therefore treated
    as one regular trading session.
    """

    df = normalize_ohlcv(frame)

    df["session"] = (
        df.index.normalize()
    )

    df["session_bar"] = (
        df.groupby(
            "session"
        )
        .cumcount()
    )

    df["bars_in_session"] = (
        df.groupby(
            "session"
        )["close"]
        .transform("size")
    )

    df["session_progress"] = (
        df["session_bar"] + 1
    ) / df["bars_in_session"]

    return df


def add_session_vwap_proxy(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hourly-OHLCV approximation of regular-session VWAP.

    This is intentionally named VWAP PROXY because
    true trade-level VWAP requires finer intraday data
    or vendor-provided VWAP.

    Proxy:
        cumulative(HLC3 * Volume)
        /
        cumulative(Volume)

    Resets every trading session.
    """

    df = frame.copy()

    if "session" not in df.columns:
        df = add_session_metadata(df)

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    dollar_proxy = (
        typical_price
        * df["volume"]
    )

    cumulative_dollar = (
        dollar_proxy
        .groupby(
            df["session"]
        )
        .cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(
            df["session"]
        )
        .cumsum()
        .replace(
            0,
            np.nan,
        )
    )

    df["session_vwap_proxy"] = (
        cumulative_dollar
        / cumulative_volume
    )

    df["above_session_vwap"] = (
        df["close"]
        > df["session_vwap_proxy"]
    )

    df["bars_above_vwap_ratio"] = (
        df["above_session_vwap"]
        .astype(float)
        .groupby(
            df["session"]
        )
        .expanding()
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    return df


def add_slot_relative_volume(
    frame: pd.DataFrame,
    *,
    lookback_sessions: int = 20,
    min_history_sessions: int = 10,
) -> pd.DataFrame:
    """
    Intraday volume must be compared against the SAME
    session slot, not against arbitrary neighbouring
    hourly bars.

    Example:
        today's first hour
        vs
        prior sessions' first hours.

    This handles the U-shaped intraday volume curve
    much better than a simple rolling hourly average.

    Current-session volume is excluded from its own
    baseline using shift(1).
    """

    df = frame.copy()

    if "session_bar" not in df.columns:
        df = add_session_metadata(df)

    baseline = (
        df.groupby(
            "session_bar",
            group_keys=False,
        )["volume"]
        .transform(
            lambda series:
                series.shift(1)
                .rolling(
                    window=(
                        lookback_sessions
                    ),
                    min_periods=(
                        min_history_sessions
                    ),
                )
                .median()
        )
    )

    df[
        "slot_median_volume_prior"
    ] = baseline

    df["slot_relative_volume"] = (
        df["volume"]
        /
        baseline.replace(
            0,
            np.nan,
        )
    )

    #
    # Expected cumulative volume up to the current
    # point in the session.
    #
    df[
        "expected_volume_to_now"
    ] = (
        baseline
        .groupby(
            df["session"]
        )
        .cumsum()
    )

    df[
        "actual_volume_to_now"
    ] = (
        df["volume"]
        .groupby(
            df["session"]
        )
        .cumsum()
    )

    df["session_relative_volume"] = (
        df["actual_volume_to_now"]
        /
        df[
            "expected_volume_to_now"
        ].replace(
            0,
            np.nan,
        )
    )

    return df


def add_hourly_price_behavior(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Describe price acceptance inside each session.
    """

    df = frame.copy()

    if "session" not in df.columns:
        df = add_session_metadata(df)

    df = add_close_location(df)

    session_open = (
        df.groupby(
            "session"
        )["open"]
        .transform("first")
    )

    df["session_open"] = (
        session_open
    )

    df["session_high_to_now"] = (
        df["high"]
        .groupby(
            df["session"]
        )
        .cummax()
    )

    df["session_low_to_now"] = (
        df["low"]
        .groupby(
            df["session"]
        )
        .cummin()
    )

    session_range = (
        df["session_high_to_now"]
        - df["session_low_to_now"]
    )

    df["session_close_location"] = (
        (
            df["close"]
            - df["session_low_to_now"]
        )
        /
        session_range.replace(
            0,
            np.nan,
        )
    )

    df["session_return_from_open"] = (
        df["close"]
        /
        session_open.replace(
            0,
            np.nan,
        )
        - 1.0
    )

    previous_close_in_session = (
        df.groupby(
            "session"
        )["close"]
        .shift(1)
    )

    first_bar = (
        previous_close_in_session
        .isna()
    )

    directional_return = (
        df["close"]
        /
        previous_close_in_session
        - 1.0
    )

    directional_return.loc[
        first_bar
    ] = (
        df.loc[
            first_bar,
            "close",
        ]
        /
        df.loc[
            first_bar,
            "open",
        ]
        - 1.0
    )

    df["hourly_directional_return"] = (
        directional_return
    )

    previous_high = (
        df.groupby(
            "session"
        )["high"]
        .shift(1)
    )

    previous_low = (
        df.groupby(
            "session"
        )["low"]
        .shift(1)
    )

    df["hourly_higher_high"] = (
        df["high"]
        > previous_high
    )

    df["hourly_higher_low"] = (
        df["low"]
        > previous_low
    )

    df["hourly_lower_high"] = (
        df["high"]
        < previous_high
    )

    df["hourly_lower_low"] = (
        df["low"]
        < previous_low
    )

    return df


def add_hourly_volume_balance(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare volume occurring during upward movement
    with volume occurring during pullbacks.

    Lower down/up ratio = healthier pullback behaviour.
    """

    df = frame.copy()

    required = {
        "session",
        "hourly_directional_return",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        df = add_hourly_price_behavior(
            df
        )

    up_volume = df["volume"].where(
        df[
            "hourly_directional_return"
        ] > 0,
        0.0,
    )

    down_volume = df["volume"].where(
        df[
            "hourly_directional_return"
        ] < 0,
        0.0,
    )

    df["session_up_volume"] = (
        up_volume
        .groupby(
            df["session"]
        )
        .cumsum()
    )

    df["session_down_volume"] = (
        down_volume
        .groupby(
            df["session"]
        )
        .cumsum()
    )

    ratio = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    has_up_volume = (
        df["session_up_volume"] > 0
    )

    ratio.loc[
        has_up_volume
    ] = (
        df.loc[
            has_up_volume,
            "session_down_volume",
        ]
        /
        df.loc[
            has_up_volume,
            "session_up_volume",
        ]
    )

    df[
        "session_down_up_volume_ratio"
    ] = ratio

    return df


def build_hourly_flow_features(
    frame: pd.DataFrame,
    *,
    lookback_sessions: int = 20,
    min_history_sessions: int = 10,
) -> pd.DataFrame:
    """
    Phase 3D hourly capital-flow primitives.

    No Big List scoring occurs here.
    """

    df = add_session_metadata(
        frame
    )

    df = add_session_vwap_proxy(
        df
    )

    df = add_slot_relative_volume(
        df,
        lookback_sessions=(
            lookback_sessions
        ),
        min_history_sessions=(
            min_history_sessions
        ),
    )

    df = add_hourly_price_behavior(
        df
    )

    df = add_hourly_volume_balance(
        df
    )

    return df


def _structure_balance(
    session: pd.DataFrame,
) -> int:
    """
    Positive:
        HH + HL dominate.

    Negative:
        LH + LL dominate.
    """

    positive = int(
        session[
            "hourly_higher_high"
        ].sum()
        +
        session[
            "hourly_higher_low"
        ].sum()
    )

    negative = int(
        session[
            "hourly_lower_high"
        ].sum()
        +
        session[
            "hourly_lower_low"
        ].sum()
    )

    return positive - negative


def summarize_hourly_flow(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    daily_atr: float,
    lookback_sessions: int = 20,
    min_history_sessions: int = 10,
    late_bars: int = 2,
) -> dict[str, Any]:
    """
    Summarize only information available by `as_of`.

    1H confirmation does NOT hard-reject a valid
    Daily setup.

    It produces:
        SUPPORTIVE
        NEUTRAL
        NEGATIVE

    for later READY / DEVELOPING classification.
    """

    if daily_atr <= 0:
        raise ValueError(
            "daily_atr must be positive."
        )

    as_of = pd.Timestamp(as_of)

    visible = frame[
        frame.index <= as_of
    ].copy()

    if visible.empty:
        raise ValueError(
            "No hourly data available by as_of."
        )

    features = (
        build_hourly_flow_features(
            visible,
            lookback_sessions=(
                lookback_sessions
            ),
            min_history_sessions=(
                min_history_sessions
            ),
        )
    )

    latest_session_id = (
        features["session"].iloc[-1]
    )

    session = features[
        features["session"]
        == latest_session_id
    ].copy()

    latest = session.iloc[-1]

    current_close = float(
        latest["close"]
    )

    current_vwap = float(
        latest["session_vwap_proxy"]
    )

    close_vs_vwap_atr = (
        current_close
        - current_vwap
    ) / daily_atr

    session_close_location = (
        latest[
            "session_close_location"
        ]
    )

    bars_above_vwap_ratio = float(
        latest[
            "bars_above_vwap_ratio"
        ]
    )

    session_relative_volume = (
        latest[
            "session_relative_volume"
        ]
    )

    latest_slot_rvol = (
        latest[
            "slot_relative_volume"
        ]
    )

    down_up_volume_ratio = (
        latest[
            "session_down_up_volume_ratio"
        ]
    )

    structure_balance = (
        _structure_balance(
            session
        )
    )

    number_late_bars = min(
        late_bars,
        len(session),
    )

    late = session.tail(
        number_late_bars
    )

    late_start_open = float(
        late["open"].iloc[0]
    )

    late_end_close = float(
        late["close"].iloc[-1]
    )

    late_return_atr = (
        late_end_close
        - late_start_open
    ) / daily_atr

    late_close_location = float(
        late[
            "clv_01"
        ].mean()
    )

    positive = 0
    negative = 0

    #
    # VWAP position
    #
    if close_vs_vwap_atr >= 0.10:
        positive += 1

    elif close_vs_vwap_atr <= -0.10:
        negative += 1

    #
    # Time spent above VWAP
    #
    if bars_above_vwap_ratio >= 0.60:
        positive += 1

    elif bars_above_vwap_ratio <= 0.40:
        negative += 1

    #
    # Price acceptance within today's range
    #
    if (
        not pd.isna(
            session_close_location
        )
        and
        session_close_location >= 0.65
    ):
        positive += 1

    elif (
        not pd.isna(
            session_close_location
        )
        and
        session_close_location <= 0.35
    ):
        negative += 1

    #
    # Micro structure
    #
    if structure_balance >= 1:
        positive += 1

    elif structure_balance <= -1:
        negative += 1

    #
    # Pullback-volume behaviour
    #
    if not pd.isna(
        down_up_volume_ratio
    ):
        if (
            down_up_volume_ratio
            <= 0.80
        ):
            positive += 1

        elif (
            down_up_volume_ratio
            >= 1.25
        ):
            negative += 1

    #
    # High participation means more when combined
    # with where price actually closes.
    #
    if not pd.isna(
        session_relative_volume
    ):
        if (
            session_relative_volume
            >= 1.10
            and
            not pd.isna(
                session_close_location
            )
            and
            session_close_location
            >= 0.65
        ):
            positive += 1

        elif (
            session_relative_volume
            >= 1.10
            and
            not pd.isna(
                session_close_location
            )
            and
            session_close_location
            <= 0.35
        ):
            negative += 1

    #
    # Late-session behaviour
    #
    if (
        late_return_atr >= 0.10
        and
        late_close_location >= 0.65
    ):
        positive += 1

    elif (
        late_return_atr <= -0.10
        and
        late_close_location <= 0.35
    ):
        negative += 1

    if (
        positive >= 4
        and
        positive - negative >= 2
    ):
        confirmation = "SUPPORTIVE"

    elif (
        negative >= 4
        and
        negative - positive >= 2
    ):
        confirmation = "NEGATIVE"

    else:
        confirmation = "NEUTRAL"

    return {
        "session":
            latest_session_id,

        "bars_observed":
            len(session),

        "close_vs_vwap_atr":
            float(
                close_vs_vwap_atr
            ),

        "bars_above_vwap_ratio":
            bars_above_vwap_ratio,

        "session_relative_volume":
            (
                None
                if pd.isna(
                    session_relative_volume
                )
                else float(
                    session_relative_volume
                )
            ),

        "latest_slot_relative_volume":
            (
                None
                if pd.isna(
                    latest_slot_rvol
                )
                else float(
                    latest_slot_rvol
                )
            ),

        "session_close_location":
            (
                None
                if pd.isna(
                    session_close_location
                )
                else float(
                    session_close_location
                )
            ),

        "session_return_from_open":
            float(
                latest[
                    "session_return_from_open"
                ]
            ),

        "structure_balance":
            int(
                structure_balance
            ),

        "down_up_volume_ratio":
            (
                None
                if pd.isna(
                    down_up_volume_ratio
                )
                else float(
                    down_up_volume_ratio
                )
            ),

        "late_return_atr":
            float(
                late_return_atr
            ),

        "late_close_location":
            float(
                late_close_location
            ),

        "positive_evidence_count":
            positive,

        "negative_evidence_count":
            negative,

        "confirmation":
            confirmation,
    }