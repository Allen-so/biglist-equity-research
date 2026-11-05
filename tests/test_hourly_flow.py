import numpy as np
import pandas as pd

from src.features.hourly_flow import (
    add_session_metadata,
    add_session_vwap_proxy,
    add_slot_relative_volume,
    build_hourly_flow_features,
    summarize_hourly_flow,
)


def make_hourly_data(
    sessions: int = 12,
    bars_per_session: int = 4,
) -> pd.DataFrame:
    records = []
    timestamps = []

    dates = pd.bdate_range(
        "2026-01-01",
        periods=sessions,
    )

    for session_number, date in enumerate(
        dates
    ):
        for bar in range(
            bars_per_session
        ):
            timestamp = (
                date
                + pd.Timedelta(
                    hours=10 + bar
                )
            )

            base = (
                100.0
                + session_number * 0.10
            )

            open_price = (
                base + bar * 0.10
            )

            close_price = (
                open_price + 0.05
            )

            timestamps.append(
                timestamp
            )

            records.append(
                {
                    "open":
                        open_price,
                    "high":
                        close_price + 0.10,
                    "low":
                        open_price - 0.10,
                    "close":
                        close_price,
                    "volume":
                        1_000_000.0,
                }
            )

    return pd.DataFrame(
        records,
        index=pd.DatetimeIndex(
            timestamps
        ),
    )


def test_session_bar_resets():
    frame = make_hourly_data(
        sessions=2,
        bars_per_session=4,
    )

    result = add_session_metadata(
        frame
    )

    first_session = (
        result.iloc[:4]
    )

    second_session = (
        result.iloc[4:]
    )

    assert (
        first_session[
            "session_bar"
        ].tolist()
        == [0, 1, 2, 3]
    )

    assert (
        second_session[
            "session_bar"
        ].tolist()
        == [0, 1, 2, 3]
    )


def test_session_vwap_resets_each_day():
    dates = pd.bdate_range(
        "2026-01-01",
        periods=2,
    )

    timestamps = [
        dates[0]
        + pd.Timedelta(hours=10),
        dates[0]
        + pd.Timedelta(hours=11),

        dates[1]
        + pd.Timedelta(hours=10),
        dates[1]
        + pd.Timedelta(hours=11),
    ]

    frame = pd.DataFrame(
        {
            "open": [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            "high": [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            "low": [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            "close": [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            "volume": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        },
        index=pd.DatetimeIndex(
            timestamps
        ),
    )

    result = (
        add_session_vwap_proxy(
            add_session_metadata(
                frame
            )
        )
    )

    assert (
        result[
            "session_vwap_proxy"
        ].iloc[0]
        == 100.0
    )

    assert (
        result[
            "session_vwap_proxy"
        ].iloc[2]
        == 200.0
    )


def test_slot_relative_volume_matches_same_hour():
    frame = make_hourly_data(
        sessions=6,
        bars_per_session=4,
    )

    #
    # Double volume on the final session's
    # first bar only.
    #
    final_session = (
        frame.index.normalize()
        == frame.index[
            -1
        ].normalize()
    )

    first_bar_index = (
        frame.loc[
            final_session
        ].index[0]
    )

    frame.loc[
        first_bar_index,
        "volume",
    ] = 2_000_000.0

    metadata = (
        add_session_metadata(
            frame
        )
    )

    result = (
        add_slot_relative_volume(
            metadata,
            lookback_sessions=4,
            min_history_sessions=3,
        )
    )

    assert (
        result.loc[
            first_bar_index,
            "slot_relative_volume",
        ]
        == 2.0
    )


def test_slot_baseline_excludes_current_bar():
    frame = make_hourly_data(
        sessions=6,
        bars_per_session=4,
    )

    final_session = (
        frame.index.normalize()
        == frame.index[
            -1
        ].normalize()
    )

    first_bar_index = (
        frame.loc[
            final_session
        ].index[0]
    )

    frame.loc[
        first_bar_index,
        "volume",
    ] = 10_000_000.0

    result = (
        add_slot_relative_volume(
            add_session_metadata(
                frame
            ),
            lookback_sessions=4,
            min_history_sessions=3,
        )
    )

    assert (
        result.loc[
            first_bar_index,
            "slot_median_volume_prior",
        ]
        == 1_000_000.0
    )

    assert (
        result.loc[
            first_bar_index,
            "slot_relative_volume",
        ]
        == 10.0
    )


def test_hourly_flow_features_exist():
    frame = make_hourly_data()

    result = (
        build_hourly_flow_features(
            frame,
            lookback_sessions=5,
            min_history_sessions=3,
        )
    )

    expected = {
        "session_vwap_proxy",
        "slot_relative_volume",
        "session_relative_volume",
        "session_close_location",
        "hourly_higher_high",
        "hourly_higher_low",
        "session_down_up_volume_ratio",
    }

    assert expected.issubset(
        result.columns
    )


def test_supportive_hourly_flow():
    frame = make_hourly_data(
        sessions=12,
        bars_per_session=4,
    )

    latest_session = (
        frame.index.normalize()
        == frame.index[
            -1
        ].normalize()
    )

    indices = frame.loc[
        latest_session
    ].index

    bars = [
        {
            "open": 100.0,
            "high": 101.2,
            "low": 99.8,
            "close": 101.0,
            "volume": 1_500_000.0,
        },
        {
            "open": 101.0,
            "high": 102.2,
            "low": 100.8,
            "close": 102.0,
            "volume": 1_600_000.0,
        },
        {
            "open": 102.0,
            "high": 103.2,
            "low": 101.8,
            "close": 103.0,
            "volume": 1_700_000.0,
        },
        {
            "open": 103.0,
            "high": 104.3,
            "low": 102.8,
            "close": 104.1,
            "volume": 1_800_000.0,
        },
    ]

    for index, bar in zip(
        indices,
        bars,
    ):
        for column, value in (
            bar.items()
        ):
            frame.loc[
                index,
                column,
            ] = value

    result = summarize_hourly_flow(
        frame,
        as_of=indices[-1],
        daily_atr=4.0,
        lookback_sessions=5,
        min_history_sessions=3,
    )

    assert (
        result[
            "confirmation"
        ]
        == "SUPPORTIVE"
    )

    assert (
        result[
            "positive_evidence_count"
        ]
        >= 4
    )


def test_negative_hourly_flow():
    frame = make_hourly_data(
        sessions=12,
        bars_per_session=4,
    )

    latest_session = (
        frame.index.normalize()
        == frame.index[
            -1
        ].normalize()
    )

    indices = frame.loc[
        latest_session
    ].index

    bars = [
        {
            "open": 104.0,
            "high": 104.2,
            "low": 102.8,
            "close": 103.0,
            "volume": 1_500_000.0,
        },
        {
            "open": 103.0,
            "high": 103.2,
            "low": 101.8,
            "close": 102.0,
            "volume": 1_600_000.0,
        },
        {
            "open": 102.0,
            "high": 102.2,
            "low": 100.8,
            "close": 101.0,
            "volume": 1_700_000.0,
        },
        {
            "open": 101.0,
            "high": 101.2,
            "low": 99.7,
            "close": 99.9,
            "volume": 1_800_000.0,
        },
    ]

    for index, bar in zip(
        indices,
        bars,
    ):
        for column, value in (
            bar.items()
        ):
            frame.loc[
                index,
                column,
            ] = value

    result = summarize_hourly_flow(
        frame,
        as_of=indices[-1],
        daily_atr=4.0,
        lookback_sessions=5,
        min_history_sessions=3,
    )

    assert (
        result[
            "confirmation"
        ]
        == "NEGATIVE"
    )

    assert (
        result[
            "negative_evidence_count"
        ]
        >= 4
    )


def test_as_of_blocks_future_hourly_bars():
    frame = make_hourly_data(
        sessions=12,
        bars_per_session=4,
    )

    latest_session = (
        frame.index.normalize()
        == frame.index[
            -1
        ].normalize()
    )

    indices = frame.loc[
        latest_session
    ].index

    as_of = indices[1]

    result_full = (
        summarize_hourly_flow(
            frame,
            as_of=as_of,
            daily_atr=4.0,
            lookback_sessions=5,
            min_history_sessions=3,
        )
    )

    truncated = frame[
        frame.index <= as_of
    ]

    result_truncated = (
        summarize_hourly_flow(
            truncated,
            as_of=as_of,
            daily_atr=4.0,
            lookback_sessions=5,
            min_history_sessions=3,
        )
    )

    assert (
        result_full[
            "bars_observed"
        ]
        ==
        result_truncated[
            "bars_observed"
        ]
    )

    assert (
        result_full[
            "close_vs_vwap_atr"
        ]
        ==
        result_truncated[
            "close_vs_vwap_atr"
        ]
    )

    assert (
        result_full[
            "confirmation"
        ]
        ==
        result_truncated[
            "confirmation"
        ]
    )