import numpy as np
import pandas as pd

from src.regime.market_regime import (
    calculate_bollinger_context,
    calculate_index_trend,
    calculate_leadership_breadth,
    calculate_market_breadth,
    calculate_market_regime,
    calculate_risk_participation,
    classify_regime_score,
)


def make_trend(
    *,
    periods: int = 280,
    daily_return: float = 0.005,
    start: float = 100.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2025-01-01",
        periods=periods,
    )

    close = (
        start
        * (
            1.0 + daily_return
        )
        ** np.arange(
            periods
        )
    )

    return pd.DataFrame(
        {
            "open":
                close.copy(),

            "high":
                close * 1.01,

            "low":
                close * 0.99,

            "close":
                close,

            "volume":
                np.full(
                    periods,
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def make_universe(
    returns: list[float],
) -> dict[str, pd.DataFrame]:
    return {
        f"S{i}":
            make_trend(
                daily_return=value
            )
        for i, value in enumerate(
            returns
        )
    }


def test_strong_index_trend_gets_full_points():
    spy = make_trend(
        daily_return=0.005
    )

    qqq = make_trend(
        daily_return=0.006
    )

    result = calculate_index_trend(
        spy,
        qqq,
        as_of=spy.index[-1],
    )

    assert np.isclose(
        result[
            "index_trend_points"
        ],
        30.0,
    )


def test_weak_index_trend_gets_zero_points():
    spy = make_trend(
        daily_return=-0.004
    )

    qqq = make_trend(
        daily_return=-0.005
    )

    result = calculate_index_trend(
        spy,
        qqq,
        as_of=spy.index[-1],
    )

    assert np.isclose(
        result[
            "index_trend_points"
        ],
        0.0,
    )


def test_strong_breadth_scores_higher():
    strong = make_universe(
        [
            0.005,
            0.004,
            0.006,
            0.003,
            0.005,
        ]
    )

    weak = make_universe(
        [
            -0.005,
            -0.004,
            -0.006,
            -0.003,
            -0.005,
        ]
    )

    as_of = next(
        iter(
            strong.values()
        )
    ).index[-1]

    strong_result = (
        calculate_market_breadth(
            strong,
            as_of=as_of,
        )
    )

    weak_result = (
        calculate_market_breadth(
            weak,
            as_of=as_of,
        )
    )

    assert (
        strong_result[
            "breadth_points"
        ]
        >
        weak_result[
            "breadth_points"
        ]
    )


def test_leadership_breadth():
    universe = make_universe(
        [
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
        ]
    )

    rs = pd.DataFrame(
        {
            "symbol": [
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
            ],

            "rs_20d_pct": [
                100.0,
                90.0,
                85.0,
                50.0,
                20.0,
            ],
        }
    )

    as_of = universe[
        "S0"
    ].index[-1]

    result = (
        calculate_leadership_breadth(
            universe,
            rs,
            as_of=as_of,
        )
    )

    assert (
        result[
            "leadership_count"
        ]
        == 3
    )

    assert (
        result[
            "leadership_breadth_points"
        ]
        > 0
    )


def test_iwm_outperformance_improves_risk_participation():
    universe = make_universe(
        [
            0.003,
            0.004,
            0.005,
            0.002,
            0.003,
        ]
    )

    spy = make_trend(
        daily_return=0.002
    )

    strong_iwm = make_trend(
        daily_return=0.005
    )

    weak_iwm = make_trend(
        daily_return=-0.001
    )

    as_of = spy.index[-1]

    strong = (
        calculate_risk_participation(
            universe,
            spy_frame=spy,
            iwm_frame=strong_iwm,
            as_of=as_of,
        )
    )

    weak = (
        calculate_risk_participation(
            universe,
            spy_frame=spy,
            iwm_frame=weak_iwm,
            as_of=as_of,
        )
    )

    assert (
        strong[
            "risk_participation_points"
        ]
        >
        weak[
            "risk_participation_points"
        ]
    )


def test_regime_thresholds():
    assert (
        classify_regime_score(
            80
        )
        == "STRONG_RISK_ON"
    )

    assert (
        classify_regime_score(
            65
        )
        == "CONSTRUCTIVE"
    )

    assert (
        classify_regime_score(
            50
        )
        == "SELECTIVE"
    )

    assert (
        classify_regime_score(
            30
        )
        == "DEFENSIVE"
    )


def test_bollinger_context_exists():
    frame = make_trend(
        daily_return=0.003
    )

    result = (
        calculate_bollinger_context(
            frame,
            as_of=frame.index[-1],
        )
    )

    assert not pd.isna(
        result[
            "bb_percent_b"
        ]
    )

    assert not pd.isna(
        result[
            "bb_bandwidth"
        ]
    )

    assert not pd.isna(
        result[
            "bb_bandwidth_pct"
        ]
    )


def test_future_crash_is_not_visible():
    frame = make_trend(
        daily_return=0.004
    )

    as_of = frame.index[-2]

    crashed = frame.copy()

    crashed.iloc[
        -1,
        crashed.columns.get_loc(
            "open"
        ),
    ] = 10.0

    crashed.iloc[
        -1,
        crashed.columns.get_loc(
            "high"
        ),
    ] = 10.5

    crashed.iloc[
        -1,
        crashed.columns.get_loc(
            "low"
        ),
    ] = 9.5

    crashed.iloc[
        -1,
        crashed.columns.get_loc(
            "close"
        ),
    ] = 10.0

    full = (
        calculate_index_trend(
            crashed,
            crashed,
            as_of=as_of,
        )
    )

    truncated = (
        calculate_index_trend(
            frame.iloc[:-1],
            frame.iloc[:-1],
            as_of=as_of,
        )
    )

    assert np.isclose(
        full[
            "index_trend_points"
        ],
        truncated[
            "index_trend_points"
        ],
    )


def test_complete_strong_market_is_risk_on():
    universe = make_universe(
        [
            0.006,
            0.005,
            0.004,
            0.007,
            0.005,
        ]
    )

    spy = make_trend(
        daily_return=0.004
    )

    qqq = make_trend(
        daily_return=0.006
    )

    iwm = make_trend(
        daily_return=0.007
    )

    rs = pd.DataFrame(
        {
            "symbol": [
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
            ],

            "rs_20d_pct": [
                100.0,
                95.0,
                90.0,
                85.0,
                80.0,
            ],
        }
    )

    result = (
        calculate_market_regime(
            universe,
            spy_frame=spy,
            qqq_frame=qqq,
            iwm_frame=iwm,
            rs_snapshot=rs,
            as_of=spy.index[-1],
        )
    )

    assert (
        result[
            "regime_status"
        ]
        == "COMPLETE"
    )

    assert (
        result[
            "regime_score"
        ]
        >= 75.0
    )

    assert (
        result[
            "regime"
        ]
        == "STRONG_RISK_ON"
    )


def test_regime_score_never_exceeds_100():
    universe = make_universe(
        [
            0.010,
            0.011,
            0.012,
            0.013,
            0.014,
        ]
    )

    spy = make_trend(
        daily_return=0.008
    )

    qqq = make_trend(
        daily_return=0.010
    )

    iwm = make_trend(
        daily_return=0.015
    )

    rs = pd.DataFrame(
        {
            "symbol": [
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
            ],

            "rs_20d_pct": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
        }
    )

    result = (
        calculate_market_regime(
            universe,
            spy_frame=spy,
            qqq_frame=qqq,
            iwm_frame=iwm,
            rs_snapshot=rs,
            as_of=spy.index[-1],
        )
    )

    assert (
        result[
            "regime_score"
        ]
        <= 100.0
    )