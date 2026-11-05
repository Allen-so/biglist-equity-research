import numpy as np
import pandas as pd

from src.features.market_group import (
    add_group_fit,
    build_industry_momentum,
    build_market_fit_snapshot,
    build_market_group_snapshot,
    build_within_group_leadership,
    stress_relative_strength,
)


def make_stock(
    returns: list[float],
    *,
    start: float = 100.0,
) -> pd.DataFrame:
    close = [
        start
    ]

    for value in returns:
        close.append(
            close[-1]
            * (
                1.0 + value
            )
        )

    close = np.asarray(
        close,
        dtype=float,
    )

    dates = pd.bdate_range(
        "2026-01-01",
        periods=len(close),
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
                    len(close),
                    1_000_000.0,
                ),
        },
        index=dates,
    )


def test_stress_rs_rewards_resilience():
    benchmark_returns = (
        [-0.01, 0.005]
        * 40
    )

    strong_returns = (
        [-0.002, 0.006]
        * 40
    )

    weak_returns = (
        [-0.02, 0.004]
        * 40
    )

    benchmark = make_stock(
        benchmark_returns
    )

    strong = make_stock(
        strong_returns
    )

    weak = make_stock(
        weak_returns
    )

    as_of = benchmark.index[-1]

    strong_result = (
        stress_relative_strength(
            strong,
            benchmark,
            as_of=as_of,
            lookback_sessions=60,
            min_stress_sessions=5,
        )
    )

    weak_result = (
        stress_relative_strength(
            weak,
            benchmark,
            as_of=as_of,
            lookback_sessions=60,
            min_stress_sessions=5,
        )
    )

    assert (
        strong_result[
            "stress_excess_return"
        ]
        >
        weak_result[
            "stress_excess_return"
        ]
    )


def test_future_crash_is_not_visible():
    benchmark = make_stock(
        [-0.005, 0.005]
        * 40
    )

    stock = make_stock(
        [-0.002, 0.006]
        * 40
    )

    as_of = stock.index[-2]

    #
    # Huge future crash AFTER as_of.
    #
    stock.iloc[
        -1,
        stock.columns.get_loc(
            "open"
        ),
    ] = 40.0

    stock.iloc[
        -1,
        stock.columns.get_loc(
            "high"
        ),
    ] = 41.0

    stock.iloc[
        -1,
        stock.columns.get_loc(
            "low"
        ),
    ] = 39.0

    stock.iloc[
        -1,
        stock.columns.get_loc(
            "close"
        ),
    ] = 40.0

    full = (
        stress_relative_strength(
            stock,
            benchmark,
            as_of=as_of,
        )
    )

    truncated = (
        stress_relative_strength(
            stock.iloc[:-1],
            benchmark.iloc[:-1],
            as_of=as_of,
        )
    )

    assert np.isclose(
        full[
            "stress_excess_return"
        ],
        truncated[
            "stress_excess_return"
        ],
        equal_nan=True,
    )


def test_market_fit_ranks_resilient_stock_higher():
    benchmark = make_stock(
        [-0.01, 0.005]
        * 40
    )

    strong = make_stock(
        [-0.002, 0.01]
        * 40
    )

    weak = make_stock(
        [-0.02, 0.002]
        * 40
    )

    result = (
        build_market_fit_snapshot(
            {
                "STRONG": strong,
                "WEAK": weak,
            },
            benchmark_frame=benchmark,
            as_of=benchmark.index[-1],
        )
        .set_index(
            "symbol"
        )
    )

    assert (
        result.loc[
            "STRONG",
            "market_fit_pct",
        ]
        >
        result.loc[
            "WEAK",
            "market_fit_pct",
        ]
    )


def make_rs_snapshot():
    return pd.DataFrame(
        {
            "symbol": [
                "A1",
                "A2",
                "B1",
                "B2",
            ],

            "return_5d": [
                0.10,
                0.08,
                0.01,
                0.00,
            ],

            "return_20d": [
                0.25,
                0.20,
                0.03,
                0.01,
            ],

            "return_60d": [
                0.50,
                0.40,
                0.05,
                0.02,
            ],

            "rs_20d_pct": [
                100.0,
                80.0,
                60.0,
                40.0,
            ],

            "rs_60d_pct": [
                100.0,
                85.0,
                55.0,
                35.0,
            ],
        }
    )


def make_metadata():
    return pd.DataFrame(
        {
            "symbol": [
                "A1",
                "A2",
                "B1",
                "B2",
            ],

            "industry": [
                "Software",
                "Software",
                "Retail",
                "Retail",
            ],
        }
    )


def test_stronger_industry_ranks_higher():
    rs = make_rs_snapshot()

    groups = (
        build_industry_momentum(
            rs,
            make_metadata(),
            minimum_group_size=2,
        )
        .set_index(
            "industry"
        )
    )

    assert (
        groups.loc[
            "Software",
            "industry_momentum_pct",
        ]
        >
        groups.loc[
            "Retail",
            "industry_momentum_pct",
        ]
    )


def test_group_fit_attaches_to_stocks():
    result = add_group_fit(
        make_rs_snapshot(),
        make_metadata(),
    )

    software = result[
        result["industry"]
        == "Software"
    ]

    assert (
        software[
            "group_fit_pct"
        ].nunique()
        == 1
    )

    assert (
        software[
            "group_fit_pct"
        ].notna()
        .all()
    )


def test_within_group_leader_ranks_higher():
    rs = make_rs_snapshot()

    velocity = pd.DataFrame(
        {
            "symbol": [
                "A1",
                "A2",
                "B1",
                "B2",
            ],

            "historical_velocity_pct": [
                95.0,
                70.0,
                60.0,
                30.0,
            ],
        }
    )

    result = (
        build_within_group_leadership(
            rs,
            velocity,
            make_metadata(),
        )
        .set_index(
            "symbol"
        )
    )

    assert (
        result.loc[
            "A1",
            "within_group_leadership_pct",
        ]
        >
        result.loc[
            "A2",
            "within_group_leadership_pct",
        ]
    )

    assert (
        result.loc[
            "B1",
            "within_group_leadership_pct",
        ]
        >
        result.loc[
            "B2",
            "within_group_leadership_pct",
        ]
    )


def test_single_stock_group_is_not_ranked():
    rs = pd.DataFrame(
        {
            "symbol": [
                "AAA",
            ],

            "return_5d": [
                0.10,
            ],

            "return_20d": [
                0.20,
            ],

            "return_60d": [
                0.40,
            ],

            "rs_20d_pct": [
                100.0,
            ],

            "rs_60d_pct": [
                100.0,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "symbol": [
                "AAA",
            ],

            "industry": [
                "Tiny Industry",
            ],
        }
    )

    velocity = pd.DataFrame(
        {
            "symbol": [
                "AAA",
            ],

            "historical_velocity_pct": [
                100.0,
            ],
        }
    )

    leadership = (
        build_within_group_leadership(
            rs,
            velocity,
            metadata,
            minimum_group_size=2,
        )
    )

    assert pd.isna(
        leadership[
            "within_group_leadership_pct"
        ].iloc[0]
    )


def test_unified_snapshot_has_required_columns():
    benchmark = make_stock(
        [-0.01, 0.005]
        * 40
    )

    universe = {
        "A1":
            make_stock(
                [-0.002, 0.012]
                * 40
            ),

        "A2":
            make_stock(
                [-0.004, 0.009]
                * 40
            ),

        "B1":
            make_stock(
                [-0.01, 0.006]
                * 40
            ),

        "B2":
            make_stock(
                [-0.015, 0.004]
                * 40
            ),
    }

    rs = make_rs_snapshot()

    velocity = pd.DataFrame(
        {
            "symbol": [
                "A1",
                "A2",
                "B1",
                "B2",
            ],

            "historical_velocity_pct": [
                95.0,
                75.0,
                55.0,
                35.0,
            ],
        }
    )

    result = (
        build_market_group_snapshot(
            universe,
            benchmark_frame=benchmark,
            as_of=benchmark.index[-1],
            rs_snapshot=rs,
            velocity_snapshot=velocity,
            metadata=make_metadata(),
        )
    )

    required = {
        "market_fit_pct",
        "group_fit_pct",
        "within_group_leadership_pct",
        "stress_rs_pct",
        "industry_momentum_pct",
    }

    assert required.issubset(
        result.columns
    )