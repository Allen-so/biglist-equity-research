import numpy as np
import pandas as pd

from src.features.relative_strength import (
    add_rank_delta,
    build_rs_snapshot,
    rank_level_score,
    relative_total_return,
)


def make_stock(
    *,
    periods: int = 100,
    start: float = 100.0,
    end: float = 120.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.linspace(
        start,
        end,
        periods,
    )

    return pd.DataFrame(
        {
            "open":
                close * 0.998,

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


def test_cross_sectional_rank_order():
    strong = make_stock(
        end=160.0,
    )

    medium = make_stock(
        end=125.0,
    )

    weak = make_stock(
        end=95.0,
    )

    universe = {
        "STRONG": strong,
        "MEDIUM": medium,
        "WEAK": weak,
    }

    as_of = strong.index[-1]

    result = build_rs_snapshot(
        universe,
        as_of=as_of,
    )

    indexed = (
        result.set_index(
            "symbol"
        )
    )

    assert (
        indexed.loc[
            "STRONG",
            "rs_20d_pct",
        ]
        >
        indexed.loc[
            "MEDIUM",
            "rs_20d_pct",
        ]
    )

    assert (
        indexed.loc[
            "MEDIUM",
            "rs_20d_pct",
        ]
        >
        indexed.loc[
            "WEAK",
            "rs_20d_pct",
        ]
    )

    assert (
        indexed.loc[
            "STRONG",
            "rs_20d_pct",
        ]
        == 100.0
    )


def test_persistent_rs_uses_all_windows():
    a = make_stock(
        end=160.0,
    )

    b = make_stock(
        end=120.0,
    )

    c = make_stock(
        end=90.0,
    )

    result = build_rs_snapshot(
        {
            "A": a,
            "B": b,
            "C": c,
        },
        as_of=a.index[-1],
    )

    row = (
        result
        .set_index(
            "symbol"
        )
        .loc["A"]
    )

    expected = (
        0.15
        * row["rs_5d_pct"]
        +
        0.20
        * row["rs_10d_pct"]
        +
        0.30
        * row["rs_20d_pct"]
        +
        0.35
        * row["rs_60d_pct"]
    )

    assert np.isclose(
        row["persistent_rs"],
        expected,
    )


def test_future_spike_is_not_visible():
    stock_a = make_stock(
        periods=100,
        end=120.0,
    )

    stock_b = make_stock(
        periods=100,
        end=125.0,
    )

    as_of = stock_a.index[-2]

    #
    # Huge spike AFTER as_of.
    #
    stock_a.iloc[
        -1,
        stock_a.columns.get_loc(
            "open"
        ),
    ] = 198.0

    stock_a.iloc[
        -1,
        stock_a.columns.get_loc(
            "high"
        ),
    ] = 202.0

    stock_a.iloc[
        -1,
        stock_a.columns.get_loc(
            "low"
        ),
    ] = 197.0

    stock_a.iloc[
        -1,
        stock_a.columns.get_loc(
            "close"
        ),
    ] = 200.0

    result = build_rs_snapshot(
        {
            "A": stock_a,
            "B": stock_b,
        },
        as_of=as_of,
    )

    indexed = (
        result.set_index(
            "symbol"
        )
    )

    assert (
        indexed.loc[
            "B",
            "rs_20d_pct",
        ]
        >
        indexed.loc[
            "A",
            "rs_20d_pct",
        ]
    )


def test_benchmark_relative_return():
    stock_return = 0.20
    benchmark_return = 0.10

    result = relative_total_return(
        stock_return,
        benchmark_return,
    )

    expected = (
        1.20 / 1.10 - 1.0
    )

    assert np.isclose(
        result,
        expected,
    )


def test_benchmark_columns_are_added():
    stock = make_stock(
        end=140.0,
    )

    benchmark = make_stock(
        end=110.0,
    )

    result = build_rs_snapshot(
        {
            "AAA": stock,
        },
        as_of=stock.index[-1],
        benchmarks={
            "SPY": benchmark,
        },
    )

    assert (
        "relative_vs_spy_20d"
        in result.columns
    )

    assert (
        result[
            "relative_vs_spy_20d"
        ].iloc[0]
        > 0
    )


def test_rank_delta_detects_improvement():
    current = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],
            "rs_20d_pct": [
                95.0,
                60.0,
            ],
            "rs_60d_pct": [
                90.0,
                70.0,
            ],
        }
    )

    past = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],
            "rs_20d_pct": [
                70.0,
                65.0,
            ],
            "rs_60d_pct": [
                80.0,
                75.0,
            ],
        }
    )

    result = add_rank_delta(
        current,
        past,
    )

    indexed = (
        result.set_index(
            "symbol"
        )
    )

    assert (
        indexed.loc[
            "AAA",
            "rs_20d_rank_delta",
        ]
        == 25.0
    )

    assert (
        indexed.loc[
            "AAA",
            "rs_60d_rank_delta",
        ]
        == 10.0
    )

    assert (
        indexed.loc[
            "BBB",
            "rs_20d_rank_delta",
        ]
        == -5.0
    )


def test_rank_level_weighting():
    snapshot = pd.DataFrame(
        {
            "rs_20d_pct": [
                80.0,
            ],
            "rs_60d_pct": [
                90.0,
            ],
        }
    )

    result = rank_level_score(
        snapshot
    )

    expected = (
        0.40 * 80.0
        + 0.60 * 90.0
    )

    assert np.isclose(
        result.iloc[0],
        expected,
    )