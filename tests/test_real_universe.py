import numpy as np
import pandas as pd

from src.data.yahoo_provider import (
    _normalize_yahoo_frame,
)
from src.universe.market_gates import (
    evaluate_market_gate,
)
from src.universe.us_equities import (
    filter_biglist_universe,
    parse_nasdaq_listed,
    parse_other_listed,
    to_yahoo_symbol,
)


def test_yahoo_symbol_conversion():
    assert (
        to_yahoo_symbol(
            "BRK.A"
        )
        == "BRK-A"
    )

    assert (
        to_yahoo_symbol(
            "AAPL"
        )
        == "AAPL"
    )


def test_parse_nasdaq_directory():
    text = (
        "Symbol|Security Name|Market Category|"
        "Test Issue|Financial Status|Round Lot Size|ETF\n"
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N\n"
        "QQQ|ETF Product|Q|N|N|100|Y\n"
        "File Creation Time: 1005202622:00||||||\n"
    )

    result = parse_nasdaq_listed(
        text
    )

    assert (
        result[
            "symbol"
        ].tolist()
        == [
            "AAPL",
            "QQQ",
        ]
    )

    assert (
        result[
            "exchange"
        ].iloc[0]
        == "NASDAQ"
    )


def test_parse_other_directory():
    text = (
        "ACT Symbol|Security Name|Exchange|"
        "CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol\n"
        "IBM|IBM Common Stock|N|IBM|N|100|N|IBM\n"
        "XYZ|XYZ Common Stock|A|XYZ|N|100|N|XYZ\n"
        "SPY|SPDR ETF|P|SPY|Y|100|N|SPY\n"
        "File Creation Time: 1005202622:00|||||||\n"
    )

    result = parse_other_listed(
        text
    )

    assert (
        result.loc[
            result[
                "symbol"
            ]
            == "IBM",
            "exchange",
        ].iloc[0]
        == "NYSE"
    )

    assert (
        result.loc[
            result[
                "symbol"
            ]
            == "XYZ",
            "exchange",
        ].iloc[0]
        == "NYSE_AMERICAN"
    )


def test_static_universe_excludes_bad_security_types():
    frame = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "BAC$L",
                "SCE$L",
                "BRK.B",
            ],

            "security_name": [
                "AAA Common Stock",
                "BBB Warrants",
                "CCC Acquisition Corp.",
                "DDD Common Stock",
                "Bank of America Non Cumulative Perpetual Conv Pfd Ser L",
                "SCE TRUST VI",
                "Berkshire Hathaway Inc. Class B Common Stock",
            ],

            "exchange": [
                "NASDAQ",
                "NASDAQ",
                "NASDAQ",
                "NYSE",
                "NYSE",
                "NYSE",
                "NYSE",
            ],

            "test_issue": [
                "N",
                "N",
                "N",
                "Y",
                "N",
                "N",
                "N",
            ],

            "financial_status": [
                "N",
                "N",
                "N",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],

            "is_etf": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )

    result = (
        filter_biglist_universe(
            frame
        )
        .set_index(
            "symbol"
        )
    )

    assert bool(
        result.loc[
            "AAA",
            "static_universe_pass",
        ]
    )

    assert not bool(
        result.loc[
            "BBB",
            "static_universe_pass",
        ]
    )

    assert not bool(
        result.loc[
            "CCC",
            "static_universe_pass",
        ]
    )

    assert not bool(
        result.loc[
            "DDD",
            "static_universe_pass",
        ]
    )


    assert not bool(
        result.loc[
            "BAC$L",
            "static_universe_pass",
        ]
    )

    assert (
        result.loc[
            "BAC$L",
            "static_exclusion_reason",
        ]
        == "SYMBOL:DOLLAR_SECURITY"
    )

    assert not bool(
        result.loc[
            "SCE$L",
            "static_universe_pass",
        ]
    )

    assert bool(
        result.loc[
            "BRK.B",
            "static_universe_pass",
        ]
    )

    assert (
        result.loc[
            "BRK.B",
            "yahoo_symbol",
        ]
        == "BRK-B"
    )


def make_market_frame(
    *,
    periods: int = 100,
    price: float = 20.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=periods,
    )

    close = np.full(
        periods,
        price,
    )

    return pd.DataFrame(
        {
            "open":
                close,

            "high":
                close + 0.5,

            "low":
                close - 0.5,

            "close":
                close,

            "volume":
                np.full(
                    periods,
                    volume,
                ),
        },
        index=dates,
    )


def test_market_gate_passes_liquid_stock():
    frame = make_market_frame(
        price=20.0,
        volume=1_000_000.0,
    )

    result = evaluate_market_gate(
        frame
    )

    assert result[
        "market_gate_pass"
    ]

    assert (
        result[
            "median_dollar_volume_20d"
        ]
        >= 8_000_000
    )


def test_market_gate_rejects_low_price():
    frame = make_market_frame(
        price=3.0,
        volume=10_000_000.0,
    )

    result = evaluate_market_gate(
        frame
    )

    assert not result[
        "market_gate_pass"
    ]

    assert (
        "PRICE_LT_4"
        in result[
            "market_gate_reason"
        ]
    )


def test_market_gate_rejects_low_liquidity():
    frame = make_market_frame(
        price=10.0,
        volume=100_000.0,
    )

    result = evaluate_market_gate(
        frame
    )

    assert not result[
        "market_gate_pass"
    ]

    assert (
        "MEDIAN_DOLLAR_VOLUME_LT_8M"
        in result[
            "market_gate_reason"
        ]
    )


def test_market_gate_rejects_short_history():
    frame = make_market_frame(
        periods=40,
        price=20.0,
        volume=1_000_000.0,
    )

    result = evaluate_market_gate(
        frame
    )

    assert not result[
        "market_gate_pass"
    ]

    assert (
        "HISTORY_LT_60"
        in result[
            "market_gate_reason"
        ]
    )


def test_normalize_yahoo_frame():
    dates = pd.bdate_range(
        "2026-01-01",
        periods=5,
    )

    raw = pd.DataFrame(
        {
            "Open": [
                10,
                11,
                12,
                13,
                14,
            ],

            "High": [
                11,
                12,
                13,
                14,
                15,
            ],

            "Low": [
                9,
                10,
                11,
                12,
                13,
            ],

            "Close": [
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
            ],

            "Volume": [
                100,
                200,
                300,
                400,
                500,
            ],
        },
        index=dates,
    )

    result = (
        _normalize_yahoo_frame(
            raw
        )
    )

    assert list(
        result.columns
    ) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert len(
        result
    ) == 5

def test_daily_cache_downloaded_before_close_is_refreshed_after_close(tmp_path, monkeypatch):
    import os
    from src.data import yahoo_provider
    folder = tmp_path / '1d'
    folder.mkdir()
    path = folder / 'AAA.parquet'
    path.touch()
    downloaded = pd.Timestamp('2026-08-10 15:30', tz='America/New_York').timestamp()
    os.utime(path, (downloaded, downloaded))
    now = pd.Timestamp('2026-08-10 16:15', tz='America/New_York').timestamp()
    monkeypatch.setattr(yahoo_provider, 'time', lambda: now)
    assert not yahoo_provider.YahooMarketDataProvider._cache_fresh(path, ttl_minutes=720)
    os.utime(path, (now, now))
    assert yahoo_provider.YahooMarketDataProvider._cache_fresh(path, ttl_minutes=720)


def test_static_universe_excludes_investment_funds_but_not_funding_name():
    frame = pd.DataFrame(
        {
            "symbol": [
                "ARCC",
                "BCAT",
                "SAFE",
            ],
            "security_name": [
                "Ares Capital Corporation - Closed End Fund",
                "BlackRock Capital Allocation Term Trust Common Shares of Beneficial Interest",
                "Safe Funding Corporation Common Stock",
            ],
            "exchange": [
                "NASDAQ",
                "NYSE",
                "NASDAQ",
            ],
            "test_issue": [
                "N",
                "N",
                "N",
            ],
            "financial_status": [
                "N",
                pd.NA,
                "N",
            ],
            "is_etf": [
                False,
                False,
                False,
            ],
        }
    )

    result = (
        filter_biglist_universe(frame)
        .set_index("symbol")
    )

    assert not bool(
        result.loc[
            "ARCC",
            "static_universe_pass",
        ]
    )

    assert (
        result.loc[
            "ARCC",
            "static_exclusion_reason",
        ]
        == "NAME:FUND"
    )

    assert not bool(
        result.loc[
            "BCAT",
            "static_universe_pass",
        ]
    )

    assert (
        result.loc[
            "BCAT",
            "static_exclusion_reason",
        ]
        == "NAME:TERM TRUST"
    )

    assert bool(
        result.loc[
            "SAFE",
            "static_universe_pass",
        ]
    )


def test_static_universe_excludes_capital_corp_spac_pattern():
    frame = pd.DataFrame(
        {
            "symbol": [
                "CCXI",
                "NORMAL",
            ],
            "security_name": [
                "Churchill Capital Corp XI - Class A Ordinary Shares",
                "Normal Capital Corporation Common Stock",
            ],
            "exchange": [
                "NASDAQ",
                "NYSE",
            ],
            "test_issue": [
                "N",
                "N",
            ],
            "financial_status": [
                "N",
                "N",
            ],
            "is_etf": [
                False,
                False,
            ],
        }
    )

    result = (
        filter_biglist_universe(frame)
        .set_index("symbol")
    )

    assert not bool(
        result.loc[
            "CCXI",
            "static_universe_pass",
        ]
    )

    assert (
        result.loc[
            "CCXI",
            "static_exclusion_reason",
        ]
        == "SPAC_SHELL"
    )

    assert bool(
        result.loc[
            "NORMAL",
            "static_universe_pass",
        ]
    )
