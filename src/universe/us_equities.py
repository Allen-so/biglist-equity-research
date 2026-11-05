from __future__ import annotations

from io import StringIO
from typing import Any
import re

import pandas as pd
import requests


NASDAQ_LISTED_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/nasdaqlisted.txt"
)

OTHER_LISTED_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/otherlisted.txt"
)


EXCLUDED_NAME_PATTERNS = (
    "WARRANT",
    "WARRANTS",
    "RIGHT",
    "RIGHTS",
    "UNIT",
    "UNITS",
    "PREFERRED",
    "PREFERENCE",
    "DEPOSITARY SHARES",
    "DEPOSITARY SHARE",
    "AMERICAN DEPOSITARY",
    "ADR",
    "EXCHANGE TRADED FUND",
    "EXCHANGE-TRADED FUND",
    " ETF",
    " ETN",
    "DEBT SECURITY",
    "SENIOR NOTE",
    "SENIOR NOTES",
    "SUBORDINATED NOTE",
    "SUBORDINATED NOTES",
    "BOND",
)


SPAC_NAME_PATTERNS = (
    "ACQUISITION CORP",
    "ACQUISITION CORPORATION",
    "ACQUISITION CO.",
    "BLANK CHECK",
)


def _download_text(
    url: str,
    *,
    timeout: float = 20.0,
) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent":
                "biglist-engine/1.0"
        },
    )

    response.raise_for_status()

    return response.text


def _parse_pipe_file(
    text: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        StringIO(text),
        sep="|",
        dtype=str,
    )

    #
    # Nasdaq Trader places:
    #
    # File Creation Time: ...
    #
    # as the final row.
    #
    first_column = (
        frame.columns[0]
    )

    mask = (
        ~frame[
            first_column
        ]
        .astype(str)
        .str.startswith(
            "File Creation Time"
        )
    )

    return (
        frame.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def _normalize_symbol(
    symbol: Any,
) -> str:
    return (
        str(symbol)
        .strip()
        .upper()
    )


def to_yahoo_symbol(
    symbol: str,
) -> str:
    """
    Nasdaq/NYSE class-share notation commonly uses:

        BRK.A

    Yahoo convention uses:

        BRK-A
    """

    return (
        _normalize_symbol(
            symbol
        )
        .replace(
            ".",
            "-",
        )
    )


def _security_name_reason(
    security_name: str,
) -> str | None:
    name = str(
        security_name
    ).upper()

    for pattern in (
        EXCLUDED_NAME_PATTERNS
    ):
        if pattern in name:
            return (
                f"NAME:{pattern}"
            )

    # Closed-end funds, BDCs, interval funds,
    # and similar investment-company vehicles
    # are outside the Big List operating-equity
    # universe. Match FUND as a standalone word
    # so names containing FUNDING are not affected.
    if re.search(
        r"\bFUND\b",
        name,
    ):
        return "NAME:FUND"

    if "TERM TRUST" in name:
        return "NAME:TERM TRUST"

    # Some Nasdaq SPACs omit "Acquisition" or
    # "Blank Check" from the security name and
    # appear as "Capital Corp ... Ordinary Shares".
    if (
        re.search(
            r"\bCAPITAL CORP\.?\b",
            name,
        )
        and re.search(
            r"\bORDINARY\s+SHARES?\b",
            name,
        )
    ):
        return "SPAC_SHELL"

    for pattern in (
        SPAC_NAME_PATTERNS
    ):
        if pattern in name:
            return "SPAC_SHELL"

    return None


def parse_nasdaq_listed(
    text: str,
) -> pd.DataFrame:
    raw = _parse_pipe_file(
        text
    )

    required = {
        "Symbol",
        "Security Name",
        "Test Issue",
    }

    missing = required.difference(
        raw.columns
    )

    if missing:
        raise ValueError(
            "nasdaqlisted.txt missing columns: "
            f"{sorted(missing)}"
        )

    result = pd.DataFrame(
        {
            "symbol":
                raw[
                    "Symbol"
                ].map(
                    _normalize_symbol
                ),

            "security_name":
                raw[
                    "Security Name"
                ].astype(str),

            "exchange":
                "NASDAQ",

            "test_issue":
                raw[
                    "Test Issue"
                ]
                .fillna("N")
                .str.upper(),

            "financial_status":
                (
                    raw[
                        "Financial Status"
                    ]
                    if "Financial Status"
                    in raw.columns
                    else pd.Series(
                        pd.NA,
                        index=raw.index,
                    )
                ),

            "is_etf":
                (
                    raw["ETF"]
                    .fillna("N")
                    .str.upper()
                    .eq("Y")
                    if "ETF"
                    in raw.columns
                    else False
                ),
        }
    )

    result[
        "source"
    ] = "NASDAQ_TRADER"

    return result


def parse_other_listed(
    text: str,
) -> pd.DataFrame:
    raw = _parse_pipe_file(
        text
    )

    required = {
        "ACT Symbol",
        "Security Name",
        "Exchange",
        "Test Issue",
    }

    missing = required.difference(
        raw.columns
    )

    if missing:
        raise ValueError(
            "otherlisted.txt missing columns: "
            f"{sorted(missing)}"
        )

    exchange_map = {
        "N": "NYSE",
        "A": "NYSE_AMERICAN",
        "P": "NYSE_ARCA",
        "Z": "BATS",
        "V": "IEX",
    }

    result = pd.DataFrame(
        {
            "symbol":
                raw[
                    "ACT Symbol"
                ].map(
                    _normalize_symbol
                ),

            "security_name":
                raw[
                    "Security Name"
                ].astype(str),

            "exchange":
                raw[
                    "Exchange"
                ]
                .map(
                    exchange_map
                ),

            "test_issue":
                raw[
                    "Test Issue"
                ]
                .fillna("N")
                .str.upper(),

            "financial_status":
                pd.NA,

            "is_etf":
                (
                    raw[
                        "ETF"
                    ]
                    .fillna("N")
                    .str.upper()
                    .eq("Y")
                    if "ETF"
                    in raw.columns
                    else False
                ),
        }
    )

    result[
        "source"
    ] = "NASDAQ_TRADER"

    return result


def filter_biglist_universe(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Static security-type universe filter.

    Keep only:

        NASDAQ
        NYSE
        NYSE American

    Exclude:

        Test securities
        ETF
        Warrants
        Rights
        Units
        Preferred
        ADR / Depositary shares
        obvious debt securities
        obvious SPAC shells

    Price / liquidity / history gates are applied
    AFTER OHLCV is downloaded.
    """

    result = metadata.copy()

    result[
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result[
        "security_name"
    ] = (
        result[
            "security_name"
        ]
        .fillna("")
        .astype(str)
    )

    allowed_exchange = (
        result[
            "exchange"
        ]
        .isin(
            [
                "NASDAQ",
                "NYSE",
                "NYSE_AMERICAN",
            ]
        )
    )

    normal_test_status = (
        result[
            "test_issue"
        ]
        .fillna("N")
        .str.upper()
        .eq("N")
    )

    not_etf = (
        ~result[
            "is_etf"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "static_exclusion_reason"
    ] = (
        result[
            "security_name"
        ]
        .map(
            _security_name_reason
        )
    )

    # NYSE-style "$" symbols in the source directory
    # represent preferred shares, notes, debentures,
    # trust securities, or similar non-common-equity
    # instruments. Big List is a common-equity universe.
    dollar_symbol = (
        result["symbol"]
        .str.contains(
            "$",
            regex=False,
        )
    )

    result.loc[
        (
            result["static_exclusion_reason"].isna()
            & dollar_symbol
        ),
        "static_exclusion_reason",
    ] = "SYMBOL:DOLLAR_SECURITY"

    no_name_exclusion = (
        result[
            "static_exclusion_reason"
        ]
        .isna()
    )

    result[
        "static_universe_pass"
    ] = (
        allowed_exchange
        &
        normal_test_status
        &
        not_etf
        &
        no_name_exclusion
    )

    result[
        "yahoo_symbol"
    ] = (
        result[
            "symbol"
        ]
        .map(
            to_yahoo_symbol
        )
    )

    return (
        result
        .sort_values(
            [
                "static_universe_pass",
                "exchange",
                "symbol",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def load_us_equity_directory(
    *,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """
    Download today's real US listed-security directory.
    """

    nasdaq_text = _download_text(
        NASDAQ_LISTED_URL,
        timeout=timeout,
    )

    other_text = _download_text(
        OTHER_LISTED_URL,
        timeout=timeout,
    )

    nasdaq = parse_nasdaq_listed(
        nasdaq_text
    )

    other = parse_other_listed(
        other_text
    )

    combined = pd.concat(
        [
            nasdaq,
            other,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "symbol",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return filter_biglist_universe(
        combined
    )