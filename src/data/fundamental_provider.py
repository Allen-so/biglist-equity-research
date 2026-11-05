from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from time import time
from typing import Any, Iterable

import pandas as pd

from src.universe.us_equities import (
    to_yahoo_symbol,
)


def _load_yfinance():
    try:
        import yfinance as yf

    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: "
            "pip install yfinance"
        ) from exc

    return yf


from src.data.checkpoint import remote, atomic_json

def _atomic_financial(frame, path):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class YahooFundamentalProvider:
    """
    V1 company metadata + fundamental provider.

    Sources:
        Yahoo Finance via yfinance

    Cached independently from OHLCV because
    fundamentals change much less frequently.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path = (
            "data/cache/fundamentals"
        ),
        metadata_cache_hours: int = 168,
        financial_cache_hours: int = 24,
    ) -> None:

        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.transport_failures = []
        self.metadata_cache_hours = (
            metadata_cache_hours
        )

        self.financial_cache_hours = (
            financial_cache_hours
        )

    def _symbol_dir(
        self,
        symbol: str,
    ) -> Path:

        safe = (
            to_yahoo_symbol(
                symbol
            )
            .replace(
                "/",
                "_",
            )
        )

        path = (
            self.cache_dir
            / safe
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    @staticmethod
    def _fresh(
        path: Path,
        *,
        ttl_hours: int,
    ) -> bool:

        if not path.exists():
            return False

        age = (
            time()
            - path.stat().st_mtime
        )

        return (
            age
            <= ttl_hours * 3600
        )

    def _last_financial_snapshot(self, path, symbol, reason):
        # Keep the acquisition timestamp unchanged; bounded fallback is auditable.
        if self._fresh(path, ttl_hours=45 * 24):
            try:
                frame = pd.read_parquet(path)
                print(f"Financial cache fallback {symbol} {path.name} captured={path.stat().st_mtime} cause={reason}", flush=True)
                return frame
            except Exception as exc:
                print(f"Unreadable financial cache {symbol} {path.name}: {exc}", flush=True)
        print(f"Financial data missing {symbol} {path.name}: {reason}", flush=True)
        return pd.DataFrame()

    def get_company_metadata(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:

        path = (
            self._symbol_dir(
                symbol
            )
            / "metadata.json"
        )

        if (
            not refresh
            and
            self._fresh(
                path,
                ttl_hours=(
                    self.metadata_cache_hours
                ),
            )
        ):
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                cached = json.load(file)
            # Empty responses from a transient Yahoo failure must not be reused
            # for a week. Incomplete industry metadata is retried on the next run.
            if cached.get("industry"):
                return cached

        yf = _load_yfinance()

        ticker = yf.Ticker(
            to_yahoo_symbol(
                symbol
            )
        )

        try:
            info = remote(self.cache_dir.parent / "yahoo", ticker.get_info)

        except Exception as exc:
            self.transport_failures.append((symbol, type(exc).__name__))
            info = {}

        metadata = {
            "symbol":
                str(symbol),

            "sector":
                info.get(
                    "sector"
                ),

            "industry":
                info.get(
                    "industry"
                ),

            "long_name":
                info.get(
                    "longName"
                ),

            "short_name":
                info.get(
                    "shortName"
                ),

            "quote_type":
                info.get(
                    "quoteType"
                ),

            "country":
                info.get(
                    "country"
                ),

            "currency":
                info.get(
                    "currency"
                ),

            "market_cap":
                info.get(
                    "marketCap"
                ),
        }

        if not info:
            # Keep this data unavailable without poisoning the long-lived cache.
            return metadata

        atomic_json(path, metadata)

        return metadata

    def get_quarterly_income_statement(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:

        path = (
            self._symbol_dir(
                symbol
            )
            / "quarterly_income.parquet"
        )

        if (
            not refresh
            and
            self._fresh(
                path,
                ttl_hours=(
                    self.financial_cache_hours
                ),
            )
        ):
            return pd.read_parquet(
                path
            )

        yf = _load_yfinance()

        ticker = yf.Ticker(
            to_yahoo_symbol(
                symbol
            )
        )

        try:
            frame = (
                remote(self.cache_dir.parent / "yahoo", ticker.get_income_stmt,
                    freq="quarterly"
                )
            )

        except Exception as exc:
            frame = self._last_financial_snapshot(path, symbol, type(exc).__name__)
            if frame.empty:
                self.transport_failures.append((symbol, type(exc).__name__))
            return frame

        if frame is None or frame.empty:
            return self._last_financial_snapshot(path, symbol, "empty Yahoo response")

        if not frame.empty:
            frame.columns = pd.to_datetime(
                frame.columns
            )

            frame = frame.loc[
                :,
                sorted(
                    frame.columns,
                    reverse=True,
                )
            ]

            _atomic_financial(frame, path)

        return frame

    def get_earnings_history(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:

        path = (
            self._symbol_dir(
                symbol
            )
            / "earnings_events_v2.parquet"
        )

        if (
            not refresh
            and
            self._fresh(
                path,
                ttl_hours=(
                    self.financial_cache_hours
                ),
            )
        ):
            frame = pd.read_parquet(
                path
            )

            frame.index = pd.to_datetime(
                frame.index
            )

            return frame

        yf = _load_yfinance()

        ticker = yf.Ticker(
            to_yahoo_symbol(
                symbol
            )
        )

        try:
            frame = (
                remote(self.cache_dir.parent / "yahoo", ticker.get_earnings_dates, limit=12)
            )

        except Exception as exc:
            frame = self._last_financial_snapshot(path, symbol, type(exc).__name__)
            if frame.empty:
                self.transport_failures.append((symbol, type(exc).__name__))
            return frame

        if frame is None or frame.empty:
            return self._last_financial_snapshot(path, symbol, "empty Yahoo response")

        if not frame.empty:
            frame.index = pd.to_datetime(frame.index)
            if frame.index.tz is not None:
                frame.index = frame.index.tz_convert("America/New_York").tz_localize(None)
            frame.index.name = "Earnings Date"

            frame = frame.sort_index(
                ascending=False
            )

            _atomic_financial(frame, path)

        return frame

    def build_metadata_snapshot(
        self,
        symbols: Iterable[str],
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Build metadata suitable for:

            Group Fit
            Within-Group Leadership
        """

        records = []

        for symbol in symbols:
            try:
                metadata = (
                    self.get_company_metadata(
                        symbol,
                        refresh=refresh,
                    )
                )

            except Exception:
                metadata = {
                    "symbol":
                        str(symbol),

                    "sector":
                        None,

                    "industry":
                        None,
                }

            records.append(
                metadata
            )

        return pd.DataFrame(
            records
        )