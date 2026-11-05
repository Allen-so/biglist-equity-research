from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """
    Standard interface between external market-data sources
    and the Big List Engine.

    The screening engine must not depend directly on a
    specific data vendor.
    """

    @abstractmethod
    def get_universe(self) -> pd.DataFrame:
        """
        Expected minimum columns:

        symbol
        security_type
        exchange
        sector
        industry
        """

        raise NotImplementedError

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return split-adjusted daily OHLCV data.
        """

        raise NotImplementedError

    @abstractmethod
    def get_hourly_bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return regular-session hourly OHLCV data.
        """

        raise NotImplementedError

    @abstractmethod
    def get_fundamentals(
        self,
        symbol: str,
    ) -> dict:
        """
        Return standardized fundamental fields.

        Missing data must remain missing.
        Do not fabricate substitute values here.
        """

        raise NotImplementedError