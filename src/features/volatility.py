"""Volatility feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _coerce_series(data: Any, price_col: str = 'close') -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data, pd.DataFrame):
        return data[price_col].astype(float)
    if isinstance(data, dict):
        return pd.Series(data, dtype=float)
    return pd.Series(list(data), dtype=float)


@dataclass
class VolatilityFeature:
    window: int = 20
    annualization_factor: float = 252.0

    def calculate(self, data: Any, price_col: str = 'close') -> pd.Series:
        series = _coerce_series(data, price_col=price_col)
        returns = series.pct_change().fillna(0.0)
        std = returns.rolling(window=self.window, min_periods=2).std().fillna(0.0)
        return (std * (self.annualization_factor ** 0.5)) * 100.0

    def score(self, data: Any, price_col: str = 'close') -> pd.Series:
        return self.calculate(data, price_col=price_col)

    def __call__(self, data: Any, price_col: str = 'close') -> pd.Series:
        return self.calculate(data, price_col=price_col)


def compute_volatility(data: Any, window: int = 20, price_col: str = 'close', annualization_factor: float = 252.0) -> pd.Series:
    return VolatilityFeature(window=window, annualization_factor=annualization_factor).calculate(data, price_col=price_col)


def volatility(data: Any, window: int = 20, price_col: str = 'close', annualization_factor: float = 252.0) -> pd.Series:
    return compute_volatility(data, window=window, price_col=price_col, annualization_factor=annualization_factor)
