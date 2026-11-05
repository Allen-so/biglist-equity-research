"""Retention / persistence feature calculations."""

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
class RetentionFeature:
    window: int = 20
    min_periods: int = 1

    def calculate(self, data: Any, price_col: str = 'close') -> pd.Series:
        series = _coerce_series(data, price_col=price_col)
        returns = series.pct_change().fillna(0.0)
        if len(returns) < self.window:
            return pd.Series([0.0] * len(returns), index=returns.index, dtype=float)
        rolling = returns.rolling(window=self.window, min_periods=self.min_periods)
        retention = rolling.apply(lambda x: float((x > 0).mean()) if len(x) else 0.0)
        return retention * 100.0

    def score(self, data: Any, price_col: str = 'close') -> pd.Series:
        return self.calculate(data, price_col=price_col)

    def __call__(self, data: Any, price_col: str = 'close') -> pd.Series:
        return self.calculate(data, price_col=price_col)


def compute_retention(data: Any, window: int = 20, price_col: str = 'close') -> pd.Series:
    return RetentionFeature(window=window).calculate(data, price_col=price_col)


def retention(data: Any, window: int = 20, price_col: str = 'close') -> pd.Series:
    return compute_retention(data, window=window, price_col=price_col)
