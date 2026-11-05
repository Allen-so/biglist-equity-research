"""Extension / breakout gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _coerce_series(data: Any, price_col: str = 'close') -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data, pd.DataFrame):
        if price_col not in data.columns:
            raise KeyError(f'Missing required column: {price_col!r}')
        return data[price_col].astype(float)
    if isinstance(data, dict):
        return pd.Series(data, dtype=float)
    return pd.Series(list(data), dtype=float)


@dataclass
class ExtensionGate:
    lookback: int = 20
    minimum_extension: float = 0.0

    def evaluate(self, data: Any, price_col: str = 'close') -> bool:
        series = _coerce_series(data, price_col=price_col)
        if len(series) < self.lookback:
            return False
        recent = series.iloc[-1]
        prior_high = series.iloc[-self.lookback:-1].max() if len(series) > self.lookback else series.iloc[0]
        extension = ((recent - prior_high) / prior_high) * 100.0 if prior_high else 0.0
        return bool(extension >= self.minimum_extension)

    def passes(self, data: Any, price_col: str = 'close') -> bool:
        return self.evaluate(data, price_col=price_col)

    def __call__(self, data: Any, price_col: str = 'close') -> bool:
        return self.evaluate(data, price_col=price_col)


def evaluate_extension(data: Any, lookback: int = 20, minimum_extension: float = 0.0, price_col: str = 'close') -> bool:
    return ExtensionGate(lookback=lookback, minimum_extension=minimum_extension).evaluate(data, price_col=price_col)


def extension_gate(data: Any, lookback: int = 20, minimum_extension: float = 0.0, price_col: str = 'close') -> bool:
    return evaluate_extension(data, lookback=lookback, minimum_extension=minimum_extension, price_col=price_col)
