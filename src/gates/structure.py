"""Structure gate for trend quality."""

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
class StructureGate:
    min_trend_strength: float = 0.0
    window: int = 20

    def evaluate(self, data: Any, price_col: str = 'close') -> bool:
        series = _coerce_series(data, price_col=price_col)
        if len(series) < 2:
            return False
        slope = (series.iloc[-1] - series.iloc[0]) / max(len(series) - 1, 1)
        drift = (series.pct_change().fillna(0.0).rolling(window=self.window, min_periods=1).mean().iloc[-1]) * 100.0
        return bool(drift >= self.min_trend_strength or slope > 0)

    def passes(self, data: Any, price_col: str = 'close') -> bool:
        return self.evaluate(data, price_col=price_col)

    def __call__(self, data: Any, price_col: str = 'close') -> bool:
        return self.evaluate(data, price_col=price_col)


def evaluate_structure(data: Any, min_trend_strength: float = 0.0, window: int = 20, price_col: str = 'close') -> bool:
    return StructureGate(min_trend_strength=min_trend_strength, window=window).evaluate(data, price_col=price_col)


def structure_gate(data: Any, min_trend_strength: float = 0.0, window: int = 20, price_col: str = 'close') -> bool:
    return evaluate_structure(data, min_trend_strength=min_trend_strength, window=window, price_col=price_col)
