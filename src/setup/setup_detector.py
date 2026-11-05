"""Simple setup-pattern detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
class SetupDetector:
    lookback: int = 20
    min_breakout: float = 2.0

    def detect(self, data: Any, price_col: str = 'close') -> Dict[str, Any]:
        series = _coerce_series(data, price_col=price_col)
        if len(series) < 3:
            return {'name': 'none', 'score': 0.0, 'passed': False, 'details': {'reason': 'too_short'}}
        recent = series.iloc[-1]
        prior_high = series.iloc[-self.lookback:-1].max() if len(series) > self.lookback else series.iloc[:-1].max()
        breakout_pct = ((recent - prior_high) / prior_high) * 100.0 if prior_high else 0.0
        score = max(0.0, breakout_pct)
        passed = breakout_pct >= self.min_breakout
        return {'name': 'breakout' if passed else 'base', 'score': score, 'passed': passed, 'details': {'breakout_pct': breakout_pct, 'prior_high': float(prior_high)}}

    def __call__(self, data: Any, price_col: str = 'close') -> Dict[str, Any]:
        return self.detect(data, price_col=price_col)


def detect_setup(data: Any, lookback: int = 20, min_breakout: float = 2.0, price_col: str = 'close') -> Dict[str, Any]:
    return SetupDetector(lookback=lookback, min_breakout=min_breakout).detect(data, price_col=price_col)


def setup_detector(data: Any, lookback: int = 20, min_breakout: float = 2.0, price_col: str = 'close') -> Dict[str, Any]:
    return detect_setup(data, lookback=lookback, min_breakout=min_breakout, price_col=price_col)
