"""Relative-volume feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _coerce_frame(data: Any, price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise TypeError(f'Unsupported input type: {type(data)!r}')


@dataclass
class VolumeFeature:
    window: int = 20
    price_col: str = 'close'
    volume_col: str = 'volume'

    def calculate(self, data: Any) -> pd.Series:
        frame = _coerce_frame(data, price_col=self.price_col, volume_col=self.volume_col)
        dollar_volume = frame[self.price_col].astype(float) * frame[self.volume_col].astype(float)
        relative = dollar_volume.rolling(window=self.window, min_periods=1).mean()
        return relative

    def score(self, data: Any) -> pd.Series:
        return self.calculate(data)

    def __call__(self, data: Any) -> pd.Series:
        return self.calculate(data)


def compute_volume(data: Any, window: int = 20, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    return VolumeFeature(window=window, price_col=price_col, volume_col=volume_col).calculate(data)


def volume(data: Any, window: int = 20, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    return compute_volume(data, window=window, price_col=price_col, volume_col=volume_col)
