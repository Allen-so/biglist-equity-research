"""Tradability gate (liquidity + volatility)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _coerce_frame(data: Any, price_col: str = 'close', volume_col: str = 'volume'):
    if isinstance(data, dict):
        return data
    if hasattr(data, 'columns'):
        return data
    raise TypeError(f'Unsupported input type: {type(data)!r}')


@dataclass
class TradabilityGate:
    minimum_dollar_volume: float = 1_000_000.0
    minimum_volatility: float = 0.0
    price_col: str = 'close'
    volume_col: str = 'volume'

    def evaluate(self, data: Any) -> bool:
        frame = _coerce_frame(data)
        if self.price_col not in frame.columns or self.volume_col not in frame.columns:
            raise KeyError(f'Tradability gate requires {self.price_col!r} and {self.volume_col!r} columns.')
        dollar_volume = (frame[self.price_col].astype(float) * frame[self.volume_col].astype(float)).mean()
        volatility = frame[self.price_col].astype(float).pct_change().fillna(0.0).std() * 100.0
        return bool(dollar_volume >= self.minimum_dollar_volume and volatility >= self.minimum_volatility)

    def passes(self, data: Any) -> bool:
        return self.evaluate(data)

    def __call__(self, data: Any) -> bool:
        return self.evaluate(data)


def evaluate_tradability(data: Any, minimum_dollar_volume: float = 1_000_000.0, minimum_volatility: float = 0.0, price_col: str = 'close', volume_col: str = 'volume') -> bool:
    return TradabilityGate(minimum_dollar_volume=minimum_dollar_volume, minimum_volatility=minimum_volatility, price_col=price_col, volume_col=volume_col).evaluate(data)


def tradability_gate(data: Any, minimum_dollar_volume: float = 1_000_000.0, minimum_volatility: float = 0.0, price_col: str = 'close', volume_col: str = 'volume') -> bool:
    return evaluate_tradability(data, minimum_dollar_volume=minimum_dollar_volume, minimum_volatility=minimum_volatility, price_col=price_col, volume_col=volume_col)
