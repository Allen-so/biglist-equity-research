"""Liquidity gate."""

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
class LiquidityGate:
    minimum_dollar_volume: float = 1_000_000.0
    price_col: str = 'close'
    volume_col: str = 'volume'

    def evaluate(self, data: Any) -> bool:
        frame = _coerce_frame(data)
        if self.price_col not in frame.columns or self.volume_col not in frame.columns:
            raise KeyError(f'Liquidity gate requires {self.price_col!r} and {self.volume_col!r} columns.')
        average_dollar_volume = (frame[self.price_col].astype(float) * frame[self.volume_col].astype(float)).mean()
        return bool(average_dollar_volume >= self.minimum_dollar_volume)

    def passes(self, data: Any) -> bool:
        return self.evaluate(data)

    def __call__(self, data: Any) -> bool:
        return self.evaluate(data)


def evaluate_liquidity(data: Any, minimum_dollar_volume: float = 1_000_000.0, price_col: str = 'close', volume_col: str = 'volume') -> bool:
    return LiquidityGate(minimum_dollar_volume=minimum_dollar_volume, price_col=price_col, volume_col=volume_col).evaluate(data)


def liquidity_gate(data: Any, minimum_dollar_volume: float = 1_000_000.0, price_col: str = 'close', volume_col: str = 'volume') -> bool:
    return evaluate_liquidity(data, minimum_dollar_volume=minimum_dollar_volume, price_col=price_col, volume_col=volume_col)
