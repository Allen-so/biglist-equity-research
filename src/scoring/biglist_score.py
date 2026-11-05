"""Composite score for ranking candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class BigListScore:
    weights: Dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                'relative_strength': 0.25,
                'retention': 0.20,
                'structure': 0.20,
                'velocity': 0.15,
                'volatility': 0.10,
                'volume': 0.10,
            }

    def calculate(self, feature_scores: Mapping[str, Any]) -> float:
        total = 0.0
        for name, weight in self.weights.items():
            score = feature_scores.get(name, 0.0)
            if isinstance(score, pd.Series):
                score = float(score.fillna(0.0).mean())
            total += _safe_float(score) * weight
        return float(total)

    def score(self, feature_scores: Mapping[str, Any]) -> float:
        return self.calculate(feature_scores)

    def __call__(self, feature_scores: Mapping[str, Any]) -> float:
        return self.calculate(feature_scores)


def calculate_biglist_score(feature_scores: Mapping[str, Any], weights: Dict[str, float] | None = None) -> float:
    return BigListScore(weights=weights).calculate(feature_scores)


def biglist_score(feature_scores: Mapping[str, Any], weights: Dict[str, float] | None = None) -> float:
    return calculate_biglist_score(feature_scores, weights=weights)
