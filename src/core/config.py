from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(filename: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "config" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must contain a mapping: {path}"
        )

    return data


def load_thresholds() -> dict[str, Any]:
    return load_yaml("thresholds.yaml")


def load_weights() -> dict[str, Any]:
    return load_yaml("weights.yaml")


def validate_weights(weights: dict[str, Any]) -> None:
    factors = weights.get("factors")

    if not isinstance(factors, dict):
        raise ValueError("weights.yaml is missing 'factors'.")

    total = sum(float(value) for value in factors.values())

    if abs(total - 100.0) > 1e-9:
        raise ValueError(
            f"Big List factor weights must total 100. "
            f"Current total: {total}"
        )