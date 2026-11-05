import numpy as np
import pandas as pd

from src.features.acceptance import add_progress_retention


def _frame(values):
    return pd.DataFrame(
        {"close": values},
        index=pd.date_range(
            "2026-01-01",
            periods=len(values),
        ),
    )


def test_zero_impulse_with_full_history_scores_zero_retention():
    out = add_progress_retention(
        _frame([100] * 11),
        impulse_window=10,
    )

    assert (
        out["progress_retention_10"].iloc[-1]
        == 0.0
    )


def test_decline_with_full_history_scores_zero_retention():
    out = add_progress_retention(
        _frame(
            [100,99,98,97,96,95,94,93,92,91,90]
        ),
        impulse_window=10,
    )

    assert (
        out["progress_retention_10"].iloc[-1]
        == 0.0
    )


def test_insufficient_history_remains_missing():
    out = add_progress_retention(
        _frame([100] * 10),
        impulse_window=10,
    )

    assert np.isnan(
        out["progress_retention_10"].iloc[-1]
    )
