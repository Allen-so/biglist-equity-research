from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def evaluate_market_gate(
    frame: pd.DataFrame,
    *,
    min_price: float = 4.0,
    min_median_dollar_volume_20d: float = (
        8_000_000.0
    ),
    min_history_days: int = 60,
) -> dict[str, Any]:
    """
    Frozen Big List basic universe gates.

        Price >= $4
        20D median dollar volume >= $8M
        Minimum 60 sessions of history
    """

    if frame is None or frame.empty:
        return {
            "market_gate_pass":
                False,

            "market_gate_reason":
                "NO_PRICE_DATA",

            "last_price":
                np.nan,

            "median_dollar_volume_20d":
                np.nan,

            "history_days":
                0,
        }

    history_days = len(
        frame
    )

    last_price = float(
        frame[
            "close"
        ].iloc[-1]
    )

    dollar_volume = (
        frame[
            "close"
        ]
        *
        frame[
            "volume"
        ]
    )

    if len(
        dollar_volume
    ) >= 20:
        median_dollar_volume = float(
            dollar_volume
            .tail(20)
            .median()
        )

    else:
        median_dollar_volume = np.nan

    reasons = []

    if (
        history_days
        < min_history_days
    ):
        reasons.append(
            "HISTORY_LT_60"
        )

    if (
        pd.isna(
            last_price
        )
        or
        last_price < min_price
    ):
        reasons.append(
            "PRICE_LT_4"
        )

    if (
        pd.isna(
            median_dollar_volume
        )
        or
        median_dollar_volume
        <
        min_median_dollar_volume_20d
    ):
        reasons.append(
            "MEDIAN_DOLLAR_VOLUME_LT_8M"
        )

    return {
        "market_gate_pass":
            len(
                reasons
            )
            == 0,

        "market_gate_reason":
            (
                "|".join(
                    reasons
                )
                if reasons
                else None
            ),

        "last_price":
            last_price,

        "median_dollar_volume_20d":
            median_dollar_volume,

        "history_days":
            history_days,
    }


def apply_market_gates(
    metadata: pd.DataFrame,
    daily_data: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Join static directory filtering with
    real market-data gates.
    """

    records = []

    for _, row in (
        metadata.iterrows()
    ):
        symbol = str(
            row[
                "symbol"
            ]
        )

        static_pass = bool(
            row.get(
                "static_universe_pass",
                False,
            )
        )

        if not static_pass:
            market_result = {
                "market_gate_pass":
                    False,

                "market_gate_reason":
                    "STATIC_UNIVERSE_FAIL",

                "last_price":
                    np.nan,

                "median_dollar_volume_20d":
                    np.nan,

                "history_days":
                    0,
            }

        else:
            market_result = (
                evaluate_market_gate(
                    daily_data.get(
                        symbol,
                        pd.DataFrame(),
                    )
                )
            )

        final_pass = (
            static_pass
            and
            bool(
                market_result[
                    "market_gate_pass"
                ]
            )
        )

        records.append(
            {
                **row.to_dict(),
                **market_result,

                "universe_pass":
                    bool(
                        final_pass
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    return (
        result
        .sort_values(
            [
                "universe_pass",
                "median_dollar_volume_20d",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )