from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _ticker_lines(
    symbols: Iterable[str],
    *,
    per_line: int = 8,
) -> list[str]:
    values = [
        str(symbol)
        for symbol in symbols
    ]

    if not values:
        return [
            "—"
        ]

    lines = []

    for start in range(
        0,
        len(values),
        per_line,
    ):
        chunk = values[
            start:
            start + per_line
        ]

        lines.append(
            ", ".join(
                chunk
            )
        )

    return lines


def _symbols(
    frame: pd.DataFrame,
) -> list[str]:
    if (
        frame is None
        or frame.empty
        or "symbol"
        not in frame.columns
    ):
        return []

    return (
        frame[
            "symbol"
        ]
        .astype(str)
        .tolist()
    )


def format_biglist_terminal(
    result: dict[str, Any],
    *,
    include_watch: bool = True,
    show_regime: bool = True,
) -> str:
    """
    Concise front-end output.

    Detailed scores remain in CSV / Parquet / audit.
    The user-facing terminal remains deliberately
    simple.
    """

    biglist = result[
        "biglist"
    ]

    ready = result[
        "ready"
    ]

    developing = result[
        "developing"
    ]

    watch = result[
        "watch"
    ]

    lines: list[str] = []

    lines.append(
        f"BIG LIST — {len(biglist)}"
    )

    if show_regime:
        regime = result.get(
            "regime",
            "UNKNOWN",
        )

        capacity = result.get(
            "capacity",
            "?"
        )

        regime_score = result.get(
            "regime_score",
            None,
        )

        if regime_score is None:
            lines.append(
                f"{regime} | CAPACITY {capacity}"
            )

        else:
            lines.append(
                f"{regime} "
                f"({float(regime_score):.1f}) "
                f"| CAPACITY {capacity}"
            )

    lines.append("")

    lines.append(
        f"READY — {len(ready)}"
    )

    lines.extend(
        _ticker_lines(
            _symbols(
                ready
            )
        )
    )

    lines.append("")

    lines.append(
        f"DEVELOPING — {len(developing)}"
    )

    lines.extend(
        _ticker_lines(
            _symbols(
                developing
            )
        )
    )

    if include_watch:
        lines.append("")

        lines.append(
            f"WATCH LIST — {len(watch)}"
        )

        lines.extend(
            _ticker_lines(
                _symbols(
                    watch
                )
            )
        )

    return "\n".join(
        lines
    )