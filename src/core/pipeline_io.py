from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd


def build_run_id(
    *,
    as_of: pd.Timestamp | None = None,
) -> str:
    """
    Deterministic human-readable run identifier.
    """

    if as_of is None:
        timestamp = pd.Timestamp.now()

    else:
        timestamp = pd.Timestamp(
            as_of
        )

    return timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )


# NYSE published trading calendars, 2026-2028. Unknown years fail closed.
_CLOSED = {
    2026: "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25",
    2027: "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24",
    2028: "01-17 02-21 04-14 05-29 06-19 07-04 09-04 11-23 12-25",
}
_EARLY = {"2026-11-27", "2026-12-24", "2027-11-26", "2028-07-03", "2028-11-24"}


def session_close(day):
    day = pd.Timestamp(day).normalize()
    if day.year not in _CLOSED or day.weekday() >= 5 or day.strftime("%m-%d") in _CLOSED[day.year].split():
        return None
    hour = 13 if str(day.date()) in _EARLY else 16
    return day.tz_localize("America/New_York") + pd.Timedelta(hours=hour)


def latest_completed_session(benchmark_frame, now=None):
    import os
    now = now or os.environ.get("BIGLIST_STARTED_AT") or pd.Timestamp.now(tz="America/New_York")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Expected timezone-aware clock")
    if benchmark_frame is not None:
        for day, row in benchmark_frame.sort_index(ascending=False).iterrows():
            day = pd.Timestamp(day).tz_localize(None).normalize()
            close = session_close(day)
            if close is not None and close + pd.Timedelta(minutes=30) <= now and pd.notna(row["close"]) and row["close"] > 0:
                return day
    raise ValueError("No confirmed completed Yahoo benchmark session")


def select_universe_symbols(
    universe_snapshot: pd.DataFrame,
) -> list[str]:
    """
    Return only stocks that passed BOTH:

        static security filtering
        market-data gates
    """

    required = {
        "symbol",
        "universe_pass",
    }

    missing = required.difference(
        universe_snapshot.columns
    )

    if missing:
        raise ValueError(
            "Universe snapshot missing columns: "
            f"{sorted(missing)}"
        )

    passed = universe_snapshot[
        universe_snapshot[
            "universe_pass"
        ].astype(bool)
    ]

    return (
        passed[
            "symbol"
        ]
        .astype(str)
        .tolist()
    )


def subset_daily_data(
    daily_data: Mapping[
        str,
        pd.DataFrame,
    ],
    symbols: list[str],
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Restrict downloaded daily data to the actual
    Big List investable universe.
    """

    result = {}

    for symbol in symbols:
        frame = daily_data.get(
            symbol
        )

        if (
            frame is None
            or frame.empty
        ):
            continue

        result[
            str(symbol)
        ] = frame

    return result


def export_preflight_artifacts(
    *,
    universe_snapshot: pd.DataFrame,
    output_dir: str | Path = "outputs",
    run_id: str,
) -> dict[str, Path]:
    """
    Persist deterministic production preflight
    artifacts.

    Files:

        universe_latest.csv
        universe_<run_id>.csv
        universe_latest.parquet
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_csv = (
        output_dir
        / "universe_latest.csv"
    )

    historical_csv = (
        output_dir
        / f"universe_{run_id}.csv"
    )

    latest_parquet = (
        output_dir
        / "universe_latest.parquet"
    )

    universe_snapshot.to_csv(
        latest_csv,
        index=False,
    )

    universe_snapshot.to_csv(
        historical_csv,
        index=False,
    )

    universe_snapshot.to_parquet(
        latest_parquet,
        index=False,
    )

    return {
        "latest_csv":
            latest_csv,

        "historical_csv":
            historical_csv,

        "latest_parquet":
            latest_parquet,
    }


def format_preflight_summary(
    *,
    directory_count: int,
    static_pass_count: int,
    downloaded_count: int,
    universe_pass_count: int,
    as_of: pd.Timestamp,
) -> str:
    """
    Terminal output for the production data preflight.
    """

    lines = [
        "BIG LIST ENGINE — DATA PREFLIGHT",
        "",
        f"AS OF                 {pd.Timestamp(as_of).date()}",
        "",
        f"LISTED DIRECTORY      {directory_count}",
        f"STATIC UNIVERSE PASS  {static_pass_count}",
        f"DAILY DATA AVAILABLE  {downloaded_count}",
        f"FINAL UNIVERSE PASS   {universe_pass_count}",
        "",
        "STATUS                OK",
    ]

    return "\n".join(
        lines
    )