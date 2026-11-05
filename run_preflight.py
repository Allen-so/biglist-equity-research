from __future__ import annotations

import argparse
from src.data.checkpoint import target_day

from src.core.pipeline_io import (
    build_run_id,
    export_preflight_artifacts,
    format_preflight_summary,
    latest_completed_session,
    session_close,
    select_universe_symbols,
)
from src.data.yahoo_provider import (
    YahooMarketDataProvider,
)
from src.universe.market_gates import (
    apply_market_gates,
)
from src.universe.us_equities import (
    load_us_equity_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Big List Engine production "
            "data preflight."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignore valid local price cache "
            "and redownload market data."
        ),
    )

    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help=(
            "Limit symbols for smoke testing. "
            "Example: --max-symbols 100"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/preflight",
        help=(
            "Directory for preflight-only "
            "universe artifacts."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(
        "Loading US security directory..."
    )

    directory = (
        load_us_equity_directory()
    )

    directory_count = len(
        directory
    )

    static = directory[
        directory[
            "static_universe_pass"
        ]
    ].copy()

    static_pass_count = len(
        static
    )

    if args.max_symbols is not None:
        if args.max_symbols < 1:
            raise ValueError(
                "--max-symbols must be >= 1."
            )

        static = static.head(
            args.max_symbols
        ).copy()

    symbols = (
        static[
            "symbol"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Static candidates: {len(symbols)}"
    )

    provider = (
        YahooMarketDataProvider()
    )

    #
    # Always resolve market as-of from SPY.
    #
    print(
        "Loading SPY benchmark..."
    )

    provider.expected_session = target_day()
    spy = provider.get_daily(
        "SPY",
        refresh=args.refresh,
    )

    as_of = (
        latest_completed_session(
            spy
        )
    )

    print(
        f"Market as-of: {as_of.date()}"
    )

    if as_of != provider.expected_session:
        raise RuntimeError(f"Benchmark behind target {provider.expected_session.date()}")
    provider.benchmark_sessions = spy.loc[:as_of].index
    close = session_close(as_of)
    print(f"確認收盤 {close:%Y-%m-%d %H:%M %Z} / 香港 {close.tz_convert('Asia/Hong_Kong'):%Y-%m-%d %H:%M}", flush=True)
    print(
        "Downloading / loading daily data..."
    )

    daily_data = (
        provider.download_daily_batch(
            symbols,
            batch_size=(
                args.batch_size
            ),
            refresh=args.refresh,
        )
    )

    daily_data = {symbol: frame.loc[frame.index <= as_of] for symbol, frame in daily_data.items()}
    from src.data.coverage import assess, require_coverage
    from pathlib import Path
    daily_data, coverage_audit, coverage = assess(symbols, daily_data, provider)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    coverage_audit.to_csv(Path(args.output_dir) / "coverage_audit_latest.csv", index=False)
    print("Coverage", coverage, flush=True)
    require_coverage(coverage)

    downloaded_count = len(
        daily_data
    )

    #
    # Important:
    # apply_market_gates() expects the metadata
    # rows we actually attempted to download.
    #
    universe_snapshot = (
        apply_market_gates(
            static,
            daily_data,
        )
    )

    universe_symbols = (
        select_universe_symbols(
            universe_snapshot
        )
    )

    universe_pass_count = len(
        universe_symbols
    )

    run_id = build_run_id(
        as_of=as_of
    )

    paths = (
        export_preflight_artifacts(
            universe_snapshot=(
                universe_snapshot
            ),
            output_dir=(
                args.output_dir
            ),
            run_id=run_id,
        )
    )

    print()
    print(
        format_preflight_summary(
            directory_count=(
                directory_count
            ),
            static_pass_count=(
                static_pass_count
            ),
            downloaded_count=(
                downloaded_count
            ),
            universe_pass_count=(
                universe_pass_count
            ),
            as_of=as_of,
        )
    )

    print()
    print(
        "ARTIFACTS"
    )

    for name, path in (
        paths.items()
    ):
        print(
            f"{name}: {path}"
        )


if __name__ == "__main__":
    main()