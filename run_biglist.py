from src.data.checkpoint import target_day
"""Production Big List runner using the existing Phase 2–7 model APIs."""
import argparse
from time import monotonic
from pathlib import Path

from src.core.pipeline_io import latest_completed_session
from src.core.pipeline import run_pipeline, export_pipeline
from src.data.yahoo_provider import YahooMarketDataProvider
from src.data.fundamental_provider import YahooFundamentalProvider
from src.output.formatter import format_biglist_terminal
from src.universe.us_equities import load_us_equity_directory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--max-symbols', type=int)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).resolve().parent / 'data/cache')
    args = parser.parse_args(argv)
    if args.max_symbols is not None and args.max_symbols < 1:
        parser.error('--max-symbols must be >= 1')
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = monotonic()
    print('Loading real US universe...', flush=True)
    metadata = load_us_equity_directory()
    directory_count = len(metadata)
    metadata = metadata.loc[metadata.static_universe_pass].copy()
    eligible_count = len(metadata)
    if args.max_symbols is not None:
        metadata = metadata.head(args.max_symbols)
        print('SMOKE TEST: ranks, groups and breadth use a limited universe.', flush=True)
    provider = YahooMarketDataProvider(cache_dir=args.cache_dir / 'yahoo')
    fundamental = YahooFundamentalProvider(cache_dir=args.cache_dir / 'fundamentals')
    provider.expected_session = target_day()
    benchmarks = {s: provider.get_daily(s, refresh=args.refresh)
                  for s in ('SPY', 'QQQ', 'IWM')}
    if benchmarks['SPY'].empty:
        raise ValueError('No completed SPY daily session available')
    as_of = latest_completed_session(benchmarks['SPY'])
    if as_of != provider.expected_session:
        raise RuntimeError(f"Benchmark behind target {provider.expected_session.date()}")
    provider.benchmark_sessions = benchmarks["SPY"].loc[:as_of].index
    provider.expected_session = as_of
    print(f'Completed session: {as_of.date()}; loading {len(metadata)} symbols...', flush=True)
    daily = provider.download_daily_batch(metadata.symbol.tolist(), batch_size=args.batch_size, refresh=args.refresh)
    from src.data.coverage import assess, require_coverage
    from pathlib import Path
    daily, coverage_audit, coverage = assess(metadata.symbol.tolist(), daily, provider)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    coverage_audit.to_csv(Path(args.output_dir) / "coverage_audit_latest.csv", index=False)
    print("Coverage", coverage, flush=True)
    require_coverage(coverage)

    result = run_pipeline(metadata=metadata, daily_data=daily, benchmarks=benchmarks,
        fundamental_provider=fundamental, hourly_provider=provider, as_of=as_of, refresh=args.refresh,
        workers=args.workers, progress=lambda message: print(message, flush=True))
    # Daily coverage has already passed the fixed-pool threshold.
    # A cooldown prevents new requests, not publication of validated data.
    transport_failures = [item for item in fundamental.transport_failures if item[1] != "KeyError"]
    if transport_failures:
        raise RuntimeError(f"Financial data transport incomplete: {transport_failures[:10]}")
    print(f"Financial schema gaps: {fundamental.transport_failures}", flush=True)
    incomplete = result["full"].loc[result["full"].hourly_data_status != "AVAILABLE"]
    print(f"Hourly MISSING: {len(incomplete)}; original model missing-data rules retained", flush=True)
    result["run_metadata"] = dict(scope="FULL" if args.max_symbols is None else "SMOKE",
        directory_count=directory_count, static_eligible=eligible_count, coverage=coverage,
        hourly_missing=len(incomplete), downloaded_count=len(daily), elapsed_seconds=round(monotonic()-started, 2))
    output_dir = args.output_dir / 'smoke' if args.max_symbols is not None else args.output_dir
    paths = export_pipeline(result, output_dir)
    print(format_biglist_terminal(result))
    print(f"Scanned: {len(result['full'])}; score complete: {int(result['full'].scoring_complete.sum())}")
    for name, path in paths.items():
        print(f'{name}: {path}')
    return result


if __name__ == '__main__':
    main()
