"""Phase 8B orchestration; model rules remain in their existing modules."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.core.admission import evaluate_candidate_admission
from src.core.finalize import build_final_lists
from src.core.scoring import build_score_snapshot
from src.features.acceptance import build_acceptance_features, evaluate_zone_interaction
from src.features.basic import build_basic_daily_features
from src.features.confluence import collect_structural_evidence, build_confluence_zones
from src.features.fundamental_delta import build_fundamental_snapshot
from src.features.hourly_flow import summarize_hourly_flow
from src.features.market_group import build_market_group_snapshot
from src.features.multitimeframe import build_weekly_features
from src.features.quality_factors import build_quality_factor_snapshot
from src.features.relative_strength import build_rs_snapshot, add_rank_delta
from src.features.setup import calculate_setup_features
from src.features.structure import detect_swings
from src.features.tradability import calculate_tradability_features
from src.features.trendlines import find_trendlines
from src.features.velocity import build_velocity_snapshot
from src.features.vwap import build_avwap_features
from src.features.zones import build_zones
from src.regime.market_regime import calculate_market_regime
from src.universe.market_gates import apply_market_gates


def completed_daily(frame, *, now=None):
    """Exclude today's partial US session; daily timestamps are NY session dates."""
    now = pd.Timestamp.now(tz='America/New_York') if now is None else pd.Timestamp(now)
    now = now.tz_localize('America/New_York') if now.tzinfo is None else now.tz_convert('America/New_York')
    cutoff = now.tz_localize(None).normalize()
    if now.hour < 16:
        cutoff -= pd.Timedelta(days=1)
    return frame.loc[frame.index.normalize() <= cutoff].copy()


def _merge(left, right):
    # Shared evidence (as_of, RS columns) is already authoritative on the left.
    columns = ['symbol'] + [c for c in right if c != 'symbol' and c not in left]
    return left.merge(right[columns], on='symbol', how='left', validate='one_to_one')


def _structure(frame, as_of):
    swings = detect_swings(frame)
    latest = frame.iloc[-1]
    zones = build_zones(swings, as_of=as_of, current_price=float(latest.close),
                        current_atr=float(latest.atr_14))
    lines = pd.concat([find_trendlines(frame, swings, as_of=as_of, kind=k)
                       for k in ('HIGH', 'LOW')], ignore_index=True)
    return swings, zones, lines


def _symbol_features(payload):
    symbol, raw, as_of, candidate_row, hourly_payload = payload
    frame = build_acceptance_features(build_basic_daily_features(raw))
    swings, zones, lines = _structure(frame, as_of)
    weekly = build_weekly_features(raw)
    weekly = weekly.loc[weekly.index <= as_of]  # completed weekly bars only
    _, weekly_zones, weekly_lines = _structure(weekly, as_of)
    evidence = collect_structural_evidence(current_price=float(frame.close.iloc[-1]),
        current_atr=float(frame.atr_14.iloc[-1]), daily_frame=frame, weekly_frame=weekly,
        daily_zones=zones, weekly_zones=weekly_zones,
        daily_trendlines=lines, weekly_trendlines=weekly_lines)
    confluence = build_confluence_zones(evidence, current_price=float(frame.close.iloc[-1]),
                                        current_atr=float(frame.atr_14.iloc[-1]))
    anchors, avwap = build_avwap_features(frame, as_of=as_of, swings=swings, acceptance_frame=frame)
    setup = calculate_setup_features(frame, as_of=as_of, swings=swings, rs_row=candidate_row)
    tradability = calculate_tradability_features(frame, as_of=as_of, setup_row=setup,
                                                 zones=zones, confluence=confluence, trendlines=lines)
    row = {**frame.iloc[-1].to_dict(), **candidate_row,
           'symbol': symbol, **setup, **tradability,
           'range_pct_20': frame.daily_range_pct.tail(20).median(),
           'daily_zone_count': len(zones), 'weekly_zone_count': len(weekly_zones),
           'daily_trendline_count': len(lines), 'weekly_trendline_count': len(weekly_lines),
           'confluence_count': len(confluence), 'avwap_anchor_count': len(anchors),
           'avwap_evidence': avwap.to_json(orient='records', date_format='iso'),
           **evaluate_zone_interaction(frame, zones, current_atr=float(frame.atr_14.iloc[-1]))}
    row.update(hourly_confirmation='NEUTRAL', hourly_data_status='MISSING')
    if hourly_payload is not None:
        try:
            hourly, error = hourly_payload
            if error:
                raise ValueError(error)
            hourly = hourly.loc[hourly.index < as_of + pd.Timedelta(days=1)]
            if hourly.empty or hourly.index[-1].normalize() != as_of:
                raise ValueError('Missing current-session hourly data')
        except Exception as exc:
            row['hourly_error'] = str(exc)
        else:
            summary = summarize_hourly_flow(hourly, as_of=as_of + pd.Timedelta(hours=23, minutes=59),
                                            daily_atr=float(frame.atr_14.iloc[-1]))
            row.update({f'hourly_{k}': v for k, v in summary.items()})
            row['hourly_data_status'] = 'AVAILABLE'
    return row, swings


def run_pipeline(*, metadata, daily_data, benchmarks, fundamental_provider,
                 hourly_provider=None, as_of, refresh=False, workers=1, progress=None):
    """Run existing APIs over one frozen session and one cross-sectional universe."""
    progress = progress or (lambda message: None)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    as_of = pd.Timestamp(as_of).normalize()
    benchmarks = {s: f.loc[f.index <= as_of].copy() for s, f in benchmarks.items()}
    for symbol in ('SPY', 'QQQ', 'IWM'):
        if symbol not in benchmarks or benchmarks[symbol].empty or benchmarks[symbol].index[-1] != as_of:
            raise ValueError(f'{symbol}: missing benchmark session {as_of.date()}')
    if metadata.empty or metadata.symbol.duplicated().any():
        raise ValueError('Universe metadata must contain unique symbols and cannot be empty.')
    visible = {s: f.loc[f.index <= as_of].copy() for s, f in daily_data.items()}
    universe = apply_market_gates(metadata, visible)
    for i, row in universe.iterrows():
        f = visible.get(row.symbol)
        if row.universe_pass and f.index[-1] != as_of:
            universe.loc[i, ['universe_pass', 'market_gate_pass', 'market_gate_reason']] = [False, False, 'STALE_DAILY_DATA']
    symbols = universe.loc[universe.universe_pass, 'symbol'].tolist()
    if not symbols:
        raise ValueError('No symbols passed current-session universe gates.')
    daily = {s: visible[s] for s in symbols}
    progress(f"Universe passed: {len(daily)}/{len(metadata)}; calculating ranks and regime")
    rs = build_rs_snapshot(daily, as_of=as_of, benchmarks=benchmarks)
    # Configured RS20 delta: 10 sessions; RS60 delta: 20 sessions.
    sessions = benchmarks['SPY'].index
    for window, lookback in ((20, 10), (60, 20)):
        if len(sessions) <= lookback:
            raise ValueError('Insufficient benchmark history for rank delta.')
        past = build_rs_snapshot(daily, as_of=sessions[-lookback-1], benchmarks=benchmarks)
        rs = add_rank_delta(rs, past, windows=(window,))
    velocity = build_velocity_snapshot(daily, as_of=as_of)
    regime = calculate_market_regime(daily, spy_frame=benchmarks['SPY'],
        qqq_frame=benchmarks['QQQ'], iwm_frame=benchmarks['IWM'], rs_snapshot=rs, as_of=as_of)
    if regime['regime_status'] != 'COMPLETE':
        raise ValueError(f"Incomplete market regime: {regime['missing_regime_components']}")
    def fundamental_one(symbol):
        return build_fundamental_snapshot([symbol], provider=fundamental_provider,
                                           daily_data=daily, refresh=refresh)

    def hourly_one(symbol):
        try:
            return hourly_provider.get_hourly(symbol, refresh=refresh), None
        except Exception as exc:
            return pd.DataFrame(), str(exc)

    progress(f"Loading fundamentals with {workers} workers")
    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(fundamental_one, symbols), 1):
            parts.append(part)
            if i % 25 == 0 or i == len(symbols):
                progress(f"Fundamentals {i}/{len(symbols)}")
        fundamentals = pd.concat(parts, ignore_index=True)
        hourly_data = {}
        if hourly_provider is not None:
            for i, (symbol, data) in enumerate(zip(symbols, pool.map(hourly_one, symbols)), 1):
                hourly_data[symbol] = data
                if i % 25 == 0 or i == len(symbols):
                    progress(f"Hourly data {i}/{len(symbols)}")
    groups = build_market_group_snapshot(daily, benchmark_frame=benchmarks['SPY'],
        as_of=as_of, rs_snapshot=rs, velocity_snapshot=velocity, metadata=fundamentals)
    snapshot = _merge(_merge(_merge(rs, velocity), groups), fundamentals).set_index('symbol')
    rows = []
    structures = {}
    payloads = [(symbol, raw, as_of, snapshot.loc[symbol].to_dict(), hourly_data.get(symbol))
                for symbol, raw in daily.items()]
    def collect(iterator):
        for position, (row, swings) in enumerate(iterator, 1):
            rows.append(row)
            structures[row['symbol']] = swings
            if position % 25 == 0 or position == len(symbols):
                progress(f"Structure / setup / tradability {position}/{len(symbols)}")
    if workers == 1:
        collect(map(_symbol_features, payloads))
    else:
        # Each symbol is independent here; cross-sectional ranks are already frozen.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_symbol_features, payloads, chunksize=1))
    snapshot = build_quality_factor_snapshot(pd.DataFrame(rows))
    admitted = []
    for row in snapshot.to_dict('records'):
        symbol = row['symbol']
        row.update(evaluate_candidate_admission(daily[symbol], as_of=as_of, candidate_row=row,
                                               swings=structures[symbol], setup_row=row, tradability_row=row))
        admitted.append(row)
    progress("Scoring and finalizing")
    scored = build_score_snapshot(pd.DataFrame(admitted), regime=regime['regime'])
    result = build_final_lists(scored, regime=regime['regime'], regime_score=regime['regime_score'])
    result.update(universe=universe, regime_details=regime, as_of=as_of)
    return result


def export_pipeline(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in [('biglist', 'biglist'), ('watch', 'watchlist'), ('full', 'audit'),
                      ('rejected', 'rejected'), ('universe', 'universe')]:
        path = output_dir / f'{name}_latest.csv'
        result[key].to_csv(path, index=False)
        paths[name] = path
    full = result['full'].copy()
    for column in full.columns:
        full[column] = full[column].map(lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict, tuple)) else x)
    paths['full_scan'] = output_dir / 'full_scan_latest.parquet'
    full.to_parquet(paths['full_scan'], index=False)
    paths['regime'] = output_dir / 'regime_latest.json'
    paths['regime'].write_text(json.dumps(result['regime_details'], default=str, indent=2))
    paths['summary'] = output_dir / 'run_summary_latest.json'
    summary = {
        'as_of': str(result['as_of']),
        'universe_candidates': len(result['universe']),
        'universe_pass': int(result['universe'].universe_pass.sum()),
        'scanned': len(result['full']),
        'scoring_complete': int(result['full'].scoring_complete.sum()),
        'fundamental_value': int(result['full'].fundamental_delta_status.eq('VALUE').sum()),
        'hourly_available': int(result['full'].hourly_data_status.eq('AVAILABLE').sum()),
        'biglist': len(result['biglist']),
        'watch': len(result['watch']),
        **result.get('run_metadata', {}),
    }
    paths['summary'].write_text(json.dumps(summary, default=str, indent=2))
    return paths
