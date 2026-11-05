import numpy as np
import pandas as pd
import pytest
from src.core.pipeline import run_pipeline, export_pipeline, completed_daily
from src.features.relative_strength import build_rs_snapshot
from src.output.formatter import format_biglist_terminal
from test_fundamental_delta import make_income_statement, make_earnings_history

class Fundamentals:
    def get_company_metadata(self, symbol, **kwargs):
        return {'symbol': symbol, 'industry': 'Software', 'sector': 'Technology'}
    def get_quarterly_income_statement(self, symbol, **kwargs):
        return make_income_statement()
    def get_earnings_history(self, symbol, **kwargs):
        return make_earnings_history()

def inputs():
    dates = pd.bdate_range(end='2026-08-10', periods=320)
    def frame(speed):
        x = np.arange(len(dates))
        close = 30 * np.exp(speed*x + .04*np.sin(x/4 + 3))
        return pd.DataFrame(dict(open=close, close=close, high=close*1.025,
                                 low=close*.975, volume=1e6+2e5*np.sin(x)), index=dates)
    daily = {s: frame(v) for s, v in [('AAA', .003), ('BBB', .002), ('CCC', .001)]}
    return dict(metadata=pd.DataFrame({'symbol': list(daily), 'static_universe_pass': True}),
        daily_data=daily, benchmarks={s: frame(.0005) for s in ['SPY', 'QQQ', 'IWM']},
        fundamental_provider=Fundamentals(), as_of=dates[-1])

def test_real_module_pipeline_and_exports(tmp_path):
    args = inputs()
    result = run_pipeline(**args)
    full = result['full'].set_index('symbol')
    assert set(full.index) == {'AAA', 'BBB', 'CCC'}
    assert full.scoring_complete.all(), full.missing_factors.tolist()
    assert full.total_score.between(0, 100).all()
    assert full.hourly_data_status.eq('MISSING').all()
    assert not any(c.endswith(('_x', '_y')) for c in full)
    assert len(result['biglist']) <= result['capacity']
    assert result['biglist'].admitted.all()
    for window, days in [(20, 10), (60, 20)]:
        past = build_rs_snapshot(args['daily_data'], as_of=args['benchmarks']['SPY'].index[-days-1]).set_index('symbol')
        pd.testing.assert_series_equal(full[f'rs_{window}d_rank_delta'].sort_index(),
            (full[f'rs_{window}d_pct']-past[f'rs_{window}d_pct']).sort_index(), check_names=False)
    paths = export_pipeline(result, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert len(pd.read_parquet(paths['full_scan'])) == 3
    for exported in [pd.read_csv(paths['audit']), pd.read_parquet(paths['full_scan'])]:
        assert exported.data_status.eq('COMPLETE').all()
        pd.testing.assert_series_equal(exported.data_status, result['full'].data_status)
    assert 'BIG LIST' in format_biglist_terminal(result)

def test_future_bars_do_not_change_results():
    args = inputs()
    expected = run_pipeline(**args)['full']
    for mapping in [args['daily_data'], args['benchmarks']]:
        for symbol, frame in mapping.items():
            future = frame.tail(1).copy()
            future.index += pd.Timedelta(days=1)
            future[['open', 'high', 'low', 'close']] *= 10
            mapping[symbol] = pd.concat([frame, future])
    pd.testing.assert_frame_equal(expected, run_pipeline(**args)['full'])

def test_missing_fundamentals_cannot_receive_complete_score():
    class Missing(Fundamentals):
        def get_quarterly_income_statement(self, symbol, **kwargs):
            raise RuntimeError('Provider unavailable')
    args = inputs()
    args['fundamental_provider'] = Missing()
    full = run_pipeline(**args)['full']
    assert not full.scoring_complete.any()
    assert full.total_score.isna().all()
    assert full.fundamental_error.eq('Provider unavailable').all()

def test_stale_stock_is_excluded_and_stale_benchmark_stops_run():
    args = inputs()
    args['daily_data']['CCC'] = args['daily_data']['CCC'].iloc[:-1]
    result = run_pipeline(**args)
    assert 'CCC' not in result['full'].symbol.tolist()
    assert result['universe'].set_index('symbol').loc['CCC', 'market_gate_reason'] == 'STALE_DAILY_DATA'
    args['benchmarks']['QQQ'] = args['benchmarks']['QQQ'].iloc[:-1]
    with pytest.raises(ValueError, match='QQQ'):
        run_pipeline(**args)

def test_partial_session_is_excluded():
    frame = inputs()['daily_data']['AAA']
    assert completed_daily(frame, now='2026-08-10 15:59').index[-1] == pd.Timestamp('2026-08-07')
    assert completed_daily(frame, now='2026-08-10 16:05').index[-1] == pd.Timestamp('2026-08-10')

def test_empty_universe_stops_without_publishing():
    args = inputs()
    args['metadata']['static_universe_pass'] = False
    with pytest.raises(ValueError, match='No symbols'):
        run_pipeline(**args)

def test_no_confirmed_swings_does_not_crash_admission():
    from src.core.admission import calculate_structure_evaluation
    args = inputs()
    result = calculate_structure_evaluation(args['daily_data']['AAA'], as_of=args['as_of'], swings=pd.DataFrame())
    assert 'structure_score' in result


def test_hourly_current_session_is_used_and_future_session_excluded():
    from src.features.hourly_flow import summarize_hourly_flow
    from src.features.basic import add_atr
    args = inputs()
    daily = args['daily_data']['AAA']
    hourly = daily.tail(20).copy()
    hourly.index = hourly.index + pd.Timedelta(hours=15, minutes=30)
    future = hourly.tail(1).copy()
    future.index += pd.Timedelta(days=1)
    class Hourly:
        def get_hourly(self, symbol, **kwargs):
            return pd.concat([hourly, future])
    args['hourly_provider'] = Hourly()
    full = run_pipeline(**args)['full'].set_index('symbol')
    expected = summarize_hourly_flow(hourly, as_of=hourly.index[-1], daily_atr=float(add_atr(daily).atr_14.iloc[-1]))
    assert full.loc['AAA', 'hourly_confirmation'] == expected['confirmation']
    assert full.hourly_data_status.eq('AVAILABLE').all()

def test_yahoo_statement_aliases_preserve_financial_values():
    from src.features.fundamental_delta import calculate_income_statement_delta
    original = make_income_statement()
    yahoo = original.rename(index={'Total Revenue': 'TotalRevenue', 'Operating Income': 'OperatingIncome'})
    assert calculate_income_statement_delta(yahoo) == calculate_income_statement_delta(original)


def test_yahoo_quarter_dates_are_not_earnings_announcement_dates():
    from src.features.fundamental_delta import calculate_eps_surprise, calculate_fundamental_delta
    history = make_earnings_history()
    history.index.name = 'quarter'
    surprise = calculate_eps_surprise(history)
    assert np.isclose(surprise['eps_surprise'], .2)
    assert pd.isna(surprise['latest_earnings_date'])
    result = calculate_fundamental_delta(quarterly_income=make_income_statement(),
        earnings_history=history, daily_frame=inputs()['daily_data']['AAA'])
    assert 'earnings_reaction' in result['fundamental_missing_components']

def test_parallel_fetch_keeps_identical_model_results():
    args = inputs()
    serial = run_pipeline(**args)['full']
    parallel = run_pipeline(**args, workers=3)['full']
    pd.testing.assert_frame_equal(serial, parallel)
