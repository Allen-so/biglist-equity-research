from types import SimpleNamespace

import pandas as pd
import numpy as np

from src.data import fundamental_provider as module
from src.features.fundamental_delta import build_fundamental_snapshot
from test_pipeline import Fundamentals, inputs
from test_fundamental_delta import make_income_statement


def test_provider_uses_event_dates_and_ignores_legacy_quarter_cache(tmp_path, monkeypatch):
    folder = tmp_path / 'AAA'
    folder.mkdir()
    pd.DataFrame({'epsActual': [999]}, index=pd.DatetimeIndex(['2026-06-30'], name='quarter')).to_parquet(folder/'earnings_history.parquet')
    calls = []
    events = pd.DataFrame({'EPS Estimate': [1.0], 'Reported EPS': [1.2]},
        index=pd.DatetimeIndex(['2026-07-30 16:00'], tz='America/New_York', name='Earnings Date'))
    class Ticker:
        def get_earnings_dates(self, limit):
            calls.append(limit)
            return events.copy()
    monkeypatch.setattr(module, '_load_yfinance', lambda: SimpleNamespace(Ticker=lambda s: Ticker()))
    provider = module.YahooFundamentalProvider(cache_dir=tmp_path)
    first = provider.get_earnings_history('AAA')
    second = provider.get_earnings_history('AAA')
    assert calls == [12]
    assert first.index[0] == pd.Timestamp('2026-07-30 16:00')
    assert first.index.name == 'Earnings Date'
    pd.testing.assert_frame_equal(first, second)


def test_five_quarters_plus_real_earnings_event_clears_existing_coverage_gate():
    class Events(Fundamentals):
        def get_quarterly_income_statement(self, symbol, **kwargs):
            return make_income_statement().iloc[:, :5]
        def get_earnings_history(self, symbol, **kwargs):
            return pd.DataFrame({'EPS Estimate': [1.0], 'Reported EPS': [1.2]},
                index=pd.DatetimeIndex(['2026-07-30 16:00'], name='Earnings Date'))
    args = inputs()
    row = build_fundamental_snapshot(['AAA'], provider=Events(), daily_data=args['daily_data']).iloc[0]
    assert np.isclose(row.fundamental_coverage_weight, .85)
    assert row.fundamental_delta_status == 'VALUE'
    assert row.fundamental_missing_components == ['revenue_acceleration']
    assert row.earnings_reaction_date == pd.Timestamp('2026-07-31')


def test_future_events_with_actual_values_are_excluded():
    class Events(Fundamentals):
        def get_earnings_history(self, symbol, **kwargs):
            return pd.DataFrame({'EPS Estimate': [1.0, 1.0], 'Reported EPS': [99, 1.2]},
                index=pd.DatetimeIndex(['2026-08-11 16:00', '2026-07-30 16:00'], tz='America/New_York', name='Earnings Date'))
    row = build_fundamental_snapshot(['AAA'], provider=Events(), daily_data=inputs()['daily_data']).iloc[0]
    assert np.isclose(row.eps_surprise, .2)
    assert row.latest_earnings_date == pd.Timestamp('2026-07-30 16:00')


def test_empty_metadata_cache_is_retried(tmp_path, monkeypatch):
    import json
    folder = tmp_path/'AAA'
    folder.mkdir()
    (folder/'metadata.json').write_text(json.dumps({'symbol': 'AAA', 'industry': None, 'sector': None}))
    class Ticker:
        def get_info(self):
            return {'industry': 'Software', 'sector': 'Technology'}
    monkeypatch.setattr(module, '_load_yfinance', lambda: SimpleNamespace(Ticker=lambda s: Ticker()))
    provider = module.YahooFundamentalProvider(cache_dir=tmp_path)
    assert provider.get_company_metadata('AAA')['industry'] == 'Software'
    assert json.loads((folder/'metadata.json').read_text())['industry'] == 'Software'


def test_failed_metadata_is_not_cached(tmp_path, monkeypatch):
    class Ticker:
        def get_info(self):
            raise RuntimeError('Temporary source failure')
    monkeypatch.setattr(module, '_load_yfinance', lambda: SimpleNamespace(Ticker=lambda s: Ticker()))
    provider = module.YahooFundamentalProvider(cache_dir=tmp_path)
    assert provider.get_company_metadata('AAA')['industry'] is None
    assert not (tmp_path/'AAA/metadata.json').exists()
