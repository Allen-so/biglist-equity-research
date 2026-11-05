import json
import os
import time
from pathlib import Path
import pandas as pd
import pytest
from src.data.hourly_chart import chart_frame
from src.data.yahoo_provider import YahooMarketDataProvider
from src.data.fundamental_provider import YahooFundamentalProvider

def fixture():
    return json.loads((Path(__file__).parent/'fixtures'/'MAX-hourly.json').read_text())

def test_dst_and_completed_regular_sessions():
    frame=chart_frame(fixture(),'MAX',float('inf'))
    for date in ['2026-10-30','2026-11-03']:
        session=frame.loc[date]
        assert len(session)==7
        assert session.index[0].strftime('%H:%M')=='09:30'
        assert session.index[-1].strftime('%H:%M')=='15:30'
    cutoff=pd.Timestamp('2026-11-03 15:59',tz='America/New_York').timestamp()
    assert chart_frame(fixture(),'MAX',cutoff).index[-1].date()<pd.Timestamp('2026-11-03').date()

def test_incomplete_bar_and_api_error():
    x=fixture()
    x['chart']['result'][0]['indicators']['quote'][0]['close'][-2]=None
    assert chart_frame(x,'MAX',float('inf')).index[-1].day==2
    with pytest.raises(ValueError):
        chart_frame({'chart':{'error':'upstream failure'}},'MAX',time.time())

def test_wrong_date_in_fresh_cache(tmp_path,monkeypatch):
    p=YahooMarketDataProvider(cache_dir=tmp_path)
    bars=chart_frame(fixture(),'MAX',float('inf'))
    p._write_cache(bars.iloc[:-7],p._cache_path('MAX',interval='1h'))
    p.expected_session=pd.Timestamp('2026-11-03')
    monkeypatch.setattr('src.data.hourly_chart.fetch_hourly',lambda _:bars)
    assert p.get_hourly('MAX').index[-1]==bars.index[-1]
    monkeypatch.setattr('src.data.hourly_chart.fetch_hourly',lambda _:pytest.fail('fresh cache should be used'))
    assert len(p.get_hourly('MAX'))==len(bars)

def test_financial_cache_bounds(tmp_path):
    p=YahooFundamentalProvider(cache_dir=tmp_path)
    f=tmp_path/'statement.parquet'
    assert p._last_financial_snapshot(f,'MAX','API failure').empty
    pd.DataFrame({'revenue':[1]}).to_parquet(f)
    stamp=time.time()-30*86400
    os.utime(f,(stamp,stamp))
    assert p._last_financial_snapshot(f,'MAX','API failure').iloc[0,0]==1
    stamp-=16*86400
    os.utime(f,(stamp,stamp))
    assert p._last_financial_snapshot(f,'MAX','API failure').empty
