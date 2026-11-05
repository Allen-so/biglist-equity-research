import time
import multiprocessing
from pathlib import Path
import pandas as pd
import pytest
from src.data import yahoo_provider as y
from src.data.coverage import require_coverage

def hang():
    time.sleep(600)

def test_hard_timeout_reaps():
    before={p.pid for p in multiprocessing.active_children()}
    start=time.monotonic()
    with pytest.raises(TimeoutError,match='TIMEOUT'):
        y._bounded_call(hang,wall_seconds=.4)
    elapsed=time.monotonic()-start
    assert elapsed<4
    assert {p.pid for p in multiprocessing.active_children()}<=before
    print('hard timeout elapsed',elapsed)

@pytest.mark.parametrize('cached',[95,99,100])
def test_remaining_gaps_all_attempted(tmp_path,monkeypatch,cached):
    p=y.YahooMarketDataProvider(cache_dir=tmp_path)
    symbols=[str(i) for i in range(100)]
    frame=pd.DataFrame({'Open':[1.],'High':[1.],'Low':[1.],'Close':[1.],'Volume':[1.]},index=pd.to_datetime(['2026-11-04']))
    monkeypatch.setattr(p,'_daily_hit',lambda path:int(path.stem)<cached)
    monkeypatch.setattr(p,'_read_cache',lambda path:frame)
    monkeypatch.setattr(p,'_write_cache',lambda *a:None)
    monkeypatch.setattr(y.clock,'sleep',lambda _:None)
    calls=[]
    def download(call,**kw):
        calls.extend(kw['tickers']);return frame
    monkeypatch.setattr(y,'_bounded_call',download)
    result=p.download_daily_batch(symbols,batch_size=1)
    assert calls==symbols[cached:]
    assert len(result)==100
    print('coverage',cached,'attempted',calls)

@pytest.mark.parametrize('cached',[94,95,99])
def test_429_stops_preserves_fixed_pool(tmp_path,monkeypatch,cached):
    p=y.YahooMarketDataProvider(cache_dir=tmp_path)
    monkeypatch.setattr(p,'_daily_hit',lambda path:int(path.stem)<cached)
    monkeypatch.setattr(p,'_read_cache',lambda path:pd.DataFrame({'close':[1]}))
    calls=[]
    def limited(*args,**kw):
        calls.append(kw['tickers']);raise RuntimeError('HTTP 429')
    monkeypatch.setattr(y,'_bounded_call',limited)
    result=p.download_daily_batch(map(str,range(100)),batch_size=1)
    assert len(calls)==1 and len(result)==cached
    assert (tmp_path/'cooldown.json').exists()
    if cached<95:
        with pytest.raises(RuntimeError):require_coverage(dict(pool_total=100,valid_daily=len(result)))
    else:require_coverage(dict(pool_total=100,valid_daily=len(result)))

@pytest.mark.parametrize('count,passes',[(9499,False),(9500,True)])
def test_exact_boundary(count,passes):
    if passes:require_coverage(dict(pool_total=10000,valid_daily=count))
    else:
        with pytest.raises(RuntimeError):require_coverage(dict(pool_total=10000,valid_daily=count))

def test_budget_bounds_remaining_attempts(tmp_path,monkeypatch):
    p=y.YahooMarketDataProvider(cache_dir=tmp_path)
    monkeypatch.setenv('BIGLIST_COVERAGE_REQUESTS','2')
    monkeypatch.setattr(p,'_daily_hit',lambda path:int(path.stem)<95)
    monkeypatch.setattr(p,'_read_cache',lambda path:pd.DataFrame({'close':[1]}))
    monkeypatch.setattr(y.clock,'sleep',lambda _:None)
    calls=[]
    def empty(*args,**kw):calls.extend(kw['tickers']);return pd.DataFrame()
    monkeypatch.setattr(y,'_bounded_call',empty)
    assert len(p.download_daily_batch(map(str,range(100)),batch_size=1))==95
    assert calls==['95','96']

def test_timeout_preserves_checkpoint_and_audit(tmp_path,monkeypatch):
    from src.data.coverage import assess
    p=y.YahooMarketDataProvider(cache_dir=tmp_path)
    p.expected_session=pd.Timestamp('2026-11-04')
    dates=pd.bdate_range(end=p.expected_session,periods=60)
    frame=pd.DataFrame({c:[1.]*60 for c in ['open','high','low','close','volume']},index=dates)
    p._write_cache(frame,p._cache_path('GOOD',interval='1d'))
    receipt=y.checkpoint.receipt_path(p._cache_path('GOOD',interval='1d'),p.expected_session)
    before=receipt.read_bytes()
    def timeout(*args,**kw):raise TimeoutError('TIMEOUT')
    monkeypatch.setattr(y,'_bounded_call',timeout)
    data=p.download_daily_batch(['GOOD','BAD'])
    valid,audit,summary=assess(['GOOD','BAD'],data,p)
    assert list(valid)==['GOOD'] and receipt.read_bytes()==before
    assert audit.set_index('ticker').loc['BAD','status']=='TIMEOUT'
    progress=__import__('json').loads((tmp_path/'progress/2026-11-04/daily.json').read_text())
    assert progress['complete']==1 and progress['failures']=={'BAD':'TIMEOUT'}

def test_daily_hang_preserves_latest_and_releases_lock(tmp_path):
    import shutil,subprocess,sys,os,fcntl
    root=Path(y.__file__).parents[2]
    for name in ['Logs','Reports']: (tmp_path/name).mkdir()
    (tmp_path/'Reports/Latest').symlink_to(root/'Reports/Latest')
    shutil.copyfile(root/'daily.py',tmp_path/'daily.py')
    script=tmp_path/'run_daily.sh';script.write_text('#!/bin/sh\nsleep 600\n');script.chmod(0o755)
    started=time.monotonic()
    run=subprocess.run([sys.executable,'daily.py'],cwd=tmp_path,env={**os.environ,'BIGLIST_RUN_TIMEOUT':'.5'},capture_output=True,timeout=5)
    assert run.returncode!=0 and time.monotonic()-started<5
    assert b'TIMEOUT' in run.stdout
    assert (tmp_path/'Reports/Latest').resolve()==(root/'Reports/Latest').resolve()
    with (tmp_path/'Logs/run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
