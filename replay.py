import sys, runpy
import pandas as pd
from src.data.yahoo_provider import YahooMarketDataProvider as Y
import src.data.fundamental_provider as fund
import src.universe.us_equities as universe
def directory():
    return pd.read_csv('archived_outputs/universe_20261007_000000.csv',usecols=list(range(10)),keep_default_na=False)
def bars(self,s,interval):
    path=self._cache_path(s,interval=interval)
    if not path.exists(): return pd.DataFrame()
    frame=self._read_cache(path)
    return frame.loc[frame.index.strftime('%Y-%m-%d') <= '2026-10-07']
def batch(self,symbols,**kw):
    return {s:f for s in symbols if not (f:=bars(self,s,'1d')).empty}
class Missing:
    def Ticker(self,s): return self
def replay():
    universe.load_us_equity_directory=directory
    Y.get_daily=lambda self,s,**kw: bars(self,s,'1d')
    Y.get_hourly=lambda self,s,**kw: bars(self,s,'1h')
    Y.download_daily_batch=batch
    fund.YahooFundamentalProvider._fresh=staticmethod(lambda path,**kw:path.exists())
    fund._load_yfinance=lambda:Missing()
    sys.argv=sys.argv[1:]
    runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__': replay()
