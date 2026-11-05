"""Read Yahoo regular-session hourly chart bars with their exchange schedule."""
import threading
import time
from urllib.parse import quote
import pandas as pd
from curl_cffi import requests

_lock = threading.Lock()
_next_request = 0.0

def chart_frame(payload, symbol, now):
    chart = payload['chart']
    if chart.get('error'):
        raise ValueError(str(chart['error']))
    data = chart['result'][0]
    meta = data['meta']
    if meta['symbol'].upper() != symbol.upper() or meta['dataGranularity'] not in ('1h','60m'):
        raise ValueError('Unexpected chart symbol or interval')
    index = pd.to_datetime(data['timestamp'], unit='s', utc=True)
    frame = pd.DataFrame(data['indicators']['quote'][0], index=index)
    sessions = meta['tradingPeriods']
    if isinstance(sessions, dict):
        sessions = sessions['regular']
    slots = []
    for group in sessions:
        for period in group:
            if period['end'] <= now:
                slots.extend(range(period['start'], period['end'], 3600))
    expected = pd.to_datetime(slots, unit="s", utc=True)
    if len(expected) == 0:
        raise ValueError("No completed Yahoo trading session")
    frame = frame.reindex(pd.to_datetime(slots, unit='s', utc=True)).dropna(subset=['open','high','low','close','volume'])
    last_session = expected[-1].tz_convert("America/New_York").date()
    required = expected[expected.tz_convert("America/New_York").date == last_session]
    if not required.isin(frame.index).all():
        frame = frame.loc[frame.index < required[0]]
    frame.index = frame.index.tz_convert('America/New_York').tz_localize(None)
    return frame.sort_index()

def fetch_hourly(symbol):
    global _next_request
    with _lock:
        time.sleep(max(0, _next_request - time.monotonic()))
        _next_request = time.monotonic() + .45
    response = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe=''),
        params={'range':'60d','interval':'1h','includePrePost':'false'},impersonate='chrome',timeout=25)
    if response.status_code == 429:
        exc = RuntimeError("Yahoo chart HTTP 429")
        exc.response = response
        raise exc
    response.raise_for_status()
    return chart_frame(response.json(),symbol,time.time())
