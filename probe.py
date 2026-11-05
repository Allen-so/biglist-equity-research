from curl_cffi import requests
import pandas as pd,time,json
from pathlib import Path
out=Path('../../outputs/repair'); session=requests.Session(impersonate='chrome')
for s in 'MAX EVER DFIN FIGS HNGE SPY QQQ'.split():
 r=session.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{s}',params={'range':'1mo','interval':'1h','includePrePost':'false'},timeout=20); print(s,r.status_code,flush=True)
 if r.ok:
  (out/f'{s}-chart.json').write_bytes(r.content)
  x=r.json()['chart']['result'][0]; idx=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert('America/New_York'); print(idx[-8:],flush=True)
 else: print(r.text[:200]);break
 time.sleep(.5)
