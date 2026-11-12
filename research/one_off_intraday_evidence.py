"""One-off targeted 1h chart evidence for GPT Big List analyst; not a score model."""
import yfinance as yf,pandas as pd,datetime as dt,time,os,json
from pathlib import Path
ROOT=Path("intraday_evidence");ROOT.mkdir(exist_ok=True)
TICKERS=["KVYO","COCO","PRCH","SG","RCEL","FIGS","APPN","INTA","PAY","TYL","BLKB","DXC","ACN","LQDA","GRNT","CLBT","TDOC","NXPI","BMI","FTDR","DKS","RDWR","TENB","OLMA","GLOB","PLSE","CNXC","SITE","TIMB","ICLR","MYGN","BB","DOCS","OLLI","CELH","RRX","KD"]
END="2026-11-12";START="2026-10-01"
statuses=[];frames=[]
for i in range(0,len(TICKERS),5):
 chunk=TICKERS[i:i+5]
 try:
  d=yf.download(chunk,start=START,end=END,interval="60m",auto_adjust=False,group_by="ticker",
                progress=False,threads=True,timeout=30)
  for sym in chunk:
   try:
    x=d[sym].copy() if isinstance(d.columns,pd.MultiIndex) else d.copy()
    x=x.dropna(subset=["Close"])
    x.index=pd.to_datetime(x.index)
    if x.index.tz is not None:x.index=x.index.tz_convert("America/New_York")
    x=x.loc[(x.index.tz_localize(None)<pd.Timestamp("2026-11-12"))]
    x=x.reset_index()
    x.rename(columns={x.columns[0]:"bar_datetime_eastern"},inplace=True)
    x["symbol"]=sym
    if len(x)>40:frames.append(x[["symbol","bar_datetime_eastern","Open","High","Low","Close","Volume"]])
    statuses.append({"symbol":sym,"bars":len(x),"last":str(x.bar_datetime_eastern.max() if len(x) else None)})
    print("BAR_RESULT",sym,len(x),"firstlast",str(x.bar_datetime_eastern.min() if len(x) else None),str(x.bar_datetime_eastern.max() if len(x) else None),flush=True)
   except Exception as e:
    statuses.append({"symbol":sym,"bars":0,"error":str(e)[:140]}); print("SYM_ERR",sym,str(e)[:160],flush=True)
 except Exception as e:
  print("GROUP_ERR",chunk,str(e)[:150],flush=True)
 time.sleep(1.5)
if frames:pd.concat(frames,ignore_index=True).to_csv(ROOT/"targeted_1h_ohlcv.csv.gz",index=False,compression="gzip")
(ROOT/"intraday_coverage.json").write_text(json.dumps(statuses,indent=2))
print("TOTAL",len(statuses),"VALID",sum(x['bars']>40 for x in statuses),"ROWS",sum(x['bars'] for x in statuses),flush=True)
