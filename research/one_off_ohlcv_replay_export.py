"""One-off data export for GPT Big List 26-layer diagnostic replay, NOT a stock selection model."""
from pathlib import Path
import pandas as pd, numpy as np, json,sys
from datetime import datetime
import yfinance as yf
ROOT=Path("diagnostic_files"); ROOT.mkdir(exist_ok=True)
history_files=list(Path("history").rglob("history_*.parquet"))
print("HISTORY_FILES",len(history_files),[x.name for x in history_files],flush=True)
if len(history_files)!=4:raise RuntimeError("Need four source OHLCV shards")
audits=list(Path("audit").rglob("all_deep_price_volume_evidence.csv"))
if not audits:audits=list(Path("audit").rglob("full_15_factor_100pt_scoring_evidence.csv"))
print("AUDITS",len(audits),[str(x) for x in audits],flush=True)
if not audits:raise RuntimeError("Need last scan complete candidate set")
targets=set(pd.read_csv(audits[0],usecols=["symbol"]).symbol.astype(str))
targets.update(["APPN","KD","CRGY","ARQT","FSLY","KNSA","PRCH","NXPI","KVYO"])
print("TARGETS",len(targets),flush=True)
parts=[]
for fname in history_files:
    frame=pd.read_parquet(fname)
    frame["symbol"]=frame.symbol.astype(str)
    sl=frame.loc[frame.symbol.isin(targets),["symbol","date","Open","High","Low","Close","Volume"]]
    print("DATA_PART",fname.name,"total",len(frame),"extracted",len(sl),flush=True)
    parts.append(sl)
data=pd.concat(parts,ignore_index=True)
data["date"]=pd.to_datetime(data.date).dt.tz_localize(None).dt.normalize()
data=data.loc[data.date<=pd.Timestamp("2026-11-11")]
if len(data)==0:raise RuntimeError("No histories extracted")
data.sort_values(["symbol","date"]).to_csv(ROOT/"replay_ohlcv_all_eligible.csv.gz",index=False,compression="gzip")
print("EXPORTED",len(data),"rows",data.symbol.nunique(),"symbols",flush=True)
benchmark_symbols=["SPY","QQQ","IWM","RSP","SMH","SOXX","XBI","XLK","XLF","XLE","XLI","XLB","XLV","XLP","XLU","XLY","XLC","HYG","LQD","TLT","IGV","XHB","KRE","VIXY"]
bf=[]
for k in range(0,len(benchmark_symbols),8):
    chunk=benchmark_symbols[k:k+8]
    x=yf.download(chunk,start="2026-01-01",end="2026-11-12",progress=False,threads=True,auto_adjust=False,group_by="ticker")
    for t in chunk:
        try:
            v=x[t].reset_index() if isinstance(x.columns,pd.MultiIndex) else x.reset_index()
            v["symbol"]=t
            v.rename(columns={"Date":"date"},inplace=True)
            v["date"]=pd.to_datetime(v["date"]).dt.tz_localize(None).dt.normalize()
            v=v.loc[(v.date<=pd.Timestamp("2026-11-11")) & (v.Close.notna()),["symbol","date","Open","High","Low","Close","Volume"]]
            if len(v)>=60:bf.append(v)
            print("BENCHMARK",t,len(v),v.date.max() if len(v) else None,flush=True)
        except Exception as e:print("BENCHMARK_FAIL",t,repr(e),flush=True)
if bf:pd.concat(bf,ignore_index=True).to_csv(ROOT/"replay_benchmarks.csv.gz",index=False,compression="gzip")
if not bf or not any((x.symbol=="SPY").any() for x in bf):
    raise RuntimeError("SPY benchmark missing for true Stress RS")
m={"target_eligible_count":len(targets),"ohlcv_symbols":int(data.symbol.nunique()),
   "ohlcv_rows":len(data),"benchmark_symbols":[t for v in bf for t in v.symbol.unique()],
   "asof":"2026-11-11"}
(ROOT/"replay_metadata.json").write_text(json.dumps(m,indent=2))
print("SUCCESS",m,flush=True)
