#!/usr/bin/env python3
"""Experimental full-universe implementation of GPT Big List Formal Baseline v1.0.

No user-mentioned tickers or previous lists are used to construct or rank the universe.
Fixed factor weights. Explicit missing-data and 26-step audit. This is a newly
coded operationalization, NOT the user's separate local Big List engine.
"""
import io
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
ASOF = pd.Timestamp("2026-11-11")
START = "2025-08-01"
END = "2026-11-12"
OUT = Path("gpt_biglist_outputs")
OUT.mkdir(exist_ok=True)
WEIGHTS = {
    "Market Fit":4, "Theme Fit":3, "Structure Quality":8,
    "Monster Quality":7, "Volatility":7, "Historical Velocity":8,
    "Productive Volatility":8, "Supply / Demand":7,
    "Progress Retention":7, "Persistent RS":10,
    "Rank Level + Rank Delta":6, "Within-Theme Leadership":5,
    "Fundamental Delta / Catalyst":5, "Setup Quality":10,
    "Tradability":5
}
assert sum(WEIGHTS.values()) == 100
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (compatible; GPTBigListResearch/1.0)"
BE = ["SPY","QQQ","IWM","RSP","SMH","^VIX","^TNX","XLK","XLV","XLF","XLI","XLE","XLY","XLP","XLU","XLC","XLB"]
AUDIT = []

def report_step(i, title, count, note=""):
    entry={"step":i,"title":title,"count":None if count is None else int(count),"note":note}
    AUDIT.append(entry)
    print("STEP",i,title,"count",count,note,flush=True)

def load_universe():
    print("UNIVERSE_SOURCE marketparquet Nov11",flush=True)
    u="https://marketparquet.com/api/data/download/stock_daily/2026-11-11.parquet"
    r=SESSION.get(u,timeout=80)
    r.raise_for_status()
    x=pd.read_parquet(io.BytesIO(r.content))
    assert len(x)>5000, "MarketParquet coverage unexpectedly small"
    x=x[x.asset_type.astype(str).str.lower()=="stock"].copy()
    x=x.sort_values("symbol").drop_duplicates("symbol").set_index("symbol",drop=False)
    meta_url="https://raw.githubusercontent.com/zyhe16/top-us-stock-tickers/main/data/v2/tickers.csv"
    try:
        m=pd.read_csv(io.BytesIO(SESSION.get(meta_url,timeout=90).content))
        m=m.drop_duplicates("symbol").set_index("symbol")
        for c in ["name","sector","industry","country"]:
            if c in m.columns:
                x[c]=x["symbol"].map(m[c].to_dict())
        print("METADATA",len(m),"fields",list(m.columns),flush=True)
    except Exception as e:
        print("METADATA_FAILED",str(e)[:200],flush=True)
    if "name" not in x: x["name"]=""
    if "sector" not in x: x["sector"]="Unknown"
    if "industry" not in x: x["industry"]=""
    x["name"]=x["name"].fillna("").astype(str)
    x["sector"]=x["sector"].fillna("Unknown").astype(str)
    x["industry"]=x["industry"].fillna("").astype(str)
    # Exclusions use security TYPE and instrument-name patterns, never theme/size/old list.
    bad=re.compile(r"\b(warrants?|rights?|preferred|depositary shares representing preferred|senior notes|notes due|units? (?:consisting|comprising|of beneficial)|etns?|etfs?|exchange traded funds?|closed.end funds?|acquisition (?:corp|corporation|co|company|ltd)|blank.check)\b",re.I)
    x["scope_reason"]=np.where(x["name"].str.contains(bad,regex=True),"NON_COMMON_OR_SPAC_SHELL","PASS_STOCK_TYPE")
    # Symbols without third-party names remain eligible and are disclosed as unclassified.
    x.to_csv(OUT/"universe_all_asof.csv",index=False)
    x=x[x.scope_reason=="PASS_STOCK_TYPE"].copy()
    report_step(0,"Scope and fresh full-universe reset",len(x),"All matching traded stock symbols; no prior watchlists, size or themes")
    return x

def get_history(syms, batch=50):
    histories={}
    status={}
    began=time.time()
    for start in range(0,len(syms),batch):
        b=syms[start:start+batch]
        y=[s.replace(".","-") for s in b]
        try:
            d=yf.download(y,start=START,end=END,auto_adjust=False,group_by="ticker",
                          progress=False,threads=5,timeout=25,repair=False)
        except Exception as e:
            print("DOWNLOAD_BLOCK_ERR",start,str(e)[:160],flush=True)
            d=pd.DataFrame()
        for ticker,yahoo_symbol in zip(b,y):
            try:
                if isinstance(d.columns,pd.MultiIndex):
                    if yahoo_symbol in d.columns.get_level_values(0):
                        z=d[yahoo_symbol].copy()
                    elif yahoo_symbol in d.columns.get_level_values(1):
                        z=d.xs(yahoo_symbol,level=1,axis=1).copy()
                    else:
                        raise KeyError("no ticker columns")
                elif len(b)==1:
                    z=d.copy()
                else:
                    raise KeyError("no multiindex")
                if not set(["Open","High","Low","Close","Volume"]).issubset(z.columns):
                    raise KeyError("OHLCV missing")
                z=z[["Open","High","Low","Close","Volume"]].copy()
                z.index=pd.to_datetime(z.index).tz_localize(None).normalize()
                z=z.loc[z.index<=ASOF].dropna(subset=["Close","High","Low","Volume"])
                z=z[z.Close>0]
                z=z[~z.index.duplicated(keep="last")].sort_index()
                if len(z)<80:
                    status[ticker]="INSUFFICIENT_80_BARS"
                    continue
                if z.index[-1] < ASOF:
                    status[ticker]="STALE_HISTORY"
                    continue
                histories[ticker]=z
                status[ticker]="AVAILABLE"
            except Exception as e:
                status[ticker]=type(e).__name__
        if start%500==0 or start+batch>=len(syms):
            print("HISTORY_PROGRESS",start+len(b),"/",len(syms),
                  "success",len(histories),"seconds",round(time.time()-began),flush=True)
    return histories,status

def price_and_volume_features(k,z,spy,regime):
    c=z.Close.astype(float); h=z.High.astype(float); l=z.Low.astype(float); v=z.Volume.astype(float)
    p=float(c.iloc[-1]);n=len(z)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=float(tr.tail(20).mean())
    atr=max(0.005,atr)
    ref=spy.reindex(c.index).ffill()
    daily=c.pct_change()
    spr=ref.pct_change()
    def rt(w,shift=0):
        if n<=w+shift:return np.nan
        return float(c.iloc[-1-shift]/c.iloc[-1-w-shift]-1)
    def ma(w):
        return float(c.tail(w).mean()) if n>=w else np.nan
    def oldrs(w,shift=0):
        if n<=w+shift:return np.nan
        sr=(c.iloc[-1-shift]/c.iloc[-1-w-shift]-1)
        try:
            s0=float(ref.iloc[-1-shift]); s1=float(ref.iloc[-1-w-shift])
            return float(sr-(s0/s1-1))
        except Exception:return np.nan
    downmask=(spr<0)&spr.notna()
    stress=(daily[downmask]-spr[downmask]).tail(30)
    upv=float(v[daily>0].tail(30).sum())
    dnv=float(v[daily<0].tail(30).sum())
    vol20=float(v.tail(20).mean())
    volprev=float(v.iloc[-20:-5].mean()) if n>=25 else vol20
    volume5=float(v.tail(5).mean())
    range5=float((h-l).tail(5).mean())
    range_prev=float((h-l).iloc[-20:-5].mean()) if n>=20 else range5
    hl=(float(l.tail(8).min())>float(l.iloc[-24:-8].min())) if n>=25 else False
    m10,m20,m50,m200=ma(10),ma(20),ma(50),ma(200)
    prev20_high=float(h.iloc[-21:-1].max())
    high52=float(h.tail(252).max())
    low60=float(l.tail(60).min())
    high60=float(h.tail(60).max())
    high20=float(h.tail(20).max())
    low20=float(l.tail(20).min())
    rise60=(p/low60-1) if low60>0 else 0
    retention=np.clip((p-low60)/max(1e-6,high60-low60),0,1)
    constructive=bool(p>m20 and (np.isnan(m50) or p>m50))
    failure=(n>=200 and p<m50 and p<m200 and m50<m200 and rt(20)<-.07 and rt(60)<-.12)
    partial_repair=(p>m20 and rt(10)>0 and (daily.tail(10).sum()>0))
    broken=bool(failure and not partial_repair)
    compressed=(range5/max(.01,range_prev)<.87 and volume5/max(1,volprev)<.88)
    rvol=float(v.iloc[-1]/max(1,vol20))
    expansion=bool((rise60>.16 or rt(20)>.12) and (v.tail(30).max()/max(1,vol20)>1.6))
    # Pivot is the prior 20-session high, not the current high.
    dist_pivot=(prev20_high-p)/p
    pivot_atr=(p-prev20_high)/atr
    ext20=(p-m20)/atr if m20>0 else np.nan
    # Prior completed supports; use closest *genuine* prior 10D support below price.
    candidates=[float(l.tail(10).min()),m20,m50,low20]
    candidates=[q for q in candidates if np.isfinite(q) and q<p]
    support=max(candidates) if candidates else np.nan
    stop=(support-.12*atr) if np.isfinite(support) else np.nan
    stop_r=(p-stop) if np.isfinite(stop) and stop<p else np.nan
    stop_atr=stop_r/atr if np.isfinite(stop_r) else np.nan
    # Measured overhead resistance; if at price discovery, only an explicit ATR scenario,
    # not a factual target, is available.
    overhead=(high52-p) if high52>p else 4*atr
    rr=(overhead/stop_r) if np.isfinite(stop_r) and stop_r>0 else np.nan
    ready=(constructive and -.4<=(prev20_high-p)/atr<=.8 and
           np.isfinite(ext20) and ext20<2.2 and
           (compressed or hl or (p>prev20_high and rvol>=1.4)))
    developing=(constructive and hl and .8<(prev20_high-p)/atr<=1.8 and
                (compressed or volume5<volprev))
    actionable=bool((ready or developing) and
                    np.isfinite(stop_atr) and .55<=stop_atr<=2.5 and
                    np.isfinite(rr) and rr>=3.0 and
                    np.isfinite(ext20) and ext20<2.5 and
                    p>=4.0 and float((c*v).tail(20).median())>=8e6)
    pureext=bool(np.isfinite(ext20) and (ext20>2.5 or pivot_atr>2.5))
    if expansion and compressed and hl:
        setup="ECR/VCP"
    elif ready:
        setup="READY_PULLBACK_OR_PIVOT"
    elif developing:
        setup="DEVELOPING_HL"
    elif (np.isfinite(m50) and p>m50 and c.iloc[-2]<m50):
        setup="50MA_RECLAIM"
    else:setup="NO_ACTIONABLE_SETUP"
    maxrun=float((c/c.cummin()-1).tail(120).max())
    step15=float((daily.tail(20).abs().sum()))
    progress=abs(rt(20))/max(1e-8,step15) if np.isfinite(rt(20)) else 0
    distribution=int(((daily<-.025)&(v>v.rolling(20).mean())).tail(30).sum())
    return dict(symbol=k,bars=n,asof=str(z.index[-1].date()),price=p,volume=float(v.iloc[-1]),
        dollarvol20=p*vol20,median_dollarvol20=float((c*v).tail(20).median()),atr20=atr,adr20=float(((h-l)/c.shift(1)).tail(20).mean()),
        rvol=rvol,ma10=m10,ma20=m20,ma50=m50,ma200=m200,
        ret5=rt(5),ret10=rt(10),ret20=rt(20),ret60=rt(60),ret120=rt(120),ret250=rt(250),
        old_rs20=oldrs(20,20),old_rs60=oldrs(60,20),old_rs120=oldrs(120,20),
        rs5=oldrs(5),rs20=oldrs(20),rs60=oldrs(60),rs120=oldrs(120),
        stress_rs=float(stress.mean()) if len(stress)>=3 else np.nan,
        upvol_ratio=upv/max(1,upv+dnv),distribution=distribution,
        r5_range_ratio=range5/max(.01,range_prev),v5_ratio=volume5/max(1,volprev),
        high52=high52,high60=high60,low60=low60,high20=high20,low20=low20,
        retention=float(retention),maxrun=maxrun,productivity=progress,
        higher_low=hl,compressed=compressed,expansion=expansion,
        broken=broken,structure_pass=not broken,constructive=constructive,
        pivot=prev20_high,dist_pivot=dist_pivot,pivot_atr=pivot_atr,
        ext20atr=ext20,support=support,stop=stop,stop_atr=stop_atr,
        rr=float(rr) if np.isfinite(rr) else np.nan,
        pure_extended=pureext,ready=bool(ready),developing=bool(developing),
        actionable=actionable,setup=setup,
        channel="MATURE" if rt(60)>.22 and retention>.55 and p>m50 else
                "NEW_MONSTER" if expansion and hl else "UNDETERMINED")

def pct(s, neutral=0):
    return pd.to_numeric(s,errors="coerce").rank(pct=True).fillna(neutral)

def clip01(s,low,high):
    return ((pd.to_numeric(s,errors="coerce")-low)/(high-low)).clip(0,1).fillna(0)

def score_all(frame,regime):
    f=frame.copy()
    now=(.15*pct(f.rs5)+.30*pct(f.rs20)+.35*pct(f.rs60)+.20*pct(f.rs120))
    old=(.20*pct(f.old_rs20)+.50*pct(f.old_rs60)+.30*pct(f.old_rs120))
    f["rank_level"]=now.rank(pct=True)
    f["rank_delta"]=now-old
    # Sector return ranks and within-sector strength are computed from ALL covered companies.
    sec=f.groupby("sector")["ret20"].median()
    f["sector20"]=f["sector"].map(sec)
    f["theme_rank"]=f.groupby("sector")["rs60"].rank(pct=True).fillna(0)
    f["factor_Market Fit"]=clip01(f.stress_rs,-.004,.009)*.6+clip01(f.rs20,-.1,.16)*.4
    f["factor_Theme Fit"]=pct(f.sector20)*.55+pct(f.groupby("sector")["ret60"].transform("median"))*.45
    f["factor_Structure Quality"]=(clip01(f.price/f.ma50,.95,1.15)*.35+
       clip01(f.ma50/f.ma200,.94,1.13)*.25+clip01(f.retention,.3,.9)*.2+
       f.higher_low.astype(float)*.2)
    f["factor_Monster Quality"]=pct(f.maxrun)*.4+pct(f.retention)*.35+f.constructive.astype(float)*.25
    f["factor_Volatility"]=(clip01(f.adr20,.014,.055)*.7+
        (1-clip01(f.adr20,.12,.25))*.3)
    f["factor_Historical Velocity"]=(pct(f.ret5)*.20+pct(f.ret20)*.35+pct(f.ret60)*.30+pct(f.ret120)*.15)
    f["factor_Productive Volatility"]=clip01(f.productivity,.08,.45)*.55+clip01(f.retention,.2,.9)*.45
    f["factor_Supply / Demand"]=(clip01(f.upvol_ratio,.35,.75)*.5+
        clip01(f.rvol,.7,2.2)*.25+
        (1-clip01(f.distribution,1,6))*.25)
    f["factor_Progress Retention"]=clip01(f.retention,.2,.9)*.75+f.higher_low.astype(float)*.25
    f["factor_Persistent RS"]=(.85*(pct(f.rs5)*.20+pct(f.rs20)*.35+pct(f.rs60)*.45)+
        .15*clip01(f.stress_rs,-.005,.009))
    f["factor_Rank Level + Rank Delta"]=f.rank_level*.6+clip01(f.rank_delta,-.30,.30)*.4
    f["factor_Within-Theme Leadership"]=f.theme_rank
    f["factor_Fundamental Delta / Catalyst"]=0.0
    f["factor_Setup Quality"]=(f.higher_low.astype(float)*.20+
        f.compressed.astype(float)*.25+
        clip01(1-(f.dist_pivot.abs()/.10),0,1)*.30+
        (f.ready|f.developing).astype(float)*.25)
    f["factor_Tradability"]=(clip01(f.rr,1,4)*.4+
        (1-clip01(f.ext20atr,1,3))*.2+
        f.actionable.astype(float)*.4)
    for k in WEIGHTS:
        f["score_"+k]=(f["factor_"+k]*WEIGHTS[k]).round(3)
    f["score_base"]=sum(f["score_"+k] for k in WEIGHTS)
    return f

def fundamental_delta(t):
    try:
        info=yf.Ticker(t.replace(".","-")).info
        if not isinstance(info,dict):raise ValueError("no info")
        rev=info.get("revenueGrowth")
        eps=info.get("earningsGrowth")
        mar=info.get("operatingMargins")
        cash=info.get("totalCash")
        debt=info.get("totalDebt")
        name=info.get("longName") or info.get("shortName")
        industry=info.get("industry")
        data_count=sum(isinstance(x,(float,int)) and np.isfinite(x) for x in [rev,eps,mar])
        if data_count==0: return 0.0,"NO_VERIFIED_FUNDAMENTAL_DELTA",industry,name
        score=(min(1,max(0,(rev or 0)/.30))*.5+
               min(1,max(0,(eps or 0)/.7))*.30+
               min(1,max(0,((mar or 0)+.10)/.30))*.2)
        return float(score),"YAHOO_INFO",industry,name
    except Exception as e:
        return 0.0,"FUND_DATA_BLOCKED_"+type(e).__name__,None,None

def classify_theme(row):
    ind=str(row.get("industry","")).lower()
    sec=str(row.get("sector","Unknown")).lower()
    sym=str(row.symbol)
    # Theme taxonomy is applied AFTER scoring and final cut; never affects candidate admission.
    if any(x in ind for x in ["biotech","pharma","biological"]):return "Biotech / Pharma"
    if any(x in ind for x in ["semiconductor","integrated circuit"]):return "Semiconductors / Equipment"
    if any(x in ind for x in ["software","application","saas"]):return "Software / Application"
    if any(x in ind for x in ["oil","gas","energy"]):return "Energy / Oilfield"
    if any(x in ind for x in ["bank","capital market","financial","insurance"]):return "Financials / Fintech"
    if any(x in ind for x in ["aerospace","defense"]):return "Aerospace / Defense"
    if "health" in sec:return "Healthcare"
    if "technology" in sec:return "Technology / Hardware"
    if "industrial" in sec:return "Industrials"
    if "consumer" in sec:return "Consumer"
    if "finance" in sec:return "Financials / Fintech"
    if "energy" in sec:return "Energy / Oilfield"
    if "utility" in sec:return "Utilities / Power"
    return str(row.get("sector","Other"))

def run():
    started=time.time()
    uni=load_universe()
    stage=os.environ.get("BIGLIST_STAGE","full")
    syms=uni.symbol.tolist()
    if stage=="download":
        shard=int(os.environ.get("BIGLIST_SHARD","0"))
        n_shards=int(os.environ.get("BIGLIST_N_SHARDS","4"))
        partition=syms[shard::n_shards]
        print("SHARD",shard,"/",n_shards,"INPUT",len(partition),flush=True)
        hist,status=get_history(partition)
        parts=[]
        for symbol,z in hist.items():
            q=z.reset_index().rename(columns={z.index.name or "index":"date"})
            q["symbol"]=symbol
            parts.append(q[["symbol","date","Open","High","Low","Close","Volume"]])
        if parts:
            pd.concat(parts,ignore_index=True).to_parquet(OUT/("history_%d.parquet"%shard),index=False)
        with (OUT/("history_status_%d.json"%shard)).open("w") as o:json.dump(status,o)
        print("SHARD_COMPLETE",shard,"input",len(partition),"downloaded",len(hist),
              "failed",len(partition)-len(hist),flush=True)
        return
    report_step(1,"Market regime inputs",None,"SPY QQQ IWM RSP VIX TNX and sector ETFs")
    rep= yf.download(BE,start=START,end=END,auto_adjust=False,progress=False,
                     group_by="ticker",threads=8,timeout=25)
    ben={}
    for b in BE:
        try:
            z=rep[b]["Close"].dropna()
            z.index=pd.to_datetime(z.index).tz_localize(None).normalize()
            ben[b]=z.loc[z.index<=ASOF]
        except Exception:ben[b]=pd.Series(dtype=float)
    # Yahoo's shared SQLite timezone cache can transiently lock when multi-threading benchmarks.
    # Retry each required benchmark on its own, with threads disabled.
    for b in BE:
        v=ben.get(b,pd.Series(dtype=float))
        if len(v)>=80 and v.index[-1]>=ASOF:continue
        try:
            d1=yf.download(b,start=START,end=END,auto_adjust=False,progress=False,
                           threads=False,timeout=25,group_by="ticker")
            if isinstance(d1.columns,pd.MultiIndex):
                if b in d1.columns.get_level_values(0):d1=d1[b]
                else:d1=d1.xs(b,level=1,axis=1)
            q=d1["Close"].dropna()
            q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
            ben[b]=q.loc[q.index<=ASOF]
            print("BENCHMARK_RETRY",b,len(ben[b]),flush=True)
        except Exception as e:print("BENCHMARK_ERROR",b,repr(e)[:180],flush=True)
    if len(ben.get("SPY",[]))<80 or ben["SPY"].index[-1] <ASOF:
        raise RuntimeError("SPY benchmark not available as of final session")
    spy=ben["SPY"]
    if stage=="score":
        parts=sorted(OUT.glob("history_*.parquet"))
        states=sorted(OUT.glob("history_status_*.json"))
        print("MERGE_PARTITIONS",len(parts),"status_files",len(states),flush=True)
        if len(parts)<4 or len(states)<4:
            raise RuntimeError("Not all four full-market shards are present")
        panel=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
        panel["date"]=pd.to_datetime(panel.date).dt.tz_localize(None).dt.normalize()
        hist={}
        for symbol,z in panel.groupby("symbol",sort=False):
            hist[symbol]=z.set_index("date")[["Open","High","Low","Close","Volume"]].sort_index()
        status={}
        for p in states:
            with p.open() as q:status.update(json.load(q))
        if len(status)!=len(uni):
            raise RuntimeError("Incomplete union of shard status: %d/%d"%(len(status),len(uni)))
    else:
        hist,status=get_history(syms)
    covered=len(hist)
    report_step(2,"Data coverage / Regime capacity",covered,"History >=80 and fresh Nov11; subsequent gates only on covered symbols")
    cov=pd.DataFrame({"symbol":list(status),"history_status":[status[s] for s in status]})
    cov=cov.merge(uni[["symbol","name","sector"]].reset_index(drop=True),on="symbol",how="left")
    cov.to_csv(OUT/"data_coverage.csv",index=False)
    feat=[]
    for j,(s,z) in enumerate(hist.items()):
        try:
            d=price_and_volume_features(s,z,spy,"MIXED")
            d["sector"]=uni.loc[s,"sector"]
            d["name"]=uni.loc[s,"name"]
            d["industry"]=uni.loc[s,"industry"]
            feat.append(d)
        except Exception as e:
            print("FEATURE_BLOCKED",s,type(e).__name__,str(e)[:100],flush=True)
    df=pd.DataFrame(feat)
    if df.empty:raise RuntimeError("No usable history")
    breadth=float((df.price>df.ma50).mean())
    def above(b):
        c=ben.get(b,pd.Series(dtype=float))
        return bool(len(c)>=50 and c.iloc[-1]>c.tail(50).mean())
    vix=float(ben["^VIX"].iloc[-1]) if len(ben["^VIX"]) else np.nan
    risk_on=above("SPY") and above("QQQ") and breadth>.55 and (np.isnan(vix) or vix<23)
    risk_off=(not above("SPY") and not above("QQQ") and breadth<.35)
    regime="RISK_ON" if risk_on else "DEFENSIVE" if risk_off else "SELECTIVE_MIXED"
    capacity=40 if risk_on else 15 if risk_off else 25
    minimum=75 if risk_on else 82 if risk_off else 80
    print("REGIME",regime,"CAPACITY",capacity,"THRESHOLD",minimum,
          "breadth50",round(breadth,3),"VIX",vix,flush=True)
    report_step(3,"Theme and rotation map",len(df),"All priced stocks, no thematic pre-selection")
    report_step(4,"Existing Leader channel",int((df.channel=="MATURE").sum()),"Evaluated on all priced stocks")
    report_step(5,"New Monster channel",int((df.channel=="NEW_MONSTER").sum()),"Evaluated on all priced stocks")
    df["hard_fail"]=df.broken.astype(bool)
    report_step(6,"Stage4/broken structure hard gate",int((~df.hard_fail).sum()),"No narrative override for broken charts")
    for i,name in [(7,"Market Fit"),(8,"Theme Fit"),(9,"Structure Quality"),(10,"Monster Quality"),(11,"Volatility"),
        (12,"Historical Velocity"),(13,"Productive Volatility"),(14,"Velocity Decay"),
        (15,"Supply / Demand"),(16,"Progress Retention"),(17,"Persistent RS"),
        (18,"Rank Level + Rank Delta"),(19,"Within-Theme Leadership"),(20,"Fundamental Delta / Catalyst")]:
        report_step(i,name,len(df),"Computations on all covered symbols; fundamental verification on potential passes after fixed-factor pre-scoring" if i==20 else "Cross-sectional calculation, no manual ticker selection")
    f=score_all(df,regime)
    f["tech_possible"]=((f.score_base+WEIGHTS["Fundamental Delta / Catalyst"]>=minimum)&
        (~f.hard_fail)&f.actionable&(~f.pure_extended)&(f.channel!="UNDETERMINED"))
    # Only retrieve slow company fundamentals for stocks that can still mathematically PASS
    # the full 100-point model. This does not select prior to cross-sectional scoring.
    may=f[f.tech_possible].sort_values("score_base",ascending=False).copy()
    print("FUNDAMENTAL_ELIGIBLE",len(may),flush=True)
    info_status={}
    for idx,s in enumerate(may.symbol.tolist()):
        val,source,industry,longname=fundamental_delta(s)
        f.loc[f.symbol==s,"factor_Fundamental Delta / Catalyst"]=val
        info_status[s]=source
        if industry:f.loc[f.symbol==s,"industry"]=industry
        if longname:f.loc[f.symbol==s,"name"]=longname
        if idx%20==0:print("FUND_PROGRESS",idx+1,"of",len(may),flush=True)
    f["fundamental_status"]=f.symbol.map(info_status).fillna("TECHNICAL_GATE_ALREADY_FAILED")
    f["score_Fundamental Delta / Catalyst"]=(
        f["factor_Fundamental Delta / Catalyst"]*WEIGHTS["Fundamental Delta / Catalyst"]).round(3)
    f["score"]=sum(f["score_"+k] for k in WEIGHTS)
    report_step(21,"Ready Now / Developing setup recognition",int((f.ready|f.developing).sum()))
    report_step(22,"Expansion-Contraction-Reexpansion recognition",int((f.setup=="ECR/VCP").sum()))
    report_step(23,"Setup Quality",int((f["factor_Setup Quality"]>=.6).sum()))
    report_step(24,"Tradability five tests",int(f.actionable.sum()),"ATR extension, pivot, stop, overhead and 3R")
    report_step(25,"Pure extended hard exclusion",int((~f.pure_extended).sum()))
    report_step(26,"New Candidate Gate",int(((f.channel=="NEW_MONSTER")&(f.tech_possible)).sum()),
        "Second channel included before Final Cut")
    f["qualified"]=(~f.hard_fail)&(~f.pure_extended)&f.actionable&(
        f["factor_Setup Quality"]>=.6)&(f.score>=minimum)&(
        f.channel!="UNDETERMINED")&(f.fundamental_status=="YAHOO_INFO")
    f["rejection_reason"]=np.select([
        f.hard_fail,f.pure_extended,~f.actionable,
        f["factor_Setup Quality"]<.6,
        f.channel=="UNDETERMINED",
        f.score<minimum,
        f.fundamental_status!="YAHOO_INFO"],
        ["BROKEN_STRUCTURE","PURE_EXTENDED","TRADABILITY_OR_RR_FAIL",
         "SETUP_QUALITY_FAIL","NO_MONSTER_CHANNEL","SCORE_BELOW_REGIME_THRESHOLD",
         "FUNDAMENTAL_DATA_UNVERIFIED"], default="PASS")
    f["theme"]=f.apply(classify_theme,axis=1)
    f=f.sort_values("score",ascending=False)
    winners=f[f.qualified].head(capacity).copy()
    f["final_biglist"]=f.symbol.isin(winners.symbol)
    f.to_csv(OUT/"full_market_scored.csv",index=False)
    winners.to_csv(OUT/"final_biglist.csv",index=False)
    f.head(250).to_csv(OUT/"top250_audit.csv",index=False)
    rotation=(f.groupby("sector",dropna=False).agg(
        n=("symbol","count"),median_ret5=("ret5","median"),median_ret20=("ret20","median"),
        median_ret60=("ret60","median"),above50=("constructive","mean"),
        leaders=("qualified","sum")).reset_index()
        .sort_values("median_ret20",ascending=False))
    rotation.to_csv(OUT/"sector_rotation.csv",index=False)
    summary={
        "as_of":"2026-11-11 16:00 ET",
        "model":"GPT Big List Formal Baseline v1.0 - independent operationalization",
        "not_local_biglist":True,
        "weights":WEIGHTS,
        "source_universe":"MarketParquet 2026-11-11 stock_daily and Nasdaq metadata",
        "full_ohlcv_snapshot":int(len(pd.read_csv(OUT/"universe_all_asof.csv"))),
        "scope_eligible":int(len(uni)),
        "history_fresh_and_80_bars":int(len(hist)),
        "features_complete":int(len(df)),
        "history_unavailable":int(len(uni)-len(hist)),
        "regime":regime,"capacity":capacity,"threshold":minimum,
        "breadth_above50":round(breadth,4),"VIX":vix,
        "ready":int(f.ready.sum()),"developing":int(f.developing.sum()),
        "fundamental_attempts":int(len(may)),
        "hard_structural_fail":int(f.hard_fail.sum()),
        "pure_extended":int(f.pure_extended.sum()),
        "tradability_pass":int(f.actionable.sum()),
        "formal_quality_pass_pre_capacity":int(f.qualified.sum()),
        "final_count":int(len(winners)),
        "final_tickers":winners.symbol.tolist(),
        "status_counts":dict(Counter(status.values())),
        "rotation":rotation.to_dict("records"),
        "rejection_counts":dict(Counter(f.rejection_reason)),
        "steps":AUDIT,"duration_seconds":round(time.time()-started,1),
        "caveats":[
            "Independent operationalization: v1.1 implementation rules not available",
            "Fundamental Delta based on readily available Yahoo company metrics, not a full filing/news event audit",
            "Entry/stop/reward scenarios are model-based, not guaranteed trading outcomes",
            "Only fresh Nov11 symbols with >=80 historical bars can receive complete derived features",
            "Latest market cross-section is from MarketParquet; history from Yahoo Finance"]
    }
    with open(OUT/"summary.json","w") as o:json.dump(summary,o,indent=2,ensure_ascii=False,default=str)
    print("RUN_SUMMARY",json.dumps({k:summary[k] for k in
       ["full_ohlcv_snapshot","scope_eligible","history_fresh_and_80_bars","features_complete",
        "regime","capacity","threshold","fundamental_attempts","formal_quality_pass_pre_capacity",
        "final_count","final_tickers","duration_seconds"]}),flush=True)
    print("FINAL_BY_THEME",winners.groupby("theme").symbol.apply(list).to_dict(),flush=True)
    print("SECTOR_ROTATION",rotation.head(12).to_string(index=False),flush=True)
    print("REJECTION_COUNTS",summary["rejection_counts"],flush=True)

if __name__=="__main__":
    run()
