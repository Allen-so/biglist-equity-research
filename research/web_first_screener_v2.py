#!/usr/bin/env python3
"""GPT Big List web-first *preflight* v0.3 (NOT final 26-factor scoring).

Uses the public web site's US stock scanner for cheap technical columns.
Does not bypass authentication or paywalls; fails closed when feed is blocked.
Every US-listed common-stock-like ticker is evaluated across 9 OR lanes.
Exports candidate and coverage audit; no score, final eligibility or buy signal.
"""
from __future__ import annotations
import csv,json,math,re,sys,time,datetime,random
from collections import Counter,defaultdict
from pathlib import Path
import requests

ASOF = "2026-11-11"
OUTPUT = Path("gpt_webfirst_outputs")
OUTPUT.mkdir(parents=True,exist_ok=True)
TV_URL="https://scanner.tradingview.com/america/scan"
COLUMNS=[
"name","description","exchange","type","close","SMA20","SMA50","SMA200",
"Perf.W","Perf.1M","Perf.3M","Perf.6M","Volatility.D",
"relative_volume_10d_calc","average_volume_10d_calc","volume",
"market_cap_basic","sector","industry","ATR","RSI","change"
]
HEADERS={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127.0.0.0 Safari/537.36",
         "Accept":"application/json","Origin":"https://www.tradingview.com","Referer":"https://www.tradingview.com/screener/"}
US_EXCH={"NASDAQ","NYSE","AMEX","NYSE MKT"}
INSTRUMENT_NOT_COMMON=re.compile(r"\b(warrants?|units? consisting|subscription rights?|depositary shares representing preferred|"
                                 r"preferred stock|preferred shares|pfd|perp pfd|deposit(?:ary)? shs repr|"
                                 r"cum conv|subordinated debentures|bonds|senior notes|exchange.traded fund|"
                                 r"etfs?|etns?|acquisition corp(?:oration)?|acquisition co(?:mpany)?)\b",re.I)

def fnum(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def ratio(p,ma):
    p=fnum(p);ma=fnum(ma)
    return p/ma if p is not None and ma is not None and ma>0 else None

def gt(x,bound):
    return x is not None and x>bound

def ge(x,bound):
    return x is not None and x>=bound

def between(x,lo,hi):
    return x is not None and lo<=x<=hi

def pct(x):
    x=fnum(x)
    return x/100 if x is not None else None

def get_tv_all():
    ss=requests.Session()
    raw=[]
    step=4000
    total=None
    begin=time.monotonic()
    for offset in range(0,20000,step):
        req={"filter":[{"left":"exchange","operation":"in_range","right":["NYSE","NASDAQ","AMEX"]},{"left":"type","operation":"in_range","right":["stock","dr"]}],
             "symbols":{"query":{"types":[]}},
             "columns":COLUMNS,
             "sort":{"sortBy":"name","sortOrder":"asc"},
             "range":[offset,offset+step]}
        last_exc=None
        for attempt in range(3):
            try:
                response=ss.post(TV_URL,json=req,headers=HEADERS,timeout=30)
                response.raise_for_status()
                payload=response.json()
                if not isinstance(payload,dict) or "data" not in payload:
                    raise ValueError("Malformed screener response")
                break
            except Exception as e:
                last_exc=e
                time.sleep(2+attempt*3)
        else:
            raise RuntimeError(f"Screener unreachable after retry: {last_exc!r}")
        if total is None:total=int(payload["totalCount"])
        if int(payload["totalCount"])!=total:
            raise RuntimeError("Source total changed during paging; restart needed")
        block=payload.get("data",[])
        for entry in block:
            arr=entry.get("d")
            if arr is None or len(arr)!=len(COLUMNS):continue
            row=dict(zip(COLUMNS,arr))
            row["id"]=entry.get("s")
            raw.append(row)
        print("FETCH_PAGE",offset,"rows",len(block),"total",total,flush=True)
        if offset+len(block)>=total:break
        if not block:raise RuntimeError("Empty page before reported total")
        time.sleep(.2)
    full_ids=[r["id"] for r in raw]
    if len(raw)!=total or len(full_ids)!=len(set(full_ids)):
        raise RuntimeError(f"Coverage failure: rows={len(raw)}, distinct={len(set(full_ids))}, expected={total}")
    print("WEB_FULL_POOL",len(raw),"elapsed",round(time.monotonic()-begin,1),flush=True)
    return raw

def write_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})

def is_us_listed(row):
    return row.get("exchange") in US_EXCH and row.get("type") in {"stock","dr"}

def get_scope_reason(row):
    if not is_us_listed(row):return "NOT_TARGET_US_EXCHANGE_OR_TYPE"
    name=str(row.get("description") or "")
    if INSTRUMENT_NOT_COMMON.search(name):return "NON_COMMON_INSTRUMENT_NAME"
    if not row.get("name"):return "MISSING_TICKER"
    if not gt(fnum(row.get("close")),0):return "NO_POSITIVE_PRICE"
    return "SCOPE_PROVISIONAL_PASS"

def label(row,sector_median):
    p=fnum(row["close"]); q20=ratio(p,row["SMA20"]);q50=ratio(p,row["SMA50"]);q200=ratio(p,row["SMA200"])
    r5=pct(row["Perf.W"]);r20=pct(row["Perf.1M"]);r60=pct(row["Perf.3M"])
    rvol=fnum(row["relative_volume_10d_calc"]);daily_vol=pct(row["Volatility.D"])
    sector=str(row.get("sector") or "Unknown")
    med=sector_median.get(sector)
    B={}
    B["A_MATURE_LEADER"]=between(q50,1.015,1000) and gt(r60,.12) and gt(r20,-.08) and ge(q20,.92)
    B["B_TURNAROUND"]=(
        between(q50,.94,1.16) and ge(q20,.96) and gt(r5,.025) and gt(rvol,1.15)
        or between(q200,.94,1.10) and gt(r20,.08) and gt(rvol,1.15))
    B["C_PREBREAKOUT"]=between(q20,.94,1.065) and gt(r5,-.07) and between(rvol,0,1.15) and gt(r20,-.12)
    B["D_SECTOR_RS_PROXY"]=med is not None and r20 is not None and r20>med+.07 and gt(r5,-.03) and ge(q20,.94)
    B["E_ABNORMAL_DEMAND"]=gt(rvol,1.8) and gt(r5,.06) and gt(r20,-.15) and gt(daily_vol,.025)
    B["F_RECOVERY_RESCUE"]=gt(r20,.12) and ge(q20,.88) and ge(q50,.86)
    B["G_YOUNG_OR_SHORT_HISTORY"]=(q200 is None) and ge(q20,.95) and gt(r20,.02) and gt(rvol,1.1)
    B["H_MODERATE_REPAIR"]=between(q20,.95,1.08) and gt(q50,.98) and gt(r20,.03) and gt(rvol,1)
    B["I_STRONG_LOW_VOLUME_HOLD"]=gt(q20,1.04) and gt(q50,1.03) and gt(r5,0) and gt(r20,0) and between(rvol,0,1.2)
    return [k for k,v in B.items() if v]

def cross_check_close(rows):
    """Independent completed-session crosscheck; exceptions are held separately."""
    try:
        import io,pandas as pd
        u=f"https://marketparquet.com/api/data/download/stock_daily/{ASOF}.parquet"
        response=requests.get(u,timeout=20);response.raise_for_status()
        y=pd.read_parquet(io.BytesIO(response.content))
        y=y[y.asset_type.astype(str).str.lower()=="stock"]
        ref={str(x.symbol):float(x.close) for x in y.itertuples() if x.close>0}
        checks=[];outliers=[];matched=set()
        for r in rows:
            k=r.get("name");price=fnum(r.get("close"))
            if k in ref and price is not None:
                delta=abs(price-ref[k])/ref[k]
                checks.append(delta);matched.add(k)
                if delta>=.005:
                    outliers.append(dict(symbol=k,exchange=r.get("exchange"),marketparquet=ref[k],
                                         tradingview=price,relative_gap=delta,stage=r.get("status")))
        write_csv(OUTPUT/"price_discrepancies.csv",outliers,[
            "symbol","exchange","marketparquet","tradingview","relative_gap","stage"])
        only_mp=[{"symbol":s,"marketparquet_close":p} for s,p in ref.items() if s not in matched]
        write_csv(OUTPUT/"coverage_gaps_mp_only.csv",only_mp,["symbol","marketparquet_close"])
        if not checks:raise RuntimeError("No overlapping independent prices")
        good=len(checks)-len(outliers)
        result={"independent_source":"MarketParquet","reference_date":ASOF,
                "marketparquet_total":len(ref),
                "matched":len(checks),"within_0_5pct":good,
                "pct_within_0_5pct":good/len(checks),
                "source_missing_or_unmatched":len(only_mp),
                "price_outliers":len(outliers),
                "max_relative_gap":max(checks),
                "_outliers":[r["symbol"] for r in outliers]}
        if result["pct_within_0_5pct"]<.90:
            raise RuntimeError("As-of mismatch vs completed-session reference")
        return result
    except Exception as e:
        return {"asof_verified":False,"error":str(e),"price_outliers":None}

def main():
    rows=get_tv_all()
    scopes=Counter()
    for r in rows:
        r["scope_status"]=get_scope_reason(r)
        scopes[r["scope_status"]]+=1
    eligible=[r for r in rows if r["scope_status"]=="SCOPE_PROVISIONAL_PASS"]
    assert len(eligible)>3000,f"US stocks scope too narrow {len(eligible)}"
    sector_returns=defaultdict(list)
    for r in eligible:
        value=pct(r.get("Perf.1M"))
        if value is not None:sector_returns[str(r.get("sector") or "Unknown")].append(value)
    sector_median={k:float(__import__("statistics").median(v)) for k,v in sector_returns.items() if len(v)>=5}
    channels=Counter();selected=[];deferred=[];excluded=[]
    for r in eligible:
        lane_names=label(r,sector_median)
        r["lanes"]=";".join(lane_names)
        r["lane_count"]=len(lane_names)
        avg=fnum(r.get("average_volume_10d_calc"))
        price=fnum(r.get("close"))
        r["dollar_volume_proxy"]=price*avg if avg is not None and price is not None else None
        r["relative_volume"]=fnum(r.get("relative_volume_10d_calc"))
        for lane in lane_names:channels[lane]+=1
        if not lane_names:
            r["status"]="NO_FAST_LANE";excluded.append(r)
        elif price is None or price<3 or not ge(r["dollar_volume_proxy"],2e6):
            r["status"]="LIQUIDITY_OR_PRICE_HOLD";deferred.append(r)
        else:
            r["status"]="WEB_PREFLIGHT_CANDIDATE";selected.append(r)
    # Even excluded candidates should be sampled for surveillance of false negatives.
    rng=random.Random(26061112)
    shadow=rng.sample(excluded,min(len(excluded),max(100,int(.07*len(excluded))))) if excluded else []
    for r in shadow:r["shadow_sample"]=True
    verify=cross_check_close(eligible)
    bad_quotes=set(verify.pop("_outliers",[]))
    if bad_quotes:
        old_selected=selected
        selected=[]
        for r in old_selected:
            if r["name"] in bad_quotes:
                r["status"]="PRICE_SOURCE_MISMATCH_HOLD"
                deferred.append(r)
            else:selected.append(r)
        print("CROSS_SOURCE_PRICE_HOLD",len(old_selected)-len(selected),flush=True)
    common=[
      "id","name","description","exchange","type","sector","industry","close",
      "SMA20","SMA50","SMA200","Perf.W","Perf.1M","Perf.3M","Perf.6M",
      "relative_volume_10d_calc","average_volume_10d_calc","volume","ATR","Volatility.D",
      "market_cap_basic","lanes","lane_count","dollar_volume_proxy","status"]
    write_csv(OUTPUT/"screener_full_raw.csv",rows,common+["scope_status"])
    write_csv(OUTPUT/"web_preflight_candidates.csv",selected,common)
    write_csv(OUTPUT/"web_preflight_liquidity_hold.csv",deferred,common)
    write_csv(OUTPUT/"shadow_audit_sample.csv",shadow,common)
    with (OUTPUT/"web_preflight_summary.json").open("w") as f:
        json.dump({
          "asof_requested":ASOF,
          "product":"GPT Big List - Web First preflight v0.3",
          "is_formal_26_layer_final":False,
          "source":"TradingView website public scan endpoint; unofficial/availability not guaranteed",
          "security_types":["stock","dr"],
          "allowed_exchanges":["NYSE","NASDAQ","AMEX"],
          "source_total":len(rows),
          "scope":dict(scopes),
          "scope_provisional_pass":len(eligible),
          "lane_counts":dict(channels),
          "or_union_with_lanes":len(selected)+len(deferred),
          "deep_candidates":len(selected),
          "liquidity_hold":len(deferred),
          "no_lane":len(excluded),
          "shadow_sample":len(shadow),
          "sector_candidates":dict(Counter(r.get("sector") or "Unknown" for r in selected)),
          "asof_price_crosscheck":verify,
          "non_binding_policy":"Do not change the 15 canonical factor weights; initial lanes are discovery only.",
          "data_policy":"Unknown indicator/fundamental is UNKNOWN, never a numerical zero score.",
          "ranking_policy":"Full-universe relative ranks should be obtained from full feed before deep selection, not ranks only on selected candidates.",
          "limitations":["Unverified security-type classification for some instruments","Web fields are snapshot metrics, not full time-series evidence","RVOL 10d vendor definition differs from historical model","Stress RS lane is sector-return proxy, not true down-day Stress RS","Not yet full implementation of 26-layer Big List"],},
          f,ensure_ascii=False,indent=2,default=str)
    print("PREFLIGHT_SUMMARY",json.dumps({"raw":len(rows),"scope":len(eligible),
      "union":len(selected)+len(deferred),"deep_candidates":len(selected),
      "deferred":len(deferred),"no_lane":len(excluded),"channels":dict(channels),
      "crosscheck":verify}),flush=True)
    print("TOP_EXAMPLES_BY_LANE")
    for lane in channels:
        subset=[r for r in selected if lane in r["lanes"].split(";")]
        subset.sort(key=lambda r:(fnum(r["Perf.1M"]) or -999),reverse=True)
        print(lane,[r["name"] for r in subset[:8]],flush=True)

if __name__=="__main__":
    main()
