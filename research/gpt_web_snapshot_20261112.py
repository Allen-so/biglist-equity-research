"""Read-only web screener data acquisition for GPT Big List 2026-11-12 premarket replay.
This code contains NO GPT Big List model weights or selection criteria.
"""
import json,time,requests,csv,io,re
from pathlib import Path
from collections import Counter
import pandas as pd
out=Path("web_snapshot_20261112");out.mkdir(exist_ok=True)
URL="https://scanner.tradingview.com/america/scan"
COLUMNS=["name","description","exchange","type","close","SMA20","SMA50","SMA200",
 "Perf.W","Perf.1M","Perf.3M","Perf.6M","Volatility.D","relative_volume_10d_calc",
 "average_volume_10d_calc","volume","market_cap_basic","sector","industry","ATR","RSI",
 "change","earnings_per_share_diluted_ttm","total_revenue_yoy_growth_ttm","net_income_yoy_growth_ttm",
 "gross_margin","operating_margin","earnings_release_date"]
HEADERS={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
"Origin":"https://www.tradingview.com","Referer":"https://www.tradingview.com/screener/"}
FILTER=[{"left":"exchange","operation":"in_range","right":["NYSE","NASDAQ","AMEX"]},
        {"left":"type","operation":"in_range","right":["stock","dr"]}]
def request_page(cols,a,b):
    q={"filter":FILTER,"symbols":{"query":{"types":[]}},"columns":cols,
       "sort":{"sortBy":"name","sortOrder":"asc"},"range":[a,b]}
    r=requests.post(URL,headers=HEADERS,json=q,timeout=35);r.raise_for_status()
    p=r.json()
    if not isinstance(p,dict) or not isinstance(p.get("data"),list):raise RuntimeError(str(p)[:1000])
    return p
try:
    t=request_page(COLUMNS,0,20);cols=COLUMNS
    print("ALL_FIELDS_AVAILABLE",len(COLUMNS),flush=True)
except Exception as e:
    print("FULL_COLUMN_SET_FAIL",str(e)[:250],flush=True)
    cols=COLUMNS[:22]
all_rows=[];all_total=None
for a in range(0,10000,3000):
    page=request_page(cols,a,a+3000)
    if all_total is None:all_total=page["totalCount"]
    if page["totalCount"]!=all_total:raise RuntimeError("count shifted during scan")
    for x in page["data"]:
        if len(x.get("d",[]))!=len(cols):raise RuntimeError("missing columns")
        row=dict(zip(cols,x["d"]));row["vendor_id"]=x["s"];all_rows.append(row)
    print("PAGE",a,len(page["data"]),"total",all_total,flush=True)
    if len(all_rows)>=all_total:break
assert len(all_rows)==all_total,("not full",len(all_rows),all_total)
assert len({r["vendor_id"] for r in all_rows})==len(all_rows),"dup identifiers"
names=[r.get("description","") or "" for r in all_rows]
bans=re.compile(r"\b(warrant|units? consisting|subscription rights|preferred stock|preferred shares|"
                r"depositary shares representing preferred|perp pfd|cum conv|subordinated debentures|"
                r"senior notes|exchange.traded fund|etfs?|etns?|acquisition corp|acquisition co)\b",re.I)
for r in all_rows:r["scope_provisional"]="NON_COMMON" if bans.search(str(r.get("description") or "")) else "SCOPE_PROVISIONAL"
pd.DataFrame(all_rows).to_csv(out/"all_us_web_universe.csv.gz",index=False,compression="gzip")
universe=[r for r in all_rows if r["scope_provisional"]=="SCOPE_PROVISIONAL"]
# provider reconciliation to asof 11/11 completed OHLCV, no reading intraday as completed daily.
ref=requests.get("https://marketparquet.com/api/data/download/stock_daily/2026-11-11.parquet",timeout=30)
ref.raise_for_status()
rdf=pd.read_parquet(io.BytesIO(ref.content))
rdf=rdf[rdf.asset_type.astype(str).str.lower()=="stock"]
close_ref={str(x.symbol):float(x.close) for x in rdf.itertuples() if x.close>0}
gaps=[];matches=[];mismatch=[]
for r in universe:
    t=r["name"]
    try:
        x=float(r["close"]);y=close_ref[t]
        delta=abs(x-y)/y
        matches.append((t,delta))
        if delta>.005:mismatch.append((t,x,y,delta))
    except Exception: gaps.append(t)
match_df=pd.DataFrame(mismatch,columns=["symbol","web_close","marketparquet_11_11_close","relative_gap"])
match_df.to_csv(out/"price_discrepancies.csv",index=False)
snapshot={"raw_count":len(all_rows),"scope_provisional":len(universe),
          "source":"TradingView public website screener scanner, not guaranteed API",
          "data_asof":"2026-11-11 regular-market close (must confirm against independent source)",
          "scan_time_utc":pd.Timestamp.now(tz="UTC").isoformat(),
          "type_counts":dict(Counter(r["type"] for r in universe)),
          "columns":cols,"matched_price_count":len(matches),
          "price_agreement_under_half_percent":sum(d<.005 for _,d in matches),
          "price_mismatch_over_half_percent":len(mismatch),
          "missing_crossvendor_reference":len(gaps),
          "note":"Universe discovery fresh; never use prior picks as universe"}
(out/"web_snapshot_summary.json").write_text(json.dumps(snapshot,indent=2,default=str))
print("SNAPSHOT",json.dumps(snapshot),flush=True)
