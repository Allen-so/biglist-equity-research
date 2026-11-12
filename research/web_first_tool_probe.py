"""Test public, non-authenticated screener endpoints. Does not bypass login/captcha."""
import requests,re,json
from bs4 import BeautifulSoup
s=requests.Session()
s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127.0.0.0 Safari/537.36", "Accept-Language":"en-US,en;q=0.9"})
cases={
"finviz_mature":"https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa,sh_avgvol_o300,sh_price_o4",
"finviz_turnaround":"https://finviz.com/screener.ashx?v=111&f=ta_sma50_crossa,sh_price_o4",
"finviz_rvol":"https://finviz.com/screener.ashx?v=111&f=sh_relvol_o1.5,sh_price_o4",
"chartmill":"https://www.chartmill.com/stock/stock-screener",
"tradingview":"https://www.tradingview.com/screener/",
"stockanalysis":"https://stockanalysis.com/stocks/screener/",
}
for name,url in cases.items():
 try:
  r=s.get(url,timeout=18)
  soup=BeautifulSoup(r.content,"html.parser")
  text=soup.get_text(" ",strip=True)
  tickers=re.findall(r'href=["\'](?:quote\.ashx\?t=|/quote/)([A-Z][A-Z0-9.-]{0,5})',r.text)
  print("PROBE",name,"status",r.status_code,"bytes",len(r.content),
        "tickers",len(set(tickers)),"titles",soup.title.get_text(" ",strip=True)[:90] if soup.title else None,
        "captcha",bool(re.search(r'captcha|cloudflare|verify you are human',text[:600],re.I)),
        "excerpt",text[:160],flush=True)
 except Exception as e:
  print("PROBE_ERROR",name,type(e).__name__,str(e)[:200],flush=True)
# TradingView's own site reports CSV export in an authenticated UI. Probe the generic scanner
# endpoint for whether unauthenticated automated access exists; do not build on it if blocked.
try:
 url="https://scanner.tradingview.com/america/scan"
 j={"filter":[{"left":"type","operation":"equal","right":"stock"},
              {"left":"close","operation":"greater","right":4}],
     "symbols":{"query":{"types":[]}},"columns":["name","close","SMA50","SMA200","relative_volume_10d_calc"],"sort":{"sortBy":"name","sortOrder":"asc"},"range":[0,20]}
 r=s.post(url,json=j,timeout=18,headers={"Content-Type":"application/json","Origin":"https://www.tradingview.com"})
 sample=r.json() if "json" in r.headers.get("content-type","") else None
 print("PROBE","tradingview_scanner","status",r.status_code,"bytes",len(r.content),
       "rows",len(sample.get("data",[])) if isinstance(sample,dict) else None,
       "totalCount",sample.get("totalCount") if isinstance(sample,dict) else None,
       "preview",str(sample)[:220] if sample is not None else r.text[:170],flush=True)
except Exception as e: print("PROBE_ERROR","tradingview_scanner",type(e).__name__,str(e)[:160],flush=True)

COLS=[
"name","description","exchange","type","close","SMA20","SMA50","SMA200",
"Perf.W","Perf.1M","Perf.3M","Perf.6M","Volatility.D",
"relative_volume_10d_calc","average_volume_10d_calc","volume",
"market_cap_basic","sector","industry","ATR","RSI","change",
]
for cols in [COLS,["name","description","close","SMA20","SMA50","SMA200","Perf.1M","Perf.3M",
     "relative_volume_10d_calc","volume","market_cap_basic","sector","industry"]]:
 try:
  j={"filter":[{"left":"type","operation":"equal","right":"stock"}],
     "symbols":{"query":{"types":[]}},"columns":cols,"sort":{"sortBy":"name","sortOrder":"asc"},"range":[0,6500]}
  r=s.post("https://scanner.tradingview.com/america/scan",json=j,timeout=45,
      headers={"Content-Type":"application/json","Origin":"https://www.tradingview.com"})
  j2=r.json() if "json" in r.headers.get("content-type","") else {}
  print("TV_BULK","status",r.status_code,"columns",len(cols),"total",j2.get("totalCount"),
        "rows",len(j2.get("data",[])), "bytes",len(r.content),"error",str(j2)[:450] if not r.ok else None,flush=True)
  if r.ok:
   print("TV_FIELDS",cols,flush=True)
   print("TV_FIRST",j2.get("data",[])[:2],flush=True)
   print("TV_LAST",j2.get("data",[])[-2:],flush=True)
 except Exception as e: print("TV_BULK_ERROR",type(e).__name__,str(e)[:180],flush=True)

for tag,extra in [
 ("type_stock_dr", [{"left":"type","operation":"in_range","right":["stock","dr"]}]),
 ("exchange_only", [{"left":"exchange","operation":"in_range","right":["NYSE","NASDAQ","AMEX"]}]),
 ("venue_and_stockdr", [{"left":"exchange","operation":"in_range","right":["NYSE","NASDAQ","AMEX"]},{"left":"type","operation":"in_range","right":["stock","dr"]}])
]:
 try:
  req={"filter":extra,"symbols":{"query":{"types":[]}},"columns":["name","description","type","exchange","close"],"range":[0,10]}
  rr=s.post("https://scanner.tradingview.com/america/scan",json=req,timeout=20,headers={"Origin":"https://www.tradingview.com"})
  j=rr.json()
  print("VARIANT",tag,"status",rr.status_code,"count",j.get("totalCount"),"data",j.get("data",[])[:2],"error",str(j)[:160] if not rr.ok else None,flush=True)
 except Exception as e:print("VARIANT_ERROR",tag,str(e)[:160],flush=True)
try:
 req={"filter":[],"symbols":{"tickers":["NASDAQ:ASML","NASDAQ:ARM","NASDAQ:RYAAY","NYSE:RTO","NYSE:TIMB","NYSE:MNSO","NASDAQ:ABVX"]},
      "columns":["name","description","type","exchange","close"],"range":[0,20]}
 rr=s.post("https://scanner.tradingview.com/america/scan",json=req,timeout=20,headers={"Origin":"https://www.tradingview.com"})
 print("ADR_TICKER_PROBE",rr.status_code,str(rr.json())[:2000],flush=True)
except Exception as e:print("ADR_PROBE_ERROR",str(e)[:150],flush=True)
