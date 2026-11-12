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
