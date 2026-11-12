#!/usr/bin/env python3
"""GPT Big List 2026-11-11 deep technical audit, web-first source.

Unlike the user's separate local Big List, this is an isolated GPT screening
experiment; it uses only the snapshot universe and raw OHLCV. No previously
computed Big List scores, prior qualified list, or hard gate flags are read.
"""
import numpy as np, pandas as pd, json, math, pathlib
from collections import Counter
OUT=pathlib.Path('gpt_deep_out');OUT.mkdir(exist_ok=True)
raw=pd.read_csv('web_inputs/screener_full_raw.csv',low_memory=False)
raw=raw[raw.scope_status=='SCOPE_PROVISIONAL_PASS'].drop_duplicates('name')
cand=pd.read_csv('web_inputs/web_preflight_candidates.csv',low_memory=False)
pars=list(pathlib.Path('history_inputs').glob('history_*.parquet'))
print('HIST_PARTITIONS',len(pars),flush=True)
if len(pars)!=4:raise RuntimeError('Missing historical partitions')
bars=pd.concat([pd.read_parquet(i) for i in pars],ignore_index=True)
bars['date']=pd.to_datetime(bars['date'])
bars=bars.sort_values(['symbol','date'])
assert bars.symbol.nunique()>5000
bars['prev_close']=bars.groupby('symbol')['Close'].shift()
bars['daily_return']=bars.Close/bars.prev_close-1
# Each market-session median is a transparent equal-weight stress proxy; NOT SPY.
benchmark=bars.groupby('date').daily_return.median().rename('ew_market_return')
ranks=raw.copy()
for name,col in [('r5','Perf.W'),('r20','Perf.1M'),('r60','Perf.3M'),('r120','Perf.6M')]:
 ranks[name]=pd.to_numeric(ranks[col],errors='coerce').rank(pct=True)
ranks['sector_rank']=ranks.groupby('sector')['Perf.3M'].rank(pct=True)
ranks['rank_level']=(.20*ranks.r5+.35*ranks.r20+.45*ranks.r60)
ranks['rank_delta_proxy']=ranks.rank_level-ranks.r120
ranks=ranks.set_index('name')
selected=set(cand.name.astype(str))
out=[];touched=set()
for sym,z in bars.groupby('symbol',sort=False):
 if sym not in selected:continue
 touched.add(sym)
 z=z.sort_values('date')
 if len(z)<80 or z.date.iloc[-1]!=pd.Timestamp('2026-11-11'):continue
 if sym not in ranks.index:continue
 v=ranks.loc[sym]
 if isinstance(v,pd.DataFrame):v=v.iloc[0]
 c=z.Close.astype(float).to_numpy()
 h=z.High.astype(float).to_numpy()
 l=z.Low.astype(float).to_numpy()
 o=z.Open.astype(float).to_numpy()
 vol=z.Volume.astype(float).to_numpy()
 n=len(c)
 if not np.isfinite(c[-1]) or c[-1]<=0:continue
 price=float(c[-1])
 if abs(price-float(v['close']))/price>.005:
  # Price mismatch is a separate data issue, not an automatic technical failure.
  continue
 def mom(k):
  return float(c[-1]/c[-k-1]-1) if n>k else float('nan')
 def ma(k):
  return float(np.mean(c[-k:])) if n>=k else float('nan')
 tr=np.maximum.reduce([h-l,abs(h-np.r_[c[0],c[:-1]]),abs(l-np.r_[c[0],c[:-1]])])
 atrw=float(pd.Series(tr).ewm(alpha=1/20,adjust=False,min_periods=20).mean().iloc[-1])
 atr14=float(pd.Series(tr).ewm(alpha=1/14,adjust=False,min_periods=14).mean().iloc[-1])
 if not np.isfinite(atrw) or atrw<=0:continue
 adr20=float(np.nanmean((h[-20:]-l[-20:])/np.r_[c[-21:-1]])) if n>21 else float('nan')
 volavg20=float(np.mean(vol[-20:]))
 usd_median=float(np.median(c[-20:]*vol[-20:]))
 upvol=float(vol[-30:][np.diff(c[-31:])>0].sum()) if n>30 else 0
 downvol=float(vol[-30:][np.diff(c[-31:])<0].sum()) if n>30 else 0
 vol_flow=upvol/max(upvol+downvol,1)
 last20_r=np.diff(c[-21:])/c[-21:-1] if n>21 else np.array([])
 mr=benchmark.reindex(z.date).to_numpy()
 mask=(mr[-61:]<-0.004) if n>=61 else (mr<-.004)
 sr=np.diff(c[-61:])/c[-61:-1] if n>=61 else np.diff(c)/c[:-1]
 mrret=mr[-60:] if n>=61 else mr[1:]
 valid=(mrret<-.004)&np.isfinite(sr)&np.isfinite(mrret)
 stress=float(np.mean(sr[valid]-mrret[valid])) if valid.sum()>=3 else float('nan')
 retention=(price-float(np.min(l[-60:])))/max(.01,float(np.max(h[-60:])-np.min(l[-60:])))
 advance20=max(mom(20),-.5)
 productive=abs(advance20)/(sum(abs(last20_r))+1e-9) if len(last20_r)==20 else 0
 m10,m20,m50,m200=ma(10),ma(20),ma(50),ma(200)
 prev20high=float(np.max(h[-21:-1]))
 high52=float(np.max(h[-252:]))
 low60=float(np.min(l[-60:]))
 high60=float(np.max(h[-60:]))
 rt60=mom(60)
 r5=mom(5)
 r20=mom(20)
 # 3-point swing low qualification (consecutive 2 bars on each side)
 swings=[i for i in range(2,n-2) if l[i]==min(l[i-2:i+3])]
 # Higher low in the last 50 sessions, three points and max slope required
 late=[i for i in swings if i>=n-65]
 recent_lows=[(i,float(l[i])) for i in late[-4:]]
 three_point=bool(len(recent_lows)>=3 and recent_lows[-1][1]>recent_lows[-2][1]>recent_lows[-3][1])
 higher_low=bool(len(recent_lows)>=2 and recent_lows[-1][1]>=recent_lows[-2][1])
 # Require near-term contraction both RANGE and VOLUME, compared with prior 10-20 days.
 short_range=float(np.mean(h[-5:]-l[-5:]))
 prev_range=float(np.mean(h[-20:-5]-l[-20:-5]))
 vol5=float(np.mean(vol[-5:]))
 volprev=float(np.mean(vol[-20:-5]))
 contraction=short_range/max(.01,prev_range)
 vratio=vol5/max(1,volprev)
 compressed=bool(contraction<.88 and vratio<.95)
 recent_impulse=bool(mom(60)>.17 and (np.max(vol[-60:-6])>1.5*volavg20))
 # ordered first demand impulse and subsequent shrink must be independently evidenced.
 ecr_developing=bool(recent_impulse and higher_low and compressed and retention>.52)
 reexpansion=bool(ecr_developing and vol[-1]/max(1,volavg20)>1.35 and
                   c[-1]>np.max(h[-10:-1]) and c[-1]>=o[-1])
 wtrend=float(np.mean(c[-5:])/np.mean(c[-30:-5])) if n>35 else float('nan')
 broken=bool(price<m50*.91 and (np.isnan(m200) or price<m200*.93)
             and r20<-.14 and r5<-.04 and not (price>m20 and r5>.06))
 extended=bool((price-m20)/atrw>3.0 or (price-prev20high)/atrw>2.6)
 # Stop candidates must be structural, without inventing an ATR-only stop.
 sups=[(float(np.min(l[-10:])), '10D_LOW'),(m20,'SMA20'),(m50,'SMA50')]
 sups=[(s,lab) for s,lab in sups if np.isfinite(s) and s<price]
 sups=sorted(sups,key=lambda a:price-a[0])
 stop=float('nan');stop_basis='UNKNOWN';stopATR=float('nan')
 for support,lab in sups:
  # 0.15 ATR below observed structure, not an invented support line
  candidate=support-.15*atrw
  dist=(price-candidate)/atrw
  if .8<=dist<=2.8:
   stop=candidate;stop_basis=lab;stopATR=dist;break
 # If above 52-week high, don't invent a four-ATR target.
 resistances=[float(v) for v in (high52,high60,prev20high) if float(v)>price+.002*atrw]
 target=min(resistances) if resistances else float('nan')
 # Some minor pivots may lie very near; use 52-week overhead as a SCENARIO, but mark source.
 major_target=high52 if high52>price else float('nan')
 rr_nearest=(target-price)/(price-stop) if np.isfinite(target) and np.isfinite(stop) else float('nan')
 rr_major=(major_target-price)/(price-stop) if np.isfinite(major_target) and np.isfinite(stop) else float('nan')
 tradability=bool(np.isfinite(rr_major) and rr_major>=3 and
                  np.isfinite(stopATR) and .8<=stopATR<=2.8 and
                  abs(price-prev20high)/atrw<=2.5 and
                  (price-m20)/atrw<=2.5)
 # developing second-pivot structure: current price within 0-1.8 ATR under major pivot
 near_pivot=bool(-.5<=(price-prev20high)/atrw<=1.8)
 setup_ready=bool(near_pivot and price>m20 and (higher_low or compressed) and not extended and tradability)
 setup_developing=bool(price>m20 and higher_low and recent_impulse and not extended and (compressed or near_pivot))
 distribution=int(np.sum((last20_r<-.025)&(vol[-20:]>volavg20*1.3))) if len(last20_r)==20 else 0
 velocity_decay=bool(r5<0 and r20>0 and vratio>1.25) or (r20<0 and rt60>.10 and retention<.5)
 out.append(dict(symbol=sym,price=price,bars=n,atr_wilder20=atrw,atr_wilder14=atr14,adr20=adr20,
  vol20=volavg20,median_dollar_volume20=usd_median,m10=m10,m20=m20,m50=m50,m200=m200,
  r5=r5,r20=r20,r60=rt60,r120=mom(120),high52=high52,high60=high60,low60=low60,pivot=prev20high,
  stress_equalweight_proxy=stress,stress_bear_days=int(valid.sum()),up_volume_ratio=vol_flow,
  retention=retention,productive_ratio=productive,three_point_higher_lows=three_point,
  higher_low=higher_low,compression_range_ratio=contraction,compression_volume_ratio=vratio,
  compressed=compressed,prior_demand_impulse=recent_impulse,ecr_developing=ecr_developing,
  reexpansion=reexpansion,velocity_decay=velocity_decay,
  stage4_structural_fail=broken,pure_extended=extended,
  distribution=distribution,stop=stop,stop_basis=stop_basis,stop_atr=stopATR,
  overhead_nearest=target,overhead_52w=major_target,
  rr_nearest=rr_nearest,rr_52w_scenario=rr_major,
  setup_ready=setup_ready,setup_developing=setup_developing,
  tradability_check=tradability,near_pivot=near_pivot,
  rank_level=float(v.get('rank_level',np.nan)),rank_delta_proxy=float(v.get('rank_delta_proxy',np.nan)),
  sector_rank=float(v.get('sector_rank',np.nan)),
  web_ret5=float(v.get('Perf.W',np.nan)),web_ret20=float(v.get('Perf.1M',np.nan)),
  web_ret60=float(v.get('Perf.3M',np.nan)),web_ret120=float(v.get('Perf.6M',np.nan)),
  lanes=str(v.get('lanes','')),web_sector=str(v.get('sector','')),company=str(v.get('description',''))
 ))
print('DEEPCOVER',len(out),'of',len(selected),'market_history',bars.symbol.nunique(),flush=True)
q=pd.DataFrame(out)
q.to_csv(OUT/'all_deep_price_volume_evidence.csv',index=False)
pool=q[(q.price>=4)&(q.median_dollar_volume20>=8e6)&~q.stage4_structural_fail&~q.pure_extended]
pool=pool.copy()
# This is a research PRIORITY, never canonical 100-point scoring.
pool['tech_priority']=(pool.rank_level*17+pool.sector_rank*6+
   np.clip(pool.retention,0,1)*14+
   np.clip(pool.adr20/.08,0,1)*9+
   pool.higher_low.astype(float)*10+
   pool.compressed.astype(float)*8+
   np.clip(pool.productive_ratio/.45,0,1)*6+
   np.clip(pool.up_volume_ratio,0,1)*5+
   pool.stress_equalweight_proxy.fillna(0).clip(-.01,.025)*350+
   pool.setup_ready.astype(float)*9+pool.setup_developing.astype(float)*5+
   pool.tradability_check.astype(float)*8 -
   pool.velocity_decay.astype(float)*7)
pool=pool.sort_values('tech_priority',ascending=False)
pool.to_csv(OUT/'deep_eligible_ranked_not_final.csv',index=False)
pool.head(100).to_csv(OUT/'top100_deep_technical_audit.csv',index=False)
summary={'initial_web_scope':len(raw),'web_preflight':len(selected),'raw_market_history_stocks':bars.symbol.nunique(),
         'deep_with_fresh_OHLCV':len(q),'deep_price_4_turnover8M_and_structure_pass':len(pool),
         'ready_setup':int(pool.setup_ready.sum()),'developing_setup':int(pool.setup_developing.sum()),
         'verified_structural_3R_candidate':int(pool.tradability_check.sum()),
         'ecr_developing':int(pool.ecr_developing.sum()),
         'reexpansion':int(pool.reexpansion.sum()),'three_point_higher_lows':int(pool.three_point_higher_lows.sum()),
         'asof':'2026-11-11 16:00 ET','stress_RS_base':'equal-weight daily market median, not SPY',
         'not_a_formal_final_list':True}
(OUT/'stage_summary.json').write_text(json.dumps(summary,indent=2))
print('STAGE_SUMMARY',json.dumps(summary),flush=True)
print('TOP50',pool.head(50)[['symbol','price','tech_priority','adr20','r20','r60',
 'retention','stress_equalweight_proxy','compressed','higher_low','rr_52w_scenario',
 'stop_atr','setup_ready','setup_developing','company']].to_string(index=False),flush=True)
