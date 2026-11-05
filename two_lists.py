import json
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.core.scoring import FACTOR_WEIGHTS


def render(dest, log):
    summary = json.loads((dest/'run_summary_latest.json').read_text())
    s = pd.read_parquet(dest/'full_scan_latest.parquet').set_index('symbol')
    if (not s.index.is_unique or summary['scope'] != 'FULL' or len(s) != summary['scanned'] or
        s.as_of.ne(pd.Timestamp(summary['as_of'])).any() or
        s.scoring_complete.sum() != summary['scoring_complete']):
        raise ValueError('不完整或日期不一致')
    points = s[[f'{key}_points' for key in FACTOR_WEIGHTS]]
    denominator = points.notna().mul(list(FACTOR_WEIGHTS.values())).sum(axis=1)
    if not np.allclose(denominator, s.available_maximum_points, equal_nan=True):
        raise ValueError('計分分母不一致')
    s['shadow_score'] = points.sum(axis=1).where(s.scoring_complete)
    for name in ['total_score','shadow_score']:
        s[name+'_rank'] = s[name].rank(ascending=False, method='min')
    s['coverage_note'] = np.where(denominator.eq(100),'100/100','部分因子缺失')
    fallback = {}
    for sym, file, ts in re.findall(r'Financial cache fallback (\S+) (\S+) captured=([\d.]+)',log.read_text()):
        fallback.setdefault(sym, []).append(file+' '+datetime.fromtimestamp(float(ts), timezone.utc).isoformat())
    s['來源'] = [', '.join(fallback.get(sym, ['UNKNOWN'])) for sym in s.index]
    s['發布時點'] = 'UNKNOWN'
    valid = (s.scoring_complete & s.close.gt(0) & s.atr_14.gt(0) & s.low.gt(0) &
             s.high.ge(s[['open','close']].max(axis=1)) & s.low.le(s[['open','close']].min(axis=1)) &
             s.median_dollar_volume_20d.gt(0) & s.fundamental_delta_status.eq('VALUE') &
             s.fundamental_delta_pct.ge(50) & ~s.structure_hard_fail & s.admission_channel_pass &
             s.rs_20d_pct.ge(75) & s.supply_demand_pct.ge(50))
    entry = pd.to_numeric(s.get('entry_price', pd.Series(np.nan,index=s.index)), errors='coerce')
    s["entry_price"] = entry
    s.loc[s.hourly_data_status.ne("AVAILABLE"), "hourly_confirmation"] = "MISSING"
    risk = entry - s.stop_price
    r = (pd.to_numeric(s.resistance_price,errors='coerce')-entry)/risk.where(risk.gt(0))
    q = (valid & s.admitted & s.score_gate_pass & s.final_classification.eq('READY') &
         s.hourly_data_status.eq('AVAILABLE') & s.hourly_confirmation.eq('SUPPORTIVE') &
         s.stop_price.gt(0) & r.ge(3) & np.isfinite(r))
    early = (s.total_score.ge(s.regime_minimum_score-3) & s.pivot_distance_atr.le(1) &
             s.tradability_gate_pass & s.reward_risk.ge(3) & s.pivot_cluster_count.ge(2))
    tracked = s.final_classification.isin(['READY','DEVELOPING','WATCH','EXTENDED'])
    possible = valid & ~q & (s.reward_risk.isna() | s.reward_risk.ge(3)) & (early | (tracked & s.total_score.ge(s.regime_minimum_score-3)))
    s['核查R'] = r
    s['估算R'] = s.reward_risk
    s['升級條件'] = ['；'.join(filter(None,[
        '等新Base／回調' if row.final_classification=='EXTENDED' else '',
        '等有效Setup' if not row.setup_gate_pass else '',
        '等小時支持（目前 '+str(row.hourly_confirmation)+'）' if row.hourly_confirmation!='SUPPORTIVE' else '',
        '重新確認樞紐 '+str(row.pivot_price)+'、日週結構與成交量；須達原READY並以實際入場、停損、目標核查≥3R'])) for row in s.itertuples()]
    s['失效條件'] = ['跌破支撐 '+str(x) if pd.notna(x) else '支撐待確認' for x in s.support_reference]
    cols = 'total_score available_maximum_points coverage_note missing_factors fundamental_delta_status fundamental_delta_pct fundamental_missing_components 來源 發布時點 rs_20d_pct median_dollar_volume_20d atr_14 pivot_price pivot_distance_atr extension_atr daily_trendline_count weekly_trendline_count hourly_confirmation hourly_data_status relative_volume_20 supply_demand_pct setup_state rejection_reasons final_classification classification_reason entry_price stop_price resistance_price 估算R 核查R 升級條件 失效條件'.split()
    s['待確認項數'] = ((~s.setup_gate_pass).astype(int) + (~s.tradability_gate_pass).astype(int) +
                    s.hourly_confirmation.ne('SUPPORTIVE').astype(int) + r.fillna(0).lt(3).astype(int))
    cols += ['待確認項數','tradability_gate_pass','structure_score','support_hold','pivot_cluster_count','fundamental_coverage_weight']
    candidates = s[possible].sort_values(['待確認項數','fundamental_delta_pct','structure_score','supply_demand_pct','rs_20d_pct','pivot_distance_atr','估算R','atr_pct'],ascending=[True,False,False,False,False,True,False,True],kind='stable').head(12)
    from html import escape
    from src.core.pipeline_io import session_close
    close = session_close(summary["as_of"])
    hk = close.tz_convert("Asia/Hong_Kong").strftime("%m/%d %H:%M") if close else "UNKNOWN"
    html = ["""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Big List · Daily Research</title><style>
    :root{color-scheme:light dark;--bg:#f6f6f4;--card:white;--ink:#252827;--muted:#535d58;--line:#d5d9d6}
    @media(prefers-color-scheme:dark){:root{--bg:#191b1a;--card:#242725;--ink:#f1f4f1;--muted:#b5c0b9;--line:#484f4a}}
    *{box-sizing:border-box}body{background:var(--bg);color:var(--ink);max-width:1000px;margin:auto;padding:20px clamp(14px,4vw,48px);font:15px/1.55 system-ui,sans-serif}h1{font-size:clamp(23px,4vw,32px);letter-spacing:-.045em;margin:4px 0}small,p{color:var(--muted)}nav{position:sticky;top:0;background:var(--bg);padding:9px 0;border-block:1px solid var(--line);z-index:1}a{color:inherit;margin-right:20px}h2{font-size:19px;scroll-margin-top:54px}article{background:var(--card);border:1px solid var(--line);padding:13px 17px;margin:8px 0;border-radius:12px}article strong{font-size:19px}article p{margin:4px 0}summary{cursor:pointer}dl{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);font-size:14px}dt,dd{overflow-wrap:anywhere;margin:0;padding:4px;border-bottom:1px solid var(--line)}
    </style><header><small>美股研究　/　收盤版</small><h1>Big List · Daily Research</h1></header>""",
    f"<p>美東 EOD {summary['as_of'][:10]}　·　香港收盤 {hk}<br>全市場 {summary['directory_count']:,} → 靜態合格 {summary['static_eligible']:,} → 市場門檻 {summary['universe_pass']:,} → 完整評分 {summary['scoring_complete']:,}</p>",
    f'<nav><a href="#cand">候選 {len(candidates)}</a><a href="#qual">符合 {int(q.sum())}</a>僅供研究</nav>']
    for name,frame,anchor in [('候選名單',candidates,'cand'),('符合名單',s[q],'qual')]:
        frame[cols].to_csv(dest/(name+'.csv'),na_rep='UNKNOWN')
        html.append(f'<h2 id="{anchor}">{name} · {len(frame)}</h2>')
        if frame.empty:
            html.append('<p>本次無符合條件的股票。</p>')
        for sym,row in frame.iterrows():
            def value(key):
                v = row[key]
                if pd.isna(v):
                    return 'UNKNOWN'
                return escape(f'{v:.2f}' if isinstance(v,float) else str(v))
            html.append(f"<article><strong>{escape(str(sym))}</strong>　{value('total_score')} 分　·　覆蓋 {value('available_maximum_points')}/100<p>收盤 {value('close')}　·　RS {value('rs_20d_pct')}　·　距樞紐 {value('pivot_distance_atr')} ATR</p><p>型態 {value('setup_state')}　·　小時線 {value('hourly_confirmation')}　·　核實 3R：{value('核查R')}</p><p>研究觀察　·　{value('final_classification')}<br>{escape(re.sub(r'(?<=\d)\.\d{3,}', lambda m: f'{float(m.group()):.2f}'[1:], str(row['升級條件'])))}</p><details><summary>詳細資料</summary><p>基本面快取回退上限 45 日；發布時點未證。延伸幅度與距樞紐分開計算。</p><dl>")
            html.extend(f"<dt>{escape(str(k))}</dt><dd>{value(k)}</dd>" for k in row.index)
            html.append("</dl></details></article>")
    (dest/'研究報告.html').write_text(''.join(html))
    s['結果'] = np.select([q,s.index.isin(candidates.index),~valid,s.reward_risk.lt(3),~(early|tracked)],['符合','候選','基本面／結構／量價或資料不足','估算不足3R','未達早期結構條件'],default='分數或容量限制')
    s[s.score_gate_pass][['total_score','available_maximum_points','missing_factors','shadow_score','total_score_rank','shadow_score_rank','regime_minimum_score','結果','rejection_reasons','估算R']].to_csv(dest/'score-audit.csv')

_original_render = render
def render(dest, log):
    _original_render(dest, log)
    from html import escape
    page = dest / '研究報告.html'
    text = page.read_text()
    summary = json.loads((dest / 'run_summary_latest.json').read_text())
    articles = re.findall(r'<article>(.*?)</article>', text, re.S)
    groups = []
    for filename, label in [('候選名單.csv', '候選'), ('符合名單.csv', '符合')]:
        frame = pd.read_csv(dest / filename)
        cards = []
        for symbol in frame.iloc[:, 0].astype(str):
            body = next(a for a in articles if a.startswith('<strong>'+escape(symbol)+'</strong>'))
            detail = body[body.index('<details>'):].replace('<details><summary>詳細資料</summary>', '', 1)
            detail = detail.rsplit('</details>', 1)[0]
            cards.append('<details class="stock"><summary>'+escape(symbol)+'</summary>'+detail+'</details>')
        groups.append('<section><h2>'+label+' '+str(len(frame))+'</h2><div class="grid">'+''.join(cards)+'</div></section>')
    c = summary.get('coverage', {})
    note = 'Coverage '+format(c.get('coverage_pct', 0), '.2f')+'%'
    page.write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Big List</title><style>*{box-sizing:border-box}body{max-width:1040px;margin:48px auto;padding:0 24px;color:#222;background:#fff;font:15px/1.6 system-ui,sans-serif}h1{font-size:24px;margin:0}h2{font-size:17px;margin:30px 0 12px}p,footer{color:#777;font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px 24px}.stock{min-width:0;border-bottom:1px solid #eee;padding:12px 0}.stock>summary{font-size:20px;font-weight:600;cursor:pointer;list-style:none}.stock[open]{grid-column:1/-1}dl{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);font-size:13px}dt,dd{margin:0;padding:5px;overflow-wrap:anywhere}footer{margin-top:40px}a{color:inherit}@media(max-width:600px){body{margin:24px auto;padding:0 20px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><header><h1>Big List</h1><p>美股收盤 '+escape(summary['as_of'][:10])+'</p></header>'+''.join(groups)+'<footer>'+note+'<details><summary>資料說明</summary><p>未覆蓋股票 '+str(c.get('excluded_unassessed', 0))+'。本次為部分 Universe；資料覆蓋率並非預測準確率。100% 為目標，95% 為完成有界補抓後的發布下限。</p><a href="coverage_audit_latest.csv">逐股覆蓋審計</a> · <a href="候選名單.csv">候選 CSV</a> · <a href="符合名單.csv">符合 CSV</a> · <a href="full_scan_latest.parquet">完整原始評分</a></details></footer></html>')
