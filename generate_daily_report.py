"""Render the two official CSV lists offline; never run or alter the model."""
from pathlib import Path
from html import escape
import json
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def txt(value):
    if value is None or pd.isna(value):
        return "—"
    return str(value).strip() or "—"


def num(value, digits=1, prefix=""):
    try:
        return "—" if pd.isna(value) else f"{prefix}{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return txt(value)


def read_list(path, allowed):
    # Membership comes only from this CSV. Preserve source order and classification.
    frame = pd.read_csv(path, dtype={"symbol": str}, keep_default_na=False)
    required = {"symbol", "final_classification"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name}: missing columns {sorted(required - set(frame.columns))}")
    return frame.loc[frame["final_classification"].isin(allowed)].copy()


def read_metadata(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_card(row):
    def field(name):
        return escape(txt(row.get(name)))

    def number(name, digits=1, prefix=""):
        return escape(num(row.get(name), digits, prefix))

    classification = field("final_classification")
    return f'''<article class="stock-card">
      <div class="top-row"><div class="ticker">{field("symbol")}</div>
      <span class="score">Score <b>{number("total_score")}</b></span></div>
      <div class="details"><span class="pill {classification.lower()}">{classification}</span>
      <span class="price">{number("current_price", 2, "$")}</span></div>
      <div class="entry">Entry <span>{field("entry_proximity_state")}</span></div>
      <div class="stats"><span>RS20 <b>{number("rs_20d_pct")}</b></span>
      <span>RS60 <b>{number("rs_60d_pct")}</b></span></div>
    </article>'''


def render_section(label, frame):
    cards = "".join(make_card(row) for _, row in frame.iterrows())
    body = f'<div class="grid">{cards}</div>' if cards else '<p class="empty">No stocks in this list.</p>'
    return f'<section><div class="section-head"><h2>{label}</h2><span class="count">{len(frame)}</span></div>{body}</section>'


STYLE = """
* {box-sizing:border-box}
body {margin:0;background:#f5f5f7;color:#202124;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.page {width:min(1160px,calc(100% - 32px));margin:24px auto 40px}
.eyebrow {font-size:10px;font-weight:700;letter-spacing:.12em;color:#737782}
h1 {font-size:25px;letter-spacing:-.03em;margin:5px 0 12px}
.summary {display:flex;flex-wrap:wrap;gap:8px 20px;font-size:12px;color:#747780}
.summary b {font-weight:600;color:#373b43;margin-left:5px}
section {margin-top:24px}
.section-head {display:flex;align-items:center;gap:8px;margin-bottom:10px}
h2 {font-size:13px;letter-spacing:.06em;margin:0}
.count {background:#e7e8eb;border-radius:20px;padding:3px 7px;font-size:10px}
.grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.stock-card {background:white;border:1px solid #e6e7eb;border-radius:11px;padding:12px 13px;min-width:0}
.top-row {display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.ticker {font-size:23px;font-weight:750;letter-spacing:-.025em;overflow-wrap:anywhere}
.score {font-size:10px;color:#737782;white-space:nowrap}
.score b {font-size:12px;font-weight:500;color:#555b65}
.details {display:flex;align-items:center;justify-content:space-between;gap:6px;margin-top:7px}
.pill {border-radius:5px;padding:3px 5px;font-size:9px;font-weight:600}
.ready {background:#e5f5ea;color:#176a36}
.developing {background:#e9efff;color:#2c5ca7}
.watch {background:#fff2d8;color:#8a5900}
.price {font-size:12px;color:#646a74}
.entry {font-size:10px;color:#858993;margin-top:9px;overflow-wrap:anywhere}
.entry span {color:#646a74;margin-left:4px}
.stats {display:flex;gap:16px;border-top:1px solid #eff0f2;padding-top:7px;margin-top:8px;font-size:10px;color:#858993}
.stats b {color:#646a74;font-weight:500;margin-left:3px}
.empty {font-size:12px;color:#737782;padding:12px 0}
@media(max-width:900px) {.grid {grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:650px) {.grid {grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:380px) {.grid {grid-template-columns:1fr}}
"""


def main():
    biglist = read_list(OUT / "biglist_latest.csv", {"READY", "DEVELOPING"})
    watchlist = read_list(OUT / "watchlist_latest.csv", {"WATCH"})
    summary = read_metadata(OUT / "run_summary_latest.json")
    market = read_metadata(OUT / "regime_latest.json")
    as_of = txt(summary.get("as_of", market.get("as_of")))[:10]
    regime = txt(market.get("regime"))
    scanned = summary.get("scanned")
    complete = summary.get("scoring_complete")
    coverage = "—"
    if isinstance(scanned, (int, float)) and scanned > 0 and isinstance(complete, (int, float)):
        coverage = f"{complete / scanned * 100:.2f}%"
    metadata = (("as_of", as_of), ("regime", regime), ("scanned", num(scanned, 0)), ("coverage", coverage))
    header = "".join(f'<span>{label}<b>{escape(value)}</b></span>' for label, value in metadata)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Lists</title><style>{STYLE}</style></head><body><main class="page">
<header><div class="eyebrow">BIG LIST ENGINE</div><h1>Daily Lists</h1><div class="summary">{header}</div></header>
{render_section("BIG LIST", biglist)}
{render_section("WATCH LIST", watchlist)}
</main></body></html>'''
    path = OUT / "daily_report_latest.html"
    path.write_text(html, encoding="utf-8")
    print("Created:", path)
    for label, frame in (("BIG LIST", biglist), ("WATCH LIST", watchlist)):
        print(f"{label} ({len(frame)}): {', '.join(frame['symbol']) or '—'}")


if __name__ == "__main__":
    main()
