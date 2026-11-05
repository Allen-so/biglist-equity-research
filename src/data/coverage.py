"""Fixed original-pool coverage; never prediction accuracy."""
import pandas as pd

def assess(symbols, data, provider):
    rows=[]
    valid={}
    for symbol in symbols:
        path=provider._cache_path(symbol, interval="1d")
        frame=data.get(symbol)
        status="SKIPPED_NO_DATA"
        detail="No validated target-session receipt"
        if frame is not None and not frame.empty and provider._daily_hit(path):
            frame=frame.loc[:provider.expected_session]
            if len(frame)>=60:
                status="DATA_AVAILABLE";detail="Validated target EOD, receipt and session window";valid[symbol]=frame
        elif path.exists():
            try:
                cached=provider._read_cache(path).loc[:provider.expected_session]
                if len(cached)<60: status="INSUFFICIENT_HISTORY";detail="Fewer than original 60 daily bars"
                elif provider.expected_session not in cached.index: status="MISSING_EOD";detail="Target completed session absent"
                else: status="INVALID_RECEIPT_OR_WINDOW";detail="Receipt/schema/session continuity not validated"
            except Exception as exc: detail=str(exc)
        if symbol not in valid and getattr(provider, "download_failures", {}).get(symbol) == "TIMEOUT":
            status="TIMEOUT";detail="Hard batch wall-clock deadline exceeded"
        rows.append(dict(ticker=symbol,status=status,reason=detail))
    audit=pd.DataFrame(rows)
    total=len(symbols); count=len(valid)
    summary=dict(pool_total=total,valid_daily=count,coverage_pct=100*count/total if total else 0,
                 excluded_unassessed=total-count,threshold_pct=95,partial_universe=count<total,
                 insufficient_history=int((audit.status=="INSUFFICIENT_HISTORY").sum()))
    summary["coverage_definition"] = "Validated >=60 daily bars / original daily static pool; short histories excluded from numerator"
    summary["status_counts"] = audit.status.value_counts().to_dict()
    return valid,audit,summary

def require_coverage(summary):
    if not summary["pool_total"] or summary["valid_daily"]*100 < summary["pool_total"]*95:
        raise RuntimeError(f"Daily coverage below 95%: {summary}")
