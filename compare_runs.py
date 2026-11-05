"""Compare two immutable full-run directories after both lists are finalized."""
import json
import sys
from pathlib import Path
import pandas as pd


def snapshot(path):
    summary = json.loads((path / 'run_summary_latest.json').read_text())
    frame = pd.read_parquet(path / 'full_scan_latest.parquet').set_index('symbol', verify_integrity=True)
    date = pd.Timestamp(summary['as_of'])
    if (summary['scope'] != 'FULL' or len(frame) != summary['scanned']
            or not frame.as_of.eq(date).all() or frame.scoring_complete.sum() != summary['scoring_complete']):
        raise ValueError('快照不完整')
    state = pd.Series('Outside', index=frame.index)
    for label, name in [('Candidate','候選名單'), ('Qualified','符合名單')]:
        symbols = pd.read_csv(path / (name + '.csv')).symbol
        if symbols.duplicated().any() or not state.loc[symbols].eq('Outside').all():
            raise ValueError('名單重複')
        state.loc[symbols] = label
    return date, state, frame


def transition(before, after):
    if after == 'Outside':
        return 'Dropped'
    if before == after:
        return 'Retained'
    if before == 'Outside':
        return 'New'
    return 'Moved to Qualified' if after == 'Qualified' else 'Downgraded'


def compare(old, new):
    a, prev, left = snapshot(old)
    b, curr, right = snapshot(new)
    if b < a:
        raise ValueError('交易日倒序')
    syms = prev[prev != 'Outside'].index.union(curr[curr != 'Outside'].index)
    result = pd.DataFrame(index=syms)
    for tag, state, frame in [('before',prev,left),('after',curr,right)]:
        result[tag] = state.reindex(syms, fill_value='Outside')
        cols = ['total_score','available_maximum_points','missing_factors','setup_state','hourly_confirmation','hourly_data_status','rs_20d_pct','relative_volume_20','pivot_distance_atr','entry_price','stop_price','resistance_price','reward_risk','fundamental_delta_status','fundamental_delta_pct','rejection_reasons','classification_reason']
        result = result.join(frame.reindex(index=syms,columns=cols).add_prefix(tag+'_'))
    result['transition'] = [transition(x,y) for x,y in zip(result.before,result.after)]
    result['comparison'] = 'same-session vintage' if a == b else 'different sessions'
    result['old_session'],result['new_session'] = a,b
    return result


if __name__ == '__main__':
    compare(*map(Path,sys.argv[1:3])).to_csv(sys.argv[3])
