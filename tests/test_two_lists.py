import json
from pathlib import Path
import pandas as pd
import pytest
from two_lists import render


@pytest.mark.parametrize('change,qual,cand',[
    ({},True,False),({'resistance_price':None},False,True),
    ({'hourly_data_status':'MISSING'},False,True),
    ({'entry_price':None},False,True),({'high':0},False,False),
    ({'fundamental_delta_status':'UNKNOWN'},False,False),
    ({'reward_risk':0.27,'resistance_price':29},False,False)])
def test_daily(tmp_path,change,qual,cand):
    import os
    source=Path(os.environ.get('BIGLIST_FIXTURE',Path(__file__).resolve().parents[1]/'Reports/20261104-045112-12999'))
    base=pd.read_parquet(source/'full_scan_latest.parquet')
    row=base[base.symbol.eq('CBLL')].copy()
    summary=json.loads((source/'run_summary_latest.json').read_text())
    log=tmp_path/'log'
    log.touch()
    for date, edits in [('2026-11-03',{}),('2026-11-04',dict(final_classification='READY',admitted=True,setup_gate_pass=True,
            hourly_confirmation='SUPPORTIVE',entry_price=28,stop_price=27,resistance_price=31,**{}))]:
        frame=row.assign(as_of=pd.Timestamp(date),**edits)
        if edits:
            for key,value in change.items():
                frame[key]=value
        frame.to_parquet(tmp_path/'full_scan_latest.parquet')
        (tmp_path/'run_summary_latest.json').write_text(json.dumps(dict(summary,as_of=date,scanned=1,scoring_complete=1)))
        render(tmp_path,log)
        assert len(pd.read_csv(tmp_path/'候選名單.csv'))==(cand if edits else True)
        assert len(pd.read_csv(tmp_path/'符合名單.csv'))==(qual if edits else False)


def test_same_date_denominator():
    from src.core.scoring import FACTOR_WEIGHTS
    import numpy as np
    for score,industry_points,expected in [(86.6569,4.986352,84.7107),(83.0245,5.76046,82.143)]:
        reduced=100-FACTOR_WEIGHTS['group_fit']-FACTOR_WEIGHTS['group_leadership']
        assert np.isclose(score*reduced/100+industry_points,expected,atol=0.001)
