import pytest
from src.data.coverage import require_coverage
@pytest.mark.parametrize("count,passes", [(949,False),(950,True),(1000,True),(0,False)])
def test_fixed_pool_threshold(count, passes):
    summary=dict(pool_total=1000,valid_daily=count)
    if passes: require_coverage(summary)
    else:
        with pytest.raises(RuntimeError): require_coverage(summary)

def test_partial_disclosure(tmp_path):
    import json, shutil
    from pathlib import Path
    from two_lists import render
    source=Path(__file__).resolve().parents[1]/'Reports/20261104-045112-12999'
    shutil.copyfile(source/'full_scan_latest.parquet',tmp_path/'full_scan_latest.parquet')
    summary=json.loads((source/'run_summary_latest.json').read_text())
    summary['coverage']=dict(coverage_pct=95.1,valid_daily=4828,pool_total=5076,excluded_unassessed=248)
    (tmp_path/'run_summary_latest.json').write_text(json.dumps(summary))
    log=tmp_path/'run.log';log.touch()
    render(tmp_path,log)
    html=(tmp_path/'研究報告.html').read_text()
    assert 'Coverage 95.10%' in html
    assert '100% 為目標，95% 為完成有界補抓後的發布下限' in html
    assert '未覆蓋股票 248' in html
    assert '資料覆蓋率並非預測準確率' in html
    assert 'coverage_audit_latest.csv' in html
    assert '<details><summary>資料說明</summary>' in html
    assert html.index('Coverage 95.10%') > html.index('<header>')
