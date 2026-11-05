import json, subprocess, sys, shutil
from pathlib import Path
import pandas as pd
import pytest

engine=Path(__file__).resolve().parents[1]
import os
base=Path(os.environ.get('BIGLIST_PUBLICATION_FIXTURE', engine/'Reports/20261104-045112-12999-two-lists-v1'))
here=Path(__file__).resolve().parent

@pytest.mark.parametrize('offset,invalid',[(0,False),(1,False),(1,True)])
def test_publication(tmp_path,offset,invalid):
    for d in ('Logs','Reports','outputs'):
        (tmp_path/d).mkdir()
    (tmp_path/'Reports/Latest').symlink_to(base)
    (tmp_path/'src').symlink_to(engine/'src')
    for file in [engine/'daily.py',engine/'compare_runs.py',engine/'two_lists.py']:
        shutil.copyfile(file,tmp_path/file.name)
    sh=tmp_path/'run_daily.sh'
    day = (pd.Timestamp('2026-11-03') + pd.Timedelta(days=offset)).date()
    sh.write_text(f'#!/bin/sh\necho "Market as-of: {day}"\necho "Completed session: {day}; fixture"\n')
    sh.chmod(0o755)
    out=tmp_path/'outputs'
    shutil.copyfile(base/'audit_latest.csv',out/'audit_latest.csv')
    frame=pd.read_parquet(base/'full_scan_latest.parquet')
    frame.as_of+=pd.Timedelta(days=offset)
    frame.to_parquet(out/'full_scan_latest.parquet')
    info=json.loads((base/'run_summary_latest.json').read_text())
    info['as_of']=str(frame.as_of.iloc[0])
    info['scanned']+=int(invalid)
    (out/'run_summary_latest.json').write_text(json.dumps(info))
    # Freeze the fixture clock after both sessions, intercept the GUI call only.
    wrapper = """import datetime, runpy, subprocess, sys
import pandas
class Clock(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026,11,5,12,tzinfo=tz)
datetime.datetime=Clock
orig=subprocess.run
def open_fixture(args, **kw):
    if args[0]=='/usr/bin/open':
        from pathlib import Path
        assert Path(args[1]).is_file()
        Path('opened').write_text(args[1])
        return subprocess.CompletedProcess(args,0)
    return orig(args, **kw)
subprocess.run=open_fixture
runpy.run_path(sys.argv[1],run_name='__main__')
"""
    run=subprocess.run([sys.executable,'-c',wrapper,tmp_path/'daily.py'], cwd=tmp_path)
    assert (tmp_path/'opened').exists() == (not invalid)
    if not invalid:
        assert Path((tmp_path/'opened').read_text()).parent != base
    latest=(tmp_path/'Reports/Latest').resolve()
    assert bool(run.returncode)==invalid
    assert (latest==base)==(invalid or offset==0)
    if offset and not invalid:
        changes=pd.read_csv(latest/'名單變化.csv')
        assert list(changes.transition)==['Retained']*6

def test_events():
    sys.path.insert(0,str(engine))
    from compare_runs import transition
    for a,b,expected in [('Outside','Candidate','New'),('Candidate','Qualified','Moved to Qualified'),('Qualified','Candidate','Downgraded'),('Candidate','Outside','Dropped')]:
        assert transition(a,b)==expected
