import pandas as pd
import pytest
from src.core.pipeline_io import latest_completed_session

@pytest.mark.parametrize("clock,expected", [
    ("2026-11-04 12:49","2026-11-03"), ("2026-11-04 16:29","2026-11-03"),
    ("2026-11-04 16:30","2026-11-04"), ("2026-11-07 17:00","2026-11-06"),
    ("2026-11-26 17:00","2026-11-25"), ("2026-11-27 13:29","2026-11-25"),
    ("2026-11-27 13:30","2026-11-27"), ("2026-07-06 16:30","2026-07-06"),
])
def test_boundary(clock,expected):
    now=pd.Timestamp(clock,tz="America/New_York")
    bars=pd.DataFrame({"close":10},index=pd.date_range(now.date()-pd.Timedelta(days=7),now.date()))
    assert latest_completed_session(bars,now)==pd.Timestamp(expected)
