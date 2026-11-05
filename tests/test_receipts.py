import pandas as pd
import pytest
from src.data.yahoo_provider import YahooMarketDataProvider
from src.data import checkpoint

def test_restart_keeps_good_price_and_rejects_bad_replacement(tmp_path):
    day = pd.Timestamp("2026-11-03")
    original = pd.DataFrame(dict(open=[1.],high=[2.],low=[1.],close=[2.],volume=[20]),index=[day])
    p = YahooMarketDataProvider(cache_dir=tmp_path)
    p.expected_session = day
    path = p._cache_path("AAA", interval="1d")
    p._write_cache(original,path)
    q = YahooMarketDataProvider(cache_dir=tmp_path)
    q.expected_session = day
    assert list(q.download_daily_batch(["AAA"])) == ["AAA"]
    with pytest.raises(ValueError):
        q._write_cache(original.shift(freq="1D"),path)
    assert q._daily_hit(path)
    q.expected_session += pd.Timedelta(days=1)
    assert not q._daily_hit(path)

def test_rate_limit_stops_next_request(tmp_path):
    def limited():
        raise RuntimeError("HTTP 429")
    with pytest.raises(RuntimeError,match="cooldown until"):
        checkpoint.remote(tmp_path,limited)
    with pytest.raises(RuntimeError,match="cooldown until"):
        checkpoint.remote(tmp_path,lambda: pytest.fail("network used"))

@pytest.mark.parametrize("revision", [False, True])
def test_incremental_overlap_or_full_rebuild(tmp_path, monkeypatch, revision):
    import numpy as np
    from src.data import yahoo_provider as module
    from concurrent.futures import Future

    days = pd.bdate_range(end="2026-11-03", periods=500)
    values = np.arange(len(days), dtype=float) + 10
    full = pd.DataFrame(dict(open=values, high=values, low=values,
                             close=values, volume=values), index=days)
    old = full.iloc[:-1].copy()
    fetched = full.copy()
    if revision:
        fetched.loc[:, ["open", "high", "low", "close"]] *= 0.5
    calls = []
    class Executor:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args, **kw):
            future = Future()
            if fn is module._incremental_history:
                opts = args[1]
                assert opts["end"] == "2026-11-04"
                assert opts["start"] == str(old.index[-5].date())
                calls.append("gap")
                future.set_result(fetched.loc[opts["start"]:].rename(columns=str.capitalize))
            else:
                calls.append("full")
                future.set_result(fetched.rename(columns=str.capitalize))
            return future
    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(module.clock, "sleep", lambda _: None)
    monkeypatch.setattr(module, "_load_yfinance", lambda: None)
    p = YahooMarketDataProvider(cache_dir=tmp_path)
    path = p._cache_path("AAA", interval="1d")
    p._write_cache(old, path)
    p.expected_session = days[-1]
    p.benchmark_sessions = days
    result = p.download_daily_batch(["AAA"])
    pd.testing.assert_frame_equal(result["AAA"], fetched, check_freq=False)
    assert calls == (["gap", "full"] if revision else ["gap"])
    assert p._daily_hit(path)

def test_http_date_retry_after_controls_deadline(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from email.utils import format_datetime
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(checkpoint, "time", lambda: 1000)
    error = RuntimeError("busy")
    error.response = SimpleNamespace(status_code=429, headers={
        "Retry-After": format_datetime(datetime.fromtimestamp(10000, timezone.utc), usegmt=True)})
    def busy():
        raise error
    with pytest.raises(RuntimeError, match="cooldown until"):
        checkpoint.remote(tmp_path, busy)
    assert json.loads((tmp_path / "cooldown.json").read_text())["until"] >= 10000
