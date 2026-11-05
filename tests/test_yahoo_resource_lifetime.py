import errno
import os
import pandas as pd
import pytest
from src.data.yahoo_provider import YahooMarketDataProvider


def test_repeated_parquet_writes_release_descriptors(tmp_path):
    frame = pd.DataFrame({"close": [10.]})
    before = len(os.listdir("/dev/fd"))
    for _ in range(100):
        YahooMarketDataProvider._write_cache(None, frame, tmp_path / "price.parquet")
    assert len(os.listdir("/dev/fd")) <= before + 2
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "price.parquet"), frame)


def test_emfile_retry_preserves_prior_cache(tmp_path, monkeypatch):
    path = tmp_path / "price.parquet"
    path.write_bytes(b"old")
    calls = []
    def fail(frame, handle):
        calls.append(handle)
        handle.write(b"partial")
        raise OSError(errno.EMFILE, "Too many open files")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="Too many open files"):
        YahooMarketDataProvider._write_cache(None, pd.DataFrame(), path)
    assert len(calls) == 2 and all(handle.closed for handle in calls)
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]
