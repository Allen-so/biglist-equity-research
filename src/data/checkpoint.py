"""Validated daily price receipts and persistent Yahoo cooldown."""
import hashlib
import fcntl
import json
import os
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time

import pandas as pd
from src.core.pipeline_io import session_close


def target_day():
    now = pd.Timestamp(os.environ.get("BIGLIST_STARTED_AT") or pd.Timestamp.now(tz="America/New_York"))
    day = now.tz_convert("America/New_York").tz_localize(None).normalize()
    for _ in range(10):
        close = session_close(day)
        if close is not None and now >= close + pd.Timedelta(minutes=30):
            return day
        day -= pd.Timedelta(days=1)
    raise ValueError("Unsupported market calendar")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def describe(path, frame, day):
    fields = ["open", "high", "low", "close", "volume"]
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or day not in frame.index:
        raise ValueError("missing session or invalid timestamps")
    bars = frame.loc[:day, fields]
    if bars.isna().any().any() or (bars.iloc[:, :4] <= 0).any().any() or (bars.volume < 0).any():
        raise ValueError("invalid OHLCV")
    return dict(provider="yahoo", symbol=path.stem, interval="1d", eod=str(day.date()),
                last_bar=str(bars.index[-1]), rows=len(bars), schema="adjusted-repair-ohlcv-v1",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                captured_at=path.stat().st_mtime)


def receipt_path(path, day):
    return path.parent.parent / "receipts" / str(day.date()) / (path.name + ".json")


def is_limited(exc):
    return (getattr(getattr(exc, "response", None), "status_code", None) == 429 or
            any(word in (type(exc).__name__ + str(exc)).lower() for word in
                ("429", "ratelimit", "rate limit", "rate-limit", "too many requests")))


def check_cooldown(root):
    path = root / "cooldown.json"
    if path.exists():
        until = json.loads(path.read_text())["until"]
        if time() < until:
            raise RuntimeError("Yahoo 429 cooldown until " + datetime.fromtimestamp(until, timezone.utc).isoformat())


def remote(root, call, **kwargs):
    check_cooldown(root)
    try:
        return call(**kwargs)
    except Exception as exc:
        if is_limited(exc):
            retry = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
            delay = 0
            if retry:
                try:
                    delay = float(retry)
                except ValueError:
                    try:
                        delay = parsedate_to_datetime(retry).timestamp() - time()
                    except (ValueError, TypeError, OverflowError):
                        delay = 0
            path = root / "cooldown.json"
            root.mkdir(parents=True, exist_ok=True)
            with (root / "cooldown.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                previous = json.loads(path.read_text()) if path.exists() else {}
                prior_until = previous.get("until", 0)
                if prior_until > time():
                    attempt = previous["attempt"]
                else:
                    attempt = min(6, previous.get("attempt", 0)+1) if prior_until > time()-86400 else 1
                until = max(prior_until, time() + max(delay, 900 * 2**(attempt-1)) + random.uniform(0, 60))
                atomic_json(path, dict(until=until, attempt=attempt))
            check_cooldown(root)
        raise


def raise_rate_limit():
    raise RuntimeError("Yahoo HTTP 429")
