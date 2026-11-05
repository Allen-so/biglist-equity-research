from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
import json
import hashlib
import errno
import os
import resource
import time as clock
from time import time
from typing import Iterable

from src.data import checkpoint
import numpy as np
import pandas as pd

from src.features.basic import (
    normalize_ohlcv,
)
from src.universe.us_equities import (
    to_yahoo_symbol,
)


def _load_yfinance():
    try:
        import yfinance as yf

    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: "
            "pip install yfinance"
        ) from exc

    return yf


def _download_batch(**options):
    # yfinance creates internal Ticker sessions even with a caller session.
    # A short-lived process gives each batch a deterministic resource lifetime.
    # download() logs per-ticker failures instead of raising them.
    limited = []
    class RateLimitObserver(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if checkpoint.is_limited(message):
                limited.append(message)
    logger = logging.getLogger("yfinance")
    observer = RateLimitObserver()
    logger.addHandler(observer)
    try:
        result = _load_yfinance().download(**options)
    finally:
        logger.removeHandler(observer)
        observer.close()
    if limited:
        if result is None:
            raise RuntimeError("Yahoo HTTP 429")
        result.attrs["yahoo_limited"] = True
    print("Yahoo batch", len(options["tickers"]), "FD at batch end",
          len(os.listdir("/dev/fd")), "limit", resource.getrlimit(resource.RLIMIT_NOFILE), flush=True)
    return result


def _bounded_call(call, *args, wall_seconds=None, **kwargs):
    """Terminate and reap workers before returning on a hard deadline (Python 3.14)."""
    limit = float(os.environ.get("BIGLIST_BATCH_TIMEOUT", "45")) if wall_seconds is None else wall_seconds
    if limit <= 0:
        raise ValueError("Batch timeout must be positive")
    worker = ProcessPoolExecutor(max_workers=1)
    try:
        return worker.submit(call, *args, **kwargs).result(timeout=limit)
    except FutureTimeout:
        processes = list(worker._processes.values())
        worker.kill_workers()
        for process in processes:
            process.join(timeout=2)
        raise TimeoutError(f"TIMEOUT after {limit:g}s") from None
    finally:
        worker.__exit__(None, None, None)


def _incremental_history(symbol, options):
    return _load_yfinance().Ticker(symbol).history(**options)


def _normalize_yahoo_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    result = frame.copy()

    #
    # Keep only OHLCV.
    #
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    result = result.rename(
        columns=rename
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = set(
        required
    ).difference(
        result.columns
    )

    if missing:
        raise ValueError(
            "Yahoo OHLCV missing columns: "
            f"{sorted(missing)}"
        )

    result = result[
        required
    ].copy()

    result.index = pd.to_datetime(
        result.index
    )

    #
    # yfinance intraday normally returns
    # timezone-aware timestamps.
    #
    if (
        isinstance(
            result.index,
            pd.DatetimeIndex,
        )
        and
        result.index.tz is not None
    ):
        result.index = (
            result.index
            .tz_convert(
                "America/New_York"
            )
            .tz_localize(
                None
            )
        )

    result = (
        result
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
    )

    return normalize_ohlcv(
        result
    )


class YahooMarketDataProvider:
    """
    Low-cost production bootstrap provider.

    DAILY:
        adjusted daily OHLCV
        default 2 years

    HOURLY:
        1H regular-session OHLCV
        recent 60 days only

    Cached as parquet to avoid repeatedly hitting
    the upstream provider.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path = (
            "data/cache/yahoo"
        ),
        daily_period: str = "2y",
        hourly_period: str = "60d",
        daily_cache_minutes: int = 720,
        hourly_cache_minutes: int = 30,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.daily_period = (
            daily_period
        )

        self.hourly_period = (
            hourly_period
        )

        self.daily_cache_minutes = (
            daily_cache_minutes
        )

        self.hourly_cache_minutes = (
            hourly_cache_minutes
        )

    def _cache_path(
        self,
        symbol: str,
        *,
        interval: str,
    ) -> Path:
        safe = (
            to_yahoo_symbol(
                symbol
            )
            .replace(
                "/",
                "_",
            )
        )

        return (
            self.cache_dir
            / interval
            / f"{safe}.parquet"
        )

    @staticmethod
    def _cache_fresh(
        path: Path,
        *,
        ttl_minutes: int,
    ) -> bool:
        if not path.exists():
            return False

        age_seconds = (
            time()
            - path.stat().st_mtime
        )

        if age_seconds > ttl_minutes * 60:
            return False

        # A daily bar downloaded before the close remains partial in the cache,
        # even when the wall clock has subsequently passed 16:00 New York.
        # Refresh it after the close instead of treating the cached bar as final.
        now = pd.Timestamp(time(), unit="s", tz="UTC").tz_convert("America/New_York")
        if path.parent.name == "1d":
            from src.core.pipeline_io import session_close
            close = session_close(now.tz_localize(None))
            if close is not None and now >= close:
                boundary = close + pd.Timedelta(minutes=30)
                if path.stat().st_mtime < (boundary if now >= boundary else close).timestamp():
                    return False
        return True

    def _history_valid(self, frame):
        if getattr(self, "benchmark_sessions", None) is not None and len(frame.loc[:self.expected_session]) < 60:
            return False
        return not frame.empty and self._history_valid_through(frame, getattr(self, "expected_session", None))

    def _history_valid_through(self, frame, last):
        sessions = getattr(self, "benchmark_sessions", None)
        if sessions is None:
            return True
        required = sessions[(sessions >= frame.index[0]) & (sessions <= last)]
        return required.isin(frame.index).all()

    def _daily_hit(self, path):
        day = getattr(self, "expected_session", None)
        if day is None:
            return self._cache_fresh(path, ttl_minutes=self.daily_cache_minutes)
        try:
            frame = self._read_cache(path)
            if not self._history_valid(frame):
                return False
            actual = checkpoint.describe(path, frame, day)
            receipt = checkpoint.receipt_path(path, day)
            if not receipt.exists():
                from src.core.pipeline_io import session_close
                if (len(frame.loc[:day]) < 480 or
                        path.stat().st_mtime < (session_close(day) + pd.Timedelta(minutes=30)).timestamp()):
                    return False
                checkpoint.atomic_json(receipt, actual)
            saved = json.loads(receipt.read_text())
            return actual == saved
        except (OSError, ValueError, KeyError):
            return False

    def _read_cache(
        self,
        path: Path,
    ) -> pd.DataFrame:
        frame = pd.read_parquet(
            path
        )

        frame.index = pd.to_datetime(
            frame.index
        )

        return normalize_ohlcv(
            frame
        )

    def _write_cache(
        self,
        frame: pd.DataFrame,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        for attempt in range(2):
            temporary = None
            try:
                with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    frame.to_parquet(handle)
                day = getattr(self, "expected_session", None)
                if path.parent.name == "1d" and day is not None:
                    if not self._history_valid(frame):
                        raise ValueError("Missing benchmark trading sessions")
                    receipt = checkpoint.describe(temporary, frame, day)
                    receipt["symbol"] = path.stem
                temporary.replace(path)
                if path.parent.name == "1d" and day is not None:
                    checkpoint.atomic_json(checkpoint.receipt_path(path, day), receipt)
                return
            except OSError as exc:
                if exc.errno != errno.EMFILE or attempt:
                    raise
                print(f"EMFILE saving {path}; retrying once", flush=True)
                clock.sleep(1)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def get_daily(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(
            symbol,
            interval="1d",
        )

        if (
            not refresh
            and
            self._daily_hit(cache_path)
        ):
            return self._read_cache(
                cache_path
            )

        yahoo_symbol = (
            to_yahoo_symbol(
                symbol
            )
        )

        raw = checkpoint.remote(self.cache_dir, lambda **options: _bounded_call(
            _incremental_history, yahoo_symbol, options),
            timeout=float(os.environ.get("BIGLIST_REQUEST_TIMEOUT", "15")),
            period=self.daily_period,
            interval="1d",
            auto_adjust=True,
            actions=False,
            repair=True,
            prepost=False,
            raise_errors=True,
        )

        result = (
            _normalize_yahoo_frame(
                raw
            )
        )

        if not result.empty:
            self._write_cache(
                result,
                cache_path,
            )

        return result

    def get_hourly(
        self,
        symbol: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(
            symbol,
            interval="1h",
        )

        expected = getattr(self, "expected_session", None)
        def complete(frame):
            from src.core.pipeline_io import session_close
            if expected is None:
                return True
            sessions = getattr(self, "benchmark_sessions", None)
            if sessions is None:
                days = [expected]
            else:
                now = pd.Timestamp(os.environ.get("BIGLIST_STARTED_AT") or pd.Timestamp.now(tz="America/New_York"))
                today = now.tz_convert("America/New_York").tz_localize(None)
                # Yahoo's 60-session range may include the current trading day.
                count = 59 if today.normalize() > expected and session_close(today) is not None and (today.hour, today.minute) >= (9, 30) else 60
                days = sessions[sessions <= expected][-count:]
            for day in days:
                close = session_close(day)
                if close is None:
                    return False
                required = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                    close.tz_localize(None), freq="h", inclusive="left")
                if not required.isin(frame.index).all():
                    return False
            return True

        def hourly_receipt(frame, *, fresh):
            if expected is None:
                return True
            visible = frame.loc[:expected + pd.Timedelta(hours=16)]
            dest = self.cache_dir / "receipts" / str(expected.date()) / "1h" / (cache_path.name + ".json")
            actual = dict(provider="yahoo", symbol=cache_path.stem, interval="1h", eod=str(expected.date()),
                          schema="chart-rth-ohlcv-v1", rows=len(visible), last_bar=str(visible.index[-1]),
                          captured_at=cache_path.stat().st_mtime,
                          sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest())
            if not fresh and dest.exists():
                return json.loads(dest.read_text()) == actual
            checkpoint.atomic_json(dest, actual)
            return True

        if not refresh and cache_path.exists():
            try:
                cached = self._read_cache(cache_path)
                if complete(cached) and (expected is not None or self._cache_fresh(
                        cache_path, ttl_minutes=self.hourly_cache_minutes)) and hourly_receipt(cached, fresh=False):
                    return cached if expected is None else cached.loc[:expected + pd.Timedelta(hours=16)]
            except (ValueError, KeyError, OSError):
                pass

        from src.data.hourly_chart import fetch_hourly
        result = normalize_ohlcv(checkpoint.remote(self.cache_dir,
            lambda: fetch_hourly(to_yahoo_symbol(symbol))))
        if not complete(result):
            raise ValueError(f"Incomplete hourly session for {symbol}: {expected}")

        if not result.empty:
            self._write_cache(
                result,
                cache_path,
            )

            hourly_receipt(result, fresh=True)

        return result

    def download_daily_batch(
        self,
        symbols: Iterable[str],
        *,
        batch_size: int = 100,
        refresh: bool = False,
    ) -> dict[
        str,
        pd.DataFrame,
    ]:
        """
        Full-universe daily downloader.

        Uses yfinance batch downloads instead of
        thousands of sequential Ticker.history calls.
        """

        symbols = [
            str(symbol)
            for symbol in symbols
        ]

        result: dict[
            str,
            pd.DataFrame,
        ] = {}

        need_download = []

        #
        # First reuse valid cache.
        #
        for symbol in symbols:
            path = self._cache_path(
                symbol,
                interval="1d",
            )

            if (
                not refresh
                and
                self._daily_hit(path)
            ):
                try:
                    result[
                        symbol
                    ] = self._read_cache(
                        path
                    )

                    continue

                except Exception:
                    pass

            if not refresh and path.exists() and getattr(self, "expected_session", None) is not None:
                try:
                    known = self._read_cache(path).loc[:self.expected_session]
                    if self.expected_session in known.index and len(known) < 60:
                        continue  # Original history gate; never count as available.
                except (ValueError, OSError, KeyError):
                    pass
            need_download.append(
                symbol
            )

        print(f"Daily cache complete {len(result)} / update needed {len(need_download)}", flush=True)
        # Update recent daily gaps with an adjusted-price overlap. Any
        # disagreement requires a full-period fetch before replacing history.
        def progress():
            if getattr(self, "expected_session", None) is not None:
                missing = [symbol for symbol in symbols if symbol not in result]
                path = self.cache_dir / "progress" / str(self.expected_session.date()) / "daily.json"
                checkpoint.atomic_json(path, dict(eod=str(self.expected_session.date()), universe=len(symbols),
                    complete=len(result), pending=missing, failures=getattr(self, "download_failures", {})))
        progress()
        import os
        deadline = clock.monotonic() + min(900, max(1, int(os.environ.get("BIGLIST_COVERAGE_SECONDS", "600"))))
        request_limit = min(300, max(1, int(os.environ.get("BIGLIST_COVERAGE_REQUESTS", "150"))))
        requests_used = 0
        self.download_failures = {}
        def allowed():
            nonlocal requests_used
            if clock.monotonic() >= deadline or requests_used >= request_limit:
                print("補抓停止：時間或 request budget；95% 不是停止條件", flush=True)
                return False
            requests_used += 1
            return True
        pending = []
        day = getattr(self, "expected_session", None)
        for symbol in need_download:
            path = self._cache_path(symbol, interval="1d")
            if day is None or refresh or not path.exists() or len(need_download) > 250:
                pending.append(symbol)
                continue
            try:
                old = self._read_cache(path).loc[:day]
                last = old.index[-1]
                eligible = len(old) >= 480 and 0 < (day-last).days <= 14 and self._history_valid_through(old, last)
            except (ValueError, OSError, KeyError, IndexError):
                eligible = False
            if not eligible:
                pending.append(symbol)
                continue
            if not allowed():
                return result
            start = old.index[-5]
            clock.sleep(0.5)
            options = dict(start=str(start.date()), end=str((day+pd.Timedelta(days=1)).date()),
                           interval="1d", auto_adjust=True, repair=True, actions=False,
                           prepost=False, raise_errors=True, timeout=float(os.environ.get("BIGLIST_REQUEST_TIMEOUT", "15")))
            try:
                raw = checkpoint.remote(self.cache_dir, lambda: _bounded_call(
                    _incremental_history, to_yahoo_symbol(symbol), options,
                    wall_seconds=min(float(os.environ.get("BIGLIST_BATCH_TIMEOUT", "45")), max(.01, deadline-clock.monotonic()))))
            except Exception as exc:
                if checkpoint.is_limited(exc):
                    progress()
                    return result
                if isinstance(exc, TimeoutError):
                    self.download_failures[symbol] = "TIMEOUT"
                    progress()
                    continue
                raise
            patch = _normalize_yahoo_frame(raw)
            if day not in patch.index or not old.loc[start:].equals(patch.loc[:last]):
                pending.append(symbol)
                continue
            combined = pd.concat([old.loc[old.index < start], patch]).loc[day-pd.DateOffset(years=2):day]
            self._write_cache(combined, path)
            result[symbol] = combined
            progress()
        need_download = pending
        batch_size = min(20, batch_size)
        if not need_download:
            return result

        for start in range(
            0,
            len(
                need_download
            ),
            batch_size,
        ):
            if not allowed():
                return result
            batch = need_download[
                start:
                start + batch_size
            ]

            yahoo_map = {
                to_yahoo_symbol(
                    symbol
                ):
                    symbol
                for symbol in batch
            }

            try:
                raw = checkpoint.remote(self.cache_dir, lambda **kw: _bounded_call(_download_batch, wall_seconds=min(float(os.environ.get("BIGLIST_BATCH_TIMEOUT", "45")), max(.01, deadline-clock.monotonic())), **kw),
                    tickers=list(
                        yahoo_map.keys()
                    ),
                    period=self.daily_period,
                    interval="1d",
                    auto_adjust=True,
                    repair=True,
                    actions=False,
                    progress=False,
                    threads=2,
                    timeout=float(os.environ.get("BIGLIST_REQUEST_TIMEOUT", "15")),
                    group_by="ticker",
                )

            except Exception as exc:
                if checkpoint.is_limited(exc):
                    progress()
                    return result
                if isinstance(exc, TimeoutError):
                    self.download_failures.update({symbol: "TIMEOUT" for symbol in batch})
                    progress()
                    continue
                raise

            limited = raw is not None and raw.attrs.get("yahoo_limited", False)
            try:
                for ticker, symbol in yahoo_map.items():
                    if raw is None or raw.empty:
                        break
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker not in raw.columns.get_level_values(0):
                            continue
                        frame = raw[ticker]
                    elif len(yahoo_map) == 1:
                        frame = raw
                    else:
                        continue
                    try:
                        normalized = _normalize_yahoo_frame(frame)
                        if normalized.empty:
                            continue
                        self._write_cache(normalized, self._cache_path(symbol, interval="1d"))
                        result[symbol] = normalized
                    except (KeyError, ValueError) as exc:
                        print(f"{symbol}: incomplete daily data ({exc})", flush=True)
            finally:
                progress()
            if limited:
                try:
                    checkpoint.remote(self.cache_dir, checkpoint.raise_rate_limit)
                except RuntimeError:
                    return result
            print(f"Daily completed {len(result)}/{len(symbols)} remaining {len(symbols)-len(result)}", flush=True)
            clock.sleep(1)

        return result