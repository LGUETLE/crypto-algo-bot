"""
Data Fetcher
============
Binance USD-M futures history is used as a price-history proxy for whatever venue you trade on
(the trading venue is only used for live data).

Sources, in order:
  1. ccxt (Binance REST: fetch_ohlcv / fetch_funding_rate_history)
  2. fallback if the REST API is unreachable (e.g. HTTP 451 in restricted regions):
     Binance's official public archive https://data.binance.vision (same USD-M futures klines
     and funding history, SHA-256 verified against the published .CHECKSUM files)

Caching: parquet files in data/cache/.
  - fixed windows (start/end in the past) are immutable and never re-downloaded
  - "last N days" requests (lookback) expire after 24 h
Optionally merges a higher-timeframe trend column (see features/macro.py).
"""
import hashlib
import io
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd
import requests

from features.macro import add_macro_trend
from features.timeframes import to_timedelta

logger = logging.getLogger(__name__)

CACHE_DIR    = os.path.join(os.path.dirname(__file__), "cache")
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"
KLINE_COLS   = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def binance_id(symbol: str) -> str:
    """'ETH/USDT:USDT' or 'ETH/USDT' -> 'ETHUSDT'."""
    return symbol.split(":")[0].replace("/", "")


def perp_symbol(symbol: str) -> str:
    """'ETH/USDT' -> 'ETH/USDT:USDT' (ccxt unified linear perpetual)."""
    if ":" in symbol:
        return symbol
    quote = symbol.split("/")[1]
    return f"{symbol}:{quote}"


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _ms_to_utc(values: pd.Series) -> pd.DatetimeIndex:
    """Binance archives switched some files to microseconds; detect by magnitude."""
    v = pd.to_numeric(values).astype("int64")
    unit = "us" if v.max() > 10**14 else "ms"
    return pd.DatetimeIndex(pd.to_datetime(v, unit=unit, utc=True))


class DataFetcher:
    def __init__(self, exchange_id: str = "okx", cache_dir: str = CACHE_DIR, session=None):
        self.exchange_id   = exchange_id
        self.cache_dir     = cache_dir
        self._live         = None
        self._hist         = None
        self.session       = session or requests.Session()
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def hist_exchange(self):
        if self._hist is None:
            self._hist = ccxt.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
        return self._hist

    @property
    def live_exchange(self):
        """Created lazily so backtests never need the trading venue."""
        if self._live is None:
            self._live = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        return self._live

    # ── Historical OHLCV ──────────────────────────────────────────────────

    def historical(
        self,
        symbol:    str,
        timeframe: str,
        days:      int = 365,
        use_cache: bool = True,
        macro_tf:  str = None,      # e.g. "1d" -> adds macro_close / macro_trend
        start=None,                 # fixed window (preferred for research): [start, end)
        end=None,
    ) -> pd.DataFrame:
        if start is None:
            end_ts   = pd.Timestamp.now(tz="UTC").floor("D")
            start_ts = end_ts - pd.Timedelta(days=days)
        else:
            start_ts = _utc(start)
            end_ts   = _utc(end) if end is not None else pd.Timestamp.now(tz="UTC").floor("D")

        df = self.ohlcv_range(symbol, timeframe, start_ts, end_ts, use_cache)
        if macro_tf:
            # extra history so the guardian is defined from the first bar on
            macro = self.ohlcv_range(symbol, macro_tf, start_ts - pd.Timedelta(days=60), end_ts, use_cache)
            df = add_macro_trend(df, macro, macro_tf)
        return df

    def ohlcv_range(self, symbol, timeframe, start, end, use_cache=True) -> pd.DataFrame:
        """Closed candles with open time in [start, end). Index = candle open time (UTC)."""
        start, end = _utc(start), _utc(end)
        path = os.path.join(self.cache_dir,
                            f"{binance_id(symbol)}_{timeframe}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.parquet")
        if use_cache and self._cache_valid(path, end):
            df = pd.read_parquet(path)
            logger.info(f"Cache hit {os.path.basename(path)}: {len(df)} candles")
            return df

        try:
            df = self._ccxt_ohlcv(symbol, timeframe, start, end)
            source = "ccxt/binance"
        except Exception as e:                              # noqa: BLE001 - any API failure -> archive
            logger.warning(f"Binance REST unavailable ({type(e).__name__}: {str(e)[:120]}); "
                           f"using the data.binance.vision archive")
            df = self._archive_ohlcv(symbol, timeframe, start, end)
            source = "data.binance.vision"

        df = df[(df.index >= start) & (df.index < end)]
        # never return the still-forming candle: a candle is closed once open + duration <= now
        df = df[df.index + to_timedelta(timeframe) <= pd.Timestamp.now(tz="UTC")]
        if df.empty:
            raise ValueError(f"No {timeframe} data for {symbol} in [{start}, {end})")
        self._report_gaps(df, timeframe, symbol)
        df.to_parquet(path)
        logger.info(f"{symbol} {timeframe} from {source}: {len(df)} candles | {df.index[0]} - {df.index[-1]}")
        return df

    # ── Historical funding ────────────────────────────────────────────────

    def funding_history(self, symbol, start, end, use_cache=True) -> pd.Series:
        """
        Realised funding rates of the Binance USD-M perpetual, indexed by funding time (UTC,
        floored to the minute). Positive rate: longs pay shorts.
        """
        start, end = _utc(start), _utc(end)
        end  = min(end, pd.Timestamp.now(tz="UTC"))
        path = os.path.join(self.cache_dir,
                            f"funding_{binance_id(symbol)}_{start:%Y%m%d}_{end:%Y%m%d%H}.parquet")
        if use_cache and self._cache_valid(path, end):
            return pd.read_parquet(path)["funding_rate"]

        try:
            s = self._ccxt_funding(symbol, start, end)
            source = "ccxt/binance"
        except Exception as e:                              # noqa: BLE001
            logger.warning(f"Binance REST funding unavailable ({type(e).__name__}: {str(e)[:120]}); "
                           f"using the data.binance.vision archive")
            s = self._archive_funding(symbol, start, end)
            source = "data.binance.vision"

        s = s[(s.index >= start) & (s.index < end)].rename("funding_rate")
        if s.empty:
            raise ValueError(f"No funding history for {symbol} in [{start}, {end})")
        s.to_frame().to_parquet(path)
        logger.info(f"Funding {symbol} from {source}: {len(s)} events | {s.index[0]} - {s.index[-1]} | "
                    f"mean {s.mean():.6f}")
        return s

    # ── ccxt (REST) ───────────────────────────────────────────────────────

    def _ccxt_ohlcv(self, symbol, timeframe, start, end) -> pd.DataFrame:
        ex, sym = self.hist_exchange, perp_symbol(symbol)
        since, stop, rows = int(start.timestamp() * 1000), int(end.timestamp() * 1000), []
        for _ in range(5000):
            batch = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 1
            if since >= stop or len(batch) < 1000:
                break
        return self._to_dataframe(rows)

    def _ccxt_funding(self, symbol, start, end) -> pd.Series:
        ex, sym = self.hist_exchange, perp_symbol(symbol)
        since, stop, rows = int(start.timestamp() * 1000), int(end.timestamp() * 1000), []
        for _ in range(5000):
            batch = ex.fetch_funding_rate_history(sym, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1]["timestamp"] + 1
            if since >= stop or len(batch) < 1000:
                break
        idx = _ms_to_utc(pd.Series([r["timestamp"] for r in rows])).floor("min")
        s = pd.Series([float(r["fundingRate"]) for r in rows], index=idx)
        return s[~s.index.duplicated(keep="last")].sort_index()

    # ── data.binance.vision archive ───────────────────────────────────────

    def _archive_ohlcv(self, symbol, timeframe, start, end) -> pd.DataFrame:
        sym, frames = binance_id(symbol), []
        for kind, period in self._archive_periods(start, end):
            url = f"{ARCHIVE_BASE}/{kind}/klines/{sym}/{timeframe}/{sym}-{timeframe}-{period}.zip"
            raw = self._download_zip_csv(url)
            if raw is not None:
                frames.append(self.parse_archive_klines(raw))
        if not frames:
            raise ValueError(f"Archive has no klines for {sym} {timeframe}")
        df = pd.concat(frames).sort_index()
        return df[~df.index.duplicated(keep="last")]

    def _archive_funding(self, symbol, start, end) -> pd.Series:
        sym, parts = binance_id(symbol), []
        for kind, period in self._archive_periods(start, end):
            if kind != "monthly":
                continue          # funding is only archived monthly; the current month may be missing
            url = f"{ARCHIVE_BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{period}.zip"
            raw = self._download_zip_csv(url)
            if raw is not None:
                parts.append(self.parse_archive_funding(raw))
        if not parts:
            raise ValueError(f"Archive has no funding history for {sym}")
        s = pd.concat(parts).sort_index()
        return s[~s.index.duplicated(keep="last")]

    @staticmethod
    def _archive_periods(start, end):
        """Monthly files for complete months, daily files for the current (incomplete) month."""
        def month(ts):
            return ts.tz_convert("UTC").tz_localize(None).to_period("M")
        this_month = month(pd.Timestamp.now(tz="UTC"))
        out = []
        for m in pd.period_range(month(start), month(end - pd.Timedelta("1ns")), freq="M"):
            if m < this_month:
                out.append(("monthly", f"{m.year:04d}-{m.month:02d}"))
            else:
                first = max(start.floor("D"), m.start_time.tz_localize("UTC"))
                last  = min(end, pd.Timestamp.now(tz="UTC")).floor("D")
                for d in pd.date_range(first, last, freq="D", inclusive="left"):
                    out.append(("daily", f"{d:%Y-%m-%d}"))
        return out

    def _download_zip_csv(self, url: str) -> Optional[bytes]:
        """Downloads a zip + its .CHECKSUM, verifies SHA-256, returns the CSV bytes (None if 404)."""
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code == 404:
                    logger.debug(f"Archive file missing: {url}")
                    return None
                r.raise_for_status()
                c = self.session.get(url + ".CHECKSUM", timeout=30)
                c.raise_for_status()
                expected = c.text.split()[0].strip().lower()
                actual   = hashlib.sha256(r.content).hexdigest()
                if expected != actual:
                    raise ValueError(f"Checksum mismatch for {url}")
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    names = [n for n in z.namelist() if n.endswith(".csv")]
                    return z.read(names[0])
            except (requests.RequestException,) as e:
                logger.warning(f"Archive download failed ({attempt + 1}/3) {url}: {e}")
                time.sleep(2 * (attempt + 1))
        raise IOError(f"Could not download {url}")

    @staticmethod
    def parse_archive_klines(raw: bytes) -> pd.DataFrame:
        first = raw.split(b"\n", 1)[0]
        has_header = first[:1].isalpha()
        df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                         names=None if has_header else KLINE_COLS)
        df.index = _ms_to_utc(df["open_time"])
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    @staticmethod
    def parse_archive_funding(raw: bytes) -> pd.Series:
        df = pd.read_csv(io.BytesIO(raw))
        idx = _ms_to_utc(df["calc_time"]).floor("min")
        return pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=idx, name="funding_rate")

    # ── Live data (trading venue) ─────────────────────────────────────────

    def live(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Recent candles from the trading venue (last row is the still-forming candle)."""
        raw = self.live_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return self._to_dataframe(raw)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _cache_valid(path: str, end: pd.Timestamp) -> bool:
        if not os.path.exists(path):
            return False
        if end <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1):
            return True                                      # fixed past window: immutable
        return (time.time() - os.path.getmtime(path)) / 3600 < 24

    @staticmethod
    def _report_gaps(df: pd.DataFrame, timeframe: str, symbol: str):
        step = to_timedelta(timeframe)
        gaps = df.index.to_series().diff().dropna()
        gaps = gaps[gaps > step]
        if len(gaps):
            missing = int(sum(g / step - 1 for g in gaps))
            logger.warning(f"{symbol} {timeframe}: {len(gaps)} gaps, {missing} missing candles "
                           f"(largest {gaps.max()} at {gaps.idxmax()})")

    @staticmethod
    def _to_dataframe(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.astype(float)
