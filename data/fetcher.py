"""
Data Fetcher
============
Binance for historical data (free, long history) - used as a price-history proxy for
whatever venue you trade on. The trading venue is only used for live data.
Caching: parquet files in data/cache/ (max 24h old).
Optionally merges a higher-timeframe trend column (see features/macro.py).
"""
import os
import time
import logging
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from features.macro import add_macro_trend

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class DataFetcher:
    def __init__(self, exchange_id: str = "okx"):
        self.exchange_id   = exchange_id
        self._live         = None
        self.hist_exchange = ccxt.binance({"options": {"defaultType": "future"}})

    @property
    def live_exchange(self):
        """Created lazily so backtests never need the trading venue."""
        if self._live is None:
            self._live = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        return self._live

    # ── Historical data (Binance) ─────────────────────────────────────────

    def historical(
        self,
        symbol:    str,
        timeframe: str,
        days:      int = 365,
        use_cache: bool = True,
        macro_tf:  str = None,      # e.g. "1d" -> adds macro_close / macro_trend
    ) -> pd.DataFrame:
        df = self._load(symbol, timeframe, days, use_cache)
        if macro_tf:
            macro = self._load(symbol, macro_tf, days + 60, use_cache)
            df = add_macro_trend(df, macro, macro_tf)
        return df

    def _load(self, symbol: str, timeframe: str, days: int, use_cache: bool) -> pd.DataFrame:
        """Downloads OHLCV from Binance futures. BTC/USDT:USDT -> BTC/USDT."""
        binance_symbol = symbol.split(":")[0]
        cache_file = os.path.join(
            CACHE_DIR, f"{binance_symbol.replace('/', '_')}_{timeframe}_{days}d.parquet"
        )

        if use_cache and os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 24:
                df = pd.read_parquet(cache_file)
                logger.info(f"Cache hit: {len(df)} candles | {df.index[0]} - {df.index[-1]}")
                return df

        logger.info(f"Downloading {days} days of {binance_symbol} {timeframe} from Binance...")
        start = datetime.now(timezone.utc) - timedelta(days=days)
        since = int(start.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        now   = self.hist_exchange.milliseconds()

        all_ohlcv, iterations = [], 0
        while since < now and iterations < 400:    # 400 x 1000 candles covers 3y of 15m data
            iterations += 1
            try:
                batch = self.hist_exchange.fetch_ohlcv(
                    binance_symbol, timeframe=timeframe, since=since, limit=1000
                )
            except Exception as e:
                logger.warning(f"Fetch error (attempt {iterations}): {e}")
                time.sleep(2)
                continue

            if not batch:
                break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 1000:
                break
            time.sleep(0.2)

        df = self._to_dataframe(all_ohlcv)
        df = df.iloc[:-1]          # drop the still-forming last candle
        df.to_parquet(cache_file)
        logger.info(f"Total: {len(df)} candles | {df.index[0]} - {df.index[-1]}")
        return df

    # ── Live data (trading venue) ─────────────────────────────────────────

    def live(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Recent candles from the trading venue (last row is the still-forming candle)."""
        raw = self.live_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return self._to_dataframe(raw)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.astype(float)
