"""
Universe data for the portfolio strategies (docs/preregistration.md, section 3)
==============================================================================
okx_xperp_coins()       base coins of the crypto X-Perps listed on OKX Europe (public endpoint),
                        stored as a dated snapshot in config/okx_xperps_snapshot.json
binance_inventory()     every Binance USD-M symbol in the public archive with its first/last month
                        (delisted symbols included), cached in data/cache/
base_coin()             'ETHUSDT' -> 'ETH', '1000PEPEUSDT' -> 'PEPE', '1MBABYDOGEUSDT' -> 'BABYDOGE'
candidates()            symbols for the 'primary' or 'binance_wide' universe
load_daily_panel()      daily OHLCV of many symbols (threaded archive downloads)
load_funding_panel()    funding rates of many symbols, summed per day (open_t, open_t+1]
"""
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from data.fetcher import CACHE_DIR, DataFetcher

logger = logging.getLogger(__name__)

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OKX_SNAPSHOT  = os.path.join(ROOT, "config", "okx_xperps_snapshot.json")
S3_LISTING    = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
OKX_FUTURES   = "https://eea.okx.com/api/v5/public/instruments?instType=FUTURES"

STABLE_OR_INDEX = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "USD1", "PYUSD", "RLUSD",
                   "BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD", "ALLSTAR", "BTCST"}


def base_coin(symbol: str) -> str:
    s = symbol.split(":")[0].replace("/", "")
    if s.endswith("USDT"):
        s = s[:-4]
    for prefix in ("1000000", "1000", "1M"):
        if s.startswith(prefix) and len(s) > len(prefix) and s[len(prefix)].isalpha():
            return s[len(prefix):]
    return s


# ── OKX Europe X-Perps ────────────────────────────────────────────────────────

def okx_xperp_coins(refresh: bool = False, session=None) -> dict:
    """{'retrieved': iso, 'coins': [...]} - crypto X-Perps (ruleType xperp, instCategory 1)."""
    if os.path.exists(OKX_SNAPSHOT) and not refresh:
        with open(OKX_SNAPSHOT) as f:
            return json.load(f)
    s = session or requests.Session()
    data = s.get(OKX_FUTURES, timeout=30).json()["data"]
    coins = sorted({i["instFamily"].split("-")[0] for i in data
                    if i.get("ruleType") == "xperp" and i.get("instCategory") == "1"})
    snap = {"retrieved": datetime.now(timezone.utc).isoformat(), "source": OKX_FUTURES,
            "filter": "ruleType=xperp, instCategory=1", "coins": coins}
    with open(OKX_SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1)
    return snap


# ── Binance archive inventory ─────────────────────────────────────────────────

def _s3_prefixes(session, prefix: str, pattern: str) -> list:
    out, marker = [], ""
    while True:
        r = session.get(S3_LISTING, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
        r.raise_for_status()
        found = re.findall(pattern, r.text)
        out += found
        if "<IsTruncated>true</IsTruncated>" not in r.text or not found:
            return out
        marker = re.findall(r"<(?:Prefix|Key)>([^<]+)</(?:Prefix|Key)>", r.text)[-1]


def binance_inventory(refresh: bool = False, workers: int = 16) -> pd.DataFrame:
    """symbol, first_month, last_month (from the monthly 1d kline files) for every USDT symbol."""
    path = os.path.join(CACHE_DIR, "binance_um_inventory.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    session = requests.Session()
    root = "data/futures/um/monthly/klines/"
    symbols = [s for s in _s3_prefixes(session, root, rf"<Prefix>{root}([^/]+)/</Prefix>") if s.endswith("USDT")]

    def months(sym):
        keys = _s3_prefixes(requests.Session(), f"{root}{sym}/1d/",
                            rf"<Key>{root}{sym}/1d/{sym}-1d-(\d{{4}}-\d{{2}})\.zip</Key>")
        return sym, sorted(set(keys))

    with ThreadPoolExecutor(workers) as ex:
        rows = [(s, m[0], m[-1]) for s, m in ex.map(months, symbols) if m]
    inv = pd.DataFrame(rows, columns=["symbol", "first_month", "last_month"])
    inv["base"] = inv["symbol"].map(base_coin)
    inv.to_parquet(path)
    logger.info(f"Binance UM inventory: {len(inv)} USDT symbols with daily klines")
    return inv


def candidates(inventory: pd.DataFrame, okx_coins: list, mode: str = "primary",
               delisted_before: str = "2026-10") -> pd.DataFrame:
    """
    primary:      base coin is an OKX Europe crypto X-Perp today, OR the symbol was delisted from
                  Binance (last monthly file before `delisted_before`) - see preregistration 3
    binance_wide: every USDT symbol
    Stablecoins and index products are always excluded.
    """
    inv = inventory[~inventory["base"].isin(STABLE_OR_INDEX)].copy()
    inv["delisted"] = inv["last_month"] < delisted_before
    inv["on_okx"] = inv["base"].isin(set(okx_coins))
    if mode == "primary":
        return inv[inv["on_okx"] | inv["delisted"]].reset_index(drop=True)
    if mode == "binance_wide":
        return inv.reset_index(drop=True)
    raise ValueError(mode)


# ── Panels ────────────────────────────────────────────────────────────────────

def _symbol_window(row, start, end):
    s = max(pd.Timestamp(start, tz="UTC"), pd.Timestamp(row["first_month"] + "-01", tz="UTC"))
    e = min(pd.Timestamp(end, tz="UTC"),
            pd.Timestamp(row["last_month"] + "-01", tz="UTC") + pd.offsets.MonthBegin(1))
    return s, e


def load_daily_panel(cands: pd.DataFrame, start, end, workers: int = 12) -> dict:
    """{symbol: daily OHLCV} for all candidates with data inside [start, end)."""
    def one(row):
        s, e = _symbol_window(row, start, end)
        if s >= e:
            return row["symbol"], None
        try:
            f = DataFetcher("binance", use_rest=False)
            return row["symbol"], f.ohlcv_range(row["symbol"] + ":USDT", "1d", s, e)
        except (ValueError, IOError) as exc:
            logger.warning(f"{row['symbol']}: no daily data ({exc})")
            return row["symbol"], None

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(one, [r for _, r in cands.iterrows()]))
    return {k: v for k, v in out.items() if v is not None and len(v)}


def load_funding_panel(symbols: list, start, end, index: pd.DatetimeIndex, workers: int = 12) -> pd.DataFrame:
    """Funding rates summed per daily bar: events in (open_t, open_t+1] belong to day t."""
    def one(sym):
        try:
            f = DataFetcher("binance", use_rest=False)
            return sym, f.funding_history(sym + ":USDT", start, end)
        except (ValueError, IOError) as exc:
            logger.warning(f"{sym}: no funding history ({exc}) - charged zero funding")
            return sym, None

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, symbols))
    cols = {}
    for sym, s in res.items():
        if s is None or s.empty:
            continue
        # event at T belongs to the day whose open is the last open strictly before T
        day = (s.index - pd.Timedelta("1ns")).floor("D")
        cols[sym] = s.groupby(day).sum()
    panel = pd.DataFrame(cols).reindex(index).fillna(0.0)
    return panel.reindex(columns=symbols, fill_value=0.0)
