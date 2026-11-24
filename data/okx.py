"""
OKX Europe X-Perps: public, unauthenticated market data (no API keys)
=====================================================================
X-Perps are listed by the EEA OKX API as instType FUTURES with ruleType "xperp":
  instId "ETH-USD_UM_XPERP-310404" (a far-dated expiry, alias "this_five_years"),
  settleCcy USD, ctVal = coins per contract (ETH 0.001, BTC 0.0001), lotSz/minSz in contracts.
ccxt parses them correctly as soon as futures markets are loaded (its okx default only loads
spot + swap); see execution/broker.py.

Endpoints (https://eea.okx.com, public):
  /api/v5/public/instruments?instType=FUTURES     -> xperp_instruments()
  /api/v5/market/history-candles?bar=1Dutc         -> daily_candles()   (1D would roll at UTC+8)
  /api/v5/public/funding-rate-history              -> funding_history() (paginated, newest first)
"""
import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EEA = "https://eea.okx.com"


def _get(session, base, path, params, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(base + path, params=params, timeout=30)
            r.raise_for_status()
            d = r.json()
            if d.get("code") not in ("0", 0):
                raise RuntimeError(f"OKX {path}: code {d.get('code')} {d.get('msg')}")
            return d["data"]
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"OKX {path} failed ({e}), retrying")
            time.sleep(1 + attempt)


def parse_xperps(instruments: list) -> pd.DataFrame:
    """Crypto X-Perps (ruleType xperp, instCategory 1) from a raw instruments response."""
    rows = [{
        "base": i["instFamily"].split("-")[0], "inst_id": i["instId"], "ct_val": float(i["ctVal"]),
        "tick_sz": float(i["tickSz"]), "lot_sz": float(i["lotSz"]), "min_sz": float(i["minSz"]),
        "settle": i["settleCcy"], "max_lever": float(i.get("lever") or 0),
        "list_time": pd.to_datetime(int(i["listTime"]), unit="ms", utc=True), "state": i["state"],
    } for i in instruments if i.get("ruleType") == "xperp" and i.get("instCategory") == "1"]
    return pd.DataFrame(rows).sort_values("base").reset_index(drop=True)


def xperp_instruments(session=None, base_url=EEA) -> pd.DataFrame:
    s = session or requests.Session()
    return parse_xperps(_get(s, base_url, "/api/v5/public/instruments", {"instType": "FUTURES"}))


def parse_candles(rows: list) -> pd.DataFrame:
    """OKX candle rows [ts, o, h, l, c, vol(contracts), volCcy(coins), volCcyQuote, confirm]."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "vol_ccy",
                                     "vol_quote", "confirm"])
    df.index = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df.index.name = "timestamp"
    out = df[["open", "high", "low", "close", "vol_ccy", "confirm"]].astype(float)
    out = out.rename(columns={"vol_ccy": "volume"})
    return out[out["confirm"] == 1].drop(columns="confirm").sort_index()


def daily_candles(inst_id: str, start, session=None, base_url=EEA) -> pd.DataFrame:
    """Closed daily candles (UTC day, '1Dutc') from `start` to now, volume in coins."""
    s = session or requests.Session()
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    rows, after = [], None
    for _ in range(50):
        params = {"instId": inst_id, "bar": "1Dutc", "limit": "100"}
        if after:
            params["after"] = after
        batch = _get(s, base_url, "/api/v5/market/history-candles", params)
        if not batch:
            break
        rows += batch
        oldest = int(batch[-1][0])
        if oldest <= start.value // 10**6:
            break
        after = str(oldest)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = parse_candles(rows)
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index >= start]


def parse_funding(rows: list) -> pd.Series:
    s = pd.Series([float(r.get("realizedRate") or r["fundingRate"]) for r in rows],
                  index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms", utc=True),
                  name="funding_rate")
    return s[~s.index.duplicated(keep="last")].sort_index()


def funding_history(inst_id: str, start, session=None, base_url=EEA) -> pd.Series:
    """Realised funding rates (positive: longs pay), oldest first, from `start` to now."""
    s = session or requests.Session()
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    rows, after = [], None
    for _ in range(200):
        params = {"instId": inst_id, "limit": "100"}
        if after:
            params["after"] = after
        batch = _get(s, base_url, "/api/v5/public/funding-rate-history", params)
        if not batch:
            break
        rows += batch
        oldest = min(int(r["fundingTime"]) for r in batch)
        if oldest <= start.value // 10**6:
            break
        after = str(oldest)
    if not rows:
        return pd.Series(dtype=float, name="funding_rate")
    out = parse_funding(rows)
    return out[out.index >= start]
