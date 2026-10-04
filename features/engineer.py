"""
Feature Engineer
================
Computes the technical features used as decision basis.

Groups: trend, momentum, volatility, volume, market structure, time.

Usage:
  df = FeatureEngineer().transform(df)

pandas_ta results are read BY COLUMN NAME (see _pick), never by position:
  bbands -> BBL, BBM, BBU, BBB, BBP
  kc     -> KCL, KCB, KCU
  macd   -> MACD, MACDh (histogram), MACDs (signal)
  adx    -> ADX, (ADXR in 0.4.x), DMP, DMN
"""
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def _pick(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """First column of a pandas_ta result whose name starts with `prefix`.
    Selecting by name is safer than by position: pandas-ta 0.4.x inserted an ADXR column into adx()."""
    return frame[[c for c in frame.columns if str(c).startswith(prefix)][0]]


class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for step in (self._trend, self._momentum, self._volatility,
                     self._volume, self._structure, self._time):
            df = step(df)
        core = [c for c in ("rsi_14", "atr_14", "vol_ratio") if c in df.columns]
        df.dropna(subset=core, inplace=True)
        logger.info(f"Features computed: {len(df.columns)} columns, {len(df)} rows")
        return df

    # ── Trend ─────────────────────────────────────────────────────────────

    def _trend(self, df):
        for period in (9, 12, 21, 26, 50, 200):
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        df["ema_cross_dist"] = (df["ema_12"] - df["ema_26"]) / df["close"]

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["macd"], df["macd_hist"], df["macd_signal"] = (
                _pick(macd, "MACD_"), _pick(macd, "MACDh"), _pick(macd, "MACDs"))

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            df["adx"], df["dmp"], df["dmn"] = _pick(adx, "ADX_"), _pick(adx, "DMP"), _pick(adx, "DMN")

        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
        if st is not None:
            df["supertrend_dir"] = _pick(st, "SUPERTd")    # 1 = uptrend, -1 = downtrend
        return df

    # ── Momentum ──────────────────────────────────────────────────────────

    def _momentum(self, df):
        df["rsi_14"]    = ta.rsi(df["close"], length=14)
        df["rsi_7"]     = ta.rsi(df["close"], length=7)
        df["rsi_slope"] = df["rsi_14"].diff(3)

        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None:
            df["stoch_k"], df["stoch_d"] = _pick(stoch, "STOCHk"), _pick(stoch, "STOCHd")

        df["roc_5"]  = ta.roc(df["close"], length=5)
        df["roc_20"] = ta.roc(df["close"], length=20)
        df["willr"]  = ta.willr(df["high"], df["low"], df["close"], length=14)
        return df

    # ── Volatility ────────────────────────────────────────────────────────

    def _volatility(self, df):
        df["atr_14"]  = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_pct"] = df["atr_14"] / df["close"]

        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            df["bb_lower"], df["bb_mid"], df["bb_upper"] = (
                _pick(bb, "BBL"), _pick(bb, "BBM"), _pick(bb, "BBU"))
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
            df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        kc = ta.kc(df["high"], df["low"], df["close"], length=20)
        if kc is not None:
            df["kc_lower"], df["kc_upper"] = _pick(kc, "KCL"), _pick(kc, "KCU")

        # Squeeze: Bollinger Bands fully inside the Keltner Channel
        if "bb_upper" in df.columns and "kc_upper" in df.columns:
            df["squeeze"] = ((df["bb_upper"] < df["kc_upper"]) &
                             (df["bb_lower"] > df["kc_lower"])).astype(int)

        # Annualised with sqrt(365*24) -> assumes 1h candles
        df["hist_vol"] = df["close"].pct_change().rolling(20).std() * np.sqrt(365 * 24)
        return df

    # ── Volume ────────────────────────────────────────────────────────────

    def _volume(self, df):
        df["vol_ma_20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma_20"] + 1e-10)

        df["obv"]     = ta.obv(df["close"], df["volume"])
        df["obv_ema"] = ta.ema(df["obv"], length=20)

        # Rolling VWAP approximation (not a true intraday VWAP)
        df["vwap"]      = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        df["vwap_dist"] = (df["close"] - df["vwap"]) / df["close"]

        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        return df

    # ── Market structure ──────────────────────────────────────────────────

    def _structure(self, df):
        df["high_20"]   = df["high"].rolling(20).max()
        df["low_20"]    = df["low"].rolling(20).min()
        df["range_pct"] = (df["high_20"] - df["low_20"]) / df["close"]
        df["price_pct_rank"] = df["close"].rolling(20).rank(pct=True)

        df["body_size"]  = abs(df["close"] - df["open"]) / df["close"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["bull_streak"] = df["is_bullish"].rolling(3).sum()
        return df

    # ── Time ──────────────────────────────────────────────────────────────

    def _time(self, df):
        df["hour"]        = df.index.hour
        df["day_of_week"] = df.index.dayofweek    # 0 = Monday
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

        # Sessions in UTC: London 8-16, New York 13-21
        df["london_session"]  = ((df["hour"] >= 8)  & (df["hour"] < 16)).astype(int)
        df["ny_session"]      = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
        df["overlap_session"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)
        return df
