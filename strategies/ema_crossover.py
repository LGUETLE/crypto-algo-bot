"""
EMA Crossover - long and short
==============================
Long:  fast EMA crosses above slow EMA, price above EMA 200, ADX > threshold, RSI not overbought
Short: mirrored (price below EMA 200, RSI not oversold)
Exits: opposite crossover or price crossing EMA 200.

Signals: 1 = long entry, -1 = short entry, 0 = nothing.
"""
import pandas as pd
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy


class EmaCrossoverStrategy(BaseStrategy):

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        if "ema_fast" not in df.columns:
            df["ema_fast"] = ta.ema(df["close"], length=c.get("ema_fast", 12))
        if "ema_slow" not in df.columns:
            df["ema_slow"] = ta.ema(df["close"], length=c.get("ema_slow", 21))
        if "ema_200" not in df.columns:
            df["ema_200"] = ta.ema(df["close"], length=200)
        if "rsi_14" not in df.columns:
            df["rsi_14"] = ta.rsi(df["close"], length=14)
        if "atr_14" not in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        if "adx" not in df.columns:
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None:
                df["adx"] = adx.iloc[:, 0]
        if "vol_ma_20" not in df.columns:
            df["vol_ma_20"] = df["volume"].rolling(20).mean()

        core = ["ema_fast", "ema_slow", "ema_200", "rsi_14"]
        df.dropna(subset=[x for x in core if x in df.columns], inplace=True)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        cross_up   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))

        above_200 = df["close"] > df["ema_200"]
        below_200 = df["close"] < df["ema_200"]

        adx_ok = df["adx"] > c.get("adx_threshold", 20) if "adx" in df.columns \
                 else pd.Series(True, index=df.index)

        rsi_lower = c.get("rsi_lower", 40)
        rsi_long  = (df["rsi_14"] > rsi_lower) & (df["rsi_14"] < 70)
        rsi_short = (df["rsi_14"] < (100 - rsi_lower)) & (df["rsi_14"] > 30)

        vol_ok = df["volume"] > df["vol_ma_20"] * 0.7

        df["signal"] = 0
        df.loc[cross_up   & above_200 & adx_ok & rsi_long  & vol_ok, "signal"] = 1
        df.loc[cross_down & below_200 & adx_ok & rsi_short & vol_ok, "signal"] = -1

        # Always boolean (a NaN here would be truthy and close every position instantly)
        df["long_exit"]  = (cross_down | below_200).fillna(False)
        df["short_exit"] = (cross_up   | above_200).fillna(False)

        return self.apply_macro_filter(df)
