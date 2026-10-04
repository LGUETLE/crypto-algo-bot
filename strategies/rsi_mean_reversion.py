"""
RSI Mean Reversion - long and short (simplified)
================================================
Long entry:  RSI crosses below rsi_oversold   -> bounce expected
Short entry: RSI crosses above rsi_overbought -> pullback expected
Long exit:   RSI rises above rsi_exit_long
Short exit:  RSI falls below rsi_exit_short

Filter:
  - ADX < adx_max: only trade sideways markets (no strong trend)
  - No higher-timeframe filter on purpose: it would contradict counter-trend entries
"""
import pandas as pd
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        df["rsi"]      = ta.rsi(df["close"], length=int(c.get("rsi_period", 14)))
        df["rsi_fast"] = ta.rsi(df["close"], length=7)

        # ATR for stop-loss sizing
        if "atr_14" not in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # ADX - trend strength (mean reversion only works in weak trends)
        if "adx" not in df.columns:
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None:
                df["adx"] = adx.iloc[:, 0]

        if "vol_ma_20" not in df.columns:
            df["vol_ma_20"] = df["volume"].rolling(20).mean()

        # EMA 200 for regime context (information only, not a filter)
        df["ema_200"] = ta.ema(df["close"], length=200)

        core = ["rsi", "adx", "atr_14"]
        df.dropna(subset=[x for x in core if x in df.columns], inplace=True)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        rsi_os  = c.get("rsi_oversold",   30)
        rsi_ob  = c.get("rsi_overbought", 70)
        rsi_el  = c.get("rsi_exit_long",  55)
        rsi_es  = c.get("rsi_exit_short", 45)
        adx_max = c.get("adx_max",        30)

        # Sideways-market filter
        no_trend = df["adx"] < adx_max if "adx" in df.columns \
                   else pd.Series(True, index=df.index)

        # RSI crossings (the moment of crossing the threshold)
        rsi_long_entry  = (df["rsi"] < rsi_os) & (df["rsi"].shift(1) >= rsi_os)
        rsi_short_entry = (df["rsi"] > rsi_ob) & (df["rsi"].shift(1) <= rsi_ob)

        # Exit: RSI returns to the middle
        df["long_exit"]  = df["rsi"] > rsi_el
        df["short_exit"] = df["rsi"] < rsi_es

        df["signal"] = 0
        df.loc[rsi_long_entry  & no_trend, "signal"] = 1
        df.loc[rsi_short_entry & no_trend, "signal"] = -1
        return df
