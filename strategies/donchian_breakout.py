"""
Donchian Channel Breakout - long and short
==========================================
Classic trend following (turtle principle).

  Long entry:  close above the previous N-period high
  Short entry: close below the previous N-period low
  Long exit:   close below the previous short-period low (exit channel)
  Short exit:  close above the previous short-period high

Filters: ATR volatility, ADX trend strength, volume, higher-timeframe guardian.

Parameters (config):
  dc_enter, dc_exit, atr_period, atr_min_mult, adx_threshold, vol_mult
"""
import pandas as pd
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        enter = int(c.get("dc_enter", 20))
        exit_ = int(c.get("dc_exit", 10))

        # Entry channel (long period) and exit channel (short period)
        df["dc_high"]      = df["high"].rolling(enter).max()
        df["dc_low"]       = df["low"].rolling(enter).min()
        df["dc_exit_high"] = df["high"].rolling(exit_).max()
        df["dc_exit_low"]  = df["low"].rolling(exit_).min()

        # Compare against the PREVIOUS candle's channel (no look-ahead bias)
        df["dc_high_prev"]      = df["dc_high"].shift(1)
        df["dc_low_prev"]       = df["dc_low"].shift(1)
        df["dc_exit_high_prev"] = df["dc_exit_high"].shift(1)
        df["dc_exit_low_prev"]  = df["dc_exit_low"].shift(1)

        # ATR and its ratio to its own moving average
        if "atr_14" not in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"],
                                  length=int(c.get("atr_period", 14)))
        df["atr_ratio"] = df["atr_14"] / (df["atr_14"].rolling(20).mean() + 1e-10)

        if "adx" not in df.columns:
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None:
                df["adx"] = adx.iloc[:, 0]

        if "vol_ma_20" not in df.columns:
            df["vol_ma_20"] = df["volume"].rolling(20).mean()

        core = ["dc_high", "dc_low", "dc_exit_high", "dc_exit_low", "atr_14"]
        df.dropna(subset=[x for x in core if x in df.columns], inplace=True)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        breakout_long  = df["close"] > df["dc_high_prev"]
        breakout_short = df["close"] < df["dc_low_prev"]

        vol_ok    = df["atr_ratio"] > c.get("atr_min_mult", 0.5)
        adx_ok    = df["adx"] > c.get("adx_threshold", 20) if "adx" in df.columns \
                    else pd.Series(True, index=df.index)
        volume_ok = df["volume"] > df["vol_ma_20"] * c.get("vol_mult", 1.0)

        df["signal"] = 0
        df.loc[breakout_long  & vol_ok & adx_ok & volume_ok, "signal"] = 1
        df.loc[breakout_short & vol_ok & adx_ok & volume_ok, "signal"] = -1

        df["long_exit"]  = (df["close"] < df["dc_exit_low_prev"]).fillna(False)
        df["short_exit"] = (df["close"] > df["dc_exit_high_prev"]).fillna(False)

        return self.apply_macro_filter(df)
