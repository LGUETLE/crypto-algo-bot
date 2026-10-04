"""Debug: shows why no mean reversion signals are generated."""
import yaml, logging
import pandas as pd
import pandas_ta as ta
logging.basicConfig(level=logging.WARNING)

from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

df = DataFetcher(config["exchange"]).historical("BTC/USDT:USDT", "1h", days=1095)
df = FeatureEngineer(config).transform(df)

# Indicators
df["rsi"] = ta.rsi(df["close"], length=14)
adx = ta.adx(df["high"], df["low"], df["close"], length=14)
df["adx"] = adx.iloc[:, 0]

# pandas_ta bbands column order: lower, mid, upper, bandwidth, percent
bb = ta.bbands(df["close"], length=20, std=2)
df["bb_lower"] = bb.iloc[:, 0]
df["bb_mid"]   = bb.iloc[:, 1]
df["bb_upper"] = bb.iloc[:, 2]
df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

df["vol_ma_20"] = df["volume"].rolling(20).mean()
df["vol_spike"] = df["volume"] > df["vol_ma_20"] * 1.5
df.dropna(inplace=True)


def share(mask):
    return f"{mask.sum()} ({mask.mean():.1%})"


print(f"Candles total: {len(df)}")
print("\nRSI statistics:")
print(f"  RSI < 30 (oversold):    {share(df['rsi'] < 30)}")
print(f"  RSI < 35:               {share(df['rsi'] < 35)}")
print(f"  RSI > 70 (overbought):  {share(df['rsi'] > 70)}")
print(f"  RSI > 65:               {share(df['rsi'] > 65)}")

print("\nADX statistics:")
print(f"  ADX < 25 (sideways):    {share(df['adx'] < 25)}")
print(f"  ADX < 30:               {share(df['adx'] < 30)}")
print(f"  ADX < 35:               {share(df['adx'] < 35)}")

print("\nBollinger position:")
print(f"  bb_pct < 0.15 (bottom): {share(df['bb_pct'] < 0.15)}")
print(f"  bb_pct < 0.25:          {share(df['bb_pct'] < 0.25)}")
print(f"  bb_pct > 0.85 (top):    {share(df['bb_pct'] > 0.85)}")

print(f"\nVolume spike:             {share(df['vol_spike'])}")

# Combined signals
rsi_long  = (df["rsi"] < 30) & (df["rsi"].shift(1) >= 30)
rsi_short = (df["rsi"] > 70) & (df["rsi"].shift(1) <= 70)
no_trend  = df["adx"] < 30
at_lower  = df["bb_pct"] < 0.15
at_upper  = df["bb_pct"] > 0.85

print("\nCombined signals:")
print(f"  RSI cross long:                          {rsi_long.sum()}")
print(f"  + ADX < 30:                              {(rsi_long & no_trend).sum()}")
print(f"  + ADX < 30 + BB lower:                   {(rsi_long & no_trend & at_lower).sum()}")
print(f"  + ADX < 30 + BB lower + volume spike:    {(rsi_long & no_trend & at_lower & df['vol_spike']).sum()}")
print(f"\n  RSI cross short:                         {rsi_short.sum()}")
print(f"  + ADX < 30:                              {(rsi_short & no_trend).sum()}")
print(f"  + ADX < 30 + BB upper:                   {(rsi_short & no_trend & at_upper).sum()}")
print(f"  + ADX < 30 + BB upper + volume spike:    {(rsi_short & no_trend & at_upper & df['vol_spike']).sum()}")
