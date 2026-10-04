"""Debug: shows how many signals remain after each filter."""
import yaml
import pandas_ta as ta
from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

# Load data
df = DataFetcher(config["exchange"]).historical("BTC/USDT:USDT", "1h", days=365)
df = FeatureEngineer(config).transform(df)

# Indicators
df["ema_fast"] = ta.ema(df["close"], length=12)
df["ema_slow"] = ta.ema(df["close"], length=21)
df["ema_200"]  = ta.ema(df["close"], length=200)
df["rsi_14"]   = ta.rsi(df["close"], length=14)
adx = ta.adx(df["high"], df["low"], df["close"], length=14)
df["adx"] = adx.iloc[:, 0]

# pandas_ta bbands column order: lower, mid, upper
bb = ta.bbands(df["close"], length=20, std=2)
df["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]

df["vol_ma_20"] = df["volume"].rolling(20).mean()
df.dropna(inplace=True)

# In-sample data (70 %)
df_is = df.iloc[:int(len(df) * 0.7)]
print(f"IS candles: {len(df_is)}")

# Crossover without filters
cross_up = (df_is["ema_fast"] > df_is["ema_slow"]) & \
           (df_is["ema_fast"].shift(1) <= df_is["ema_slow"].shift(1))
print(f"\nCrossover up (total):      {cross_up.sum()}")

# Filters one by one
above_200 = df_is["close"] > df_is["ema_200"]
print(f"+ above_200:               {(cross_up & above_200).sum()}")
adx_ok = df_is["adx"] > 18
print(f"+ ADX > 18:                {(cross_up & above_200 & adx_ok).sum()}")
bb_ok = df_is["bb_width"] > 0.01
print(f"+ BB width > 0.01:         {(cross_up & above_200 & adx_ok & bb_ok).sum()}")
rsi_ok = (df_is["rsi_14"] > 35) & (df_is["rsi_14"] < 72)
print(f"+ RSI 35-72:               {(cross_up & above_200 & adx_ok & bb_ok & rsi_ok).sum()}")
vol_ok = df_is["volume"] > df_is["vol_ma_20"] * 0.7
print(f"+ Volume > 0.7x MA:        {(cross_up & above_200 & adx_ok & bb_ok & rsi_ok & vol_ok).sum()}")

# Without the EMA 200 filter
print(f"\nWithout EMA 200 (ADX+RSI only): {(cross_up & adx_ok & rsi_ok).sum()}")
print(f"Crossover + RSI only:           {(cross_up & rsi_ok).sum()}")
print(f"Crossover only:                 {cross_up.sum()}")

# Check the period
print(f"\nIS period: {df_is.index[0]} - {df_is.index[-1]}")
print(f"Price above EMA200: {above_200.sum()}/{len(df_is)} candles ({above_200.mean():.0%})")
print(f"ADX > 18:           {adx_ok.sum()}/{len(df_is)} candles ({adx_ok.mean():.0%})")
