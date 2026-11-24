"""
Owner-authorised ONE-TIME audit of the frozen ETH hold-out evaluation (largest equity bar).
Re-runs research.evaluate_holdout (logged as a repeated hold-out evaluation, warns), then prints
the largest bar, the trade behind it, and the raw 1h candles around it from the archive cache.
No parameters are changed.  python scripts/audit_holdout_bar.py
"""
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import research           # noqa: E402
from data.fetcher import DataFetcher        # noqa: E402

logging.basicConfig(level=logging.WARNING)
cfg = yaml.safe_load(open("config/settings.yaml"))
pair = [p for p in yaml.safe_load(open("config/pairs.yaml"))["pairs"] if p["active"] and "ETH" in p["symbol"]][0]
out = research.evaluate_holdout(cfg, pair, DataFetcher(cfg["exchange"]))
res = out["result"]
eq = res.equity
prev = eq.shift(1)
prev.iloc[0] = 10_000.0
ret = eq / prev - 1
ts = ret.idxmax()
i = eq.index.get_loc(ts)
print(f"\nAUDIT params={out['params']} total_return={res.metrics['total_return']:+.4f} trades={len(res.trades)}")
print(f"largest bar {ts}  return {ret.max():+.4%}  equity {eq.iloc[i-1]:.2f} -> {eq.iloc[i]:.2f}")
t = res.trades[(res.trades.entry_time <= ts) & (res.trades.exit_time >= ts)]
print(t[["side", "entry_time", "exit_time", "entry_price", "exit_price", "size", "pnl_usdt", "exit_reason"]].to_string())
print("notional / equity at entry:", round(float(t["size"].iloc[0] * t["entry_price"].iloc[0] / eq.iloc[i - 1]), 3))
d = research.load_research_data(cfg, pair, DataFetcher(cfg["exchange"]), final_holdout=True)
print("\nETH 1h candles around the bar (Binance archive):")
print(d.df.loc[eq.index[i - 3]:eq.index[i + 3], ["open", "high", "low", "close", "volume"]])
print("\nBTC 1h candles around the bar (cross-check):")
print(d.btc.loc[eq.index[i - 3]:eq.index[i + 3], ["open", "high", "low", "close", "volume"]])
print("\nsum trade pnl", round(res.trades.pnl_usdt.sum(), 6), "equity change", round(eq.iloc[-1] - 10_000, 6))
