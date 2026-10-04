"""
Validates the most robust parameter candidates side by side (in-sample vs. out-of-sample).
Run with: python3 validate_robust.py

CAUTION: comparing several candidates on the same OOS period is itself a form of
data snooping. Treat the result as a sanity check, not as proof.
"""
import yaml, logging
from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer
from backtesting.engine import BacktestEngine
from strategies.donchian_breakout import DonchianBreakoutStrategy

logging.basicConfig(level=logging.WARNING)    # warnings only, no spam

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

symbol   = "BTC/USDT:USDT"
macro_tf = config.get("macro_timeframe") if config.get("use_macro_filter", True) else None

print("Loading 365 days of BTC data...")
df = DataFetcher(config["exchange"]).historical(symbol, config["timeframe"], days=365, macro_tf=macro_tf)
df = FeatureEngineer(config).transform(df)

engine = BacktestEngine(config)

# Candidates to compare
candidates = [
    {"name": "Rank 1 (high return)",   "dc_enter": 30, "dc_exit": 15, "adx_threshold": 15, "atr_min_mult": 0.7},
    {"name": "Rank 4 (low drawdown)",  "dc_enter": 30, "dc_exit":  5, "adx_threshold": 25, "atr_min_mult": 0.3},
    {"name": "Conservative",           "dc_enter": 40, "dc_exit": 10, "adx_threshold": 25, "atr_min_mult": 0.5},
]

split_idx = int(len(df) * 0.7)
df_is     = df.iloc[:split_idx]
df_oos    = df.iloc[split_idx:]

print(f"\nIS:  {df_is.index[0].date()} - {df_is.index[-1].date()} ({len(df_is)} candles)")
print(f"OOS: {df_oos.index[0].date()} - {df_oos.index[-1].date()} ({len(df_oos)} candles)")
print("\n" + "=" * 80)
print(f"  {'Name':<28} {'IS Ret':>8} {'OOS Ret':>8} {'OOS DD':>8} {'OOS Sharpe':>11} {'Trades':>7}")
print("=" * 80)

for c in candidates:
    params = {k: v for k, v in c.items() if k != "name"}
    strat  = DonchianBreakoutStrategy({**config, **params})

    is_r  = engine.run(df_is,  strat, symbol, verbose=False)
    oos_r = engine.run(df_oos, strat, symbol, verbose=False)

    is_ret  = is_r.metrics.get("total_return", 0)
    oos_ret = oos_r.metrics.get("total_return", 0)
    oos_dd  = oos_r.metrics.get("max_drawdown", 0)
    oos_sh  = oos_r.metrics.get("sharpe", 0)
    oos_tr  = oos_r.metrics.get("total_trades", 0)
    flag    = "OK  " if oos_ret > 0 and oos_dd > -0.30 else "WARN" if oos_ret > 0 else "FAIL"

    print(f"  {flag} {c['name']:<26} {is_ret:>7.1%} {oos_ret:>8.1%} "
          f"{oos_dd:>8.1%} {oos_sh:>11.2f} {oos_tr:>7.0f}")

print("=" * 80)
print("\nRecommendation: copy parameters marked OK into config/settings.yaml")
print("Then confirm with: python3 main.py --mode backtest  (2+ years of data)")
