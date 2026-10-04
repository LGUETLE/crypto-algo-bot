"""
Cost analysis - shows how much transaction costs really cost.
python3 analyze_costs.py
"""
import yaml, logging
logging.basicConfig(level=logging.WARNING)

from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer
from features.macro import add_macro_trend  # noqa: F401  (macro merge happens inside the fetcher)
from backtesting.engine import BacktestEngine
from strategies.donchian_breakout import DonchianBreakoutStrategy

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

symbol = "ETH/USDT:USDT"
macro_tf = config.get("macro_timeframe") if config.get("use_macro_filter", True) else None
df = DataFetcher(config["exchange"]).historical(symbol, config["timeframe"],
                                                days=config["backtest"]["lookback_days"],
                                                macro_tf=macro_tf)
df = FeatureEngineer(config).transform(df)
strategy = DonchianBreakoutStrategy(config)

print("\n" + "=" * 60)
print("  COST ANALYSIS - ETH Donchian Breakout")
print("=" * 60)

scenarios = {
    "No costs (ideal)":        (0.0,    0.0),
    "Commission only (0.06%)": (0.0006, 0.0),
    "Commission + slippage":   (0.0006, 0.0002),
    "Higher costs (0.1%)":     (0.001,  0.0003),
}

results = {}
for name, (comm, slip) in scenarios.items():
    test_config = {**config, "backtest": {**config["backtest"], "commission": comm, "slippage": slip}}
    m = BacktestEngine(test_config).run(df, strategy, symbol=symbol, verbose=False).metrics
    results[name] = m
    print(f"\n  {name}")
    print(f"    Total return:  {m.get('total_return', 0):>8.1%}")
    print(f"    Sharpe:        {m.get('sharpe', 0):>8.2f}")
    print(f"    Profit factor: {m.get('profit_factor', 0):>8.2f}")
    print(f"    Trades:        {m.get('total_trades', 0):>8}")

no_cost   = results["No costs (ideal)"]["total_return"]
with_cost = results["Commission + slippage"]["total_return"]
n_trades  = max(results["Commission + slippage"].get("total_trades", 1), 1)
cost_impact = (no_cost - with_cost) * 10_000       # on a $10k basis

print(f"\n{'=' * 60}")
print("  Cost difference (ideal vs. realistic):")
print(f"  {no_cost:.1%} -> {with_cost:.1%}")
print(f"  Costs on $10,000 capital: ~${cost_impact:,.0f} over the test period")
print(f"  Per trade (approx.): ${cost_impact / n_trades:.1f}")
print("=" * 60)
