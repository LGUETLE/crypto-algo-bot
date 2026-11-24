"""
Cost report
===========
cost_breakdown()  decomposes one backtest into gross PnL at mid prices (before fees and slippage),
                  slippage, fees, funding and net PnL; turnover; break-even cost per round trip
cost_scenarios()  re-runs the same strategy with no costs, fees only, fees + slippage,
                  the configured costs and 2x stressed costs (replaces analyze_costs.py)

Break-even cost per round trip = (gross PnL + funding) / sum of entry notionals: the total
trading cost (fees + slippage, both sides) as a fraction of notional at which net PnL is zero.
Compare with the assumed 2 x (commission + slippage). A negative value means the strategy loses
money even before trading costs.
"""
import copy

import pandas as pd

from backtesting.engine import BacktestEngine
from features.timeframes import to_timedelta


def cost_breakdown(trades: pd.DataFrame, equity: pd.Series, capital: float, config: dict) -> dict:
    bt = config["backtest"]
    out = {"assumed_cost_rt": 2 * (bt["commission"] + bt["slippage"]), "round_trips": 0}
    if trades is None or trades.empty:
        return {**out, "gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "funding": 0.0, "net_pnl": 0.0,
                "turnover_per_year": 0.0, "break_even_cost_rt": float("nan"), "avg_notional": 0.0}
    side = trades["side"].map({"long": 1, "short": -1})
    price_pnl = (side * (trades["exit_price"] - trades["entry_price"]) * trades["size"]).sum()
    fees, slip, fund = trades["fees"].sum(), trades["slippage"].sum(), trades["funding"].sum()
    gross = price_pnl + slip                         # at mid prices, before fees and slippage
    entry_notional = (trades["size"] * trades["entry_price"]).sum()
    traded = (trades["size"] * (trades["entry_price"] + trades["exit_price"])).sum()
    bar = to_timedelta(config.get("timeframe", "1h"))
    years = (equity.index[-1] - equity.index[0] + bar) / pd.Timedelta(days=365)
    return {
        **out,
        "round_trips":        int(len(trades)),
        "gross_pnl":          float(gross),
        "slippage":           float(slip),
        "fees":               float(fees),
        "funding":            float(fund),
        "net_pnl":            float(gross - slip - fees + fund),
        "avg_notional":       float(entry_notional / len(trades)),
        "turnover_per_year":  float(traded / equity.mean() / years),
        "break_even_cost_rt": float((gross + fund) / entry_notional),
    }


def print_breakdown(c: dict, capital: float, title: str = "COST REPORT"):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for label, key in [("Gross PnL (mid prices)", "gross_pnl"), ("- slippage", "slippage"),
                       ("- fees", "fees"), ("+ funding", "funding"), ("= Net PnL", "net_pnl")]:
        v = c[key]
        print(f"  {label:<28} {v:>12,.2f}  ({v / capital:+.1%} of start capital)")
    print(f"  {'Round trips':<28} {c['round_trips']:>12}")
    print(f"  {'Avg notional per trade':<28} {c['avg_notional']:>12,.2f}")
    print(f"  {'Turnover (x equity / year)':<28} {c['turnover_per_year']:>12.1f}")
    print(f"  {'Break-even cost / round trip':<28} {c['break_even_cost_rt']:>12.4%}   "
          f"(assumed {c['assumed_cost_rt']:.4%})")
    print("=" * 70 + "\n")


SCENARIOS = [
    # label, commission multiplier, slippage multiplier, use funding
    ("No costs",                    0.0, 0.0, False),
    ("Fees only",                   1.0, 0.0, False),
    ("Fees + slippage",             1.0, 1.0, False),
    ("Configured (+ funding)",      1.0, 1.0, True),
    ("Stressed (2x fees+slippage)", 2.0, 2.0, True),
]


def cost_scenarios(config, df, strategy_factory, symbol, funding, start=None, intrabar=None) -> pd.DataFrame:
    rows = []
    for label, cm, sm, use_f in SCENARIOS:
        c = copy.deepcopy(config)
        c["backtest"]["commission"] *= cm
        c["backtest"]["slippage"] *= sm
        if not use_f:
            c["backtest"]["funding_rate_8h"] = 0.0
        r = BacktestEngine(c).run(df, strategy_factory(c), symbol, verbose=False, start=start,
                                  funding=funding if use_f else None, intrabar=intrabar)
        m = r.metrics
        rows.append({"scenario": label, "total_return": m.get("total_return", 0.0),
                     "sharpe": m.get("sharpe", 0.0), "profit_factor": m.get("profit_factor", 0.0),
                     "trades": m.get("total_trades", 0)})
    return pd.DataFrame(rows).set_index("scenario")


def print_scenarios(table: pd.DataFrame):
    print(f"  {'Scenario':<30} {'Return':>9} {'Sharpe':>7} {'PF':>6} {'Trades':>7}")
    for name, r in table.iterrows():
        print(f"  {name:<30} {r['total_return']:>9.1%} {r['sharpe']:>7.2f} {r['profit_factor']:>6.2f} "
              f"{int(r['trades']):>7}")
    print()
