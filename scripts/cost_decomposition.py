"""
Cost decomposition of the configured strategy on the RESEARCH period (hold-out excluded).
python scripts/cost_decomposition.py [--symbol ETH]

Runs the configured parameters three times: all costs on (historical funding), no fees/slippage
(funding on), no costs at all. Shows how much of the result is edge and how much is cost.
"""
import argparse
import copy
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import research                      # noqa: E402
from backtesting.engine import BacktestEngine         # noqa: E402
from data.fetcher import DataFetcher                  # noqa: E402
from strategies.registry import strategy_class        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    cfg = yaml.safe_load(open("config/settings.yaml"))
    pairs = [p for p in yaml.safe_load(open("config/pairs.yaml"))["pairs"] if p["active"]
             and (args.symbol is None or args.symbol.upper() in p["symbol"])]
    for pair in pairs:
        d = research.load_research_data(cfg, pair, DataFetcher(cfg["exchange"]), final_holdout=False)
        name = pair.get("strategy", cfg.get("strategy"))
        print(f"\n{d.symbol} | {name} | research period {d.df.index[0]:%Y-%m-%d} - {d.df.index[-1]:%Y-%m-%d}")
        for label, comm, slip, use_funding in [("all costs (historical funding)", None, None, True),
                                               ("no fees/slippage, funding on", 0.0, 0.0, True),
                                               ("no costs at all", 0.0, 0.0, False)]:
            c = copy.deepcopy(cfg)
            if comm is not None:
                c["backtest"].update(commission=comm, slippage=slip)
            if not use_funding:
                c["backtest"]["funding_rate_8h"] = 0.0
            r = BacktestEngine(c).run(d.df, strategy_class(name)(c), d.symbol, verbose=False,
                                      funding=d.funding if use_funding else None)
            t = r.trades
            gross = ((t.exit_price - t.entry_price) * t["size"] * t.side.map({"long": 1, "short": -1})).sum()
            print(f"  {label:<32} return {r.metrics['total_return']:+8.1%}  Sharpe {r.metrics['sharpe']:+6.2f}  "
                  f"trades {len(t):>4}  price PnL {gross:+9.0f}  fees {t.fees.sum():8.0f}  "
                  f"funding {t.funding.sum():+7.0f}  ambiguous trailing bars {r.metrics['ambiguous_trailing_bars']}")


if __name__ == "__main__":
    main()
