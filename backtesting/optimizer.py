"""
Parameter Optimizer
===================
Grid search over all parameter combinations.
Optimizes on in-sample data only, validates on out-of-sample data.

CAUTION: every extra grid search or OOS peek erodes the independence of the OOS
result (data snooping). Keep a final hold-out period that you never touch.
"""
import itertools
import logging
from typing import Type

import pandas as pd

from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    def __init__(self, config: dict, engine: BacktestEngine):
        self.config = config
        self.engine = engine

    def grid_search(
        self,
        df:             pd.DataFrame,
        strategy_class: Type,
        param_grid:     dict,
        metric:         str   = "sharpe",
        oos_pct:        float = 0.3,
        min_trades:     int   = 8,
        max_dd_limit:   float = -0.50,    # drop combinations with a drawdown worse than this
        symbol:         str   = "BTC/USDT:USDT",
    ) -> pd.DataFrame:
        split_idx = int(len(df) * (1 - oos_pct))
        df_is     = df.iloc[:split_idx]

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))

        logger.info(f"Grid search: {len(combos)} combinations | "
                    f"IS: {len(df_is)} candles | OOS: {len(df) - len(df_is)} candles")

        results = []
        for i, combo in enumerate(combos):
            params      = dict(zip(keys, combo))
            test_config = {**self.config, **params}
            strategy    = strategy_class(test_config)

            try:
                result = self.engine.run(df_is, strategy, symbol, verbose=False)
                m      = result.metrics
                if m.get("total_trades", 0) < min_trades:
                    continue
                if m.get("max_drawdown", -1) < max_dd_limit:
                    continue

                results.append({
                    **params,
                    "sharpe":        m.get("sharpe", 0),
                    "total_return":  m.get("total_return", 0),
                    "max_drawdown":  m.get("max_drawdown", 0),
                    "profit_factor": m.get("profit_factor", 0),
                    "win_rate":      m.get("win_rate", 0),
                    "total_trades":  m.get("total_trades", 0),
                })
            except Exception as e:
                logger.debug(f"{params}: {e}")

            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(combos)} tested...")

        if not results:
            logger.error("No valid results - adjust the parameter grid")
            return pd.DataFrame()

        df_res = pd.DataFrame(results).sort_values(metric, ascending=False).reset_index(drop=True)
        logger.info(f"Done: {len(df_res)} valid combinations")
        return df_res

    def validate_best(
        self,
        df:             pd.DataFrame,
        strategy_class: Type,
        best_params:    dict,
        oos_pct:        float = 0.3,
        symbol:         str   = "BTC/USDT:USDT",
    ) -> dict:
        split_idx = int(len(df) * (1 - oos_pct))
        df_is     = df.iloc[:split_idx]
        df_oos    = df.iloc[split_idx:]

        strategy   = strategy_class({**self.config, **best_params})
        is_result  = self.engine.run(df_is,  strategy, symbol, verbose=False)
        oos_result = self.engine.run(df_oos, strategy, symbol, verbose=False)

        print("\n" + "=" * 55)
        print("  VALIDATION: in-sample vs. out-of-sample")
        print("=" * 55)
        print(f"  Parameters: {best_params}")
        print(f"  {'Metric':<25} {'In-sample':>12} {'Out-of-sample':>14}")
        print(f"  {'-' * 52}")

        for label, key, fmt in [
            ("Sharpe",        "sharpe",        ".2f"),
            ("Total return",  "total_return",  ".1%"),
            ("Max drawdown",  "max_drawdown",  ".1%"),
            ("Profit factor", "profit_factor", ".2f"),
            ("Win rate",      "win_rate",      ".1%"),
            ("Trades",        "total_trades",  ".0f"),
        ]:
            iv = is_result.metrics.get(key, 0)
            ov = oos_result.metrics.get(key, 0)
            print(f"  {label:<25} {format(iv, fmt):>12} {format(ov, fmt):>14}")

        is_ret  = is_result.metrics.get("total_return", 0)
        oos_ret = oos_result.metrics.get("total_return", 0)
        deg     = (is_ret - oos_ret) / max(abs(is_ret), 1e-6)

        print(f"\n  Return degradation IS -> OOS: {deg:.0%}")
        is_robust = deg < 0.6 and oos_result.metrics.get("sharpe", 0) > 0.5
        if is_robust:
            print("  Robust - paper trading recommended")
        elif deg < 0.8 and oos_result.metrics.get("total_return", -1) > 0:
            print("  Moderate degradation - paper trading with caution")
        else:
            print("  High degradation - rework the strategy")
        print("=" * 55 + "\n")

        return {
            "is_metrics":  is_result.metrics,
            "oos_metrics": oos_result.metrics,
            "degradation": deg,
            "is_robust":   is_robust,
        }

    def best_params(self, results: pd.DataFrame, metric: str = "sharpe", top_n: int = 5) -> dict:
        if results.empty:
            return {}
        metric_cols = ("sharpe", "total_return", "max_drawdown", "profit_factor", "win_rate", "total_trades")
        param_cols  = [c for c in results.columns if c not in metric_cols]
        print(f"\nTop {top_n} by {metric}:")
        print(results[param_cols + [metric, "total_return", "max_drawdown", "total_trades"]]
              .head(top_n).to_string())
        return results[param_cols].iloc[0].to_dict()
