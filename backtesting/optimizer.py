"""
Parameter Optimizer
===================
grid_search()            all parameter combinations on ONE in-sample window
walk_forward_optimize()  true walk-forward optimisation: for every window the grid is searched on
                         the in-sample (IS) part only, the best parameters are frozen and traded on
                         the following out-of-sample (OOS) part; the OOS pieces are stitched into one
                         equity curve. Rolling (fixed-length IS) or anchored (expanding IS).

CAUTION: every extra grid search or OOS peek erodes the independence of the OOS result (data
snooping). The final hold-out period is handled outside this module (backtesting/research.py) and is
never passed in here.
"""
import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.metrics import calc_all
from backtesting.stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio

logger = logging.getLogger(__name__)

METRIC_COLS = ("sharpe", "total_return", "max_drawdown", "profit_factor", "win_rate", "total_trades",
               "sharpe_per_period", "n_periods", "skew", "kurtosis")


@dataclass
class Window:
    n:         int
    is_start:  pd.Timestamp
    is_end:    pd.Timestamp       # exclusive, == oos_start
    oos_start: pd.Timestamp
    oos_end:   pd.Timestamp       # exclusive


@dataclass
class WFOResult:
    mode:      str
    windows:   pd.DataFrame       # one row per window: dates, chosen params, IS and OOS metrics
    equity:    pd.Series          # stitched OOS equity
    trades:    pd.DataFrame       # all OOS trades
    metrics:   dict               # metrics of the stitched OOS equity (+ psr, dsr)
    stability: dict               # how stable the chosen parameters are across windows
    settings:  dict = field(default_factory=dict)


def make_windows(data_start, data_end, is_months: int, oos_months: int,
                 step_months: Optional[int] = None, mode: str = "rolling") -> list:
    """
    Calendar windows. data_end is exclusive. The last OOS window may be shorter than oos_months.
      rolling:  IS = [start + k*step, start + k*step + is_months)
      anchored: IS = [start,          start + k*step + is_months)
    """
    if mode not in ("rolling", "anchored"):
        raise ValueError(f"mode must be 'rolling' or 'anchored', got {mode!r}")
    step  = step_months or oos_months
    start = pd.Timestamp(data_start)
    end   = pd.Timestamp(data_end)
    out, k = [], 0
    while True:
        is_end = start + pd.DateOffset(months=is_months + k * step)
        if is_end >= end:
            break
        is_start = start if mode == "anchored" else start + pd.DateOffset(months=k * step)
        oos_end  = min(is_end + pd.DateOffset(months=oos_months), end)
        out.append(Window(k + 1, is_start, is_end, is_end, oos_end))
        k += 1
    return out


class ParameterOptimizer:
    def __init__(self, config: dict, engine: BacktestEngine, log=None):
        self.config = config
        self.engine = engine
        self.log    = log                   # backtesting.research_log.ResearchLog or None

    # ── Grid search on one window ─────────────────────────────────────────

    def evaluate_grid(self, df, strategy_class: Type, param_grid: dict, start=None, end=None,
                      funding=None, symbol="BTC/USDT:USDT") -> pd.DataFrame:
        """Runs EVERY combination on [start, end) and returns all trials (also the failed ones)."""
        keys   = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        if end is not None:
            df = df[df.index < pd.Timestamp(end)]          # hard guarantee: nothing after the window
        rows = []
        for combo in combos:
            params = dict(zip(keys, combo))
            strat  = strategy_class({**self.config, **params})
            try:
                m = self.engine.run(df, strat, symbol, verbose=False, start=start, end=end,
                                    funding=funding).metrics
            except Exception as e:                          # noqa: BLE001 - a broken combo is a failed trial
                logger.warning(f"{params}: {e}")
                m = {}
            rows.append({**params, **{k: m.get(k, np.nan) for k in METRIC_COLS}})
        out = pd.DataFrame(rows)
        out["total_trades"] = out["total_trades"].fillna(0)
        return out

    @staticmethod
    def filter_valid(trials: pd.DataFrame, min_trades: int, max_dd_limit: float) -> pd.DataFrame:
        ok = (trials["total_trades"] >= min_trades) & (trials["max_drawdown"] >= max_dd_limit) & \
             trials["sharpe"].notna()
        return trials[ok]

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
        start=None, end=None, funding=None,
    ) -> pd.DataFrame:
        """
        Valid combinations sorted by `metric` (best first). The window is [start, end) if given,
        otherwise the first (1 - oos_pct) of df.
        """
        if start is None and end is None:
            end = df.index[int(len(df) * (1 - oos_pct))]
        trials = self.evaluate_grid(df, strategy_class, param_grid, start, end, funding, symbol)
        valid  = self.filter_valid(trials, min_trades, max_dd_limit)
        logger.info(f"Grid search: {len(trials)} combinations, {len(valid)} valid")
        if valid.empty:
            logger.error("No valid results - adjust the parameter grid")
            return pd.DataFrame()
        return valid.sort_values(metric, ascending=False).reset_index(drop=True)

    def best_params(self, results: pd.DataFrame, metric: str = "sharpe", top_n: int = 5) -> dict:
        if results.empty:
            return {}
        param_cols = [c for c in results.columns if c not in METRIC_COLS]
        print(f"\nTop {top_n} by {metric}:")
        print(results[param_cols + [metric, "total_return", "max_drawdown", "total_trades"]]
              .head(top_n).to_string())
        return {k: _py(v) for k, v in results[param_cols].iloc[0].items()}

    # ── Walk-forward optimisation ─────────────────────────────────────────

    def walk_forward_optimize(
        self,
        df:             pd.DataFrame,
        strategy_class: Type,
        param_grid:     dict,
        *,
        symbol:         str = "BTC/USDT:USDT",
        funding:        Optional[pd.Series] = None,
        mode:           str = "rolling",
        is_months:      int = 12,
        oos_months:     int = 3,
        step_months:    Optional[int] = None,
        metric:         str = "sharpe",
        min_trades:     int = 10,
        max_dd_limit:   float = -0.50,
        capital:        Optional[float] = None,
        n_variants:     int = 1,          # WFO configurations tried on this data (for the OOS DSR)
        strategy_name:  Optional[str] = None,
        study_id:       Optional[str] = None,
        verbose:        bool = True,
    ) -> WFOResult:
        capital  = capital or self.engine.initial_capital
        bar      = pd.Timedelta(self.config.get("timeframe", "1h"))
        data_end = df.index[-1] + bar
        windows  = make_windows(df.index[0], data_end, is_months, oos_months, step_months, mode)
        if not windows:
            raise ValueError("Not enough data for a single walk-forward window")
        keys     = list(param_grid.keys())
        n_trials = int(np.prod([len(v) for v in param_grid.values()]))
        tf       = self.config.get("timeframe")
        name     = strategy_name or strategy_class.__name__

        rows, eq_parts, tr_parts, trial_vars = [], [], [], []
        equity_now = capital
        for w in windows:
            trials = self.evaluate_grid(df, strategy_class, param_grid, w.is_start, w.is_end, funding, symbol)
            valid  = self.filter_valid(trials, min_trades, max_dd_limit)
            sr_bar = trials["sharpe_per_period"].astype(float)
            v_sr   = float(sr_bar[np.isfinite(sr_bar)].var(ddof=1)) if np.isfinite(sr_bar).sum() > 1 else 0.0
            trial_vars.append(v_sr)

            row = {"window": w.n, "is_start": w.is_start, "is_end": w.is_end,
                   "oos_start": w.oos_start, "oos_end": w.oos_end, "n_valid": len(valid)}

            oos_df = df[df.index < w.oos_end]
            if valid.empty:
                params = None
                idx = df.index[(df.index >= w.oos_start) & (df.index < w.oos_end)]
                eq  = pd.Series(equity_now, index=idx, name="equity")    # stay flat
                oos_m, oos_t = calc_all(eq, pd.DataFrame(), equity_now,
                                        self.engine.periods_per_year(idx)), pd.DataFrame()
                row.update({k: None for k in keys})
                row.update({"is_sharpe": np.nan, "is_trades": 0, "is_dsr": np.nan})
            else:
                best   = valid.sort_values(metric, ascending=False).iloc[0]
                params = {k: _py(best[k]) for k in keys}
                row.update(params)
                row.update({
                    "is_sharpe": best["sharpe"], "is_trades": int(best["total_trades"]),
                    "is_dsr": deflated_sharpe_ratio(best["sharpe_per_period"], int(best["n_periods"]),
                                                    best["skew"], best["kurtosis"], n_trials, v_sr),
                })
                res = self.engine.run(oos_df, strategy_class({**self.config, **params}), symbol,
                                      capital=equity_now, verbose=False, start=w.oos_start,
                                      end=w.oos_end, funding=funding)
                eq, oos_m, oos_t = res.equity, res.metrics, res.trades
                if not oos_t.empty:
                    oos_t = oos_t.assign(window=w.n)

            row.update({"oos_return": oos_m.get("total_return", 0.0), "oos_sharpe": oos_m.get("sharpe", 0.0),
                        "oos_max_dd": oos_m.get("max_drawdown", 0.0),
                        "oos_trades": int(oos_m.get("total_trades", 0) or 0)})
            rows.append(row)
            eq_parts.append(eq)
            tr_parts.append(oos_t)
            if len(eq):
                equity_now = float(eq.iloc[-1])

            if self.log is not None:
                self.log.append(
                    "grid_search", study_id=study_id, strategy=name, symbol=symbol, timeframe=tf,
                    params=params, n_trials=n_trials,
                    period={"start": w.is_start, "end": w.is_end},
                    metrics={"is_sharpe": row["is_sharpe"], "is_dsr": row["is_dsr"],
                             "trial_sr_variance": v_sr, "n_valid": len(valid)},
                    mode=mode, window=w.n, selection={"metric": metric, "min_trades": min_trades,
                                                      "max_dd_limit": max_dd_limit},
                    trials=trials.to_dict(orient="records"))
            logger.info(f"WFO {mode} window {w.n}/{len(windows)}: params={params} "
                        f"OOS return {row['oos_return']:+.2%}")

        equity = pd.concat(eq_parts)
        trades = pd.concat([t for t in tr_parts if not t.empty], ignore_index=True) \
            if any(not t.empty for t in tr_parts) else pd.DataFrame()
        metrics = calc_all(equity, trades, capital, self.engine.periods_per_year(equity.index),
                           self.engine.risk_free_rate)
        v_mean  = float(np.mean(trial_vars)) if trial_vars else 0.0
        metrics["psr"] = probabilistic_sharpe_ratio(metrics["sharpe_per_period"], 0.0, metrics["n_periods"],
                                                    metrics["skew"], metrics["kurtosis"])
        metrics["dsr"] = deflated_sharpe_ratio(metrics["sharpe_per_period"], metrics["n_periods"],
                                               metrics["skew"], metrics["kurtosis"], n_variants, v_mean)
        metrics["dsr_n_variants"]    = n_variants
        metrics["trial_sr_variance"] = v_mean

        windows_df = pd.DataFrame(rows)
        stability  = param_stability(windows_df, keys)
        settings   = {"mode": mode, "is_months": is_months, "oos_months": oos_months,
                      "step_months": step_months or oos_months, "metric": metric, "min_trades": min_trades,
                      "max_dd_limit": max_dd_limit, "param_grid": param_grid, "n_trials_per_window": n_trials}
        result = WFOResult(mode, windows_df, equity, trades, metrics, stability, settings)

        if self.log is not None:
            self.log.append(
                "wfo_study", study_id=study_id, strategy=name, symbol=symbol, timeframe=tf,
                params=None, n_trials=n_trials * len(windows),
                period={"start": windows[0].is_start, "oos_start": windows[0].oos_start,
                        "end": windows[-1].oos_end},
                metrics=_summary(metrics), settings=settings, stability=stability,
                windows=windows_df.to_dict(orient="records"))
        if verbose:
            print_wfo(result)
        return result

    # ── Legacy single split (kept for comparison) ─────────────────────────

    def validate_best(self, df, strategy_class: Type, best_params: dict, oos_pct: float = 0.3,
                      symbol: str = "BTC/USDT:USDT", funding=None) -> dict:
        split     = df.index[int(len(df) * (1 - oos_pct))]
        strategy  = strategy_class({**self.config, **best_params})
        is_res    = self.engine.run(df, strategy, symbol, verbose=False, end=split, funding=funding)
        oos_res   = self.engine.run(df, strategy, symbol, verbose=False, start=split, funding=funding)
        print("\n" + "=" * 55)
        print("  VALIDATION: in-sample vs. out-of-sample (single split)")
        print("=" * 55)
        print(f"  Parameters: {best_params}")
        print(f"  {'Metric':<25} {'In-sample':>12} {'Out-of-sample':>14}")
        for label, key, fmt in [("Sharpe", "sharpe", ".2f"), ("Total return", "total_return", ".1%"),
                                ("Max drawdown", "max_drawdown", ".1%"), ("Trades", "total_trades", ".0f")]:
            print(f"  {label:<25} {format(is_res.metrics.get(key, 0), fmt):>12} "
                  f"{format(oos_res.metrics.get(key, 0), fmt):>14}")
        print("=" * 55 + "\n")
        return {"is_metrics": is_res.metrics, "oos_metrics": oos_res.metrics}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _py(v):
    """numpy scalar -> plain Python (ints stay ints)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return int(f) if f.is_integer() else f
    return v


def param_stability(windows: pd.DataFrame, keys: list) -> dict:
    """Distinct parameter sets, how often they change between windows, value counts per parameter."""
    chosen = windows.dropna(subset=keys) if keys else windows
    if chosen.empty:
        return {"n_windows": len(windows), "n_selected": 0}
    sets    = [tuple(_py(r[k]) for k in keys) for _, r in chosen.iterrows()]
    changes = sum(1 for a, b in zip(sets, sets[1:]) if a != b)
    return {
        "n_windows":       len(windows),
        "n_selected":      len(sets),
        "n_distinct_sets": len(set(sets)),
        "change_rate":     changes / max(len(sets) - 1, 1),
        "most_common_set": dict(zip(keys, max(set(sets), key=sets.count))),
        "value_counts":    {k: {str(_py(v)): int(c) for v, c in chosen[k].value_counts().items()} for k in keys},
    }


def _summary(m: dict) -> dict:
    keys = ("total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "volatility",
            "total_trades", "win_rate", "profit_factor", "psr", "dsr", "dsr_n_variants",
            "sharpe_per_period", "skew", "kurtosis", "n_periods", "trial_sr_variance")
    return {k: m.get(k) for k in keys}


def print_wfo(r: WFOResult):
    keys = [k for k in r.settings["param_grid"].keys()]
    print("\n" + "=" * 100)
    print(f"  WALK-FORWARD OPTIMISATION ({r.mode}, IS {r.settings['is_months']}m / OOS "
          f"{r.settings['oos_months']}m, {r.settings['n_trials_per_window']} combos per window)")
    print("=" * 100)
    hdr = f"  {'#':>2} {'OOS period':<23} " + " ".join(f"{k[:9]:>9}" for k in keys) + \
          f" {'IS SR':>6} {'IS DSR':>6} {'OOS ret':>8} {'OOS SR':>7} {'OOS DD':>7} {'Tr':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, w in r.windows.iterrows():
        vals = " ".join(f"{('-' if pd.isna(w[k]) else _py(w[k])):>9}" for k in keys)
        print(f"  {int(w['window']):>2} {w['oos_start']:%Y-%m-%d} - {w['oos_end']:%Y-%m-%d} {vals} "
              f"{w['is_sharpe']:>6.2f} {w['is_dsr']:>6.2f} {w['oos_return']:>8.1%} "
              f"{w['oos_sharpe']:>7.2f} {w['oos_max_dd']:>7.1%} {int(w['oos_trades']):>4}")
    m = r.metrics
    print("  " + "-" * (len(hdr) - 2))
    print(f"  Stitched OOS: return {m['total_return']:+.1%} | CAGR {m['cagr']:+.1%} | Sharpe {m['sharpe']:.2f} | "
          f"max DD {m['max_drawdown']:.1%} | trades {m.get('total_trades', 0)}")
    print(f"  PSR(SR>0) {m['psr']:.3f} | DSR {m['dsr']:.3f} (N={m['dsr_n_variants']} WFO variant(s), "
          f"V[SR]={m['trial_sr_variance']:.2e} per bar from the IS trials)")
    s = r.stability
    if s.get("n_selected"):
        print(f"  Parameter stability: {s['n_distinct_sets']} distinct sets in {s['n_selected']} windows, "
              f"change rate {s['change_rate']:.0%}, most common {s['most_common_set']}")
    print("=" * 100 + "\n")
