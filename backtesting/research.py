"""
Research pipeline
=================
load_research_data()   prices (+ guardian), funding and BTC benchmark data for one pair, with the
                       FINAL HOLD-OUT cut off unless explicitly requested
run_backtest()         fixed (configured) parameters: report, benchmarks, split consistency check
run_optimize()         walk-forward optimisation (rolling + anchored), PSR/DSR, benchmarks
evaluate_holdout()     the single final hold-out evaluation

Hold-out rule: config backtest.holdout_start splits the data into a research period
[start_date, holdout_start) and a hold-out [holdout_start, end_date). Nothing in this module reads
hold-out rows unless final_holdout=True (CLI: --final-holdout). Every hold-out evaluation is
written to the research log; a repeated evaluation of the same strategy/symbol/timeframe/hold-out
start triggers a warning, because a hold-out that has been looked at more than once is no longer
out of sample.

Strategies receive raw OHLCV (+ macro columns) exactly like the live bot (main.run_cycle); no
FeatureEngineer columns are precomputed, so the strategy computes every indicator itself with its
own parameters.
"""
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtesting.benchmark import compare, print_comparison
from backtesting.engine import BacktestEngine
from backtesting.metrics import print_report
from backtesting.optimizer import ParameterOptimizer, _py
from backtesting.research_log import DEFAULT_PATH, ResearchLog, new_study_id
from strategies.registry import strategy_class

logger = logging.getLogger(__name__)

BTC = "BTC/USDT:USDT"


class HoldoutReuseWarning(UserWarning):
    pass


@dataclass
class ResearchData:
    symbol:           str
    df:               pd.DataFrame            # OHLCV (+ macro_close / macro_trend)
    funding:          Optional[pd.Series]     # realised funding rates or None (-> constant fallback)
    btc:              pd.DataFrame            # BTC OHLCV for the benchmark
    holdout_start:    Optional[pd.Timestamp]
    includes_holdout: bool


# ── Config helpers ────────────────────────────────────────────────────────────

def _ts(x) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def research_window(config: dict):
    """(start, end) of all data, end exclusive. Fixed dates if configured, else the last N days."""
    bt = config["backtest"]
    if bt.get("start_date"):
        return _ts(bt["start_date"]), _ts(bt.get("end_date")) or pd.Timestamp.now(tz="UTC").floor("D")
    end = pd.Timestamp.now(tz="UTC").floor("D")
    return end - pd.Timedelta(days=bt["lookback_days"]), end


def holdout_start(config: dict) -> Optional[pd.Timestamp]:
    return _ts(config["backtest"].get("holdout_start"))


def cut_holdout(obj, hs: Optional[pd.Timestamp]):
    """Rows strictly before the hold-out start (works for DataFrame and Series)."""
    if hs is None or obj is None:
        return obj
    return obj[obj.index < hs]


def macro_timeframe(config: dict):
    return config.get("macro_timeframe") if config.get("use_macro_filter", True) else None


def optimize_settings(config: dict, strategy_name: str) -> dict:
    o = config.get("optimize", {})
    grids = o.get("param_grid", {})
    if strategy_name not in grids:
        raise ValueError(f"No optimize.param_grid for {strategy_name} in config")
    return {
        "metric":       o.get("metric", "sharpe"),
        "min_trades":   o.get("min_trades", 10),
        "max_dd_limit": o.get("max_dd_limit", -0.5),
        "is_months":    o.get("is_months", 12),
        "oos_months":   o.get("oos_months", 3),
        "modes":        list(o.get("modes", ["rolling", "anchored"])),
        "param_grid":   {k: list(v) for k, v in grids[strategy_name].items()},
    }


def research_log(config: dict) -> ResearchLog:
    return ResearchLog(config["backtest"].get("research_log", DEFAULT_PATH))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_research_data(config: dict, pair: dict, fetcher, final_holdout: bool = False) -> ResearchData:
    data_symbol = pair.get("data_symbol", pair["symbol"])
    tf          = config["timeframe"]
    start, end  = research_window(config)
    hs          = holdout_start(config)
    if hs is not None and not (start < hs < end):
        raise ValueError(f"holdout_start {hs} must lie inside the data window [{start}, {end})")

    df = fetcher.historical(data_symbol, tf, start=start, end=end, macro_tf=macro_timeframe(config))
    try:
        funding = fetcher.funding_history(data_symbol, start, end)
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"Historical funding unavailable for {data_symbol} ({e}); "
                       f"the engine falls back to the constant funding_rate_8h")
        funding = None
    btc = df if data_symbol.split(":")[0] == BTC.split(":")[0] else fetcher.ohlcv_range(BTC, tf, start, end)

    if not final_holdout:
        df, funding, btc = cut_holdout(df, hs), cut_holdout(funding, hs), cut_holdout(btc, hs)
        if hs is not None:
            logger.info(f"Hold-out from {hs:%Y-%m-%d} is excluded (pass --final-holdout to evaluate it)")
    return ResearchData(data_symbol, df, funding, btc, hs, final_holdout)


def _strategy_class(config, name):
    return strategy_class(name)


def _cost_per_side(config) -> float:
    return config["backtest"]["commission"] + config["backtest"]["slippage"]


def _period(eq: pd.Series) -> dict:
    return {"start": eq.index[0], "end": eq.index[-1]} if len(eq) else {}


def _benchmarks(config, engine, data: ResearchData, equity, capital, title):
    asset = None if data.symbol.split(":")[0] == BTC.split(":")[0] else data.df
    table = compare(equity, capital, data.btc, engine.periods_per_year(equity.index),
                    _cost_per_side(config), asset_ohlc=asset,
                    asset_name=data.symbol.split("/")[0], risk_free_rate=engine.risk_free_rate)
    print_comparison(table, title)
    return table


# ── Backtest with fixed parameters ───────────────────────────────────────────

def run_backtest(config: dict, pair: dict, fetcher, final_holdout: bool = False, log=None) -> dict:
    strat_name = pair.get("strategy", config.get("strategy"))
    log        = log or research_log(config)
    data       = load_research_data(config, pair, fetcher, final_holdout)
    engine     = BacktestEngine(config)
    strategy   = _strategy_class(config, strat_name)(config)
    params     = {k: config.get(k) for k in optimize_settings(config, strat_name)["param_grid"]} \
        if strat_name in config.get("optimize", {}).get("param_grid", {}) else {}

    start = data.holdout_start if final_holdout and data.holdout_start is not None else None
    if final_holdout:
        _holdout_guard(log, strat_name, data.symbol, config["timeframe"], data.holdout_start)

    label = "FINAL HOLD-OUT" if final_holdout else "RESEARCH PERIOD"
    print(f"\n{'=' * 60}\n  BACKTEST ({label}): {data.symbol} | {strat_name} | fixed config parameters\n{'=' * 60}")
    res = engine.run(data.df, strategy, data.symbol, verbose=False, start=start, funding=data.funding)
    _print_trades_summary(res)
    print_report(res.metrics)
    table = _benchmarks(config, engine, data, res.equity, engine.initial_capital,
                        f"BENCHMARKS - backtest ({label.lower()})")

    split = None
    if not final_holdout:
        split = engine.split_consistency_check(data.df, strategy, data.symbol, funding=data.funding)

    kind = "holdout" if final_holdout else "backtest"
    extra = {"holdout_start": data.holdout_start.isoformat()} if final_holdout else {}
    log.append(kind, strategy=strat_name, symbol=data.symbol, timeframe=config["timeframe"],
               params=params, period=_period(res.equity), metrics=res.metrics, n_trials=1,
               benchmarks=table.reset_index().to_dict(orient="records"),
               funding_source="historical" if data.funding is not None else "constant", **extra)
    return {"result": res, "benchmarks": table, "split_check": split}


def _print_trades_summary(res):
    t = res.trades
    if t.empty:
        print("  No trades")
        return
    for side in ("long", "short"):
        s = t[t["side"] == side]
        print(f"  {side.capitalize():<5} trades: {len(s):>4} | net PnL: {s['pnl_usdt'].sum():>10.2f} | "
              f"fees: {s['fees'].sum():>8.2f} | funding: {s['funding'].sum():>8.2f}")
    print(f"  Exits: { {k: int(v) for k, v in t['exit_reason'].value_counts().items()} }")


# ── Walk-forward optimisation ─────────────────────────────────────────────────

def _variant_key(settings: dict) -> str:
    keys = ("mode", "is_months", "oos_months", "step_months", "metric", "min_trades", "max_dd_limit",
            "param_grid")
    return json.dumps({k: settings.get(k) for k in keys}, sort_keys=True, default=str)


def count_wfo_variants(log: ResearchLog, strategy: str, symbol: str, timeframe: str,
                       current: list) -> int:
    """Distinct WFO configurations ever evaluated on this strategy/symbol/timeframe (incl. current)."""
    seen = {_variant_key(e.get("settings") or {}) for e in
            log.entries("wfo_study", strategy=strategy, symbol=symbol, timeframe=timeframe)}
    seen |= {_variant_key(s) for s in current}
    return len(seen)


def run_optimize(config: dict, pair: dict, fetcher, final_holdout: bool = False, log=None) -> dict:
    strat_name = pair.get("strategy", config.get("strategy"))
    log        = log or research_log(config)
    s          = optimize_settings(config, strat_name)
    data       = load_research_data(config, pair, fetcher, final_holdout=False)   # never the hold-out
    engine     = BacktestEngine(config)
    optimizer  = ParameterOptimizer(config, engine, log)
    cls        = _strategy_class(config, strat_name)
    tf         = config["timeframe"]

    current = [{**s, "mode": m, "step_months": s["oos_months"]} for m in s["modes"]]
    n_var   = count_wfo_variants(log, strat_name, data.symbol, tf, current)
    print(f"\n{'=' * 60}\n  WALK-FORWARD OPTIMISATION: {data.symbol} | {strat_name}\n"
          f"  research period {data.df.index[0]:%Y-%m-%d} - {data.df.index[-1]:%Y-%m-%d} "
          f"(hold-out from {data.holdout_start:%Y-%m-%d} excluded)\n{'=' * 60}")

    results = {}
    for mode in s["modes"]:
        study = new_study_id(f"wfo-{mode}")
        r = optimizer.walk_forward_optimize(
            data.df, cls, s["param_grid"], symbol=data.symbol, funding=data.funding, mode=mode,
            is_months=s["is_months"], oos_months=s["oos_months"], metric=s["metric"],
            min_trades=s["min_trades"], max_dd_limit=s["max_dd_limit"], n_variants=n_var,
            strategy_name=strat_name, study_id=study)
        table = _benchmarks(config, engine, data, r.equity, engine.initial_capital,
                            f"BENCHMARKS - stitched OOS ({mode})")
        log.append("wfo_benchmarks", study_id=study, strategy=strat_name, symbol=data.symbol, timeframe=tf,
                   period=_period(r.equity), benchmarks=table.reset_index().to_dict(orient="records"))
        results[mode] = {"wfo": r, "benchmarks": table, "study_id": study}

    out = {"wfo": results}
    if final_holdout:
        out["holdout"] = evaluate_holdout(config, pair, fetcher, log=log)
    return out


# ── Final hold-out ────────────────────────────────────────────────────────────

def _holdout_guard(log: ResearchLog, strategy, symbol, timeframe, hs):
    if hs is None:
        raise ValueError("--final-holdout needs backtest.holdout_start in the config")
    prior = log.holdout_evaluations(strategy, symbol, timeframe, hs)
    if prior:
        msg = (f"HOLD-OUT REUSE: {strategy} {symbol} {timeframe} hold-out from {hs:%Y-%m-%d} has already "
               f"been evaluated {len(prior)} time(s) (first {prior[0]['timestamp']}). "
               f"It is no longer an unbiased out-of-sample test.")
        logger.warning(msg)
        warnings.warn(msg, HoldoutReuseWarning, stacklevel=3)
    return len(prior)


def evaluate_holdout(config: dict, pair: dict, fetcher, log=None) -> dict:
    """
    Pre-registered hold-out procedure (primary WFO mode, i.e. the first in optimize.modes):
      1. fit the grid on the last IS window before holdout_start, with exactly the WFO settings
         (rolling: [hs - is_months, hs); anchored: [start_date, hs)), on data < hs only
      2. freeze the selected parameters
      3. trade them once over [holdout_start, end_date) and compare with the benchmarks
    """
    strat_name = pair.get("strategy", config.get("strategy"))
    log        = log or research_log(config)
    s          = optimize_settings(config, strat_name)
    tf         = config["timeframe"]
    full       = load_research_data(config, pair, fetcher, final_holdout=True)
    hs         = full.holdout_start
    prior      = _holdout_guard(log, strat_name, full.symbol, tf, hs)

    engine    = BacktestEngine(config)
    optimizer = ParameterOptimizer(config, engine, log=None)
    cls       = _strategy_class(config, strat_name)
    mode      = s["modes"][0]
    is_start  = full.df.index[0] if mode == "anchored" else hs - pd.DateOffset(months=s["is_months"])

    research_df = cut_holdout(full.df, hs)                 # fitting never sees hold-out rows
    research_f  = cut_holdout(full.funding, hs)
    trials = optimizer.evaluate_grid(research_df, cls, s["param_grid"], is_start, hs, research_f, full.symbol)
    valid  = optimizer.filter_valid(trials, s["min_trades"], s["max_dd_limit"])
    if valid.empty:
        raise RuntimeError("No valid parameter set in the final IS window - nothing to evaluate")
    best   = valid.sort_values(s["metric"], ascending=False).iloc[0]
    params = {k: _py(best[k]) for k in s["param_grid"]}

    print(f"\n{'=' * 60}\n  FINAL HOLD-OUT EVALUATION: {full.symbol} | {strat_name}\n"
          f"  fit ({mode}) on [{is_start:%Y-%m-%d}, {hs:%Y-%m-%d}) -> {params}\n"
          f"  hold-out [{hs:%Y-%m-%d}, {full.df.index[-1]:%Y-%m-%d %H:%M}]"
          f"{'  (REPEATED EVALUATION #' + str(prior + 1) + ')' if prior else ''}\n{'=' * 60}")

    strategy = cls({**config, **params})
    res = engine.run(full.df, strategy, full.symbol, verbose=False, start=hs, funding=full.funding)
    _print_trades_summary(res)
    print_report(res.metrics)
    table = _benchmarks(config, engine, full, res.equity, engine.initial_capital, "BENCHMARKS - final hold-out")

    log.append("holdout", strategy=strat_name, symbol=full.symbol, timeframe=tf, params=params,
               period=_period(res.equity), metrics=res.metrics, n_trials=len(trials),
               holdout_start=hs.isoformat(), prior_evaluations=prior,
               fit={"mode": mode, "is_start": is_start, "is_end": hs, "is_sharpe": best["sharpe"],
                    "settings": {k: v for k, v in s.items() if k != "modes"}},
               benchmarks=table.reset_index().to_dict(orient="records"),
               funding_source="historical" if full.funding is not None else "constant")
    return {"params": params, "result": res, "benchmarks": table, "prior_evaluations": prior}

