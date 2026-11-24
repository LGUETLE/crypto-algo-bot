"""
TSMOM research pipeline (docs/preregistration.md)
=================================================
python main.py --mode tsmom                  research period: V1-V4, W1, W2, R1, DSR (N = 7), benchmarks
python main.py --mode tsmom --final-holdout  ... and then the single hold-out evaluation (W1 procedure)

The hold-out [holdout_start, end_date) is cut from prices, volumes and funding before anything is
computed, unless the final hold-out is requested. Every configuration and the hold-out evaluation
are written to the research log; a repeated hold-out evaluation warns.
"""
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from backtesting.benchmark import information_ratio, vol_matched
from backtesting.metrics import calc_all, period_returns
from backtesting.optimizer import make_windows
from backtesting.portfolio import (DEFAULTS, VARIANTS, Panel, equal_weight_weights, membership, simulate,
                                   variant_weights)
from backtesting.research_log import ResearchLog, new_study_id
from backtesting.stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from data import universe

logger = logging.getLogger(__name__)

STRATEGY = "tsmom_portfolio"


class HoldoutReuseWarning(UserWarning):
    pass


def load_config(path="config/tsmom.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def params(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg["universe"], **cfg["signal"], **cfg["portfolio"], **cfg["costs"]}


@dataclass
class TsmomData:
    panel:   Panel
    members: pd.DataFrame
    btc:     pd.DataFrame          # daily BTC open/close for benchmarks
    mode:    str
    n_candidates: int


def load_data(cfg: dict, mode: str, final_holdout: bool = False) -> TsmomData:
    d = cfg["data"]
    start, end, hs = d["start_date"], d["end_date"], pd.Timestamp(d["holdout_start"], tz="UTC")
    p = params(cfg)
    inv = universe.binance_inventory()
    okx = universe.okx_xperp_coins()
    cands = universe.candidates(inv, okx["coins"], mode, cfg["universe"]["delisted_before"])
    logger.info(f"{mode}: {len(cands)} candidate symbols (OKX snapshot {okx['retrieved'][:10]})")
    frames = universe.load_daily_panel(cands, start, end)
    base = {s: universe.base_coin(s) for s in frames}
    panel = Panel.from_frames(frames, base=base)
    if not final_holdout:
        panel = panel.cut(hs)                                 # hold-out guard: before ANY computation
    members = membership(panel, p)
    ever = [s for s in members.columns if members[s].any()]
    fund_end = end if final_holdout else hs + pd.Timedelta(days=1)
    funding = universe.load_funding_panel(ever, start, fund_end, panel.close.index)
    panel = Panel(panel.open, panel.close, panel.volume,
                  funding.reindex(columns=panel.close.columns, fill_value=0.0), base)
    if not final_holdout:
        panel = panel.cut(hs)
    btc = frames["BTCUSDT"][["open", "close"]]
    if not final_holdout:
        btc = btc[btc.index < hs]
    return TsmomData(panel, members, btc, mode, len(cands))


# ── Building blocks ───────────────────────────────────────────────────────────

def _summary(res, extra=None) -> dict:
    m = res.metrics
    out = {k: m.get(k) for k in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
                                 "volatility", "sharpe_per_period", "skew", "kurtosis", "n_periods")}
    out.update({k: res.costs[k] for k in ("gross_pnl", "fees", "slippage", "funding", "net_pnl",
                                          "turnover_per_year")})
    out["psr"] = probabilistic_sharpe_ratio(m["sharpe_per_period"], 0.0, m["n_periods"], m["skew"], m["kurtosis"])
    return {**out, **(extra or {})}


def benchmarks(data: TsmomData, equity: pd.Series, capital: float, p: dict) -> pd.DataFrame:
    idx = equity.index
    ew_W = equal_weight_weights(data.members)
    prev = data.members.index[data.members.index < idx[0]]
    if len(prev):                                          # start invested, not at the next rebalance
        n = data.members.loc[prev[-1]].sum()
        ew_W.loc[prev[-1]] = data.members.loc[prev[-1]] / n if n else 0.0
    ew = simulate(data.panel, ew_W, {**p, "commission": 0.0, "slippage_major": 0.0, "slippage_other": 0.0,
                                     "band": 0.0}, start=idx[0], end=idx[-1] + pd.Timedelta(days=1),
                  capital=capital).equity
    btc = data.btc.reindex(idx)
    bh = capital * btc["close"] / btc["open"].iloc[0]
    vm = vol_matched(data.btc, equity, capital)
    rows = []
    for name, eq in [("Strategy", equity), ("BTC buy & hold", bh), ("Equal-weight universe", ew),
                     (f"BTC vol-matched (x{vm.attrs['scale']:.2f})", vm)]:
        m = calc_all(eq, pd.DataFrame(), capital, 365)
        row = {"line": name, "total_return": m["total_return"], "cagr": m["cagr"], "sharpe": m["sharpe"],
               "max_drawdown": m["max_drawdown"], "calmar": m["calmar"], "volatility": m["volatility"]}
        if name != "Strategy":
            row["excess_return"] = calc_all(equity, pd.DataFrame(), capital, 365)["total_return"] - m["total_return"]
            row["information_ratio"] = information_ratio(equity, eq, capital, 365)
        rows.append(row)
    return pd.DataFrame(rows).set_index("line")


def select_variant(data: TsmomData, weights: dict, is_start, is_end, p: dict):
    """Variant with the highest Sharpe on [is_start, is_end), each simulated from flat."""
    scores = {}
    for v, W in weights.items():
        r = simulate(data.panel, W, p, start=is_start, end=is_end)
        scores[v] = r.metrics.get("sharpe", float("-inf")) if len(r.equity) > 1 else float("-inf")
    best = max(sorted(scores), key=lambda v: scores[v])
    return best, scores


def wfo(data: TsmomData, weights: dict, cfg: dict, mode: str, p: dict):
    """Walk-forward over V1-V4. Weights decided on the close of day t trade on day t+1, so the row
    of day t follows the variant of the window that contains day t+1. Positions carry over."""
    v = cfg["validation"]
    idx = data.panel.close.index
    windows = make_windows(idx[0], idx[-1] + pd.Timedelta(days=1), v["is_months"], v["oos_months"], mode=mode)
    stitched = pd.DataFrame(np.nan, index=idx, columns=data.panel.close.columns)
    rows = []
    for w in windows:
        best, scores = select_variant(data, weights, w.is_start, w.is_end, p)
        trade_day = pd.Series(idx, index=idx).shift(-1)
        rows_in = (trade_day >= w.oos_start) & (trade_day < w.oos_end)
        stitched.loc[rows_in.to_numpy()] = weights[best].loc[rows_in.to_numpy()].to_numpy()
        rows.append({"window": w.n, "is_start": w.is_start, "oos_start": w.oos_start, "oos_end": w.oos_end,
                     "variant": best, **{f"is_sharpe_{k}": s for k, s in scores.items()}})
    stitched = stitched.fillna(0.0)
    res = simulate(data.panel, stitched, p, start=windows[0].oos_start)
    win = pd.DataFrame(rows)
    for i, w in win.iterrows():
        seg = res.equity[(res.equity.index >= w["oos_start"]) & (res.equity.index < w["oos_end"])]
        prev = res.equity[res.equity.index < w["oos_start"]]
        base_eq = prev.iloc[-1] if len(prev) else p["capital"]
        win.loc[i, "oos_return"] = seg.iloc[-1] / base_eq - 1 if len(seg) else 0.0
    return res, win


def _print_line(name, s):
    print(f"  {name:<28} return {s['total_return']:>+8.1%}  CAGR {s['cagr']:>+7.1%}  Sharpe {s['sharpe']:>5.2f}  "
          f"maxDD {s['max_drawdown']:>7.1%}  PSR {s['psr']:.3f}  turnover {s['turnover_per_year']:>5.1f}x/y  "
          f"fees {s['fees']:>7.0f} slip {s['slippage']:>6.0f} funding {s['funding']:>+7.0f}")


def _print_bench(t: pd.DataFrame, title):
    print(f"\n  {title}")
    for name, r in t.iterrows():
        extra = "" if pd.isna(r.get("excess_return", np.nan)) else \
            f"  excess {r['excess_return']:>+8.1%}  IR {r['information_ratio']:>5.2f}"
        print(f"    {name:<28} {r['total_return']:>+8.1%}  CAGR {r['cagr']:>+7.1%}  Sharpe {r['sharpe']:>5.2f}  "
              f"maxDD {r['max_drawdown']:>7.1%}  Calmar {r['calmar']:>5.2f}{extra}")


# ── Study ─────────────────────────────────────────────────────────────────────

def run_study(cfg: dict = None, log: ResearchLog = None) -> dict:
    cfg = cfg or load_config()
    log = log or ResearchLog(cfg["research_log"])
    p = params(cfg)
    study = new_study_id("tsmom")
    out, sr = {}, {}

    data = load_data(cfg, "primary")
    first_day = data.members.index[data.members.any(axis=1)][0]
    print(f"\n{'=' * 100}\n  TSMOM PORTFOLIO - research period {data.panel.close.index[0]:%Y-%m-%d} - "
          f"{data.panel.close.index[-1]:%Y-%m-%d} | primary universe: {data.n_candidates} candidates, "
          f"{int(data.members.any().sum())} ever members | first member day {first_day:%Y-%m-%d}\n{'=' * 100}")
    weights = {v: variant_weights(data.panel, v, data.members, p) for v in VARIANTS}

    start = first_day + pd.Timedelta(days=1)               # first day a position can exist
    for v, W in weights.items():
        res = simulate(data.panel, W, p, start=start)
        s = _summary(res, {"variant": v, **VARIANTS[v]})
        bench = benchmarks(data, res.equity, p["capital"], p)
        _print_line(f"{v} {VARIANTS[v]['signal']}/{VARIANTS[v]['direction']}", s)
        log.append("tsmom_config", study_id=study, strategy=STRATEGY, symbol="primary", timeframe="1d",
                   params={**VARIANTS[v]}, period={"start": res.equity.index[0], "end": res.equity.index[-1]},
                   metrics=s, n_trials=cfg["validation"]["n_trials"], config_id=v,
                   benchmarks=bench.reset_index().to_dict(orient="records"))
        out[v] = {"result": res, "summary": s, "benchmarks": bench}
        sr[v] = s

    for wid, mode in (("W1", "rolling"), ("W2", "anchored")):
        res, win = wfo(data, weights, cfg, mode, p)
        s = _summary(res, {"procedure": mode})
        bench = benchmarks(data, res.equity, p["capital"], p)
        _print_line(f"{wid} WFO {mode}", s)
        print("    chosen per window: " + " ".join(f"{int(r.window)}:{r.variant}" for r in win.itertuples()))
        log.append("tsmom_config", study_id=study, strategy=STRATEGY, symbol="primary", timeframe="1d",
                   params={"procedure": mode}, period={"start": res.equity.index[0], "end": res.equity.index[-1]},
                   metrics=s, n_trials=cfg["validation"]["n_trials"], config_id=wid,
                   windows=win.to_dict(orient="records"), benchmarks=bench.reset_index().to_dict(orient="records"))
        out[wid] = {"result": res, "summary": s, "windows": win, "benchmarks": bench}
        sr[wid] = s

    wide = load_data(cfg, "binance_wide")
    w_wide = {v: variant_weights(wide.panel, v, wide.members, p) for v in VARIANTS}
    res, win = wfo(wide, w_wide, cfg, "rolling", p)
    s = _summary(res, {"procedure": "rolling", "universe": "binance_wide"})
    bench = benchmarks(wide, res.equity, p["capital"], p)
    _print_line("R1 WFO rolling, Binance-wide", s)
    print("    chosen per window: " + " ".join(f"{int(r.window)}:{r.variant}" for r in win.itertuples()))
    log.append("tsmom_config", study_id=study, strategy=STRATEGY, symbol="binance_wide", timeframe="1d",
               params={"procedure": "rolling"}, period={"start": res.equity.index[0], "end": res.equity.index[-1]},
               metrics=s, n_trials=cfg["validation"]["n_trials"], config_id="R1",
               windows=win.to_dict(orient="records"), benchmarks=bench.reset_index().to_dict(orient="records"))
    out["R1"] = {"result": res, "summary": s, "windows": win, "benchmarks": bench}
    sr["R1"] = s

    # DSR: N = 7 preregistered configurations, V = variance of their per-day Sharpe ratios
    n = cfg["validation"]["n_trials"]
    v_sr = float(np.var([x["sharpe_per_period"] for x in sr.values()], ddof=1))
    print(f"\n  Deflated Sharpe Ratio (N = {n}, V[SR] = {v_sr:.2e} per day):")
    for k, x in sr.items():
        x["dsr"] = deflated_sharpe_ratio(x["sharpe_per_period"], x["n_periods"], x["skew"], x["kurtosis"], n, v_sr)
        print(f"    {k:<4} Sharpe {x['sharpe']:>5.2f}  PSR {x['psr']:.3f}  DSR {x['dsr']:.3f}")
    log.append("tsmom_dsr", study_id=study, strategy=STRATEGY, symbol="primary", timeframe="1d",
               metrics={k: {"sharpe": x["sharpe"], "psr": x["psr"], "dsr": x["dsr"]} for k, x in sr.items()},
               n_trials=n, trial_sr_variance=v_sr)
    for k in ("W1", "W2", "R1"):
        _print_bench(out[k]["benchmarks"], f"Benchmarks - {k} (identical days)")
    out["dsr"] = {k: x["dsr"] for k, x in sr.items()}
    out["study_id"] = study
    return out


# ── Hold-out ──────────────────────────────────────────────────────────────────

def evaluate_holdout(cfg: dict = None, log: ResearchLog = None) -> dict:
    cfg = cfg or load_config()
    log = log or ResearchLog(cfg["research_log"])
    p = params(cfg)
    hs = pd.Timestamp(cfg["data"]["holdout_start"], tz="UTC")
    prior = [e for e in log.entries("holdout", strategy=STRATEGY) if e.get("holdout_start") == hs.isoformat()]
    if prior:
        msg = f"HOLD-OUT REUSE: {STRATEGY} hold-out from {hs:%Y-%m-%d} already evaluated {len(prior)} time(s)"
        logger.warning(msg)
        warnings.warn(msg, HoldoutReuseWarning, stacklevel=2)

    data = load_data(cfg, "primary", final_holdout=True)
    weights = {v: variant_weights(data.panel, v, data.members, p) for v in VARIANTS}
    is_start = hs - pd.DateOffset(months=cfg["validation"]["is_months"])
    best, scores = select_variant(data, weights, is_start, hs, p)    # fit: only days < hs traded/scored
    res = simulate(data.panel, weights[best], p, start=hs)
    s = _summary(res, {"variant": best})
    bench = benchmarks(data, res.equity, p["capital"], p)
    print(f"\n{'=' * 100}\n  TSMOM FINAL HOLD-OUT {hs:%Y-%m-%d} - {res.equity.index[-1]:%Y-%m-%d}: "
          f"selected {best} on [{is_start:%Y-%m-%d}, {hs:%Y-%m-%d}) IS Sharpes {scores}"
          f"{'  (REPEATED EVALUATION)' if prior else ''}\n{'=' * 100}")
    _print_line(f"Hold-out {best}", s)
    _print_bench(bench, "Benchmarks - hold-out (identical days)")
    log.append("holdout", strategy=STRATEGY, symbol="primary", timeframe="1d", params={"variant": best, **VARIANTS[best]},
               period={"start": res.equity.index[0], "end": res.equity.index[-1]}, metrics=s, n_trials=4,
               holdout_start=hs.isoformat(), prior_evaluations=len(prior), is_sharpes=scores,
               benchmarks=bench.reset_index().to_dict(orient="records"))
    return {"variant": best, "result": res, "summary": s, "benchmarks": bench, "prior": len(prior)}
