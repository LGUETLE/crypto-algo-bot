"""
Benchmarks
==========
Every strategy result is compared with passive BTC exposure over the IDENTICAL bars:

  BTC buy & hold     buy at the open of the first bar, mark at every close, sell at the last
                     close; one commission + slippage per side, no funding (spot-like holding)
  BTC vol-matched    BTC bar returns scaled by k = std(strategy) / std(BTC) over the same bars,
                     rebalanced every bar, no costs. Answers "would simply holding BTC at the
                     strategy's risk level have done better?". k is ex post (uses the full period),
                     so this is a yardstick, not a tradable strategy.
  <asset> buy & hold optional, same rules, for the traded asset itself (context)

Reported per line: total return, CAGR, Sharpe, max drawdown, Calmar, volatility; for every
benchmark also the strategy's excess return (total and CAGR difference) and the information ratio
IR = mean(r_strategy - r_benchmark) / std(r_strategy - r_benchmark) * sqrt(bars per year).
"""
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.metrics import calc_all, period_returns


def _align(ohlc: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Benchmark candles on the strategy's bars (forward-filled over gaps in the benchmark data)."""
    if len(index) and (index[0] < ohlc.index[0] or index[-1] > ohlc.index[-1]):
        raise ValueError(f"Benchmark data [{ohlc.index[0]}, {ohlc.index[-1]}] does not cover the "
                         f"strategy period [{index[0]}, {index[-1]}]")
    out = ohlc[["open", "close"]].reindex(ohlc.index.union(index)).ffill().reindex(index)
    if out.isna().any().any():
        raise ValueError("Benchmark data does not cover the strategy period")
    return out


def buy_and_hold(ohlc: pd.DataFrame, index: pd.DatetimeIndex, capital: float,
                 cost_per_side: float = 0.0) -> pd.Series:
    px    = _align(ohlc, index)
    units = capital * (1 - cost_per_side) / px["open"].iloc[0]
    eq    = units * px["close"]
    eq.iloc[-1] *= (1 - cost_per_side)
    return eq.rename("equity")


def vol_matched(ohlc: pd.DataFrame, strategy_equity: pd.Series, capital: float) -> pd.Series:
    px   = _align(ohlc, strategy_equity.index)
    r_b  = px["close"] / px["close"].shift(1) - 1
    r_b.iloc[0] = px["close"].iloc[0] / px["open"].iloc[0] - 1
    r_s  = period_returns(strategy_equity, capital)
    sd_b = r_b.std()
    k    = r_s.std() / sd_b if sd_b > 0 else 0.0
    eq   = capital * (1 + k * r_b).cumprod()
    eq.attrs["scale"] = float(k)
    return eq.rename("equity")


def information_ratio(strategy_equity: pd.Series, bench_equity: pd.Series, capital: float,
                      ppy: float) -> float:
    diff = period_returns(strategy_equity, capital) - period_returns(bench_equity, capital)
    sd   = diff.std()
    return float(diff.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def compare(strategy_equity: pd.Series, capital: float, btc_ohlc: pd.DataFrame, ppy: float,
            cost_per_side: float = 0.0, asset_ohlc: Optional[pd.DataFrame] = None,
            asset_name: str = "Asset", risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Strategy vs. benchmarks over strategy_equity.index. One row per line, metrics as columns."""
    idx = strategy_equity.index
    lines = {"Strategy": strategy_equity,
             "BTC buy & hold": buy_and_hold(btc_ohlc, idx, capital, cost_per_side)}
    vm = vol_matched(btc_ohlc, strategy_equity, capital)
    lines[f"BTC vol-matched (x{vm.attrs['scale']:.2f})"] = vm
    if asset_ohlc is not None:
        lines[f"{asset_name} buy & hold"] = buy_and_hold(asset_ohlc, idx, capital, cost_per_side)

    s = calc_all(strategy_equity, pd.DataFrame(), capital, ppy, risk_free_rate)
    rows = []
    for name, eq in lines.items():
        m = calc_all(eq, pd.DataFrame(), capital, ppy, risk_free_rate)
        row = {"line": name, "total_return": m["total_return"], "cagr": m["cagr"], "sharpe": m["sharpe"],
               "max_drawdown": m["max_drawdown"], "calmar": m["calmar"], "volatility": m["volatility"]}
        if name != "Strategy":
            row.update({"excess_return": s["total_return"] - m["total_return"],
                        "excess_cagr": s["cagr"] - m["cagr"],
                        "information_ratio": information_ratio(strategy_equity, eq, capital, ppy)})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("line")
    out.attrs["period"] = (idx[0], idx[-1])
    return out


def print_comparison(table: pd.DataFrame, title: str = "BENCHMARK COMPARISON"):
    start, end = table.attrs.get("period", (None, None))
    print("\n" + "=" * 112)
    print(f"  {title}" + (f"  ({start:%Y-%m-%d %H:%M} - {end:%Y-%m-%d %H:%M} UTC, identical bars)" if start else ""))
    print("=" * 112)
    print(f"  {'':<26} {'Return':>9} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Vol':>7} "
          f"{'Excess':>9} {'Exc.CAGR':>9} {'IR':>6}")
    for name, r in table.iterrows():
        exc = "" if pd.isna(r.get("excess_return", np.nan)) else \
            f"{r['excess_return']:>9.1%} {r['excess_cagr']:>9.1%} {r['information_ratio']:>6.2f}"
        print(f"  {name:<26} {r['total_return']:>9.1%} {r['cagr']:>8.1%} {r['sharpe']:>7.2f} "
              f"{r['max_drawdown']:>8.1%} {r['calmar']:>7.2f} {r['volatility']:>7.1%} {exc}")
    print("  Excess / IR = strategy minus benchmark. Benchmarks: no funding; vol-matched: no costs, ex-post scale.")
    print("=" * 112 + "\n")
