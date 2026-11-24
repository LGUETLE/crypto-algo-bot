"""
Performance Metrics
===================
Computes the key figures for a backtest evaluation.

Metrics:
  Returns  - total return, CAGR
  Risk     - max drawdown, volatility, VaR
  Quality  - Sharpe, Sortino, Calmar, profit factor
  Trades   - win rate, avg win/loss, expectancy
"""
import numpy as np
import pandas as pd
from scipy import stats

from features.timeframes import to_timedelta

YEAR_SECONDS = 365 * 24 * 3600          # crypto trades 24/7: 365 days, no trading-day calendar


def periods_per_year(timeframe: str) -> float:
    """Bars per year for a timeframe string ("1h" -> 8760, "4h" -> 2190, "1d" -> 365)."""
    return YEAR_SECONDS / to_timedelta(timeframe).total_seconds()


def infer_periods_per_year(index: pd.Index) -> float:
    """Bars per year from the median bar spacing of a DatetimeIndex."""
    if len(index) < 2:
        return periods_per_year("1h")
    step = pd.Series(index).diff().median().total_seconds()
    return YEAR_SECONDS / step


def period_returns(equity: pd.Series, capital: float = None) -> pd.Series:
    """Simple returns per bar. With `capital`, the first bar's return vs. the start capital counts too."""
    if capital is not None and len(equity):
        prev = equity.shift(1)
        prev.iloc[0] = capital
        return (equity / prev - 1).astype(float)
    return equity.pct_change().dropna()


def calc_all(
    equity:           pd.Series,
    trades:           pd.DataFrame,
    capital:          float,
    periods_per_year: float = None,   # None -> inferred from the bar spacing of `equity`
    risk_free_rate:   float = 0.0,    # annual; 0 = crypto convention, same basis for benchmarks
) -> dict:
    """
    Computes all metrics.

    Args:
        equity:  equity curve (index = timestamp, values = account currency)
        trades:  trade history DataFrame (needs a pnl_usdt column)
        capital: starting capital (the first bar's return is measured against it)
        periods_per_year: bars per year used to annualise (8760 for 1h, 365 for 1d)
    """
    ppy     = periods_per_year or infer_periods_per_year(equity.index)
    ret     = _return_metrics(equity, capital, ppy)
    risk    = _risk_metrics(equity, capital, ppy)
    trade   = _trade_metrics(trades) if trades is not None and not trades.empty else {}
    quality = _quality_metrics(period_returns(equity, capital), ret, risk, ppy, risk_free_rate)

    return {**ret, **risk, **trade, **quality, "periods_per_year": ppy}


def print_report(metrics: dict):
    """Prints a formatted report."""
    print("\n" + "=" * 50)
    print("  BACKTEST REPORT")
    print("=" * 50)

    groups = {
        "Returns": ["total_return", "cagr", "best_day", "worst_day"],
        "Risk":    ["max_drawdown", "avg_drawdown", "volatility", "var_95"],
        "Quality": ["sharpe", "sortino", "calmar", "profit_factor"],
        "Trades":  ["total_trades", "win_rate", "avg_win", "avg_loss",
                    "expectancy", "avg_hold_periods"],
    }

    labels = {
        "total_return":     "Total return",
        "cagr":             "CAGR",
        "best_day":         "Best period",
        "worst_day":        "Worst period",
        "max_drawdown":     "Max drawdown",
        "avg_drawdown":     "Avg drawdown",
        "volatility":       "Volatility (ann.)",
        "var_95":           "Value at Risk 95%",
        "sharpe":           "Sharpe ratio",
        "sortino":          "Sortino ratio",
        "calmar":           "Calmar ratio",
        "profit_factor":    "Profit factor",
        "total_trades":     "Number of trades",
        "win_rate":         "Win rate",
        "avg_win":          "Avg win",
        "avg_loss":         "Avg loss",
        "expectancy":       "Expectancy",
        "avg_hold_periods": "Avg hold time (hours)",
    }

    for group, keys in groups.items():
        print(f"\n  {group}")
        print(f"  {'-' * 30}")
        for k in keys:
            v = metrics.get(k, "-")
            label = labels.get(k, k)
            if isinstance(v, float):
                if k in ("total_return", "cagr", "max_drawdown", "win_rate",
                         "volatility", "var_95", "avg_drawdown"):
                    print(f"  {label:<25} {v:>8.1%}")
                elif k in ("avg_win", "avg_loss", "expectancy"):
                    print(f"  {label:<25} {v:>8.2f}")
                else:
                    print(f"  {label:<25} {v:>8.2f}")
            else:
                print(f"  {label:<25} {str(v):>8}")
    print("=" * 50 + "\n")


# ── Internal calculations ─────────────────────────────────────────────────────

def _years(index: pd.Index, ppy: float) -> float:
    """Calendar length of the curve incl. the last bar."""
    if len(index) < 2:
        return len(index) / ppy
    span = (index[-1] - index[0]).total_seconds() + YEAR_SECONDS / ppy
    return span / YEAR_SECONDS


def _return_metrics(equity: pd.Series, capital: float, ppy: float) -> dict:
    if len(equity) == 0:
        return {"total_return": 0, "cagr": 0, "best_day": 0, "worst_day": 0, "final_equity": capital}
    total_return = (equity.iloc[-1] - capital) / capital
    n_years      = _years(equity.index, ppy)
    cagr         = (equity.iloc[-1] / capital) ** (1 / max(n_years, 1e-9)) - 1 if equity.iloc[-1] > 0 else -1.0
    returns      = period_returns(equity, capital)

    return {
        "total_return": total_return,
        "cagr":         cagr,
        "best_day":     returns.max(),     # best / worst single BAR (label kept for compatibility)
        "worst_day":    returns.min(),
        "final_equity": round(equity.iloc[-1], 2),
        "years":        n_years,
    }


def _risk_metrics(equity: pd.Series, capital: float, ppy: float) -> dict:
    if len(equity) == 0:
        return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "volatility": 0.0, "var_95": 0.0}
    returns    = period_returns(equity, capital)
    curve      = pd.concat([pd.Series([capital]), pd.Series(equity.to_numpy())], ignore_index=True)
    roll_max   = curve.cummax()
    drawdown   = (curve - roll_max) / roll_max
    max_dd     = drawdown.min()
    avg_dd     = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
    volatility = returns.std() * np.sqrt(ppy)
    var_95     = returns.quantile(0.05)

    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "volatility":   volatility,
        "var_95":       var_95,
    }


def _quality_metrics(returns: pd.Series, ret: dict, risk: dict, ppy: float, rf_annual: float) -> dict:
    """
    Sharpe  = mean(r - rf) / std(r) * sqrt(ppy)
    Sortino = mean(r - rf) / downside deviation * sqrt(ppy),
              downside deviation = sqrt(mean(min(r - rf, 0)^2)) over ALL bars
    Also returns the non-annualised Sharpe, skewness and (Pearson, non-excess) kurtosis of the
    bar returns, which the Probabilistic / Deflated Sharpe Ratio need.
    """
    if len(returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "sharpe_per_period": 0.0,
                "skew": 0.0, "kurtosis": 3.0, "n_periods": len(returns)}
    rf      = rf_annual / ppy
    excess  = returns - rf
    std     = returns.std()
    dd      = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))

    sr_bar  = excess.mean() / std if std > 0 else 0.0
    sharpe  = sr_bar * np.sqrt(ppy)
    sortino = excess.mean() / dd * np.sqrt(ppy) if dd > 0 else 0.0
    calmar  = ret["cagr"] / abs(risk["max_drawdown"]) if risk["max_drawdown"] != 0 else 0.0

    return {
        "sharpe":            round(float(sharpe), 3),
        "sortino":           round(float(sortino), 3),
        "calmar":            round(float(calmar), 3),
        "sharpe_per_period": float(sr_bar),
        "skew":              float(stats.skew(returns)) if std > 0 else 0.0,
        "kurtosis":          float(stats.kurtosis(returns, fisher=False)) if std > 0 else 3.0,
        "n_periods":         int(len(returns)),
    }


def _trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty or "pnl_usdt" not in trades.columns:
        return {}

    closed = trades[trades["pnl_usdt"].notna()]
    if closed.empty:
        return {}

    wins   = closed[closed["pnl_usdt"] > 0]["pnl_usdt"]
    losses = closed[closed["pnl_usdt"] <= 0]["pnl_usdt"]

    win_rate   = len(wins) / len(closed) if len(closed) > 0 else 0
    avg_win    = wins.mean()   if not wins.empty   else 0
    avg_loss   = losses.mean() if not losses.empty else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    gross_profit  = wins.sum()
    gross_loss    = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_hold = 0
    if "entry_time" in closed.columns and "exit_time" in closed.columns:
        hold_times = pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])
        avg_hold = hold_times.mean().total_seconds() / 3600     # hours

    return {
        "total_trades":     len(closed),
        "win_rate":         win_rate,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "expectancy":       expectancy,
        "profit_factor":    round(profit_factor, 3),
        "avg_hold_periods": round(avg_hold, 1),
    }
