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


def calc_all(
    equity:           pd.Series,
    trades:           pd.DataFrame,
    capital:          float,
    periods_per_year: int = 8760,     # hourly data: 24 * 365
) -> dict:
    """
    Computes all metrics.

    Args:
        equity:  equity curve (index = timestamp, values = account currency)
        trades:  trade history DataFrame (needs a pnl_usdt column)
        capital: starting capital
        periods_per_year: 8760 for 1h, 365 for daily, 52 for weekly data
    """
    ret     = _return_metrics(equity, capital, periods_per_year)
    risk    = _risk_metrics(equity, periods_per_year)
    trade   = _trade_metrics(trades) if not trades.empty else {}
    quality = _quality_metrics(equity, ret, risk, periods_per_year)

    return {**ret, **risk, **trade, **quality}


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

def _return_metrics(equity: pd.Series, capital: float, ppy: int) -> dict:
    if len(equity) == 0:
        return {"total_return": 0, "cagr": 0, "best_day": 0, "worst_day": 0, "final_equity": capital}
    total_return = (equity.iloc[-1] - capital) / capital
    n_years      = len(equity) / ppy
    cagr         = (equity.iloc[-1] / capital) ** (1 / max(n_years, 0.01)) - 1
    returns      = equity.pct_change().dropna()

    return {
        "total_return": total_return,
        "cagr":         cagr,
        "best_day":     returns.max(),
        "worst_day":    returns.min(),
        "final_equity": round(equity.iloc[-1], 2),
    }


def _risk_metrics(equity: pd.Series, ppy: int) -> dict:
    returns    = equity.pct_change().dropna()
    roll_max   = equity.cummax()
    drawdown   = (equity - roll_max) / roll_max
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


def _quality_metrics(equity: pd.Series, ret: dict, risk: dict, ppy: int) -> dict:
    returns  = equity.pct_change().dropna()
    rf_rate  = 0.04 / ppy                                  # 4 % risk-free rate
    excess   = returns - rf_rate
    downside = returns[returns < 0].std() * np.sqrt(ppy)

    sharpe  = (excess.mean() / returns.std() * np.sqrt(ppy)) if returns.std() > 0 else 0
    sortino = (excess.mean() * ppy / downside) if downside > 0 else 0
    calmar  = ret["cagr"] / abs(risk["max_drawdown"]) if risk["max_drawdown"] != 0 else 0

    return {
        "sharpe":  round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar":  round(calmar, 3),
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
