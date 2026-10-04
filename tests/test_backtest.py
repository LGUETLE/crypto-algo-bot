"""Tests for the backtest engine and metrics."""
import pytest
import numpy as np
import pandas as pd
from backtesting.engine import BacktestEngine
from backtesting.metrics import calc_all
from strategies.ema_crossover import EmaCrossoverStrategy

CONFIG = {
    "ema_fast": 12, "ema_slow": 26, "rsi_period": 14,
    "rsi_lower": 40, "rsi_upper": 60,
    "risk": {
        "max_risk_per_trade": 0.02, "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04, "max_open_positions": 3,
        "max_daily_loss_pct": 0.05, "leverage": 1,
    },
    "backtest": {
        "commission": 0.0006, "slippage": 0.0002, "walk_forward_pct": 0.3,
    },
}


def make_df(n=500):
    np.random.seed(0)
    close = np.cumsum(np.random.randn(n) * 100 + 1) + 40000
    close = np.maximum(close, 1000)
    idx   = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   close * 0.999, "high": close * 1.003,
        "low":    close * 0.997, "close": close,
        "volume": np.random.randint(500, 5000, n).astype(float),
    }, index=idx)


def test_backtest_runs():
    df     = make_df()
    engine = BacktestEngine(CONFIG)
    strat  = EmaCrossoverStrategy(CONFIG)
    result = engine.run(df, strat, verbose=False)
    assert result.equity is not None
    assert len(result.equity) > 0
    assert "sharpe" in result.metrics


def test_equity_starts_at_capital():
    df     = make_df()
    engine = BacktestEngine(CONFIG)
    strat  = EmaCrossoverStrategy(CONFIG)
    result = engine.run(df, strat, capital=10_000, verbose=False)
    assert result.equity.iloc[0] == pytest.approx(10_000, rel=0.05)


def test_metrics_reasonable():
    equity = pd.Series([10000, 10100, 10050, 10200, 10150],
                       index=pd.date_range("2024-01-01", periods=5, freq="1h"))
    trades = pd.DataFrame({"pnl_usdt": [100, -50, 150]})
    m = calc_all(equity, trades, 10_000)
    assert -1 < m["total_return"] < 10
    assert m["win_rate"] == pytest.approx(2 / 3, rel=0.01)
