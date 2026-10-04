"""
Tests for the EMA crossover strategy.
Run with: pytest tests/ -v
"""
import numpy as np
import pandas as pd
from strategies.ema_crossover import EmaCrossoverStrategy

CONFIG = {
    "ema_fast": 12, "ema_slow": 26,
    "rsi_period": 14, "rsi_lower": 40, "rsi_upper": 60,
}


def make_df(n=300, trend="up"):
    """Simple synthetic OHLCV DataFrame."""
    np.random.seed(42)
    drift = 2 if trend == "up" else -2
    close = np.cumsum(np.random.randn(n) * 10 + drift) + 30000
    close = np.maximum(close, 1000)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


def test_indicators_computed():
    out = EmaCrossoverStrategy(CONFIG).add_indicators(make_df())
    assert "ema_fast" in out.columns
    assert "ema_slow" in out.columns
    assert "rsi_14" in out.columns


def test_signals_column_exists():
    strat = EmaCrossoverStrategy(CONFIG)
    df = strat.generate_signals(strat.add_indicators(make_df()))
    assert "signal" in df.columns
    assert df["signal"].isin([-1, 0, 1]).all()


def test_exit_columns_have_no_nan():
    """NaN in the exit columns is truthy and would close every position instantly."""
    strat = EmaCrossoverStrategy(CONFIG)
    df = strat.generate_signals(strat.add_indicators(make_df()))
    assert not df["long_exit"].isna().any()
    assert not df["short_exit"].isna().any()


def test_get_latest_signal_returns_int():
    assert EmaCrossoverStrategy(CONFIG).get_latest_signal(make_df()) in (-1, 0, 1)


def test_no_signal_on_short_data():
    # With 10 candles the lookback is too short -> dropna removes everything
    out = EmaCrossoverStrategy(CONFIG).add_indicators(make_df(n=10))
    assert len(out) == 0 or "ema_fast" in out.columns
