"""
Look-ahead tests.

1. Fill timing: a signal computed on the close of bar t is filled at the OPEN of bar t+1
   (entries and signal exits alike).
2. Truncation: signals computed on df[:n] equal the signals computed on the full df for every
   bar < n. If a feature peeked into the future, the full-history value would differ.
"""
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from features.macro import add_macro_trend
from strategies.base_strategy import BaseStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.ema_crossover import EmaCrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

SLIP = 0.001
CONFIG = {
    "risk": {
        "max_risk_per_trade": 0.02, "stop_loss_pct": 0.5, "take_profit_pct": None,
        "atr_stop_mult": None, "trailing_activation": None, "trailing_callback": None,
        "max_open_positions": 1, "max_daily_loss_pct": 0.05, "leverage": 3,
    },
    "backtest": {"commission": 0.0, "slippage": SLIP, "funding_rate_8h": 0.0},
    "timeframe": "1h",
}


def ohlcv(n=800, seed=1, freq="1h"):
    rng   = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    high  = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low   = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    idx   = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(100, 1000, n)}, index=idx)


class ScriptedStrategy(BaseStrategy):
    """Emits fixed signals at given bar positions (no indicators, no warm-up)."""

    def __init__(self, config, entries=None, exits=None):
        super().__init__(config)
        self.entries = entries or {}          # {bar position: +1 / -1}
        self.exits   = exits or set()         # bar positions with long_exit/short_exit = True

    def add_indicators(self, df):
        df["atr_14"] = np.nan                 # forces the fixed-percent stop
        return df

    def generate_signals(self, df):
        df["signal"] = 0
        for i, s in self.entries.items():
            df.iloc[i, df.columns.get_loc("signal")] = s
        flag = np.zeros(len(df), dtype=bool)
        flag[list(self.exits)] = True
        df["long_exit"] = flag
        df["short_exit"] = flag
        return df


# ── 1. Fill timing ────────────────────────────────────────────────────────────

def test_entry_fills_at_next_bar_open():
    df = ohlcv(50)
    res = BacktestEngine(CONFIG).run(df, ScriptedStrategy(CONFIG, entries={10: 1}), verbose=False)
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[11]
    assert t["entry_price"] == pytest.approx(df["open"].iloc[11] * (1 + SLIP))


def test_short_entry_fills_at_next_bar_open():
    df = ohlcv(50)
    res = BacktestEngine(CONFIG).run(df, ScriptedStrategy(CONFIG, entries={10: -1}), verbose=False)
    t = res.trades.iloc[0]
    assert t["side"] == "short"
    assert t["entry_time"] == df.index[11]
    assert t["entry_price"] == pytest.approx(df["open"].iloc[11] * (1 - SLIP))


def test_signal_exit_fills_at_next_bar_open():
    df = ohlcv(50)
    strat = ScriptedStrategy(CONFIG, entries={10: 1}, exits={20})
    t = BacktestEngine(CONFIG).run(df, strat, verbose=False).trades.iloc[0]
    assert t["exit_reason"] == "Signal"
    assert t["exit_time"] == df.index[21]
    assert t["exit_price"] == pytest.approx(df["open"].iloc[21] * (1 - SLIP))


def test_reversal_closes_and_reopens_at_next_open():
    df = ohlcv(50)
    strat = ScriptedStrategy(CONFIG, entries={10: 1, 20: -1})
    trades = BacktestEngine(CONFIG).run(df, strat, verbose=False).trades
    assert list(trades["side"][:2]) == ["long", "short"]
    assert trades["exit_time"].iloc[0] == df.index[21]
    assert trades["entry_time"].iloc[1] == df.index[21]


def test_signal_on_last_bar_is_not_traded():
    """There is no next open to fill at."""
    df = ohlcv(50)
    res = BacktestEngine(CONFIG).run(df, ScriptedStrategy(CONFIG, entries={49: 1}), verbose=False)
    assert res.trades.empty


def test_stop_is_checked_on_the_entry_bar():
    """The position is live from the open of the entry bar, so that bar's low can stop it out."""
    df = ohlcv(50)
    df.iloc[11, df.columns.get_loc("low")] = df["open"].iloc[11] * 0.4     # crash through a 50 % stop
    res = BacktestEngine(CONFIG).run(df, ScriptedStrategy(CONFIG, entries={10: 1}), verbose=False)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "SL" and t["exit_time"] == df.index[11]


def test_equity_does_not_react_to_signal_bar_close():
    """Equity on the signal bar must still be flat: nothing is filled before the next open."""
    df = ohlcv(50)
    res = BacktestEngine(CONFIG).run(df, ScriptedStrategy(CONFIG, entries={10: 1}), verbose=False)
    assert res.equity.iloc[10] == pytest.approx(res.equity.iloc[0])


# ── 2. Truncation ─────────────────────────────────────────────────────────────

def _signals(strategy, df):
    out = strategy.generate_signals(strategy.add_indicators(df.copy()))
    cols = ["signal"] + [c for c in ("long_exit", "short_exit") if c in out.columns]
    return out[cols]


def _macro(df, tf="1d"):
    return df.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


STRATEGIES = [
    (DonchianBreakoutStrategy, {"dc_enter": 20, "dc_exit": 10, "adx_threshold": 15,
                                "atr_min_mult": 0.5, "use_macro_filter": True}),
    (EmaCrossoverStrategy,     {"ema_fast": 12, "ema_slow": 26, "rsi_lower": 40,
                                "adx_threshold": 15, "use_macro_filter": True}),
    (RSIMeanReversionStrategy, {"rsi_period": 14, "adx_max": 40}),
]


@pytest.mark.parametrize("cls,params", STRATEGIES, ids=lambda x: getattr(x, "__name__", ""))
@pytest.mark.parametrize("n", [300, 457, 650])
def test_signals_unchanged_by_future_bars(cls, params, n):
    raw   = ohlcv(800, seed=7)
    full  = add_macro_trend(raw, _macro(raw), "1d")
    trunc = add_macro_trend(raw.iloc[:n], _macro(raw.iloc[:n]), "1d")
    strat = cls({**CONFIG, **params})

    s_full  = _signals(strat, full)
    s_trunc = _signals(strat, trunc)
    common  = s_trunc.index
    assert len(common) > 50
    pd.testing.assert_frame_equal(s_trunc, s_full.loc[common], check_dtype=False)
    # the full run must have produced at least some signals for the test to mean anything
    assert (s_full["signal"] != 0).any()


def test_macro_merge_only_uses_closed_macro_candles():
    raw   = ohlcv(24 * 20, seed=3)
    macro = _macro(raw)
    out   = add_macro_trend(raw, macro, "1d")
    # every hourly bar of day D must see the close of day D-1, never day D
    for ts in out.index[30::17]:
        day_open      = ts.normalize()
        expected_prev = macro["close"].get(day_open - pd.Timedelta("1d"), np.nan)
        got           = out.loc[ts, "macro_close"]
        if np.isnan(expected_prev):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected_prev)


def test_macro_merge_ignores_a_still_forming_macro_candle():
    """A partial macro candle (as returned before the day closes) must not change any bar."""
    raw     = ohlcv(24 * 10 + 5, seed=4)              # last day has only 5 hours
    macro   = _macro(raw)                             # includes a partial last daily candle
    out     = add_macro_trend(raw, macro, "1d")
    out_cut = add_macro_trend(raw, macro.iloc[:-1], "1d")
    pd.testing.assert_series_equal(out["macro_close"], out_cut["macro_close"])


# ── 3. Engine trading window ──────────────────────────────────────────────────

def test_run_with_end_equals_run_on_truncated_data():
    """Bars at/after `end` must not influence anything before it."""
    raw  = ohlcv(900, seed=11)
    full = add_macro_trend(raw, _macro(raw), "1d")
    cfg  = {**CONFIG, "dc_enter": 20, "dc_exit": 10, "adx_threshold": 15, "atr_min_mult": 0.5}
    cfg["risk"] = {**CONFIG["risk"], "atr_stop_mult": 2.0, "stop_loss_pct": 0.02}
    eng  = BacktestEngine(cfg)
    cut  = full.index[600]

    a = eng.run(full, DonchianBreakoutStrategy(cfg), end=cut, verbose=False)
    b = eng.run(full[full.index < cut], DonchianBreakoutStrategy(cfg), verbose=False)
    assert len(a.trades) > 0
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_run_window_trades_only_inside_window():
    df  = ohlcv(200)
    eng = BacktestEngine(CONFIG)
    strat = ScriptedStrategy(CONFIG, entries={20: 1, 120: -1}, exits={60})
    res = eng.run(df, strat, start=df.index[100], end=df.index[180], verbose=False)
    assert res.equity.index[0] == df.index[100] and res.equity.index[-1] == df.index[179]
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[121]
    assert t["exit_time"] == df.index[179] and t["exit_reason"] == "End"
    # position closed at the window end -> final equity is realised cash
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["pnl_usdt"], rel=1e-9)
