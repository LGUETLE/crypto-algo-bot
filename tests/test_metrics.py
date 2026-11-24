"""Annualisation and metric definitions."""
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.metrics import calc_all, periods_per_year
from features.engineer import FeatureEngineer
from tests.test_lookahead import CONFIG, ScriptedStrategy, ohlcv


def _equity(freq, n=400, seed=0):
    rng = np.random.default_rng(seed)
    r   = rng.normal(0.001, 0.01, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.Series(10_000 * np.cumprod(1 + r), index=idx)


def _ref_sharpe(equity, capital, ppy):
    r = pd.concat([pd.Series([capital]), pd.Series(equity.to_numpy())]).pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(ppy)


@pytest.mark.parametrize("tf,ppy", [("1h", 8760), ("4h", 2190), ("1d", 365), ("15min", 35040)])
def test_periods_per_year(tf, ppy):
    assert periods_per_year(tf) == pytest.approx(ppy)


@pytest.mark.parametrize("freq,ppy", [("1h", 8760), ("4h", 2190), ("1D", 365)])
def test_sharpe_annualisation_is_inferred_from_bar_spacing(freq, ppy):
    eq = _equity(freq)
    m  = calc_all(eq, pd.DataFrame(), 10_000)
    assert m["sharpe"] == pytest.approx(_ref_sharpe(eq, 10_000, ppy), rel=1e-3)


def test_engine_uses_configured_timeframe_for_sharpe():
    df  = ohlcv(600, freq="4h")
    cfg = {**CONFIG, "timeframe": "4h"}
    res = BacktestEngine(cfg).run(df, ScriptedStrategy(cfg, entries={10: 1}), verbose=False)
    assert res.metrics["periods_per_year"] == pytest.approx(2190)
    assert res.metrics["sharpe"] == pytest.approx(_ref_sharpe(res.equity, 10_000, 2190), rel=1e-3)


def test_cagr_uses_calendar_time():
    idx = pd.date_range("2022-01-01", periods=2 * 365, freq="1D", tz="UTC")
    eq  = pd.Series(np.linspace(10_000, 20_000, len(idx)), index=idx)
    m   = calc_all(eq, pd.DataFrame(), 10_000)
    assert m["cagr"] == pytest.approx(2 ** 0.5 - 1, rel=1e-2)       # doubling in 2 years


def test_first_bar_return_counts_and_drawdown_starts_at_capital():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    eq  = pd.Series([9_000.0, 9_500.0, 9_900.0], index=idx)          # lost 10 % on bar 1
    m   = calc_all(eq, pd.DataFrame(), 10_000)
    assert m["max_drawdown"] == pytest.approx(-0.10)


def test_sortino_uses_downside_deviation():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    r   = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    eq  = pd.Series(10_000 * np.cumprod(1 + r), index=idx)
    m   = calc_all(eq, pd.DataFrame(), 10_000)
    dd  = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert m["sortino"] == pytest.approx(r.mean() / dd * np.sqrt(365), rel=1e-3)


def test_hist_vol_annualised_for_configured_timeframe():
    df = ohlcv(400, freq="4h")
    out = FeatureEngineer({"timeframe": "4h"}).transform(df)
    ref = df["close"].pct_change().rolling(20).std() * np.sqrt(2190)
    assert out["hist_vol"].iloc[-1] == pytest.approx(ref.iloc[-1])


@pytest.mark.parametrize("tf,hours", [("15m", 0.25), ("1h", 1), ("4h", 4), ("1d", 24), ("1w", 168),
                                      ("15min", 0.25)])
def test_timeframe_parsing(tf, hours):
    from features.timeframes import to_timedelta
    assert to_timedelta(tf) == pd.Timedelta(hours=hours)
