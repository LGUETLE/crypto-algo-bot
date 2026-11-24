import numpy as np
import pandas as pd
import pytest

from backtesting.benchmark import buy_and_hold, compare, information_ratio, vol_matched
from backtesting.metrics import period_returns


def _ohlc(n=500, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    close = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[40_000, close[:-1]]
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "close": close}, index=idx)


def _strategy(index, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(10_000 * np.cumprod(1 + rng.normal(0.0002, 0.002, len(index))), index=index)


def test_buy_and_hold_without_costs():
    btc = _ohlc()
    idx = btc.index[100:300]
    eq = buy_and_hold(btc, idx, 10_000)
    assert eq.index.equals(idx)
    assert eq.iloc[-1] == pytest.approx(10_000 * btc["close"].iloc[299] / btc["open"].iloc[100])


def test_buy_and_hold_costs_both_sides():
    btc = _ohlc()
    a = buy_and_hold(btc, btc.index, 10_000, 0.0)
    b = buy_and_hold(btc, btc.index, 10_000, 0.001)
    assert b.iloc[-1] == pytest.approx(a.iloc[-1] * 0.999 ** 2)


def test_vol_matched_has_strategy_volatility():
    btc = _ohlc()
    s = _strategy(btc.index)
    vm = vol_matched(btc, s, 10_000)
    assert period_returns(vm, 10_000).std() == pytest.approx(period_returns(s, 10_000).std(), rel=1e-9)
    assert vm.attrs["scale"] < 1           # strategy is much less volatile than BTC here


def test_information_ratio_against_itself_is_zero():
    btc = _ohlc()
    s = _strategy(btc.index)
    assert information_ratio(s, s, 10_000, 8760) == 0.0


def test_benchmark_aligned_over_gaps():
    btc = _ohlc().drop(_ohlc().index[50:55])          # gap in BTC data
    idx = _ohlc().index
    eq = buy_and_hold(btc, idx, 10_000)
    assert eq.index.equals(idx) and not eq.isna().any()


def test_benchmark_must_cover_period():
    btc = _ohlc(start="2024-02-01")
    with pytest.raises(ValueError):
        buy_and_hold(btc, _ohlc().index, 10_000)


def test_compare_table():
    btc, eth = _ohlc(seed=0), _ohlc(seed=5)
    s = _strategy(btc.index)
    t = compare(s, 10_000, btc, 8760, cost_per_side=0.0008, asset_ohlc=eth, asset_name="ETH")
    assert list(t.index[:2]) == ["Strategy", "BTC buy & hold"]
    assert t.index[2].startswith("BTC vol-matched") and t.index[3] == "ETH buy & hold"
    bh = t.loc["BTC buy & hold"]
    assert bh["excess_return"] == pytest.approx(t.loc["Strategy", "total_return"] - bh["total_return"])
    assert {"sharpe", "max_drawdown", "calmar", "information_ratio"} <= set(t.columns)


def test_benchmark_must_cover_end_of_period():
    """Forward-filling past the end of the benchmark data would fake flat BTC returns."""
    btc = _ohlc().iloc[:300]
    with pytest.raises(ValueError):
        buy_and_hold(btc, _ohlc().index, 10_000)
