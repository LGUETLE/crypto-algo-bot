"""
Funding in the backtest.

Rule: a funding event at time T is paid/received by the position HELD at T. Bars are labelled by
their open time; an order filled "at the open of bar T" executes just after T, so
  - a position closed at the open of bar T still pays the event at T,
  - a position opened at the open of bar T does not.
Payment = -side * size * price(T) * rate, price(T) = close of the bar ending at T.
Positive rate: longs pay, shorts receive.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from tests.test_lookahead import CONFIG, ScriptedStrategy, ohlcv

COMM = 0.001
CFG = {**CONFIG, "backtest": {"commission": COMM, "slippage": 0.0, "funding_rate_8h": 0.0001}}


def _events(df, rate, hours=(0, 8, 16)):
    idx = df.index[df.index.hour.isin(hours)]
    return pd.Series(rate, index=idx)


def _run(df, entries, exits=(), funding=None, cfg=CFG):
    strat = ScriptedStrategy(cfg, entries=entries, exits=set(exits))
    return BacktestEngine(cfg).run(df, strat, verbose=False, funding=funding)


@pytest.mark.parametrize("side,sign", [(1, -1), (-1, 1)])
def test_funding_paid_at_event_times_with_sign(side, sign):
    df = ohlcv(100)                                    # starts 2024-01-01 00:00
    # signal on bar 3 -> entry at open of bar 4 (04:00); exit signal bar 40 -> exit at open of 41
    res = _run(df, {3: side}, exits=[40], funding=_events(df, 0.001))
    t = res.trades.iloc[0]
    held = [ts for ts in df.index[df.index.hour.isin((0, 8, 16))]
            if df.index[4] < ts <= df.index[41]]       # 08:00, 16:00, 00:00, 08:00, 16:00 (41 = 17:00 day 2)
    expected = sum(sign * t["size"] * df["close"].shift(1).loc[ts] * 0.001 for ts in held)
    assert len(held) == 5
    assert t["funding"] == pytest.approx(expected)


def test_negative_rate_pays_longs():
    df = ohlcv(100)
    res = _run(df, {3: 1}, exits=[40], funding=_events(df, -0.001))
    assert res.trades.iloc[0]["funding"] > 0


def test_no_funding_between_events():
    """Held 04:00 -> 07:00: no funding timestamp inside, so no cost (old engine charged per bar)."""
    df = ohlcv(100)
    res = _run(df, {3: 1}, exits=[6], funding=_events(df, 0.001))
    assert res.trades.iloc[0]["funding"] == 0.0


def test_close_at_funding_open_pays_open_at_funding_open_does_not():
    df = ohlcv(100)
    # entry at open of bar 8 (08:00, a funding time) -> must NOT pay 08:00
    # exit at open of bar 16 (16:00, a funding time) -> MUST pay 16:00
    res = _run(df, {7: 1}, exits=[15], funding=_events(df, 0.001))
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[8] and t["exit_time"] == df.index[16]
    expected = -t["size"] * df["close"].iloc[15] * 0.001
    assert t["funding"] == pytest.approx(expected)


def test_constant_fallback_warns_and_uses_8h_events(caplog):
    df = ohlcv(100)
    with caplog.at_level(logging.WARNING):
        res = _run(df, {3: 1}, exits=[40], funding=None)
    assert any("funding" in r.message.lower() and "constant" in r.message.lower() for r in caplog.records)
    t = res.trades.iloc[0]
    held = [ts for ts in df.index[df.index.hour.isin((0, 8, 16))] if df.index[4] < ts <= df.index[41]]
    expected = sum(-t["size"] * df["close"].shift(1).loc[ts] * 0.0001 for ts in held)
    assert t["funding"] == pytest.approx(expected)


def test_trade_pnl_is_net_of_both_fees_and_funding():
    """Sum of trade PnL must reconcile exactly with the equity change."""
    df = ohlcv(300)
    res = _run(df, {3: 1, 50: -1, 51: -1, 120: 1, 121: 1}, exits=[200], funding=_events(df, 0.0005))
    assert len(res.trades) == 3
    assert res.trades["pnl_usdt"].sum() == pytest.approx(res.equity.iloc[-1] - 10_000, abs=1e-6)
    t = res.trades.iloc[0]
    gross = (t["exit_price"] - t["entry_price"]) * t["size"]
    fees  = (t["entry_price"] + t["exit_price"]) * t["size"] * COMM
    assert t["pnl_usdt"] == pytest.approx(gross - fees + t["funding"])
    assert t["fees"] == pytest.approx(fees)


def test_equity_reflects_funding_when_flat_price():
    """Constant price: the only equity change while holding is fees + funding."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "volume": 1.0}, index=idx)
    res = _run(df, {1: 1}, funding=_events(df, 0.001))
    t = res.trades.iloc[0]
    n_events = int(((idx > idx[2]) & idx.hour.isin((0, 8, 16))).sum())
    assert t["funding"] == pytest.approx(-n_events * t["size"] * 100 * 0.001)
    assert np.isclose(res.equity.iloc[-1], 10_000 + t["pnl_usdt"])
