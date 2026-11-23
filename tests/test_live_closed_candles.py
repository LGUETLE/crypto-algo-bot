"""
Live path look-ahead guard: run_cycle must decide on CLOSED candles only.

The exchange returns the still-forming candle as the last row. A breakout that exists only in
that forming candle must not trigger an order; the same breakout on a closed candle must.
No network, no .env: load_dotenv is stubbed and the broker is a fake.
"""
import importlib

import dotenv
import numpy as np
import pandas as pd
import pytest

from risk.guard import RiskGuard
from risk.position_sizer import PositionSizer
from strategies.donchian_breakout import DonchianBreakoutStrategy

SYMBOL = "ETH/USDT:USDT"
CONFIG = {
    "timeframe": "1h", "use_macro_filter": False, "strategy": "donchian_breakout",
    "dc_enter": 20, "dc_exit": 10, "adx_threshold": 0, "atr_min_mult": 0.0, "vol_mult": 0.0,
    "risk": {"max_risk_per_trade": 0.02, "stop_loss_pct": 0.02, "take_profit_pct": None,
             "atr_stop_mult": 2.0, "max_open_positions": 2, "max_total_risk_pct": 0.04,
             "min_trade_risk_pct": 0.005, "max_daily_loss_pct": 0.05, "leverage": 3},
}


class FakeOM:
    def get_position(self, symbol):
        return None


class FakeBroker:
    mode = "paper"

    def __init__(self, candles):
        self.candles = candles
        self.om      = FakeOM()
        self.orders  = []

    def sync_positions(self):            return []
    def get_balance(self):               return 10_000.0
    def get_open_positions(self):        return []
    def manage_stops(self, symbol, df):  return None
    def close_position(self, *a, **k):   return None
    def get_ohlcv(self, symbol, timeframe, limit=200):
        return self.candles.copy()
    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "x"}


def _candles(n=120):
    rng   = np.random.default_rng(5)
    close = 100 + rng.normal(0, 0.2, n)                # range-bound, no breakout
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3,
                         "close": close, "volume": 1000.0}, index=idx)


def _with_breakout_last(df):
    df = df.copy()
    df.iloc[-1, df.columns.get_indexer(["open", "high", "low", "close"])] = [100, 111, 100, 110]
    return df


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    tg = importlib.import_module("notify.telegram_bot")
    monkeypatch.setattr(tg, "TOKEN", "")
    return importlib.import_module("main")


def _run(main_module, candles):
    broker = FakeBroker(candles)
    strategies = {"donchian_breakout": DonchianBreakoutStrategy(CONFIG)}
    pairs = [{"symbol": SYMBOL, "strategy": "donchian_breakout"}]
    main_module.run_cycle(broker, strategies, RiskGuard(CONFIG, 10_000), PositionSizer(CONFIG),
                          pairs, CONFIG)
    return broker.orders


def test_breakout_in_forming_candle_is_ignored(main_module):
    assert _run(main_module, _with_breakout_last(_candles())) == []


def test_breakout_in_last_closed_candle_is_traded(main_module):
    """Positive control: same breakout, but followed by a (neutral) forming candle."""
    df = _with_breakout_last(_candles())
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + pd.Timedelta("1h")
    orders = _run(main_module, pd.concat([df, forming]))
    assert len(orders) == 1 and orders[0]["side"] == "buy"
