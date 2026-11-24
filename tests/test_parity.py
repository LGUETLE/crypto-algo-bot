"""
Backtest / paper / live parity (reference = what the exchange does).
  - trailing stop checked intrabar; with OHLC bars the ADVERSE order is assumed
  - paper and backtest fill a gapped stop at the open
  - live/paper stop is set from the actual fill price
  - live sizing uses the strategy's own ATR (configured atr_period), like the backtest
"""
import importlib

import dotenv
import numpy as np
import pandas as pd
import pytest

import execution.order_manager as om_module
from backtesting.engine import BacktestEngine
from execution.broker import Broker
from risk.stops import bar_stop_check
from tests.test_lookahead import CONFIG, ScriptedStrategy

SYMBOL = "ETH/USDT:USDT"
ACT, CB = 0.015, 0.008


# ── shared intrabar logic ────────────────────────────────────────────────────

def test_old_stop_hit_exits_at_old_stop_even_if_trail_would_ratchet():
    # low reaches the old stop AND high would arm the trail: adverse order = stop first
    px, reason, _, _ = bar_stop_check("long", 100, 95, False, 100, 103, 94, ACT, CB)
    assert px == 95 and reason == "SL"


def test_gap_through_stop_fills_at_open():
    px, reason, _, _ = bar_stop_check("long", 100, 95, False, 90, 91, 89, ACT, CB)
    assert px == 90 and reason == "SL"
    px, _, _, _ = bar_stop_check("short", 100, 105, False, 110, 111, 109, ACT, CB)
    assert px == 110


def test_trail_armed_and_hit_in_same_bar_exits_at_new_trail_long():
    px, reason, sl, active = bar_stop_check("long", 100, 95, False, 100, 103, 101.5, ACT, CB)
    assert active and sl == pytest.approx(103 * (1 - CB))
    assert reason == "TrailingSL" and px == pytest.approx(103 * (1 - CB))


def test_trail_armed_and_hit_in_same_bar_exits_at_new_trail_short():
    px, reason, sl, _ = bar_stop_check("short", 100, 105, False, 100, 98.5, 97, ACT, CB)
    assert reason == "TrailingSL" and px == pytest.approx(97 * (1 + CB))


def test_trail_ratchets_without_exit_when_range_stays_above():
    px, reason, sl, active = bar_stop_check("long", 100, 95, False, 100, 103, 102.5, ACT, CB)
    assert px is None and active and sl == pytest.approx(103 * (1 - CB))


def test_take_profit_and_stop_in_same_bar_takes_the_worse():
    px, reason, _, _ = bar_stop_check("long", 100, 95, False, 100, 106, 94, None, None, tp_price=105)
    assert reason == "SL" and px == 95


# ── engine uses it ───────────────────────────────────────────────────────────

def test_engine_exits_on_intrabar_trailing_hit():
    cfg = {**CONFIG, "risk": {**CONFIG["risk"], "trailing_activation": 0.01, "trailing_callback": 0.005}}
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    px = np.full(10, 100.0)
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)
    df.iloc[3, df.columns.get_indexer(["high", "low", "close"])] = [103.0, 100.0, 102.0]
    res = BacktestEngine(cfg).run(df, ScriptedStrategy(cfg, entries={1: 1}), verbose=False)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "TrailingSL" and t["exit_time"] == idx[3]
    assert t["exit_price"] == pytest.approx(103 * 0.995 * (1 - CONFIG["backtest"]["slippage"]))


# ── paper broker ─────────────────────────────────────────────────────────────

PAPER_CFG = {
    "exchange": "bitget",
    "risk": {"leverage": 3, "trailing_activation": ACT, "trailing_callback": CB},
    "futures": {"margin_type": "isolated"},
    "backtest": {"commission": 0.0005, "slippage": 0.001},
}


@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.setattr(om_module, "DB_PATH", str(tmp_path / "trades.db"))
    monkeypatch.setenv("TRADING_MODE", "paper")
    return Broker(PAPER_CFG)


def _candle(o, h, l, c=None):
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    return pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c if c is not None else (h + l) / 2],
                         "volume": [1.0]}, index=idx)


def test_paper_stop_gap_fills_at_open(paper):
    paper.place_order(SYMBOL, "buy", 1.0, sl_dist=5.0, entry_price=100.0, strategy="t", margin=33)
    closed = paper.manage_stops(SYMBOL, _candle(90, 91, 89))
    assert closed["exit_price"] == pytest.approx(90 * (1 - 0.001))


def test_paper_trailing_intrabar_hit_closes(paper):
    paper.place_order(SYMBOL, "buy", 1.0, sl_dist=5.0, entry_price=100.0, strategy="t", margin=33)
    fill = 100 * 1.001
    closed = paper.manage_stops(SYMBOL, _candle(fill, fill * 1.03, fill * 1.015))
    assert closed is not None and closed["exit_reason"] == "TrailingSL"
    assert closed["exit_price"] == pytest.approx(fill * 1.03 * (1 - CB) * (1 - 0.001))


def test_paper_stop_is_set_from_fill_price(paper):
    paper.place_order(SYMBOL, "buy", 1.0, sl_dist=5.0, entry_price=100.0, strategy="t", margin=33)
    pos = paper.om.get_position(SYMBOL)
    assert pos["entry_price"] == pytest.approx(100.1)
    assert pos["sl_price"] == pytest.approx(100.1 - 5.0)


# ── live order path (fake exchange, no network) ──────────────────────────────

class FakeExchange:
    markets = {"x": 1}

    def __init__(self, fill):
        self.fill, self.calls, self.open_stops = fill, [], []

    def market(self, symbol):            return {"contractSize": 0.1, "limits": {"amount": {"min": 1}}}
    def amount_to_precision(self, s, a): return round(a, 4)
    def price_to_precision(self, s, p):  return round(p, 2)
    def set_leverage(self, *a, **k):     pass
    def cancel_all_orders(self, *a, **k): pass

    def create_market_order(self, symbol, side, amount, params=None):
        self.calls.append(("entry", side, amount, dict(params or {})))
        sl = (params or {}).get("stopLoss")
        if sl:
            self.open_stops.append({"id": "attached", "triggerPrice": sl["triggerPrice"]})
        return {"id": "e1", "average": self.fill}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.calls.append(("order", side, amount, dict(params or {})))
        if "stopLossPrice" in (params or {}):
            self.open_stops.append({"id": "fillstop", "triggerPrice": params["stopLossPrice"]})
        return {"id": "o2"}

    def fetch_open_orders(self, symbol, params=None):
        return list(self.open_stops)

    def cancel_order(self, order_id, symbol, params=None):
        self.calls.append(("cancel", order_id))
        self.open_stops = [o for o in self.open_stops if o["id"] != order_id]


def test_live_entry_moves_stop_to_fill_based_level(tmp_path, monkeypatch):
    monkeypatch.setattr(om_module, "DB_PATH", str(tmp_path / "trades.db"))
    monkeypatch.setenv("TRADING_MODE", "demo")
    cfg = {**PAPER_CFG, "risk": {**PAPER_CFG["risk"], "trailing_activation": None}}
    monkeypatch.setattr(Broker, "_build_exchange", lambda self: FakeExchange(fill=101.0))
    b = Broker(cfg)
    b.place_order(SYMBOL, "buy", 1.0, sl_price=95.0, sl_dist=5.0, entry_price=100.0, strategy="t")
    ex = b.exchange
    new_stop = [c for c in ex.calls if c[0] == "order" and "stopLossPrice" in c[3]]
    assert new_stop and new_stop[0][3]["stopLossPrice"] == pytest.approx(96.0)
    assert new_stop[0][3]["reduceOnly"] is True
    # the new stop is placed BEFORE the estimate-based one is cancelled
    kinds = [c[0] for c in ex.calls]
    assert kinds.index("order") < kinds.index("cancel")
    assert [o["triggerPrice"] for o in ex.open_stops] == [96.0]
    assert b.om.get_position(SYMBOL)["sl_price"] == pytest.approx(96.0)


# ── live sizing ATR ──────────────────────────────────────────────────────────

def test_live_sizing_uses_strategy_atr(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    main = importlib.import_module("main")
    from strategies.donchian_breakout import DonchianBreakoutStrategy
    from tests.test_lookahead import ohlcv
    cfg = {"dc_enter": 20, "dc_exit": 10, "atr_period": 7, "use_macro_filter": False}
    df = ohlcv(300)
    s = DonchianBreakoutStrategy(cfg)
    expected = BacktestEngine.prepare(df, s)["atr_14"].iloc[-1]
    assert main.signal_atr(s, df) == pytest.approx(expected)
