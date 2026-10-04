"""Paper-mode broker tests (no network): shorts are stored, stops fire, trailing ratchets."""
import pandas as pd
import pytest

import execution.order_manager as om_module
from execution.broker import Broker

SYMBOL = "ETH/USDT:USDT"

CONFIG = {
    "exchange": "bitget",
    "risk": {"leverage": 3, "trailing_activation": 0.015, "trailing_callback": 0.008},
    "futures": {"margin_type": "isolated"},
    "backtest": {"commission": 0.0006, "slippage": 0.0},
}


def _candle(high, low):
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    return pd.DataFrame({"open": [100.0], "high": [high], "low": [low],
                         "close": [(high + low) / 2], "volume": [1.0]}, index=idx)


@pytest.fixture
def broker(tmp_path, monkeypatch):
    monkeypatch.setattr(om_module, "DB_PATH", str(tmp_path / "trades.db"))
    monkeypatch.setenv("TRADING_MODE", "paper")
    return Broker(CONFIG)


def test_short_is_stored_and_stopped_out(broker):
    broker.place_order(SYMBOL, "sell", 1.0, sl_price=105.0, entry_price=100.0, strategy="t", margin=33)
    pos = broker.om.get_position(SYMBOL)
    assert pos is not None and pos["side"] == "sell" and pos["tp_price"] is None

    closed = broker.manage_stops(SYMBOL, _candle(high=106.0, low=99.0))
    assert closed is not None and closed["pnl_usdt"] < 0          # short stopped out above entry
    assert broker.om.get_position(SYMBOL) is None


def test_trailing_stop_ratchets_up_for_long(broker):
    broker.place_order(SYMBOL, "buy", 1.0, sl_price=95.0, entry_price=100.0, strategy="t", margin=33)
    assert broker.manage_stops(SYMBOL, _candle(high=103.0, low=100.0)) is None
    pos = broker.om.get_position(SYMBOL)
    assert pos["trail_active"] == 1
    assert pos["sl_price"] == pytest.approx(103.0 * 0.992)

    broker.manage_stops(SYMBOL, _candle(high=102.9, low=102.5))    # lower high must not loosen the stop
    assert broker.om.get_position(SYMBOL)["sl_price"] == pytest.approx(103.0 * 0.992)


def test_paper_pnl_has_no_leverage_factor(broker):
    broker.place_order(SYMBOL, "buy", 2.0, sl_price=95.0, entry_price=100.0, strategy="t", margin=66)
    result = broker.close_position(SYMBOL, 110.0, "test")
    gross = (110.0 - 100.0) * 2.0
    fees  = (110.0 + 100.0) * 2.0 * 0.0006
    assert result["pnl_usdt"] == pytest.approx(gross - fees, abs=1e-3)


def test_paper_balance_follows_realized_pnl(broker):
    start = broker.get_balance()
    broker.place_order(SYMBOL, "buy", 1.0, sl_price=95.0, entry_price=100.0, strategy="t", margin=33)
    broker.close_position(SYMBOL, 110.0, "test")
    assert broker.get_balance() > start
