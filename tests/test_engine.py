"""Regression tests: leverage must not multiply PnL, trailing stop only moves in favour."""
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, Trade
from risk.stops import update_trailing

CONFIG = {
    "risk": {
        "max_risk_per_trade": 0.02, "stop_loss_pct": 0.02, "take_profit_pct": None,
        "atr_stop_mult": 2.0, "trailing_activation": 0.015, "trailing_callback": 0.008,
        "max_open_positions": 3, "max_daily_loss_pct": 0.05, "leverage": 3,
    },
    "backtest": {"commission": 0.0, "slippage": 0.0, "walk_forward_pct": 0.3},
}


def _trade(side="long", entry=100.0, size=2.0, sl=95.0):
    return Trade("X", side, pd.Timestamp("2024-01-01"), entry, size, sl, None)


def test_pnl_does_not_multiply_leverage():
    engine = BacktestEngine(CONFIG)
    t = engine._close(_trade(), pd.Timestamp("2024-01-02"), 110.0, "x")
    assert t.pnl_usdt == pytest.approx(20.0)      # (110 - 100) * 2, leverage 3 must not matter


def test_short_pnl_sign():
    engine = BacktestEngine(CONFIG)
    t = engine._close(_trade("short", sl=105.0), pd.Timestamp("2024-01-02"), 90.0, "x")
    assert t.pnl_usdt == pytest.approx(20.0)


def test_trailing_stop_only_moves_in_favour():
    sl, active = update_trailing("long", 100, 95, False, high=102, low=100,
                                 activation=0.015, callback=0.008)
    assert active and sl == pytest.approx(102 * 0.992)
    sl2, _ = update_trailing("long", 100, sl, True, high=101, low=100,
                             activation=0.015, callback=0.008)
    assert sl2 == sl                               # lower high must not loosen the stop


def test_trailing_not_armed_below_activation():
    sl, active = update_trailing("long", 100, 95, False, high=101, low=99,
                                 activation=0.015, callback=0.008)
    assert not active and sl == 95
