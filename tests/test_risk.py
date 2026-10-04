"""Tests for risk management."""
import pytest
from risk.position_sizer import PositionSizer
from risk.guard import RiskGuard

CONFIG = {
    "risk": {
        "max_risk_per_trade": 0.02,
        "stop_loss_pct":      0.02,
        "take_profit_pct":    0.04,
        "atr_stop_mult":      2.0,
        "max_open_positions": 3,
        "max_total_risk_pct": 0.04,
        "max_daily_loss_pct": 0.05,
        "leverage":           3,
    }
}


def _pos(side, entry, sl, size):
    return {"side": side, "entry_price": entry, "sl_price": sl, "size": size}


# ── Sizer ─────────────────────────────────────────────────────────────────────

def test_position_size_within_capital():
    result = PositionSizer(CONFIG).calculate(10_000, 50_000)
    assert result["margin"] <= 10_000 * 0.9
    assert result["sl_price"] < 50_000
    assert result["tp_price"] > 50_000


def test_sl_tp_ratio_with_percent_fallback():
    result  = PositionSizer(CONFIG).calculate(10_000, 50_000)    # no ATR -> percent stop
    sl_dist = 50_000 - result["sl_price"]
    tp_dist = result["tp_price"] - 50_000
    assert tp_dist / sl_dist == pytest.approx(2.0, rel=0.01)


def test_atr_stop_distance_and_risk():
    r = PositionSizer(CONFIG).calculate(10_000, 100.0, "long", atr=2.0)
    assert r["sl_price"] == pytest.approx(96.0)                  # 100 - 2.0 * 2.0
    assert r["size"] * r["sl_dist"] <= 10_000 * 0.02 + 1e-6      # loss at stop <= 2 % of capital


def test_short_stop_above_entry():
    r = PositionSizer(CONFIG).calculate(10_000, 100.0, "short", atr=2.0)
    assert r["sl_price"] == pytest.approx(104.0)
    assert r["tp_price"] < 100.0


def test_no_take_profit_when_disabled():
    cfg = {"risk": {**CONFIG["risk"], "take_profit_pct": None}}
    assert PositionSizer(cfg).calculate(10_000, 100.0, "long", atr=2.0)["tp_price"] is None


def test_risk_pct_override_scales_size():
    sizer = PositionSizer(CONFIG)
    full = sizer.calculate(10_000, 100.0, "long", atr=2.0)
    half = sizer.calculate(10_000, 100.0, "long", atr=2.0, risk_pct=0.01)
    assert half["size"] == pytest.approx(full["size"] / 2, rel=1e-3)


# ── Kill switch and slots ─────────────────────────────────────────────────────

def test_kill_switch_triggers():
    guard = RiskGuard(CONFIG, initial_capital=10_000)
    assert guard.killed(9_400) is True            # -6 %
    assert guard.can_open(0) is False


def test_max_positions_blocks_entry_only():
    guard = RiskGuard(CONFIG, initial_capital=10_000)
    assert guard.killed(10_000) is False          # exits stay allowed
    assert guard.can_open(3) is False


def test_no_kill_within_limit():
    guard = RiskGuard(CONFIG, initial_capital=10_000)
    assert guard.killed(9_600) is False           # -4 % is within the -5 % limit
    assert guard.can_open(1) is True


# ── Portfolio risk budget ─────────────────────────────────────────────────────

def test_open_risk_sums_stop_out_losses():
    positions = [_pos("buy", 100, 96, 10),        # 40 at the stop
                 _pos("sell", 50, 52, 20)]        # 40 at the stop
    assert RiskGuard.open_risk(positions, 10_000) == pytest.approx(0.008)


def test_trailing_stop_in_profit_counts_as_zero_risk():
    assert RiskGuard.open_risk([_pos("buy", 100, 105, 10)], 10_000) == 0.0


def test_risk_budget_shrinks_with_open_positions():
    guard = RiskGuard(CONFIG, 10_000)
    assert guard.risk_budget(10_000, []) == pytest.approx(0.04)
    one = [_pos("buy", 100, 98, 100)]             # 200 = 2 % at risk
    assert guard.risk_budget(10_000, one) == pytest.approx(0.02)
    two = one + [_pos("buy", 100, 98, 100)]       # 4 % at risk -> budget exhausted
    assert guard.risk_budget(10_000, two) == 0.0


def test_no_portfolio_cap_means_infinite_budget():
    cfg = {"risk": {k: v for k, v in CONFIG["risk"].items() if k != "max_total_risk_pct"}}
    assert RiskGuard(cfg, 10_000).risk_budget(10_000, [_pos("buy", 100, 90, 100)]) == float("inf")
