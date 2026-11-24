"""Walk-forward optimisation: window construction, IS-only fitting, OOS stitching."""
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.optimizer import ParameterOptimizer, make_windows, param_stability
from backtesting.research_log import ResearchLog
from strategies.donchian_breakout import DonchianBreakoutStrategy

CFG = {
    "timeframe": "4h", "use_macro_filter": False,
    "dc_enter": 20, "dc_exit": 10, "adx_threshold": 15, "atr_min_mult": 0.5, "vol_mult": 0.0,
    "risk": {"max_risk_per_trade": 0.02, "stop_loss_pct": 0.03, "take_profit_pct": None,
             "atr_stop_mult": 2.0, "trailing_activation": None, "trailing_callback": None,
             "max_open_positions": 1, "max_daily_loss_pct": 0.05, "leverage": 3},
    "backtest": {"commission": 0.0006, "slippage": 0.0002, "funding_rate_8h": 0.0,
                 "walk_forward_pct": 0.3},
}
GRID = {"dc_enter": [10, 30], "dc_exit": [5, 15]}


def trending(n_days=730, seed=3):
    rng = np.random.default_rng(seed)
    n = n_days * 6
    drift = np.repeat(rng.choice([-0.004, 0.004], n // 60 + 1), 60)[:n]
    close = 1000 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.004,
                         "low": np.minimum(open_, close) * 0.996, "close": close,
                         "volume": rng.uniform(100, 200, n)}, index=idx)


def _opt(log=None):
    return ParameterOptimizer(CFG, BacktestEngine(CFG), log=log)


def test_make_windows_rolling():
    w = make_windows("2021-01-01", "2022-08-01", is_months=12, oos_months=3)
    assert [(x.is_start.strftime("%Y-%m"), x.oos_start.strftime("%Y-%m"), x.oos_end.strftime("%Y-%m"))
            for x in w] == [("2021-01", "2022-01", "2022-04"), ("2021-04", "2022-04", "2022-07"),
                            ("2021-07", "2022-07", "2022-08")]          # last OOS window is partial
    assert all(x.is_end == x.oos_start for x in w)


def test_make_windows_anchored():
    w = make_windows("2021-01-01", "2022-08-01", is_months=12, oos_months=3, mode="anchored")
    assert all(x.is_start == pd.Timestamp("2021-01-01") for x in w)
    assert [x.is_end.strftime("%Y-%m") for x in w] == ["2022-01", "2022-04", "2022-07"]


def test_oos_windows_tile_without_overlap():
    w = make_windows("2021-01-01", "2025-11-01", 12, 3)
    for a, b in zip(w, w[1:]):
        assert a.oos_end == b.oos_start
    assert w[-1].oos_end == pd.Timestamp("2025-11-01")


def test_engine_has_split_consistency_check_not_walk_forward():
    assert hasattr(BacktestEngine, "split_consistency_check")
    assert not hasattr(BacktestEngine, "walk_forward")


@pytest.mark.parametrize("mode", ["rolling", "anchored"])
def test_wfo_stitches_oos_equity(mode):
    df = trending()
    r = _opt().walk_forward_optimize(df, DonchianBreakoutStrategy, GRID, mode=mode, is_months=6,
                                     oos_months=3, min_trades=3, verbose=False)
    w = r.windows
    assert len(w) == 6
    # stitched equity covers exactly the OOS periods, in order, without duplicates
    assert r.equity.index.is_monotonic_increasing and not r.equity.index.duplicated().any()
    assert r.equity.index[0] >= w["oos_start"].iloc[0]
    assert r.equity.index[-1] < w["oos_end"].iloc[-1]
    # every OOS trade lies inside its own OOS window
    for _, t in r.trades.iterrows():
        win = w[w["window"] == t["window"]].iloc[0]
        assert win["oos_start"] <= t["entry_time"] < win["oos_end"]
    # compounding: total return equals the product of the window returns
    assert (1 + r.metrics["total_return"]) == pytest.approx(np.prod(1 + w["oos_return"]), rel=1e-9)
    assert 0 <= r.metrics["psr"] <= 1 and 0 <= r.metrics["dsr"] <= 1
    assert r.stability["n_windows"] == 6


def test_wfo_params_do_not_depend_on_future_data():
    """Destroy all prices after window 3's IS end: windows 1-3 must choose the same parameters."""
    df = trending()
    opt = _opt()
    base = opt.walk_forward_optimize(df, DonchianBreakoutStrategy, GRID, is_months=6, oos_months=3,
                                     min_trades=3, verbose=False)
    cut = base.windows["is_end"].iloc[2]
    poisoned = df.copy()
    after = poisoned.index >= cut
    rng = np.random.default_rng(99)
    poisoned.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 1.5, (after.sum(), 1))
    poisoned["high"] = poisoned[["open", "high", "low", "close"]].max(axis=1)
    poisoned["low"]  = poisoned[["open", "high", "low", "close"]].min(axis=1)
    pois = opt.walk_forward_optimize(poisoned, DonchianBreakoutStrategy, GRID, is_months=6,
                                     oos_months=3, min_trades=3, verbose=False)
    cols = ["dc_enter", "dc_exit", "is_sharpe"]
    pd.testing.assert_frame_equal(base.windows[cols].iloc[:3], pois.windows[cols].iloc[:3],
                                  check_dtype=False)
    # and the poisoning did change later windows (otherwise the test proves nothing)
    assert not base.windows["oos_return"].iloc[3:].equals(pois.windows["oos_return"].iloc[3:])


def test_wfo_logs_every_grid_search_and_the_study(tmp_path):
    log = ResearchLog(str(tmp_path / "log.jsonl"))
    r = _opt(log).walk_forward_optimize(trending(400), DonchianBreakoutStrategy, GRID, is_months=6,
                                        oos_months=3, min_trades=3, study_id="s1",
                                        strategy_name="donchian_breakout", verbose=False)
    grids = log.entries("grid_search", study_id="s1")
    assert len(grids) == len(r.windows)
    assert all(g["n_trials"] == 4 and len(g["trials"]) == 4 and g["git"]["commit"] for g in grids)
    study = log.entries("wfo_study", study_id="s1")[0]
    assert study["n_trials"] == 4 * len(r.windows)
    assert study["settings"]["param_grid"] == GRID


def test_window_without_valid_params_stays_flat():
    df = trending(400)
    r = _opt().walk_forward_optimize(df, DonchianBreakoutStrategy, GRID, is_months=6, oos_months=3,
                                     min_trades=10_000, verbose=False)
    assert (r.windows["n_valid"] == 0).all()
    assert r.trades.empty
    assert r.equity.nunique() == 1


def test_param_stability_summary():
    w = pd.DataFrame({"a": [1, 1, 2, 2], "b": [5, 5, 5, 6]})
    s = param_stability(w, ["a", "b"])
    assert s["n_distinct_sets"] == 3 and s["change_rate"] == pytest.approx(2 / 3)
    assert s["value_counts"]["a"] == {"1": 2, "2": 2}


def test_py_converts_integral_floats():
    from backtesting.optimizer import _py
    assert _py(40.0) == 40 and isinstance(_py(40.0), int)
    assert _py(np.float64(0.3)) == 0.3 and _py(np.int64(5)) == 5
