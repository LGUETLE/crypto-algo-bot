"""Research pipeline: hold-out guard, logging, live parity of the data the strategy sees."""
import numpy as np
import pandas as pd
import pytest

from backtesting import research
from backtesting.research_log import ResearchLog
from features.macro import add_macro_trend
from strategies.donchian_breakout import DonchianBreakoutStrategy

HS = pd.Timestamp("2023-07-01", tz="UTC")


def _ohlcv(start, end, freq, seed):
    idx = pd.date_range(start, end, freq=freq.replace("d", "D"), tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.004, 0.004], len(idx) // 60 + 1), 60)[:len(idx)]
    close = 1000 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.004,
                         "low": np.minimum(open_, close) * 0.996, "close": close,
                         "volume": rng.uniform(100, 200, len(idx))}, index=idx)


class FakeFetcher:
    def __init__(self, poison_after=None):
        self.poison_after = poison_after

    def _maybe_poison(self, df):
        if self.poison_after is None:
            return df
        df = df.copy()
        m = df.index >= self.poison_after
        df.loc[m, ["open", "high", "low", "close"]] *= 3.0
        return df

    def ohlcv_range(self, symbol, tf, start, end, use_cache=True):
        seed = 1 if symbol.startswith("BTC") else 2
        return self._maybe_poison(_ohlcv(start, end, tf, seed))

    def historical(self, symbol, tf, days=None, use_cache=True, macro_tf=None, start=None, end=None):
        df = self.ohlcv_range(symbol, tf, start, end)
        if macro_tf:
            macro = self.ohlcv_range(symbol, macro_tf, pd.Timestamp(start) - pd.Timedelta(days=60), end)
            df = add_macro_trend(df, macro, macro_tf)
        return df

    def funding_history(self, symbol, start, end, use_cache=True):
        idx = pd.date_range(start, end, freq="8h", tz="UTC", inclusive="left")
        s = pd.Series(0.0001, index=idx)
        if self.poison_after is not None:
            s[s.index > self.poison_after] = 0.05   # the event AT hs belongs to the research period
        return s


def _config(tmp_path):
    return {
        "timeframe": "4h", "use_macro_filter": True, "macro_timeframe": "1d",
        "strategy": "donchian_breakout",
        "dc_enter": 20, "dc_exit": 10, "adx_threshold": 15, "atr_min_mult": 0.5, "vol_mult": 0.0,
        "risk": {"max_risk_per_trade": 0.02, "stop_loss_pct": 0.03, "take_profit_pct": None,
                 "atr_stop_mult": 2.0, "trailing_activation": 0.015, "trailing_callback": 0.008,
                 "max_open_positions": 1, "max_daily_loss_pct": 0.05, "leverage": 3},
        "backtest": {"start_date": "2022-01-01", "end_date": "2024-01-01",
                     "holdout_start": HS.strftime("%Y-%m-%d"), "commission": 0.0006,
                     "slippage": 0.0002, "funding_rate_8h": 0.0001, "walk_forward_pct": 0.3,
                     "research_log": str(tmp_path / "log.jsonl")},
        "optimize": {"metric": "sharpe", "min_trades": 3, "max_dd_limit": -0.9, "is_months": 6,
                     "oos_months": 3, "modes": ["rolling", "anchored"],
                     "param_grid": {"donchian_breakout": {"dc_enter": [10, 30], "dc_exit": [5, 15]}}},
    }


PAIR = {"symbol": "ETH/USDT:USDT", "data_symbol": "ETH/USDT:USDT", "strategy": "donchian_breakout"}


def test_holdout_is_cut_from_prices_funding_and_benchmark(tmp_path):
    d = research.load_research_data(_config(tmp_path), PAIR, FakeFetcher())
    assert d.df.index.max() < HS and d.funding.index.max() <= HS and d.btc.index.max() < HS
    full = research.load_research_data(_config(tmp_path), PAIR, FakeFetcher(), final_holdout=True)
    assert full.df.index.max() >= HS


def test_strategy_sees_raw_ohlcv_like_live(tmp_path):
    """No FeatureEngineer columns: otherwise strategies skip computing their own indicators."""
    d = research.load_research_data(_config(tmp_path), PAIR, FakeFetcher())
    assert set(d.df.columns) == {"open", "high", "low", "close", "volume", "macro_close", "macro_trend"}


def test_holdout_must_lie_inside_window(tmp_path):
    cfg = _config(tmp_path)
    cfg["backtest"]["holdout_start"] = "2025-01-01"
    with pytest.raises(ValueError):
        research.load_research_data(cfg, PAIR, FakeFetcher())


def test_backtest_and_optimize_ignore_holdout_data(tmp_path, capsys):
    cfg = _config(tmp_path)
    clean = research.run_optimize(cfg, PAIR, FakeFetcher())
    pois  = research.run_optimize(cfg, PAIR, FakeFetcher(poison_after=HS))
    for mode in ("rolling", "anchored"):
        a, b = clean["wfo"][mode]["wfo"], pois["wfo"][mode]["wfo"]
        assert a.equity.index.max() < HS
        pd.testing.assert_series_equal(a.equity, b.equity)
    bt_a = research.run_backtest(cfg, PAIR, FakeFetcher())
    bt_b = research.run_backtest(cfg, PAIR, FakeFetcher(poison_after=HS))
    pd.testing.assert_series_equal(bt_a["result"].equity, bt_b["result"].equity)
    assert bt_a["result"].equity.index.max() < HS


def test_holdout_fit_uses_only_pre_holdout_data(tmp_path):
    cfg = _config(tmp_path)
    a = research.evaluate_holdout(cfg, PAIR, FakeFetcher(), log=ResearchLog(str(tmp_path / "a.jsonl")))
    b = research.evaluate_holdout(cfg, PAIR, FakeFetcher(poison_after=HS),
                                  log=ResearchLog(str(tmp_path / "b.jsonl")))
    assert a["params"] == b["params"]
    assert a["result"].equity.index.min() == HS        # trading starts exactly at the hold-out


def test_repeated_holdout_evaluation_warns_and_is_logged(tmp_path):
    cfg = _config(tmp_path)
    log = ResearchLog(cfg["backtest"]["research_log"])
    first = research.evaluate_holdout(cfg, PAIR, FakeFetcher(), log=log)
    assert first["prior_evaluations"] == 0
    with pytest.warns(research.HoldoutReuseWarning):
        second = research.evaluate_holdout(cfg, PAIR, FakeFetcher(), log=log)
    assert second["prior_evaluations"] == 1
    entries = log.entries("holdout")
    assert len(entries) == 2 and entries[1]["prior_evaluations"] == 1
    assert entries[0]["holdout_start"] == HS.isoformat() and entries[0]["git"]["commit"]


def test_backtest_final_holdout_flag_counts_as_holdout_use(tmp_path):
    cfg = _config(tmp_path)
    log = ResearchLog(cfg["backtest"]["research_log"])
    research.run_backtest(cfg, PAIR, FakeFetcher(), final_holdout=True, log=log)
    with pytest.warns(research.HoldoutReuseWarning):
        research.evaluate_holdout(cfg, PAIR, FakeFetcher(), log=log)


def test_wfo_variants_counted_for_dsr(tmp_path):
    cfg = _config(tmp_path)
    log = ResearchLog(cfg["backtest"]["research_log"])
    r1 = research.run_optimize(cfg, PAIR, FakeFetcher(), log=log)
    assert r1["wfo"]["rolling"]["wfo"].metrics["dsr_n_variants"] == 2
    research.run_optimize(cfg, PAIR, FakeFetcher(), log=log)         # same settings: no new variant
    cfg["optimize"]["is_months"] = 9
    r3 = research.run_optimize(cfg, PAIR, FakeFetcher(), log=log)    # two new variants
    assert r3["wfo"]["rolling"]["wfo"].metrics["dsr_n_variants"] == 4


def test_live_window_signals_match_full_history():
    """Live computes on the last ~300 closed candles; the backtest on the full history.
    For the Donchian strategy (largest grid value dc_enter=50) both must agree."""
    df = _ohlcv("2022-01-01", "2022-07-01", "1h", 4)
    for p in ({"dc_enter": 50, "dc_exit": 15, "adx_threshold": 20, "atr_min_mult": 0.3},
              {"dc_enter": 20, "dc_exit": 5, "adx_threshold": 30, "atr_min_mult": 0.7}):
        s = DonchianBreakoutStrategy({**p, "use_macro_filter": False, "vol_mult": 1.0})
        full = s.generate_signals(s.add_indicators(df.copy()))
        mismatches = 0
        for end in range(1000, len(df), 37):
            tail = df.iloc[end - 299:end]                    # what run_cycle passes (299 closed)
            live = s.generate_signals(s.add_indicators(tail.copy()))
            ts = df.index[end - 1]
            mismatches += int(live["signal"].iloc[-1] != full.loc[ts, "signal"]
                              or live["long_exit"].iloc[-1] != full.loc[ts, "long_exit"])
        assert mismatches == 0


def test_optimize_runs_without_holdout_start(tmp_path, capsys):
    cfg = _config(tmp_path)
    del cfg["backtest"]["holdout_start"]
    out = research.run_optimize(cfg, PAIR, FakeFetcher())
    assert "rolling" in out["wfo"]


def test_funding_event_at_holdout_start_belongs_to_research_period(tmp_path):
    d = research.load_research_data(_config(tmp_path), PAIR, FakeFetcher())
    assert d.funding.index.max() == HS            # settled against the position held at HS
    assert d.df.index.max() < HS


def test_wfo_reports_forced_window_end_exits(tmp_path):
    out = research.run_optimize(_config(tmp_path), PAIR, FakeFetcher())
    m = out["wfo"]["rolling"]["wfo"].metrics
    assert "window_end_exits" in m and "window_end_exit_pnl" in m


def test_cost_report_respects_holdout_guard(tmp_path):
    cfg = _config(tmp_path)
    log = ResearchLog(cfg["backtest"]["research_log"])
    a = research.run_costs(cfg, PAIR, FakeFetcher(), log=log)
    b = research.run_costs(cfg, PAIR, FakeFetcher(poison_after=HS), log=log)
    assert a["result"].equity.index.max() < HS
    assert a["costs"] == b["costs"]
    assert list(a["scenarios"].index)[0] == "No costs" and len(log.entries("cost_report")) == 2
    research.run_costs(cfg, PAIR, FakeFetcher(), final_holdout=True, log=log)
    with pytest.warns(research.HoldoutReuseWarning):
        research.evaluate_holdout(cfg, PAIR, FakeFetcher(), log=log)


def test_intrabar_data_is_cut_at_holdout_and_poisoning_has_no_effect(tmp_path):
    cfg = _config(tmp_path)
    cfg["backtest"]["intrabar_timeframe"] = "1h"           # finer than the 4h test bars
    d = research.load_research_data(cfg, PAIR, FakeFetcher())
    assert d.intrabar is not None and d.intrabar.index.max() < HS
    a = research.run_backtest(cfg, PAIR, FakeFetcher())
    b = research.run_backtest(cfg, PAIR, FakeFetcher(poison_after=HS))
    pd.testing.assert_series_equal(a["result"].equity, b["result"].equity)
