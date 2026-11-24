"""TSMOM pipeline on a synthetic universe (no network): hold-out guard, logging, repeat warning."""
import numpy as np
import pandas as pd
import pytest

from backtesting import tsmom_research as tr
from backtesting.research_log import ResearchLog
from data import universe

HS = "2023-07-01"


def _frames(poison=False, k=16, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", "2024-01-01", freq="D", tz="UTC", inclusive="left")
    out = {}
    for j in range(k):
        close = 100 * np.exp(np.cumsum(rng.normal(rng.normal(0, 0.002), 0.04, len(idx))))
        df = pd.DataFrame({"open": np.r_[100, close[:-1]], "close": close,
                           "volume": rng.uniform(1e5, 1e6) * np.ones(len(idx))}, index=idx)
        df["high"], df["low"] = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
        if poison:
            df.loc[df.index >= HS, ["open", "high", "low", "close"]] *= np.random.default_rng(100 + j).uniform(0.2, 5)
        out[("BTC" if j == 0 else f"C{j:02d}") + "USDT"] = df
    return out


@pytest.fixture
def fake_universe(monkeypatch):
    state = {"poison": False}
    monkeypatch.setattr(universe, "binance_inventory", lambda: pd.DataFrame(
        {"symbol": list(_frames()), "first_month": "2021-01", "last_month": "2026-10",
         "base": [universe.base_coin(s) for s in _frames()]}))
    monkeypatch.setattr(universe, "okx_xperp_coins", lambda: {"retrieved": "2026-11-24", "coins":
                                                              [universe.base_coin(s) for s in _frames()]})
    monkeypatch.setattr(universe, "load_daily_panel", lambda c, s, e: _frames(state["poison"]))
    monkeypatch.setattr(universe, "load_funding_panel", lambda syms, s, e, idx: pd.DataFrame(
        0.0003, index=idx, columns=syms))
    return state


def _cfg(tmp_path):
    cfg = tr.load_config()
    cfg["data"] = {"start_date": "2021-01-01", "end_date": "2024-01-01", "holdout_start": HS}
    cfg["validation"]["is_months"], cfg["validation"]["oos_months"] = 6, 3
    cfg["research_log"] = str(tmp_path / "log.jsonl")
    return cfg


def test_study_never_sees_holdout(fake_universe, tmp_path):
    cfg = _cfg(tmp_path)
    a = tr.run_study(cfg)
    fake_universe["poison"] = True
    b = tr.run_study(cfg)
    for k in ("V1", "V2", "V3", "V4", "W1", "W2", "R1"):
        ea, eb = a[k]["result"].equity, b[k]["result"].equity
        assert ea.index.max() < pd.Timestamp(HS, tz="UTC")
        pd.testing.assert_series_equal(ea, eb)
    assert set(a["dsr"]) == {"V1", "V2", "V3", "V4", "W1", "W2", "R1"}
    log = ResearchLog(cfg["research_log"])
    assert len(log.entries("tsmom_config")) == 14 and all(e["n_trials"] == 7 for e in log.entries("tsmom_config"))


def test_holdout_selection_ignores_holdout_and_repeat_warns(fake_universe, tmp_path):
    cfg = _cfg(tmp_path)
    log = ResearchLog(cfg["research_log"])
    a = tr.evaluate_holdout(cfg, log=log)
    assert a["result"].equity.index.min() == pd.Timestamp(HS, tz="UTC") and a["prior"] == 0
    fake_universe["poison"] = True
    with pytest.warns(tr.HoldoutReuseWarning):
        b = tr.evaluate_holdout(cfg, log=log)
    assert a["variant"] == b["variant"]                 # selection uses pre-hold-out data only
    assert len(log.entries("holdout", strategy=tr.STRATEGY)) == 2
