"""Forward paper trading of tsmom-v1 with injected (synthetic) data sources: no network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from paper import tsmom_paper as tp

NOW = pd.Timestamp("2026-03-15 00:20", tz="UTC")
COINS = ["BTC", "ETH"] + [f"C{i:02d}" for i in range(14)]


def _series(coin, days=400):
    rng = np.random.default_rng(sum(map(ord, coin)))
    idx = pd.date_range(NOW.floor("D") - pd.Timedelta(days=days), NOW.floor("D"), freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, len(idx))))
    df = pd.DataFrame({"open": np.r_[100, close[:-1]], "close": close,
                       "volume": rng.uniform(1e4, 1e5) * np.ones(len(idx))}, index=idx)
    df["high"], df["low"] = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    return df[["open", "high", "low", "close", "volume"]]


def fake_instruments():
    return pd.DataFrame({"base": COINS, "inst_id": [f"{c}-USD_UM_XPERP-310404" for c in COINS]})


def fake_candles(inst_id, start):
    df = _series(inst_id.split("-")[0])               # includes TODAY's still-open candle on purpose
    return df[df.index >= start]


def fake_funding(inst_id, start):
    idx = pd.date_range(NOW.floor("D") - pd.Timedelta(days=400), NOW, freq="8h", tz="UTC")
    s = pd.Series(0.0001, index=idx, name="funding_rate")
    return s[s.index >= start]


def _spec(start):
    spec = tp.load_spec()
    spec.update({"coins": COINS, "start": start, "history_days": 300})
    return spec


def _run(state, spec, now=NOW):
    cov = tp.update_store(state, spec, now, fake_instruments, fake_candles, fake_funding)
    out = tp.compute(state, spec)
    tp.write_outputs(state, out, cov)
    return out


def test_frozen_spec_matches_amendment():
    spec = tp.load_spec()
    assert spec["name"] == "tsmom-v1" and spec["variant"] == "V1" and spec["start"] == "2026-11-25"
    assert len(spec["coins"]) == 168
    text = open("docs/preregistration.md").read()
    assert "Amendment 2" in text and "open of 2026-11-25" in text


def test_store_keeps_closed_days_only_and_appends_idempotently(tmp_path):
    spec = _spec("2026-02-01")
    tp.update_store(str(tmp_path), spec, NOW, fake_instruments, fake_candles, fake_funding)
    c = pd.read_csv(tmp_path / "data" / "candles" / "BTC.csv", index_col=0)
    assert pd.Timestamp(c.index[-1]) < NOW.floor("D")              # today's open candle not stored
    n = len(c)
    tp.update_store(str(tmp_path), spec, NOW, fake_instruments, fake_candles, fake_funding)
    assert len(pd.read_csv(tmp_path / "data" / "candles" / "BTC.csv")) == n


def test_before_start_only_reports_orders(tmp_path):
    out = _run(str(tmp_path), _spec("2026-04-01"))
    assert out["status"] == "not started" and out["orders_for_open_of"] == "2026-03-15"
    assert os.path.exists(tmp_path / "report.md")


def test_running_replay_is_consistent_and_logged_once_per_day(tmp_path):
    spec = _spec("2026-02-01")
    a = _run(str(tmp_path), spec)
    assert a["status"] == "running" and a["days"] == (NOW.floor("D") - pd.Timestamp("2026-02-01", tz="UTC")).days
    eq = pd.read_csv(tmp_path / "equity.csv", index_col=0)
    assert eq.index[0].startswith("2026-02-01")
    b = _run(str(tmp_path), spec)                                   # same day again: same numbers
    assert b["equity"] == a["equity"] and b["target_weights"] == a["target_weights"]
    with open(tmp_path / "log.jsonl") as f:
        assert len([line for line in f if line.strip()]) == 1
    s = json.load(open(tmp_path / "summary.json"))
    assert s["sharpe_ci95"][0] < s["sharpe"] < s["sharpe_ci95"][1]
    assert {b["line"] for b in s["benchmarks"]} >= {"Strategy", "BTC buy & hold", "Equal-weight universe"}
    assert all(w >= 0 for w in s["target_weights"].values())       # V1 is long-flat


def test_delisted_coin_is_reported_and_closed(tmp_path):
    spec = _spec("2026-02-01")
    _run(str(tmp_path), spec, now=NOW - pd.Timedelta(days=10))
    inst = lambda: fake_instruments().iloc[1:]                       # BTC disappears from OKX
    cov = tp.update_store(str(tmp_path), spec, NOW, inst, fake_candles, fake_funding)
    assert "BTC" in cov["missing_now"]


def test_telegram_is_skipped_without_secrets(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert tp.telegram("x") is False
