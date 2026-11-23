import json

import numpy as np
import pandas as pd

from backtesting.research_log import ResearchLog, git_revision


def test_append_and_read_back(tmp_path):
    log = ResearchLog(str(tmp_path / "log.jsonl"))
    log.append("grid_search", strategy="s", symbol="ETH", timeframe="1h",
               params={"a": np.int64(3)}, period={"start": pd.Timestamp("2024-01-01", tz="UTC")},
               metrics={"sharpe": np.float64(1.5), "pf": float("inf")}, n_trials=108, study_id="x")
    e = log.entries("grid_search")[0]
    assert e["params"] == {"a": 3}
    assert e["metrics"] == {"sharpe": 1.5, "pf": None}          # inf is not valid JSON
    assert e["n_trials"] == 108 and e["study_id"] == "x"
    assert e["period"]["start"].startswith("2024-01-01")
    assert "commit" in e["git"] and "timestamp" in e
    # one JSON object per line
    with open(tmp_path / "log.jsonl") as f:
        assert all(json.loads(line) for line in f)


def test_filter_by_fields(tmp_path):
    log = ResearchLog(str(tmp_path / "log.jsonl"))
    for sym in ("ETH", "BTC", "ETH"):
        log.append("backtest", strategy="s", symbol=sym, timeframe="1h")
    assert len(log.entries("backtest", symbol="ETH")) == 2
    assert log.entries("holdout") == []


def test_holdout_evaluations_are_counted_per_holdout_start(tmp_path):
    log = ResearchLog(str(tmp_path / "log.jsonl"))
    hs = pd.Timestamp("2025-11-01", tz="UTC")
    log.append("holdout", strategy="s", symbol="ETH", timeframe="1h", holdout_start=hs.isoformat())
    log.append("holdout", strategy="s", symbol="ETH", timeframe="1h",
               holdout_start=pd.Timestamp("2024-01-01", tz="UTC").isoformat())
    assert len(log.holdout_evaluations("s", "ETH", "1h", hs)) == 1


def test_git_revision_in_repo():
    rev = git_revision()
    assert rev["commit"] is None or len(rev["commit"]) == 40
