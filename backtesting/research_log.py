"""
Research log
============
Append-only JSONL record of every research run, so that the number of trials behind a result is
known and every number can be traced to code and data:

  timestamp (UTC), kind, study_id, strategy, symbol, timeframe, params, period, metrics,
  n_trials, git commit + dirty flag, plus free-form extra fields.

kinds used by the pipeline:
  grid_search   one parameter grid evaluated on one in-sample window (all trial results)
  wfo_study     one walk-forward optimisation (per-window params, stitched OOS metrics)
  backtest      a single backtest with fixed parameters
  holdout       an evaluation on the final hold-out period (should happen ONCE per study)
"""
import json
import logging
import math
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join("logs", "research_log.jsonl")


def git_revision(cwd: str = None) -> dict:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=cwd, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _jsonable(x):
    """Recursively converts numpy/pandas scalars, timestamps and non-finite floats for JSON."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def new_study_id(prefix: str = "study") -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


class ResearchLog:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._git = None
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    @property
    def git(self) -> dict:
        if self._git is None:
            self._git = git_revision()
        return self._git

    def append(self, kind: str, *, strategy: str, symbol: str, timeframe: str,
               params=None, period=None, metrics=None, n_trials: Optional[int] = None,
               study_id: Optional[str] = None, **extra) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind":      kind,
            "study_id":  study_id,
            "strategy":  strategy,
            "symbol":    symbol,
            "timeframe": timeframe,
            "params":    params,
            "period":    period,
            "metrics":   metrics,
            "n_trials":  n_trials,
            "git":       self.git,
            **extra,
        }
        entry = _jsonable(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=False) + "\n")
        return entry

    def entries(self, kind: str = None, **match) -> list:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if kind is not None and e.get("kind") != kind:
                    continue
                if all(e.get(k) == v for k, v in match.items()):
                    out.append(e)
        return out

    def holdout_evaluations(self, strategy: str, symbol: str, timeframe: str, holdout_start) -> list:
        hs = pd.Timestamp(holdout_start).isoformat()
        return [e for e in self.entries("holdout", strategy=strategy, symbol=symbol, timeframe=timeframe)
                if e.get("holdout_start") == hs]
