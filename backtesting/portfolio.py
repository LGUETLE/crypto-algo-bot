"""
Portfolio backtest: cross-asset time-series momentum (docs/preregistration.md)
==============================================================================
Daily bars, many assets. Index = day (candle open time, 00:00 UTC). Everything that decides a
position on day t+1 uses data up to the CLOSE of day t only:

  membership(t)  point-in-time universe from trailing 30-day quote volume (rank <= 12 to enter,
                 <= 15 to stay), >= 90 bars of history
  signal(t)      TSMOM blend of 7/14/28/56-day lookbacks: 'sign' or 'volscaled'
  weights(t)     inverse volatility, scaled to the portfolio volatility target with the 60-day
                 covariance, per-asset cap, gross cap
  simulate()     weights(t) are traded at the OPEN of day t+1 with a no-trade band; fees, slippage
                 and funding; positions in coins (units)

The single-asset engine (backtesting/engine.py) is not used or changed by this module.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.metrics import calc_all

DEFAULTS = {
    "lookbacks":      [7, 14, 28, 56],
    "vol_window":     60,
    "cov_window":     60,
    "volume_window":  30,
    "min_history":    90,
    "n_enter":        12,
    "n_stay":         15,
    "vol_target":     0.30,
    "cap_asset":      0.25,
    "cap_gross":      2.0,
    "band":           0.05,
    "commission":     0.0005,
    "slippage_major": 0.0002,
    "slippage_other": 0.0005,
    "majors":         ["BTCUSDT", "ETHUSDT"],
    "capital":        10_000.0,
}

VARIANTS = {
    "V1": {"signal": "sign",      "direction": "long_flat"},
    "V2": {"signal": "sign",      "direction": "long_short"},
    "V3": {"signal": "volscaled", "direction": "long_flat"},
    "V4": {"signal": "volscaled", "direction": "long_short"},
}


@dataclass
class Panel:
    open:    pd.DataFrame          # day x symbol
    close:   pd.DataFrame
    volume:  pd.DataFrame          # base volume
    funding: pd.DataFrame          # summed funding rate of events in (open_t, open_t+1]
    base:    dict = field(default_factory=dict)    # symbol -> base coin (dedupe)

    @classmethod
    def from_frames(cls, frames: dict, funding: Optional[pd.DataFrame] = None, base: dict = None):
        idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
        get = lambda col: pd.DataFrame({s: f[col] for s, f in frames.items()}).reindex(idx)
        fund = funding.reindex(index=idx, columns=list(frames)).fillna(0.0) if funding is not None \
            else pd.DataFrame(0.0, index=idx, columns=list(frames))
        return cls(get("open"), get("close"), get("volume"), fund, base or {})

    def cut(self, end) -> "Panel":
        """Rows strictly before `end` (hold-out guard / truncation)."""
        m = self.close.index < pd.Timestamp(end)
        return Panel(self.open[m], self.close[m], self.volume[m], self.funding[m], self.base)


# ── Universe ──────────────────────────────────────────────────────────────────

def membership(panel: Panel, p: dict = DEFAULTS) -> pd.DataFrame:
    """Bool day x symbol: member after the close of day t (trades on t+1)."""
    close, vol = panel.close, panel.volume
    dollar = (vol * close).rolling(p["volume_window"], min_periods=p["volume_window"]).mean()
    history = close.notna().cumsum()
    eligible = (history >= p["min_history"]) & close.notna() & dollar.notna()
    score = dollar.where(eligible)

    members = pd.DataFrame(False, index=close.index, columns=close.columns)
    current: list = []
    cols = list(close.columns)
    for t, row in enumerate(score.to_numpy()):
        valid = ~np.isnan(row)
        order = [cols[j] for j in np.argsort(-np.where(valid, row, -np.inf)) if valid[j]]
        # one symbol per base coin (e.g. a relisting or a 1000x contract): keep the most liquid
        seen, ranked = set(), []
        for s in order:
            b = panel.base.get(s, s)
            if b not in seen:
                seen.add(b)
                ranked.append(s)
        rank = {s: i + 1 for i, s in enumerate(ranked)}
        current = [s for s in current if rank.get(s, 10**9) <= p["n_stay"]]
        for s in ranked:
            if len(current) >= p["n_enter"]:
                break
            if s not in current and panel.base.get(s, s) not in {panel.base.get(c, c) for c in current}:
                current.append(s)
        if current:
            members.iloc[t, [cols.index(s) for s in current]] = True
    return members


# ── Signal ────────────────────────────────────────────────────────────────────

def signal(panel: Panel, kind: str, direction: str, p: dict = DEFAULTS) -> pd.DataFrame:
    logc = np.log(panel.close)
    r = logc.diff()
    sig_d = r.rolling(p["vol_window"], min_periods=p["vol_window"]).std()
    parts = []
    for L in p["lookbacks"]:
        mom = logc - logc.shift(L)
        if kind == "sign":
            parts.append(np.sign(mom))
        elif kind == "volscaled":
            parts.append((mom / (sig_d * np.sqrt(L))).clip(-2, 2) / 2)
        else:
            raise ValueError(kind)
    s = sum(parts) / len(parts)
    if direction == "long_flat":
        s = s.clip(lower=0)
    elif direction != "long_short":
        raise ValueError(direction)
    return s


# ── Weights ───────────────────────────────────────────────────────────────────

def target_weights(panel: Panel, members: pd.DataFrame, sig: pd.DataFrame, p: dict = DEFAULTS) -> pd.DataFrame:
    """Weights decided on the close of each day (notional / equity)."""
    ret = panel.close.pct_change(fill_method=None)
    vol = ret.rolling(p["vol_window"], min_periods=p["vol_window"]).std() * np.sqrt(365)
    W = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    s_arr, v_arr, m_arr = sig.to_numpy(), vol.to_numpy(), members.to_numpy()
    r_arr = ret.to_numpy()
    cw = p["cov_window"]
    for t in range(len(W)):
        live = m_arr[t] & np.isfinite(s_arr[t]) & np.isfinite(v_arr[t]) & (v_arr[t] > 0) & (s_arr[t] != 0)
        if not live.any() or t + 1 < cw:
            continue
        j = np.flatnonzero(live)
        raw = s_arr[t, j] / v_arr[t, j]
        window = pd.DataFrame(r_arr[t - cw + 1:t + 1, j])
        cov = window.cov(min_periods=cw // 2).to_numpy() * 365
        cov = np.where(np.isfinite(cov), cov, 0.0)
        pv = float(np.sqrt(max(raw @ cov @ raw, 0.0)))
        if pv <= 0:
            continue
        w = raw * p["vol_target"] / pv
        w = np.clip(w, -p["cap_asset"], p["cap_asset"])
        gross = np.abs(w).sum()
        if gross > p["cap_gross"]:
            w *= p["cap_gross"] / gross
        W.iloc[t, j] = w
    return W


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class PortfolioResult:
    equity:   pd.Series
    weights:  pd.DataFrame        # realised weights after the open trades (notional / equity)
    trades:   pd.DataFrame        # day, symbol, traded notional, fee, slippage
    costs:    dict
    metrics:  dict


def simulate(panel: Panel, weights: pd.DataFrame, p: dict = DEFAULTS, start=None, end=None,
             capital: float = None) -> PortfolioResult:
    """
    Day t (open time): 1. weights decided on the close of t-1 are traded at the open of t
    (no-trade band; a target of 0 always closes; a coin without a bar on t is closed at its last
    close, i.e. delisted), 2. mark to the close of t, 3. funding of events in (open_t, open_t+1]
    at the close of t. Trading happens only on days in [start, end); the book starts flat.
    """
    capital = capital or p["capital"]
    idx, cols = panel.close.index, list(panel.close.columns)
    mask = np.ones(len(idx), bool)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx < pd.Timestamp(end)
    days = np.flatnonzero(mask)
    O, C = panel.open.to_numpy(float), panel.close.to_numpy(float)
    F = panel.funding.reindex(index=idx, columns=cols).fillna(0.0).to_numpy(float)
    Wr = weights.reindex(index=idx, columns=cols)
    hold = Wr.isna().all(axis=1).to_numpy()          # an all-NaN row = no rebalance that day (hold)
    Wt = Wr.fillna(0.0).to_numpy(float)
    slip = np.array([p["slippage_major"] if c in p["majors"] else p["slippage_other"] for c in cols])
    fee = p["commission"]

    units = np.zeros(len(cols))
    last_px = np.full(len(cols), np.nan)
    E = float(capital)
    eq, realised, trades = [], [], []
    tot = {"fees": 0.0, "slippage": 0.0, "funding": 0.0, "traded": 0.0}

    for t in days:
        o, c = O[t], C[t]
        held = units != 0
        # delisted / missing bar: close at the last known close
        gone = held & ~np.isfinite(o)
        for j in np.flatnonzero(gone):
            notional = abs(units[j]) * last_px[j]
            cost_f, cost_s = notional * fee, notional * slip[j]
            E -= cost_f + cost_s
            tot["fees"] += cost_f; tot["slippage"] += cost_s; tot["traded"] += notional
            trades.append((idx[t], cols[j], -units[j] * last_px[j], cost_f, cost_s))
            units[j] = 0.0
        # overnight move close(t-1) -> open(t)
        ok = np.isfinite(o) & np.isfinite(last_px)
        E += float(np.sum(units[ok] * (o[ok] - last_px[ok])))
        # rebalance at the open
        if t > 0 and E > 0 and not hold[t - 1]:
            tgt = Wt[t - 1]
            cur = np.where(np.isfinite(o), units * np.nan_to_num(o) / E, 0.0)
            for j in np.flatnonzero(np.isfinite(o) & ((tgt != 0) | (units != 0))):
                close_out = tgt[j] == 0 and units[j] != 0
                if not close_out and abs(tgt[j] - cur[j]) <= p["band"]:
                    continue
                new_units = tgt[j] * E / o[j]
                d_notional = (new_units - units[j]) * o[j]
                if d_notional == 0:
                    continue
                cost_f, cost_s = abs(d_notional) * fee, abs(d_notional) * slip[j]
                E -= cost_f + cost_s
                tot["fees"] += cost_f; tot["slippage"] += cost_s; tot["traded"] += abs(d_notional)
                trades.append((idx[t], cols[j], d_notional, cost_f, cost_s))
                units[j] = new_units
        # open -> close, funding
        okc = np.isfinite(c) & np.isfinite(o)
        E += float(np.sum(units[okc] * (c[okc] - o[okc])))
        fund = -float(np.sum(units[okc] * c[okc] * F[t, okc]))
        E += fund
        tot["funding"] += fund
        last_px = np.where(np.isfinite(c), c, last_px)
        if E <= 0:                                   # ruin: nothing left to trade
            E, units[:] = 0.0, 0.0
        eq.append(E)
        realised.append(np.where(np.isfinite(c) & (E > 0), units * np.nan_to_num(c) / max(E, 1e-12), 0.0))

    eq_idx = idx[days]
    equity = pd.Series(eq, index=eq_idx, name="equity")
    tr = pd.DataFrame(trades, columns=["day", "symbol", "notional", "fee", "slippage"])
    years = len(eq_idx) / 365 if len(eq_idx) else 1.0
    gross = (equity.iloc[-1] - capital) + tot["fees"] + tot["slippage"] - tot["funding"] if len(equity) else 0.0
    costs = {**tot, "gross_pnl": gross, "net_pnl": (equity.iloc[-1] - capital) if len(equity) else 0.0,
             "turnover_per_year": tot["traded"] / max(equity.mean(), 1e-12) / years if len(equity) else 0.0}
    metrics = calc_all(equity, pd.DataFrame(), capital, periods_per_year=365) if len(equity) else {}
    return PortfolioResult(equity, pd.DataFrame(realised, index=eq_idx, columns=cols), tr, costs, metrics)


def equal_weight_weights(members: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: equal weights across members, rebalanced on the first day of each month and when
    membership changes; NaN rows (hold) in between."""
    n = members.sum(axis=1)
    W = members.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    changed = members.ne(members.shift(1)).any(axis=1)
    month_start = pd.Series(members.index.month, index=members.index).diff().fillna(1) != 0
    W[~(changed | month_start)] = np.nan
    return W


def variant_weights(panel: Panel, variant: str, members: pd.DataFrame = None, p: dict = DEFAULTS) -> pd.DataFrame:
    v = VARIANTS[variant]
    members = membership(panel, p) if members is None else members
    return target_weights(panel, members, signal(panel, v["signal"], v["direction"], p), p)
