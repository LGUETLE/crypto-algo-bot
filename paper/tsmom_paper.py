"""
Forward paper trading of tsmom-v1 (docs/preregistration.md, Amendment 2)
========================================================================
python -m paper.tsmom_paper --state <dir>      (run daily after 00:00 UTC; GitHub Actions)

Each run:
  1. appends newly CLOSED daily candles (OKX Europe X-Perps, bar 1Dutc) and realised funding of the
     frozen candidate coins to <state>/data/ (an auditable record of the data that was used)
  2. replays the frozen engine (backtesting/portfolio.py, variant V1) from the start date over the
     stored data: equity, positions, trades, costs, funding
  3. writes <state>/summary.json, equity.csv, report.md and appends one line per new day to
     log.jsonl; the target weights decided on the last close are the orders for the next open
  4. optionally sends a Telegram summary (only if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set)

Replaying instead of carrying a mutable book means a run can be repeated without double counting.
No orders are ever sent anywhere: this is paper trading on public data.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.metrics import calc_all                                   # noqa: E402
from backtesting.portfolio import DEFAULTS, Panel, membership, simulate, variant_weights   # noqa: E402
from backtesting.stats import sharpe_standard_error                        # noqa: E402
from backtesting.tsmom_research import TsmomData, benchmarks, params       # noqa: E402
from data import okx                                                       # noqa: E402

logger = logging.getLogger("paper")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_spec(root=ROOT) -> dict:
    with open(os.path.join(root, "config", "tsmom_v1.yaml")) as f:
        spec = yaml.safe_load(f)
    with open(os.path.join(root, "config", "tsmom.yaml")) as f:
        spec["tsmom"] = yaml.safe_load(f)
    with open(os.path.join(root, spec["universe_snapshot"])) as f:
        spec["coins"] = json.load(f)["coins"]
    return spec


# ── Data store (append-only CSV per coin) ─────────────────────────────────────

def _read(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def update_store(state: str, spec: dict, now: pd.Timestamp, fetch_instruments=okx.xperp_instruments,
                 fetch_candles=okx.daily_candles, fetch_funding=okx.funding_history) -> dict:
    """Appends closed candles / funding after the last stored timestamp. Returns coverage info."""
    os.makedirs(os.path.join(state, "data", "candles"), exist_ok=True)
    os.makedirs(os.path.join(state, "data", "funding"), exist_ok=True)
    inst = fetch_instruments().set_index("base")
    first_needed = pd.Timestamp(spec["start"], tz="UTC") - pd.Timedelta(days=spec["history_days"])
    today = now.floor("D")
    cov = {"listed_now": 0, "missing_now": [], "updated": 0}
    for coin in spec["coins"]:
        if coin not in inst.index:
            cov["missing_now"].append(coin)                  # delisted from OKX: no new data
            continue
        cov["listed_now"] += 1
        iid = inst.loc[coin, "inst_id"]
        cpath = os.path.join(state, "data", "candles", f"{coin}.csv")
        old = _read(cpath, ["open", "high", "low", "close", "volume"])
        since = old.index[-1] + pd.Timedelta(days=1) if len(old) else first_needed
        if since < today:
            new = fetch_candles(iid, since)
            new = new[(new.index >= since) & (new.index < today)]          # closed UTC days only
            if len(new):
                pd.concat([old, new]).sort_index().to_csv(cpath)
                cov["updated"] += 1
        fpath = os.path.join(state, "data", "funding", f"{coin}.csv")
        oldf = _read(fpath, ["funding_rate"])
        fsince = oldf.index[-1] + pd.Timedelta(seconds=1) if len(oldf) else first_needed
        newf = fetch_funding(iid, fsince)
        newf = newf[(newf.index >= fsince) & (newf.index <= now)]
        if len(newf):
            pd.concat([oldf, newf.to_frame("funding_rate")]).sort_index().to_csv(fpath)
    return cov


def load_panel(state: str, spec: dict) -> tuple:
    frames, fund = {}, {}
    for coin in spec["coins"]:
        c = _read(os.path.join(state, "data", "candles", f"{coin}.csv"), [])
        if len(c):
            frames[coin] = c.astype(float)
        f = _read(os.path.join(state, "data", "funding", f"{coin}.csv"), [])
        if len(f):
            s = f["funding_rate"].astype(float)
            fund[coin] = s.groupby((s.index - pd.Timedelta("1ns")).floor("D")).sum()   # (open_t, open_t+1]
    panel = Panel.from_frames(frames, base={c: c for c in frames})
    funding = pd.DataFrame(fund).reindex(index=panel.close.index, columns=panel.close.columns).fillna(0.0)
    return Panel(panel.open, panel.close, panel.volume, funding, panel.base), frames


# ── Run ───────────────────────────────────────────────────────────────────────

def compute(state: str, spec: dict) -> dict:
    p = {**params(spec["tsmom"]), "majors": spec["majors"]}
    panel, frames = load_panel(state, spec)
    start = pd.Timestamp(spec["start"], tz="UTC")
    members = membership(panel, p)
    W = variant_weights(panel, spec["variant"], members, p)
    last_close_day = panel.close.index[-1]
    orders_for = last_close_day + pd.Timedelta(days=1)
    targets = {c: round(float(w), 6) for c, w in W.iloc[-1].items() if w != 0}
    out = {"name": spec["name"], "variant": spec["variant"], "start": f"{start:%Y-%m-%d}",
           "last_closed_day": f"{last_close_day:%Y-%m-%d}", "orders_for_open_of": f"{orders_for:%Y-%m-%d}",
           "target_weights": targets, "members": sorted(members.columns[members.iloc[-1]]),
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if last_close_day < start:
        out["status"] = "not started"
        return out
    res = simulate(panel, W, p, start=start)
    eq = res.equity
    m = calc_all(eq, pd.DataFrame(), p["capital"], 365)
    n = len(eq)
    se = sharpe_standard_error(m.get("sharpe", 0.0), max(n, 1), 365) if n > 1 else float("nan")
    btc = frames.get(spec["benchmark"])
    out.update({
        "status": "running", "days": n, "equity": round(float(eq.iloc[-1]), 2),
        "total_return": m["total_return"], "sharpe": m.get("sharpe", 0.0), "sharpe_se": se,
        "sharpe_ci95": [m.get("sharpe", 0.0) - 1.96 * se, m.get("sharpe", 0.0) + 1.96 * se] if n > 1 else None,
        "max_drawdown": m["max_drawdown"], "costs": res.costs,
        "positions": {c: round(float(w), 6) for c, w in res.weights.iloc[-1].items() if abs(w) > 1e-9},
        "trades_last_day": res.trades[res.trades["day"] == eq.index[-1]].assign(
            day=lambda d: d["day"].astype(str)).to_dict(orient="records"),
    })
    if btc is not None and n > 1 and btc.index[0] <= eq.index[0]:
        data = TsmomData(panel, members, btc[["open", "close"]], "okx_xperps", len(spec["coins"]))
        b = benchmarks(data, eq, p["capital"], p)
        out["benchmarks"] = b.reset_index().to_dict(orient="records")
    out["_equity"] = eq
    return out


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return str(x)


def write_outputs(state: str, out: dict, coverage: dict):
    eq = out.pop("_equity", None)
    out["coverage"] = coverage
    with open(os.path.join(state, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    if eq is not None:
        eq.rename("equity").to_csv(os.path.join(state, "equity.csv"))
    log_path = os.path.join(state, "log.jsonl")
    logged = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            logged = {json.loads(line)["last_closed_day"] for line in f if line.strip()}
    if out["last_closed_day"] not in logged:
        entry = {k: out.get(k) for k in ("last_closed_day", "status", "equity", "total_return", "positions",
                                          "trades_last_day", "target_weights", "orders_for_open_of", "generated")}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")
    with open(os.path.join(state, "report.md"), "w") as f:
        f.write(report_md(out))


def report_md(o: dict) -> str:
    lines = [f"# {o['name']} paper trading", "",
             f"Variant {o['variant']} | start {o['start']} | last closed day {o['last_closed_day']} | "
             f"status **{o['status']}** | generated {o['generated']}", ""]
    if o["status"] == "running":
        lo, hi = o["sharpe_ci95"] or (float("nan"), float("nan"))
        c = o["costs"]
        lines += [f"- Equity: {o['equity']:,.2f} USDT ({o['total_return']:+.2%}) over {o['days']} days",
                  f"- Sharpe {o['sharpe']:.2f} +/- {o['sharpe_se']:.2f} (95 % interval {lo:+.2f} to {hi:+.2f})",
                  f"- Max drawdown {o['max_drawdown']:.1%}",
                  f"- Fees {c['fees']:.2f}, slippage {c['slippage']:.2f}, funding {c['funding']:+.2f}, "
                  f"turnover {c['turnover_per_year']:.1f}x / year", ""]
        if o.get("benchmarks"):
            lines += ["| Line | Return | Sharpe | Max DD |", "|---|---:|---:|---:|"]
            lines += [f"| {b['line']} | {b['total_return']:+.2%} | {b['sharpe']:.2f} | {b['max_drawdown']:.1%} |"
                      for b in o["benchmarks"]]
            lines.append("")
    lines += [f"Orders for the open of {o['orders_for_open_of']} (target weights):", ""]
    lines += [f"- {c}: {w:+.4f}" for c, w in sorted(o["target_weights"].items())] or ["- none (flat)"]
    lines += ["", "Statistically, 12 months can only confirm a Sharpe above about 1.64 (RESEARCH.md P2.5)."]
    return "\n".join(lines) + "\n"


def telegram(text: str):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text}, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    spec = load_spec()
    now = pd.Timestamp.now(tz="UTC")
    cov = update_store(args.state, spec, now)
    out = compute(args.state, spec)
    write_outputs(args.state, out, cov)
    msg = (f"{out['name']} {out['last_closed_day']}: {out['status']}"
           + (f", equity {out['equity']:,.0f} ({out['total_return']:+.1%}), Sharpe {out['sharpe']:.2f}"
              if out["status"] == "running" else "")
           + f", {len(out['target_weights'])} targets for {out['orders_for_open_of']}")
    print(msg)
    telegram(msg)


if __name__ == "__main__":
    main()
