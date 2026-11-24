"""
OKX Europe X-Perp funding vs Binance USD-M funding over the overlapping period (public data only).
python scripts/compare_funding.py [BTC ETH SOL ...]
Per coin: settlement interval, annualised mean rate (cost of a 1x long, positive = longs pay),
correlation of daily sums, mean absolute daily difference.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import okx                       # noqa: E402
from data.fetcher import DataFetcher       # noqa: E402

logging.basicConfig(level=logging.ERROR)
BINANCE_END = "2026-11-01"                 # last complete month in the Binance archive


def daily(s: pd.Series) -> pd.Series:
    return s.groupby((s.index - pd.Timedelta("1ns")).floor("D")).sum()


def main(coins):
    inst = okx.xperp_instruments().set_index("base")
    print(f"{'Coin':<6} {'OKX since':<11} {'OKX interval':<14} {'Days':>5} {'OKX %/yr':>9} {'Binance %/yr':>13} "
          f"{'Diff %/yr':>10} {'Corr(daily)':>11} {'MAD daily bp':>12}")
    rows = []
    for c in coins:
        i = inst.loc[c]
        o = okx.funding_history(i["inst_id"], i["list_time"])
        b = DataFetcher("binance", use_rest=False).funding_history(f"{c}/USDT:USDT", o.index[0].floor("D"), BINANCE_END)
        iv = o.index.to_series().diff().dropna().value_counts()
        od, bd = daily(o), daily(b)
        days = od.index.intersection(bd.index)
        days = days[(days >= o.index[0].ceil("D")) & (days < pd.Timestamp(BINANCE_END, tz="UTC") - pd.Timedelta(days=1))]
        od, bd = od.reindex(days), bd.reindex(days)
        r = {"coin": c, "since": f"{o.index[0]:%Y-%m-%d}", "interval": str(iv.index[0])[7:12] + f" ({iv.iloc[0] / iv.sum():.0%})",
             "days": len(days), "okx": od.mean() * 365, "binance": bd.mean() * 365,
             "corr": od.corr(bd), "mad_bp": (od - bd).abs().mean() * 1e4}
        rows.append(r)
        print(f"{c:<6} {r['since']:<11} {r['interval']:<14} {r['days']:>5} {r['okx']:>9.2%} {r['binance']:>13.2%} "
              f"{r['okx'] - r['binance']:>+10.2%} {r['corr']:>11.2f} {r['mad_bp']:>12.2f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(sys.argv[1:] or ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "AVAX", "LTC"])
