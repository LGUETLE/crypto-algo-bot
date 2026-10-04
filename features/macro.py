"""Higher-timeframe trend filter ("guardian") without look-ahead bias."""
import numpy as np
import pandas as pd


def add_macro_trend(df: pd.DataFrame, macro: pd.DataFrame, macro_tf: str) -> pd.DataFrame:
    """
    Adds `macro_close` and `macro_trend` (+1 price above, -1 below, 0 unknown).

    A macro candle only becomes visible once it has CLOSED (open time + duration),
    so a 1h bar never sees the still-forming daily candle.
    Both frames need a UTC DatetimeIndex (open time of each candle).
    """
    duration  = pd.Timedelta(macro_tf)          # "4h", "1d", "1w" are all valid
    available = pd.Series(macro["close"].values, index=macro.index + duration)
    available = available[~available.index.duplicated(keep="last")].sort_index()

    union       = df.index.union(available.index)
    macro_close = available.reindex(union).ffill().reindex(df.index)

    out = df.copy()
    out["macro_close"] = macro_close
    out["macro_trend"] = np.where(
        macro_close.isna(), 0.0,
        np.where(out["close"] > macro_close, 1.0, -1.0),
    )
    return out
