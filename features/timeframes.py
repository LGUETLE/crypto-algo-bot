"""ccxt timeframe strings ("15m", "1h", "4h", "1d", "1w") -> pandas Timedelta."""
import re

import pandas as pd

_UNITS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def to_timedelta(timeframe: str) -> pd.Timedelta:
    m = re.fullmatch(r"(\d+)([mhdw])", timeframe)
    if m:
        return pd.Timedelta(int(m.group(1)), _UNITS[m.group(2)])
    return pd.Timedelta(timeframe)          # already a pandas string such as "15min"
