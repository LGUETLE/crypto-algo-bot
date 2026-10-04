from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Must create df["signal"] (1 long entry, -1 short entry, 0 nothing);
        may create boolean df["long_exit"] / df["short_exit"]."""

    def apply_macro_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guardian: block entries against the higher-timeframe trend.
        macro_trend == 0 (no data) lets everything through."""
        if self.config.get("use_macro_filter", True) and "macro_trend" in df.columns:
            df.loc[(df["signal"] == 1)  & (df["macro_trend"] < 0), "signal"] = 0
            df.loc[(df["signal"] == -1) & (df["macro_trend"] > 0), "signal"] = 0
        return df

    def get_latest_signal(self, df: pd.DataFrame) -> int:
        df = self.generate_signals(self.add_indicators(df.copy()))
        return int(df["signal"].iloc[-1])

    def should_exit(self, df: pd.DataFrame, side: str) -> bool:
        """Exit flag of the last candle. NaN counts as False (bool(nan) would be True!)."""
        df = self.generate_signals(self.add_indicators(df.copy()))
        col = "long_exit" if side == "long" else "short_exit"
        if col not in df.columns:
            return False
        return bool(df[col].fillna(False).iloc[-1])
