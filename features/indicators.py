import pandas as pd
import ta  # technical analysis library
from typing import List, Dict, Any

class TechnicalIndicators:
    """Compute various technical indicators."""

    @staticmethod
    def add_sma(df: pd.DataFrame, column: str = "Close", windows: List[int] = [20, 50]) -> pd.DataFrame:
        for w in windows:
            df[f"sma_{w}"] = ta.trend.sma_indicator(df[column], window=w)
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, column: str = "Close", windows: List[int] = [12, 26]) -> pd.DataFrame:
        for w in windows:
            df[f"ema_{w}"] = ta.trend.ema_indicator(df[column], window=w)
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.DataFrame:
        df["rsi_14"] = ta.momentum.rsi(df[column], window=window)
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
        macd = ta.trend.MACD(df[column])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, column: str = "Close", window: int = 20, std: int = 2) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(df[column], window=window, window_dev=std)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=window)
        return df

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls.add_sma(df)
        df = cls.add_ema(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_atr(df)
        return df