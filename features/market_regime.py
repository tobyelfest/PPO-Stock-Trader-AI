import pandas as pd
import numpy as np

class MarketRegime:
    """Detect bull/bear/sideways regime using moving average crossovers and volatility."""

    @staticmethod
    def detect_regime(df: pd.DataFrame, short_ma: int = 20, long_ma: int = 50) -> pd.Series:
        """Return regime label for each row: 0 (bear), 1 (sideways), 2 (bull)."""
        sma_short = df["Close"].rolling(short_ma).mean()
        sma_long = df["Close"].rolling(long_ma).mean()
        # Bull: short > long
        bull = (sma_short > sma_long).astype(int)
        # Bear: short < long
        bear = (sma_short < sma_long).astype(int) * -1
        regime = bull + bear  # -1, 0, 1
        # Map to 0,1,2
        regime_map = {-1: 0, 0: 1, 1: 2}
        return regime.map(regime_map).fillna(1).astype(int)

    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        df["regime"] = MarketRegime.detect_regime(df)
        # Add rolling volatility
        df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()
        return df