import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class SupportResistance:
    """Identify local minima (support) and maxima (resistance)."""

    @staticmethod
    def find_local_extrema(df: pd.DataFrame, column: str = "Close", order: int = 5) -> tuple:
        """Find indices of local minima and maxima."""
        prices = df[column].values
        local_min = argrelextrema(prices, np.less, order=order)[0]
        local_max = argrelextrema(prices, np.greater, order=order)[0]
        return local_min, local_max

    def add_sr_features(self, df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
        """Add distance to nearest support/resistance."""
        local_min, local_max = self.find_local_extrema(df, column)
        current_price = df[column].iloc[-1]
        nearest_support = df.iloc[local_min][column].max() if len(local_min) > 0 else None
        nearest_resistance = df.iloc[local_max][column].min() if len(local_max) > 0 else None
        df["dist_to_support"] = (current_price - nearest_support) if nearest_support else np.nan
        df["dist_to_resistance"] = (nearest_resistance - current_price) if nearest_resistance else np.nan
        return df