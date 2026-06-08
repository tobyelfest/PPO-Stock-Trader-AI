import pandas as pd
from typing import List, Optional

class DataPreprocessor:
    """Normalize and scale features."""

    def __init__(self, features: List[str], lookback: int = 20):
        self.features = features
        self.lookback = lookback
        self.scalers = {}  # store (mean, std) per column

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Compute scaling parameters on training data."""
        for col in self.features:
            mean = df[col].mean()
            std = df[col].std()
            self.scalers[col] = (mean, std)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        df_norm = df.copy()
        for col in self.features:
            mean, std = self.scalers[col]
            df_norm[col] = (df_norm[col] - mean) / (std + 1e-8)
        return df_norm

    def inverse_transform(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Reverse transform a single column."""
        mean, std = self.scalers[col]
        return df[col] * std + mean

    def add_lagged_features(self, df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        """Add lagged values for given columns."""
        df_lagged = df.copy()
        for col in cols:
            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df_lagged