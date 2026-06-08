import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate raw data for completeness and anomalies."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, str]:
        """Check data quality.

        Returns:
            (is_valid, message)
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            return False, f"Missing columns: {required_cols - set(df.columns)}"

        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            return False, f"Found {missing} missing values"

        # Check negative prices
        if (df[["Open", "High", "Low", "Close"]] < 0).any().any():
            return False, "Negative prices detected"

        # Check volume non-negative
        if (df["Volume"] < 0).any():
            return False, "Negative volume detected"

        # Check High >= Low
        if (df["High"] < df["Low"]).any():
            return False, "High < Low in some rows"

        return True, "Data is valid"

    @staticmethod
    def handle_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """Fill or drop missing values."""
        if method == "ffill":
            return df.ffill().bfill()
        elif method == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")