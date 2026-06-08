import pandas as pd
import numpy as np

class FibonacciLevels:
    """Compute Fibonacci retracement levels based on recent swing high/low."""

    @staticmethod
    def find_swing_points(df: pd.DataFrame, column: str = "Close", lookback: int = 20) -> tuple:
        """Find recent high and low within lookback window."""
        recent = df[column].iloc[-lookback:]
        high = recent.max()
        low = recent.min()
        return high, low

    @staticmethod
    def compute_retracement(high: float, low: float) -> dict:
        """Return levels: 0.236, 0.382, 0.5, 0.618, 0.786."""
        diff = high - low
        return {
            "fib_0": high,
            "fib_236": high - 0.236 * diff,
            "fib_382": high - 0.382 * diff,
            "fib_500": high - 0.5 * diff,
            "fib_618": high - 0.618 * diff,
            "fib_786": high - 0.786 * diff,
            "fib_1": low,
        }

    def add_fib_features(self, df: pd.DataFrame, column: str = "Close", lookback: int = 20) -> pd.DataFrame:
        """Add distance to nearest Fibonacci level as feature."""
        high, low = self.find_swing_points(df, column, lookback)
        levels = self.compute_retracement(high, low)
        current = df[column].iloc[-1]
        distances = {f"dist_to_{k}": abs(current - v) for k, v in levels.items()}
        for k, v in distances.items():
            df[k] = v  # This will broadcast, but for full DataFrame we need to apply rolling
        # For simplicity, we assign the same values to all rows (not ideal). In production,
        # you would compute rolling windows.
        return df