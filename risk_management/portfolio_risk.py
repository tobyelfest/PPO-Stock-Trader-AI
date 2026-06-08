import numpy as np
import pandas as pd

class PortfolioRisk:
    """Compute portfolio-level risk metrics."""

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical VaR."""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Expected shortfall."""
        var = PortfolioRisk.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return drawdown.min()