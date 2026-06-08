from abc import ABC, abstractmethod

class IPositionSizer(ABC):
    @abstractmethod
    def compute_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> int:
        pass

class FixedFractionSizer(IPositionSizer):
    """Kelly-like fixed fraction of capital."""
    def compute_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> int:
        amount = capital * risk_per_trade
        return int(amount // price)

class KellySizer(IPositionSizer):
    """Simplified Kelly criterion (needs win probability and payoff)."""
    def __init__(self, win_prob: float = 0.55, win_loss_ratio: float = 1.5):
        self.win_prob = win_prob
        self.win_loss_ratio = win_loss_ratio

    def compute_size(self, capital: float, price: float, risk_per_trade: float = None) -> int:
        kelly_fraction = (self.win_prob * self.win_loss_ratio - (1 - self.win_prob)) / self.win_loss_ratio
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # cap at 25%
        amount = capital * kelly_fraction
        return int(amount // price)