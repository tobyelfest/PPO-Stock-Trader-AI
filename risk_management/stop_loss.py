from abc import ABC, abstractmethod

class IStopLoss(ABC):
    @abstractmethod
    def get_stop_price(self, entry_price: float, current_high: float = None) -> float:
        pass

class FixedStopLoss(IStopLoss):
    def __init__(self, percentage: float = 0.05):
        self.percentage = percentage

    def get_stop_price(self, entry_price: float, current_high: float = None) -> float:
        return entry_price * (1 - self.percentage)

class TrailingStopLoss(IStopLoss):
    def __init__(self, trail_percent: float = 0.03):
        self.trail_percent = trail_percent
        self.highest_price = None

    def get_stop_price(self, entry_price: float, current_high: float = None) -> float:
        if current_high is None:
            return entry_price * (1 - self.trail_percent)
        if self.highest_price is None or current_high > self.highest_price:
            self.highest_price = current_high
        return self.highest_price * (1 - self.trail_percent)