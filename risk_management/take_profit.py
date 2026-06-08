from abc import ABC, abstractmethod

class ITakeProfit(ABC):
    @abstractmethod
    def get_target_price(self, entry_price: float) -> float:
        pass

class FixedTakeProfit(ITakeProfit):
    def __init__(self, ratio: float = 2.0):
        self.ratio = ratio

    def get_target_price(self, entry_price: float) -> float:
        return entry_price * (1 + self.ratio)