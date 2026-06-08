import numpy as np
from typing import List

class RewardEngine:
    """Compute reward signal based on portfolio performance."""

    def __init__(self, reward_scaling: float = 1.0, use_sharpe: bool = True):
        self.reward_scaling = reward_scaling
        self.use_sharpe = use_sharpe
        self.returns_history = []  # store daily returns for sharpe

    def compute_reward(self,
                       current_portfolio_value: float,
                       previous_portfolio_value: float,
                       step_return: float) -> float:
        """Calculate reward for the step.

        Args:
            current_portfolio_value: total value after step
            previous_portfolio_value: total value before step
            step_return: (current - previous)/previous

        Returns:
            reward (scalar)
        """
        if self.use_sharpe:
            self.returns_history.append(step_return)
            if len(self.returns_history) > 20:
                self.returns_history.pop(0)
            if len(self.returns_history) > 1:
                sharpe = np.mean(self.returns_history) / (np.std(self.returns_history) + 1e-8)
                reward = sharpe
            else:
                reward = step_return * 100
        else:
            reward = step_return * 100  # simple return in percent

        # Penalty for negative returns
        if step_return < 0:
            reward *= 0.8
        return reward * self.reward_scaling

    def reset(self):
        self.returns_history = []