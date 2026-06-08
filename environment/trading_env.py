import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from config.config import Config
from risk_management.position_sizing import FixedFractionSizer, IPositionSizer
from risk_management.stop_loss import FixedStopLoss, IStopLoss
from risk_management.take_profit import FixedTakeProfit, ITakeProfit
from environment.reward_engine import RewardEngine

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 df: pd.DataFrame,
                 stocks: list,
                 initial_capital: float = Config.INITIAL_CAPITAL,
                 transaction_cost_pct: float = Config.TRANSACTION_COST_PCT,
                 lookback_window: int = Config.LOOKBACK_WINDOW,
                 reward_engine: Optional[RewardEngine] = None,
                 position_sizer: Optional[IPositionSizer] = None,
                 stop_loss: Optional[IStopLoss] = None,
                 take_profit: Optional[ITakeProfit] = None):
        super().__init__()
        self.df = df.sort_index()
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback = lookback_window

        self.reward_engine = reward_engine or RewardEngine()
        self.position_sizer = position_sizer or FixedFractionSizer()
        self.stop_loss = stop_loss or FixedStopLoss()
        self.take_profit = take_profit or FixedTakeProfit()

        # Hitung jumlah hari unik
        self.max_step = len(self.df.index.get_level_values(0).unique()) - self.lookback - 1
        self.current_step = self.lookback

        # Define spaces
        price_shape = (self.lookback, self.num_stocks)
        indicator_shape = (self.lookback, len(Config.INDICATORS))
        portfolio_shape = (1 + self.num_stocks,)
        risk_shape = (2,)

        self.observation_space = spaces.Dict({
            "prices": spaces.Box(low=-np.inf, high=np.inf, shape=price_shape, dtype=np.float32),
            "indicators": spaces.Box(low=-np.inf, high=np.inf, shape=indicator_shape, dtype=np.float32),
            "portfolio": spaces.Box(low=0, high=np.inf, shape=portfolio_shape, dtype=np.float32),
            "risk": spaces.Box(low=-np.inf, high=np.inf, shape=risk_shape, dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)
        
        # State awal
        self.cash = self.initial_capital
        self.holdings = {stock: 0 for stock in self.stocks}
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []

    def _get_observation(self) -> Dict[str, np.ndarray]:
        idx = self.current_step - self.lookback + 1
        
        # Get Price & Indicator Data
        prices_df = self.df["Close"].unstack(level="Ticker")
        prices = prices_df.iloc[idx: self.current_step+1].values
        norm_prices = prices / (prices[0:1] + 1e-8)

        indicator_cols = [c for c in self.df.columns if c in Config.INDICATORS]
        indicator_data = self.df[indicator_cols].unstack(level="Ticker").iloc[idx: self.current_step+1].values

        # Portfolio state
        portfolio = np.array([self.cash] + [self.holdings[s] for s in self.stocks], dtype=np.float32)

        # Risk metrics
        history = np.array(self.portfolio_history, dtype=np.float32)
        running_max = np.maximum.accumulate(history)
        current_val = history[-1]
        
        drawdown = (current_val - running_max[-1]) / (running_max[-1] + 1e-8)
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.0
        
        risk = np.array([drawdown, volatility], dtype=np.float32)

        return {
            "prices": norm_prices.astype(np.float32),
            "indicators": indicator_data.astype(np.float32),
            "portfolio": portfolio,
            "risk": risk,
        }

    def _get_current_equity(self) -> float:
        prices_df = self.df["Close"].unstack(level="Ticker")
        current_prices = prices_df.iloc[self.current_step].values
        holdings_array = np.array([self.holdings[s] for s in self.stocks])
        return float(self.cash + np.sum(holdings_array * current_prices))

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        prev_val = self._get_current_equity()
        
        # Eksekusi aksi
        reward, new_val = self._take_action(action)
        
        # Update history
        self.portfolio_history.append(new_val)
        self.returns_history.append((new_val - prev_val) / (prev_val + 1e-8))

        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False

        obs = self._get_observation()
        info = {"portfolio_value": new_val, "step": self.current_step}

        return obs, reward, terminated, truncated, info

    def _take_action(self, actions: np.ndarray) -> Tuple[float, float]:
        prices_df = self.df["Close"].unstack(level="Ticker")
        current_prices = prices_df.iloc[self.current_step].values
        prev_value = self._get_current_equity()

        for i, stock in enumerate(self.stocks):
            action = actions[i]
            price = current_prices[i]
            if action == 1:  # buy
                size = self.position_sizer.compute_size(self.cash, price)
                cost = size * price * (1 + self.transaction_cost_pct)
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[stock] += size
            elif action == 2:  # sell
                if self.holdings[stock] > 0:
                    revenue = self.holdings[stock] * price * (1 - self.transaction_cost_pct)
                    self.cash += revenue
                    self.holdings[stock] = 0

        new_value = self._get_current_equity()
        step_return = (new_value - prev_value) / (prev_value + 1e-8)
        reward = self.reward_engine.compute_reward(new_value, prev_value, step_return)
        return float(reward), float(new_value)

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        # WAJIB: Panggil super().reset untuk Gymnasium
        super().reset(seed=seed)
        
        self.current_step = self.lookback
        self.cash = self.initial_capital
        self.holdings = {stock: 0 for stock in self.stocks}
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        if self.reward_engine:
            self.reward_engine.reset()
            
        obs = self._get_observation()
        info = {"portfolio_value": self.initial_capital}
        
        # WAJIB: Kembalikan (observasi, info)
        return obs, info

    def render(self, mode="human"):
        pass