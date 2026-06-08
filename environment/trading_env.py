import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from config.config import Config
from risk_management.position_sizing import FixedFractionSizer, IPositionSizer
from risk_management.stop_loss import FixedStopLoss, IStopLoss
from risk_management.take_profit import FixedTakeProfit, ITakeProfit
from environment.reward_engine import RewardEngine

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_capital: float = Config.INITIAL_CAPITAL,
                 transaction_cost_pct: float = Config.TRANSACTION_COST_PCT,
                 lookback_window: int = Config.LOOKBACK_WINDOW,
                 reward_engine: Optional[RewardEngine] = None,
                 position_sizer: Optional[IPositionSizer] = None,
                 stop_loss: Optional[IStopLoss] = None,
                 take_profit: Optional[ITakeProfit] = None):
        super().__init__()
        
        self.df = df.sort_index().copy()
        self.lookback = lookback_window
        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = transaction_cost_pct
        
        self.reward_engine = reward_engine or RewardEngine()
        self.position_sizer = position_sizer or FixedFractionSizer()
        self.stop_loss = stop_loss or FixedStopLoss()
        self.take_profit = take_profit or FixedTakeProfit()
        
        # Pastikan lookback tidak melebihi panjang data
        if len(self.df) <= self.lookback:
            raise ValueError(f"Data terlalu pendek ({len(self.df)} baris) untuk lookback {self.lookback}")
        
        # Fitur: semua kolom numerik kecuali Open, High, Low, Volume (tapi Close tetap dipakai)
        excluded = ['Open', 'High', 'Low', 'Volume']
        self.feature_columns = [c for c in self.df.columns if c not in excluded]
        if 'Close' not in self.feature_columns:
            self.feature_columns.append('Close')
        self.feature_columns = list(set(self.feature_columns))
        
        # Validasi kolom
        missing = [c for c in self.feature_columns if c not in self.df.columns]
        if missing:
            raise KeyError(f"Kolom tidak ditemukan: {missing}")
        
        self.num_features = len(self.feature_columns)
        obs_dim = self.lookback * self.num_features + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        # State internal
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        # Pastikan step dalam batas integer
        current = int(self.current_step)
        start = current - self.lookback + 1
        end = current + 1
        
        # Ambil data dengan iloc (indeks posisi)
        data_window = self.df.iloc[start:end][self.feature_columns].values
        if data_window.shape[0] != self.lookback:
            # Fallback jika ada masalah (misal start negatif)
            data_window = self.df.iloc[-self.lookback:][self.feature_columns].values
        
        obs_flat = data_window.flatten().astype(np.float32)
        
        # Portfolio state
        current_price = self._current_price()
        total_value = self.cash + self.shares * current_price
        cash_ratio = self.cash / (self.initial_capital + 1e-8)
        shares_ratio = (self.shares * current_price) / (total_value + 1e-8) if total_value > 0 else 0.0
        portfolio_state = np.array([cash_ratio, shares_ratio], dtype=np.float32)
        
        obs = np.concatenate([obs_flat, portfolio_state])
        obs = np.nan_to_num(obs, nan=0.0)
        return obs
    
    def _current_price(self) -> float:
        return float(self.df.iloc[int(self.current_step)]['Close'])
    
    def _get_current_equity(self) -> float:
        return self.cash + self.shares * self._current_price()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_value = self._get_current_equity()
        current_price = self._current_price()
        
        # Validasi action
        action = int(action)
        if action == 1:  # BUY
            if self.cash > 0:
                max_shares = self.position_sizer.compute_size(self.cash, current_price)
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares += max_shares
                        self.entry_price = current_price
        elif action == 2:  # SELL
            if self.shares > 0:
                revenue = self.shares * current_price * (1 - self.transaction_cost_pct)
                self.cash += revenue
                self.shares = 0
        
        # Increment step
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False
        
        new_value = self._get_current_equity()
        step_return = (new_value - prev_value) / (prev_value + 1e-8)
        reward = self.reward_engine.compute_reward(new_value, prev_value, step_return)
        
        self.portfolio_history.append(new_value)
        self.returns_history.append(step_return)
        
        obs = self._get_obs()
        info = {
            "portfolio_value": new_value,
            "step": self.current_step,
            "shares": self.shares,
            "cash": self.cash
        }
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = int(self.lookback)
        self.max_step = len(self.df) - 1
        self.cash = float(self.initial_capital)
        self.shares = 0
        self.entry_price = 0.0
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        self.reward_engine.reset()
        obs = self._get_obs()
        info = {"portfolio_value": self.initial_capital}
        return obs, info
    
    def render(self, mode="human"):
        pass
