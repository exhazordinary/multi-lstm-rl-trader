import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiAssetTradingEnv(gym.Env):
    def __init__(self, dfs: dict, initial_cash=10_000, window_size=30, transaction_cost_pct=0.001):
        super(MultiAssetTradingEnv, self).__init__()

        self.dfs = {symbol: df.reset_index(drop=True) for symbol, df in dfs.items()}
        self.symbols = list(dfs.keys())
        self.num_assets = len(self.symbols)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct

        self.current_step = window_size
        self.cash = initial_cash
        self.stock_held = {symbol: 0 for symbol in self.symbols}
        self.recent_returns = []
        self.volatility_penalty_scale = 0.1

        # Action: 0 = Hold, 1 = Buy, 2 = Sell for each asset
        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)

        # Observation: (window x features x assets) + [cash, holdings]
        sample_df = next(iter(self.dfs.values()))
        self.obs_dim = window_size * sample_df.shape[1] * self.num_assets + self.num_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.stock_held = {symbol: 0 for symbol in self.symbols}
        self.recent_returns = []
        self.trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        obs = []
        for symbol in self.symbols:
            window = self.dfs[symbol].iloc[self.current_step - self.window_size:self.current_step]
            obs.extend(window.values.flatten())

        norm_cash = self.cash / self.initial_cash
        norm_holdings = [self.stock_held[symbol] / 1000 for symbol in self.symbols]

        obs.extend([norm_cash] + norm_holdings)
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        done = False
        info = {}

        prev_prices = {sym: self.dfs[sym].loc[self.current_step - 1, "Close"] for sym in self.symbols}
        prev_portfolio_value = self.cash + sum(
            self.stock_held[sym] * prev_prices[sym] for sym in self.symbols
        )

        for i, action in enumerate(actions):
            symbol = self.symbols[i]
            current_price = self.dfs[symbol].loc[self.current_step, "Close"]

            if action == 1:  # Buy
                cost_per_share = current_price * (1 + self.transaction_cost_pct)
                max_shares = int(self.cash // cost_per_share)
                if max_shares > 0:
                    total_cost = max_shares * cost_per_share
                    self.cash -= total_cost
                    self.stock_held[symbol] += max_shares
                    self.trades.append(("Buy", symbol, self.current_step, current_price))

            elif action == 2:  # Sell
                if self.stock_held[symbol] > 0:
                    sell_price = current_price * (1 - self.transaction_cost_pct)
                    total_revenue = self.stock_held[symbol] * sell_price
                    self.cash += total_revenue
                    self.trades.append(("Sell", symbol, self.current_step, current_price))
                    self.stock_held[symbol] = 0

        self.current_step += 1
        if self.current_step >= len(next(iter(self.dfs.values()))) - 1:
            done = True

        current_prices = {sym: self.dfs[sym].loc[self.current_step, "Close"] for sym in self.symbols}
        portfolio_value = self.cash + sum(
            self.stock_held[sym] * current_prices[sym] for sym in self.symbols
        )

        # === Reward Calculation ===
        step_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100
        self.recent_returns.append(step_return)
        if len(self.recent_returns) > 30:
            self.recent_returns.pop(0)

        volatility_penalty = np.std(self.recent_returns) * self.volatility_penalty_scale
        trade_penalty = -self.transaction_cost_pct * sum(1 for a in actions if a in [1, 2]) * 0.1 * 100

        reward = step_return - volatility_penalty + trade_penalty
        # ==========================

        self.total_profit = portfolio_value - self.initial_cash
        return self._get_observation(), reward, done, False, info

    def render(self):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}")
        for symbol in self.symbols:
            print(f"  {symbol}: Held={self.stock_held[symbol]}")
        print(f"  Total Profit: {self.total_profit:.2f}")
