import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_cash=10_000, window_size=30, transaction_cost_pct=0.001):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

        # FIXED: Flattened observation shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * df.shape[1] + 2,),  # flattened + cash + stock
            dtype=np.float32
        )
        self.recent_returns = []
        self.volatility_penalty_scale = 0.1  # Tune this

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.stock_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        obs = window.values.flatten()

        norm_cash = self.cash / self.initial_cash
        norm_stock = self.stock_held / 1000  # Assume max 1000 shares held for scaling

        obs = np.append(obs, [norm_cash, norm_stock])
        return obs.astype(np.float32)


    def step(self, action):
        done = False
        info = {}

        current_price = self.df.loc[self.current_step, "Close"]
        prev_price = self.df.loc[self.current_step - 1, "Close"]
        prev_portfolio_value = self.cash + self.stock_held * prev_price

        if action == 1:  # Buy
            cost_per_share = current_price * (1 + self.transaction_cost_pct)
            shares_bought = int(self.cash // cost_per_share)

            if shares_bought > 0:
                total_cost = shares_bought * cost_per_share
                self.cash -= total_cost
                self.stock_held += shares_bought
                self.total_shares_bought += shares_bought
                self.trades.append(("Buy", self.current_step, current_price))

        elif action == 2:  # Sell
            if self.stock_held > 0:
                sell_price_per_share = current_price * (1 - self.transaction_cost_pct)
                total_revenue = self.stock_held * sell_price_per_share
                self.cash += total_revenue
                self.total_shares_sold += self.stock_held
                self.trades.append(("Sell", self.current_step, current_price))
                self.stock_held = 0

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        portfolio_value = self.cash + self.stock_held * current_price

        # === Reward calculation ===
        portfolio_value = self.cash + self.stock_held * current_price
        step_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        step_return *= 100

        self.recent_returns.append(step_return)
        if len(self.recent_returns) > 30:
            self.recent_returns.pop(0)

        # Volatility penalty (std of returns)
        volatility_penalty = np.std(self.recent_returns) * self.volatility_penalty_scale

        # Trading penalty (discourage excessive trade)
        trade_penalty = 0
        if action in [1, 2]:
            trade_penalty = -self.transaction_cost_pct * 0.1 * 100

        reward = step_return - volatility_penalty + trade_penalty
        # =====================

        self.total_profit = portfolio_value - self.initial_cash

        return self._get_observation(), reward, done, False, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Cash: {self.cash:.2f}, "
            f"Held: {self.stock_held}, "
            f"Total Profit: {self.total_profit:.2f}"
        )