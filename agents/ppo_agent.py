from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv
from utils.indicators import add_indicators
import pandas as pd

class PPOTradingAgent:
    def __init__(self, csv_path: str, train_mode=True, window_size=30, split_ratio=0.8):
        self.train_mode = train_mode
        self.window_size = window_size

        df = pd.read_csv(csv_path)
        df = add_indicators(df)
        df = df.select_dtypes(include=["number"])

        split_idx = int(len(df) * split_ratio)
        if train_mode:
            self.df = df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.df = df.iloc[split_idx:].reset_index(drop=True)

        self.env = DummyVecEnv([
            lambda: TradingEnv(self.df, window_size=window_size, transaction_cost_pct=0.001)
        ])


        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/"
        )

    def train(self, timesteps: int = 100_000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path="models/ppo_trading_model"):
        self.model.save(path)

    def load(self, path="models/ppo_trading_model"):
        self.model = PPO.load(path)

    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)

    def get_env(self):
        return self.env
