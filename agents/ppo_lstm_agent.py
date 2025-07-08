import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and preprocess ===
df = pd.read_csv("data/AAPL_2020_2025.csv")
df = add_indicators(df)
df = df.select_dtypes(include=["number"])

# === Wrap env ===
def make_env():
    return TradingEnv(df)

env = DummyVecEnv([make_env])  # SB3 still requires VecEnv

# === Define LSTM-enabled PPO ===
model = PPO(
    policy="MlpLstmPolicy",
    env=env,
    verbose=1,
    n_steps=128,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.01,
    learning_rate=3e-4,
    clip_range=0.2,
    tensorboard_log="./ppo_lstm_tensorboard/"
)

# === Train ===
model.learn(total_timesteps=500_000)

# === Save model ===
model.save("models/ppo_lstm_model")
print("✅ LSTM Model saved to models/ppo_lstm_model.zip")
