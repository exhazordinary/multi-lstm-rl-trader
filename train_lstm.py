import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and prepare data ===
df = pd.read_csv("data/AAPL_2020_2025.csv")
df = add_indicators(df)
df = df.select_dtypes(include=["number"])  # Keep only numeric columns

# === Wrap env with DummyVecEnv ===
env = DummyVecEnv([lambda: TradingEnv(df)])

# === Define PPO-LSTM agent ===
model = RecurrentPPO(
    policy="MlpLstmPolicy",     # ← LSTM-enabled policy
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

# === Train the agent ===
model.learn(total_timesteps=500_000)

# === Save the model ===
model.save("models/ppo_lstm_model")
print("✅ PPO-LSTM model saved to models/ppo_lstm_model.zip")
