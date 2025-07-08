import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and preprocess multiple assets ===
symbols = ["AAPL", "MSFT", "TSLA"]
dfs = []

for symbol in symbols:
    df = pd.read_csv(f"data/{symbol}_2020_2025.csv")
    df = add_indicators(df)
    df = df.select_dtypes(include=["number"])
    dfs.append(df)

# === Concatenate multi-asset data ===
df_all = pd.concat(dfs, axis=0).reset_index(drop=True)

# === Create vectorized environment ===
env = DummyVecEnv([lambda: TradingEnv(df_all)])

# === Define and train LSTM PPO agent ===
model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=env,
    learning_rate=2.5e-5,
    n_steps=512,
    gae_lambda=0.90,
    ent_coef=0.005,
    clip_range=0.15,
    verbose=1,
    tensorboard_log="./ppo_multi_lstm_tensorboard"
)

# === Train agent ===
model.learn(total_timesteps=200_000)

# === Save model ===
model.save("models/ppo_multi_lstm")
print("✅ LSTM PPO model saved to models/ppo_multi_lstm.zip")
