import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and preprocess dataset ===
df_full = pd.read_csv("data/AAPL_2020_2025.csv")
df_full = add_indicators(df_full)
df_full = df_full.select_dtypes(include=["number"])

# === Split into train/test ===
split_idx = int(len(df_full) * 0.8)
df_test = df_full.iloc[split_idx:].reset_index(drop=True)

# === Load trained PPO model ===
model = PPO.load("models/ppo_lstm_model")

# === Create environment ===
env = TradingEnv(df_test)
obs, _ = env.reset()

done = False
portfolio_values = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    # Track portfolio value
    current_price = df_test.loc[env.current_step - 1, "Close"]
    value = env.cash + env.stock_held * current_price
    portfolio_values.append(value)

# === Calculate performance ===
initial_cash = env.initial_cash
final_value = portfolio_values[-1]
total_return = final_value - initial_cash

# === Calculate Sharpe Ratio ===
returns = np.diff(portfolio_values) / portfolio_values[:-1]  # daily returns
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0

# === Output ===
print("\n✅ Evaluation Results:")
print(f"Initial Cash: ${initial_cash:.2f}")
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
