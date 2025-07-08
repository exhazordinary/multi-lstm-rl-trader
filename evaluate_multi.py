import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and preprocess multi-asset test data ===
symbols = ["AAPL", "MSFT", "TSLA"]
dfs = []

for symbol in symbols:
    df = pd.read_csv(f"data/{symbol}_2020_2025.csv")
    df = add_indicators(df)
    df = df.select_dtypes(include=["number"])
    
    # Split: use last 20% for testing
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    dfs.append(df_test)

# Combine all test data
df_all_test = pd.concat(dfs, axis=0).reset_index(drop=True)

# === Create test environment ===
env = TradingEnv(df_all_test)
obs, _ = env.reset()

# === Load trained model ===
model = RecurrentPPO.load("models/ppo_multi_lstm")

# === Initialize recurrent hidden state ===
lstm_states = None
done = False
total_reward = 0

# === Rollout ===
while not done:
    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=np.array([done]),  # tell model if new episode starts
        deterministic=True
    )
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

# === Summary ===
final_portfolio_value = env.cash + env.stock_held * df_all_test.loc[env.current_step, "Close"]
total_return = final_portfolio_value - env.initial_cash

# === Calculate Sharpe Ratio ===
returns = pd.Series(env.recent_returns)
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0

print("\n✅ LSTM Evaluation Results:")
print(f"Initial Cash: ${env.initial_cash:,.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Total Return: ${total_return:,.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
