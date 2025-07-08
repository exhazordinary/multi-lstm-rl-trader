import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib.ppo_recurrent import RecurrentPPO
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

# === Load and prepare test data ===
symbols = ["AAPL", "MSFT", "TSLA"]
dfs = []
symbol_map = []

for symbol in symbols:
    df = pd.read_csv(f"data/{symbol}_2020_2025.csv")
    df = add_indicators(df)
    df = df.select_dtypes(include=["number"])
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    dfs.append(df_test)
    symbol_map.extend([symbol] * len(df_test))

# Combine all
df_all = pd.concat(dfs, axis=0).reset_index(drop=True)

# === Create environment ===
env = TradingEnv(df_all)
obs, _ = env.reset()

# === Load model ===
model = RecurrentPPO.load("models/ppo_multi_lstm")
lstm_state = None
done = False

# === Logs ===
prices = []
actions = []
portfolio_values = []
symbols_traded = []

while not done:
    action, lstm_state = model.predict(
        obs,
        state=lstm_state,
        episode_start=np.array([done]),
        deterministic=True
    )
    obs, reward, done, _, _ = env.step(action)

    current_price = df_all.loc[env.current_step - 1, "Close"]
    symbol_now = symbol_map[env.current_step - 1]
    portfolio_value = env.cash + env.stock_held * current_price

    prices.append(current_price)
    actions.append(int(action))
    portfolio_values.append(portfolio_value)
    symbols_traded.append(symbol_now)

# === Plotting ===
plt.figure(figsize=(16, 8))

# --- Price + Trade Points ---
plt.subplot(2, 1, 1)
plt.plot(prices, label="Price", color="blue", linewidth=1)
for i, act in enumerate(actions):
    if act == 1:  # Buy
        plt.scatter(i, prices[i], marker="^", color="green", label="Buy" if "Buy" not in plt.gca().get_legend_handles_labels()[1] else "")
    elif act == 2:  # Sell
        plt.scatter(i, prices[i], marker="v", color="red", label="Sell" if "Sell" not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title("Stock Price and Trades (Across All Symbols)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()

# --- Portfolio Value ---
plt.subplot(2, 1, 2)
plt.plot(portfolio_values, label="Portfolio Value", color="purple")
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value ($)")
plt.legend()

plt.tight_layout()
plt.show()
