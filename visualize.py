# visualize.py

from agents.ppo_agent import PPOTradingAgent
import matplotlib.pyplot as plt

# === Initialize PPO agent in evaluation mode ===
agent = PPOTradingAgent(
    csv_path="data/AAPL_2020_2025.csv",
    train_mode=False,
    window_size=30,
    split_ratio=0.8
)
agent.load("models/ppo_trading_model")

# === Get raw environment and dataframe ===
env = agent.get_env().envs[0]  # unwrap DummyVecEnv
df_test = env.df.reset_index(drop=True)
obs, _ = env.reset()

# === Run evaluation and collect logs ===
prices = []
actions = []
portfolio_values = []

done = False
while not done:
    action, _ = agent.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    current_price = df_test.loc[env.current_step - 1, "Close"]
    portfolio_value = env.cash + env.stock_held * current_price

    prices.append(current_price)
    actions.append(action)
    portfolio_values.append(portfolio_value)

# === Plotting results ===
plt.figure(figsize=(14, 6))

# --- Price chart with trade markers ---
plt.subplot(2, 1, 1)
plt.plot(prices, label="Stock Price", color="blue")
buy_plotted = sell_plotted = False

for i, act in enumerate(actions):
    if act == 1:  # Buy
        plt.scatter(i, prices[i], marker="^", color="green", label="Buy" if not buy_plotted else "")
        buy_plotted = True
    elif act == 2:  # Sell
        plt.scatter(i, prices[i], marker="v", color="red", label="Sell" if not sell_plotted else "")
        sell_plotted = True

plt.title("Stock Price with Buy/Sell Points")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()

# --- Portfolio value over time ---
plt.subplot(2, 1, 2)
plt.plot(portfolio_values, label="Portfolio Value", color="purple")
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value ($)")
plt.legend()

plt.tight_layout()
plt.show()
