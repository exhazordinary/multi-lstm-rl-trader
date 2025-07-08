# train.py

from agents.ppo_agent import PPOTradingAgent

# === Initialize PPO agent for training ===
agent = PPOTradingAgent(
    csv_path="data/AAPL_2020_2025.csv",
    train_mode=True,
    window_size=30,
    split_ratio=0.8
)

# === Train the agent ===
agent.train(timesteps=500_000)

# === Save the trained model ===
agent.save("models/ppo_trading_model")

print("✅ PPO agent training complete and model saved.")
