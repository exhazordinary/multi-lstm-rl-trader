import pandas as pd

# === Load and preprocess data ===
df = pd.read_csv("data/AAPL_2020_2025.csv")
from utils.indicators import add_indicators
df = add_indicators(df)
df = df.select_dtypes(include=["number"])

# === Use same split as evaluation ===
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].reset_index(drop=True)

# === Buy & Hold Logic ===
initial_cash = 10_000
buy_price = df_test.loc[0, "Close"]
final_price = df_test.loc[len(df_test) - 1, "Close"]

# Buy as many shares as possible at start
shares_bought = int(initial_cash // buy_price)
remaining_cash = initial_cash - (shares_bought * buy_price)

# Final portfolio value = cash + shares * final price
final_portfolio_value = remaining_cash + (shares_bought * final_price)
total_return = final_portfolio_value - initial_cash

# === Output ===
print("=== Buy & Hold Baseline ===")
print(f"Initial Cash: ${initial_cash:.2f}")
print(f"Buy Price: ${buy_price:.2f}")
print(f"Final Price: ${final_price:.2f}")
print(f"Shares Held: {shares_bought}")
print(f"Remaining Cash: ${remaining_cash:.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Total Return: ${total_return:.2f}")
