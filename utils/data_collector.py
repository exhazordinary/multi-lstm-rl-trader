import yfinance as yf

tickers = ["AAPL", "TSLA", "MSFT"]
start, end = "2020-01-01", "2025-01-01"

# Download daily price history
data = yf.download(tickers, start=start, end=end, interval="1d")

# Save individual stock CSVs
for symbol in tickers:
    df = data.xs(symbol, axis=1, level=1)  # Select symbol columns
    df.to_csv(f"data/{symbol}_2020_2025.csv")
    print(f"✅ Saved: data/{symbol}_2020_2025.csv")