import pandas as pd

# Load feature-engineered CSV
df = pd.read_csv("data/xlk_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Initialize signal column
df["signal"] = 0  # 0 = Hold, 1 = Buy, -1 = Sell

# --- BUY CONDITIONS ---
buy_cond = (
    (df["ema9"] > df["ema20"]) &
    (df["ema20"] > df["ema50"]) &
    (df["close"] > df["vwap"]) &
    (df["volume_ratio"] > 1.2) &
    (df["rsi"] > 50) & (df["rsi"] < 70)
)

# --- SELL CONDITIONS ---
sell_cond = (
    (df["ema9"] < df["ema20"]) |
    (df["close"] < df["vwap"]) |
    (df["rsi"] > 70)
)

# Apply conditions
df.loc[buy_cond, "signal"] = 1
df.loc[sell_cond, "signal"] = -1

# Save signals CSV
df.to_csv("data/xlk_signals.csv", index=False)

print("Signals generated!")
print(df[["timestamp", "close", "signal"]].tail(20))
