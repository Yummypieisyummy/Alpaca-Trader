'''
Features to be engineered:

trend features
momentum features
volume features
volatility features
relative positioning features
'''

import pandas as pd
import ta
from pathlib import Path

# Define base path - go up two levels from NVDA Trader folder
base_path = Path(__file__).parent.parent.parent

# Load your CSV from data folder
df = pd.read_csv(base_path / "data" / "tsla_5min_2019-2025_cleaned.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# === Trend ===
df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)

# === Momentum ===
df["rsi"] = ta.momentum.rsi(df["close"], window=14)
macd = ta.trend.MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# === Volume Strength ===
df["volume_avg20"] = df["volume"].rolling(20).mean()
df["volume_ratio"] = df["volume"] / df["volume_avg20"]

# === Volatility ===
df["atr"] = ta.volatility.average_true_range(
    df["high"], df["low"], df["close"], window=14
)

# === Position Relative to VWAP ===
df["vwap_distance"] = df["close"] - df["vwap"]

# === EMA Slope ===
df["ema21_slope"] = df["ema21"].diff(5)  # slope over last 5 candles

# Drop early NaNs
df = df.dropna()

# Save to data folder (go up to root level)
output_path = base_path / "data" / "tsla_features_19-25.csv"
df.to_csv(output_path, index=False)

print("Feature engineering complete.")
print(f"Saved to: {output_path}")
print(df.head())
