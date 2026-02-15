"""
NVDA 1-MINUTE DATA SCRAPER
===========================
Fetch and clean 1-minute OHLCV bars for NVDA from Alpaca API.

Example:
    python nvda_short_interval_scraper.py
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

base_path = Path(__file__).parent.parent

load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API_KEY or SECRET_KEY in API.env")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

print("Fetching 1-minute NVDA bars (2024-2025)...")
print("This may take 1-2 minutes...")

# 1-minute bars for recent 1 year (manageable size)
request = StockBarsRequest(
    symbol_or_symbols="NVDA",
    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
    start=datetime(2024, 1, 1),
    end=datetime(2025, 2, 14),
    adjustment="all"
)

bars = client.get_stock_bars(request).df

# Clean dataframe
bars = bars.reset_index()
bars = bars[bars["symbol"] == "NVDA"]

# Convert timezone to New York
bars["timestamp"] = pd.to_datetime(bars["timestamp"])
bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")

# Save raw data
bars.to_csv(base_path / "data" / "nvda_1min_2024-2025.csv", index=False)
print(f"Raw data saved: {len(bars)} bars")
print(bars.head())

# Load and clean
df = pd.read_csv(base_path / "data" / "nvda_1min_2024-2025.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

# Keep only regular trading hours (9:30–16:00)
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df = df[((df["hour"] == 9) & (df["minute"] >= 30)) | 
        ((df["hour"] > 9) & (df["hour"] < 16)) |
        ((df["hour"] == 16) & (df["minute"] == 0))]

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# Check for missing candles (1-minute gaps)
df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 60
missing_candles = df[df["time_diff"] != 1.0]
print(f"\nMissing candles found: {len(missing_candles)}")
if len(missing_candles) > 0:
    print(missing_candles[["timestamp", "time_diff"]].head(20))

# Remove helper columns
df = df.drop(columns=["hour", "minute", "time_diff"])

# Save cleaned data
df.to_csv(base_path / "data" / "nvda_1min_2024-2025_cleaned.csv", index=False)
print(f"\nCleaned data saved: {len(df)} bars")
print(df.head())

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time
print("\nTrading hours per day:")
print(df.groupby("date")[["time"]].agg(["min", "max"]))

print("\nData scraping complete!")
