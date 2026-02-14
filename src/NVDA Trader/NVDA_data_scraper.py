from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 3 years of 5-minute bars
request = StockBarsRequest(
    symbol_or_symbols="NVDA",
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=datetime(2019, 1, 1),
    end=datetime(2020, 1, 1),
    adjustment="all"
)

bars = client.get_stock_bars(request).df

# Clean dataframe
bars = bars.reset_index()
bars = bars[bars["symbol"] == "NVDA"]

# Convert timezone to New York
bars["timestamp"] = pd.to_datetime(bars["timestamp"])
bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")

# Save to CSV
bars.to_csv("nvda_5min_2019-2020.csv", index=False)

print("Download complete.")
print(bars.head())

# Load the CSV
df = pd.read_csv("nvda_5min_2019-2020.csv")
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

# Check for missing candles (5-minute gaps)
df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 60
missing_candles = df[df["time_diff"] != 5.0]
print(f"\nMissing candles found: {len(missing_candles)}")
if len(missing_candles) > 0:
    print(missing_candles[["timestamp", "time_diff"]].head(10))

# Remove helper columns
df = df.drop(columns=["hour", "minute", "time_diff"])

# Save cleaned data
df.to_csv("nvda_5min_2019-2020_cleaned.csv", index=False)
print("\nCleaning complete.")
print(df.head())

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time
print(df.groupby("date")[["time"]].min())
print(df.groupby("date")[["time"]].max())

