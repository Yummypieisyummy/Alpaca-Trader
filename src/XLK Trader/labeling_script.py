import pandas as pd
import numpy as np

# Load your feature-engineered CSV
df = pd.read_csv("data/xlk_backtest_realistic.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Parameters
future_candles = 6       # look ahead 6 candles (~30 minutes)
threshold_pct = 0.002    # 0.2% price change

labels = []

for i in range(len(df)):
    if i + future_candles < len(df):
        future_close = df.loc[i+1:i+future_candles, "close"].max()
        future_low = df.loc[i+1:i+future_candles, "close"].min()
        current_close = df.loc[i, "close"]

        up_move = (future_close - current_close) / current_close
        down_move = (future_low - current_close) / current_close

        if up_move >= threshold_pct:
            label = 1   # price will go up
        elif down_move <= -threshold_pct:
            label = -1  # price will go down
        else:
            label = 0   # price stays flat
    else:
        label = 0  # last few candles cannot look ahead

    labels.append(label)

df["label"] = labels

# Save labeled dataset
df.to_csv("data/xlk_labeled.csv", index=False)

print("Labeling complete! Sample:")
print(df[["timestamp", "close", "label"]].tail(10))