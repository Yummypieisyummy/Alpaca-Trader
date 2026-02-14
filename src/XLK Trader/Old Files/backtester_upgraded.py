import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load signals / features
df = pd.read_csv("data/xlk_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Parameters
initial_capital = 100
position_size = 0.3
atr_multiplier = 1.5          # stop-loss = ATR * multiplier
take_profit_pct = 0.01        # 1% fixed take profit
take_profit_multiplier = 2.75
max_trades_per_hour = 4
ema_short, ema_mid, ema_long = 8, 21, 50
rsi_low, rsi_high = 40, 60
volume_ratio_thresh = 1.5

# Compute ATR if not already present
if "atr" not in df.columns:
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()

# Variables
capital = initial_capital
position = 0
entry_price = 0
stop_loss = 0
equity_curve = []
trades = []
last_trade_hour = None
trade_count_this_hour = 0
buy_times = []
sell_times = []

for idx, row in df.iterrows():
    price = row["close"]
    timestamp = row["timestamp"]
    signal = 0  # default to hold

    # Calculate current hour
    current_hour = timestamp.floor('h')

    # Reset trade counter at new hour
    if last_trade_hour != current_hour:
        trade_count_this_hour = 0
        last_trade_hour = current_hour

    # === Entry Condition ===
    if position == 0 and trade_count_this_hour < max_trades_per_hour:
        # Trend confirmation: both EMA slope and VWAP distance
        vwap_distance = price - row["vwap"]
        trend_confirmed = (
            (row["ema21_slope"] > 0) and          # EMA trending up
            (vwap_distance >= 0.5 * row["atr"])   # Price well above VWAP
        )
        
        if trend_confirmed:
            buy_cond = (
                (row["ema9"] > row["ema21"] > row["ema50"]) and
                (price > row["ema21"]) and
                (price > row["vwap"]) and
                (row["volume_ratio"] > volume_ratio_thresh) and
                (row["rsi"] >= rsi_low) and (row["rsi"] <= rsi_high)
            )
            if buy_cond:
                cash_to_use = capital * position_size
                position = cash_to_use / price
                capital -= cash_to_use
                entry_price = price
                take_profit = entry_price + (row["atr"] * take_profit_multiplier)
                trade_count_this_hour += 1
                buy_times.append((timestamp, price))
                signal = 1

    # === Exit Condition ===
    if position > 0:
        exit_trade = (
            price <= stop_loss or           # ATR-based stop-loss
            price >= take_profit or         # ATR-based take profit
            price < row["vwap"] or
            row["ema9"] < row["ema21"]
        )
        if exit_trade:
            trade_pnl = (price - entry_price) * position
            trades.append(trade_pnl)
            capital += position * price
            position = 0
            entry_price = 0
            stop_loss = 0
            sell_times.append((timestamp, price))
            signal = -1

    total_equity = capital + (position * price if position > 0 else 0)
    equity_curve.append(total_equity)

# === Metrics ===
num_trades = len(trades)
winning_trades = sum(1 for t in trades if t > 0)
losing_trades = sum(1 for t in trades if t <= 0)
win_rate = winning_trades / num_trades if num_trades > 0 else 0
profit_factor = sum(t for t in trades if t > 0) / abs(sum(t for t in trades if t < 0)) if sum(t for t in trades if t < 0) != 0 else float('inf')

equity_series = pd.Series(equity_curve)
rolling_max = equity_series.cummax()
drawdowns = (equity_series - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

print("===== Upgraded Backtest Summary =====")
print(f"Final Capital: ${equity_curve[-1]:.2f}")
print(f"Number of Trades: {num_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")

# === Save equity curve ===
df["equity"] = equity_curve
df.to_csv("data/xlk_backtest_upgraded.csv", index=False)

# === Plot equity curve with buy/sell points ===
plt.figure(figsize=(14,7))
plt.plot(df["timestamp"], equity_curve, label="Equity Curve", color="blue")
plt.scatter(*zip(*buy_times), color="green", marker="^", s=50, label="Buy")
plt.scatter(*zip(*sell_times), color="red", marker="v", s=50, label="Sell")
plt.title("XLK Strategy Equity Curve with Buy/Sell Points")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.show()