import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load data ===
df = pd.read_csv("data/xlk_features_24-25.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === Parameters ===
initial_capital = 100
position_size = 0.5
atr_multiplier = 2.0
take_profit_multiplier = 2.5
max_trades_per_hour = 4
ema_short, ema_mid, ema_long = 7, 21, 54
rsi_low, rsi_high = 47, 63
volume_ratio_thresh = 1.8
buffer = 0.005  # 0.5% buffer zone
min_hold_hours = 2  # Minimum 2 hours in regime

# === Precompute indicators (safe, non-trade-altering) ===
df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
df["ema21_slope"] = df["ema21"].diff()

if "atr" not in df.columns:
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()

df["vwap_distance"] = df["close"] - df["vwap"]

# === Initialize variables ===
capital = initial_capital
position = 0
entry_price = 0
stop_loss = 0
take_profit = 0
equity_curve = []
trades = []
buy_times = []
sell_times = []
regime_mode = False  # Track if we're in 200 EMA regime
regime_trades = 0  # Count regime trades
active_trades = 0  # Count active strategy trades

last_trade_hour = None
trade_count_this_hour = 0

# Track regime entry time
regime_entry_time = None

# === Backtest Loop ===
for idx, row in df.iterrows():
    price = row["close"]
    timestamp = row["timestamp"]
    current_hour = timestamp.floor('h')

    # Reset hourly trade counter
    if last_trade_hour != current_hour:
        trade_count_this_hour = 0
        last_trade_hour = current_hour

    # === REGIME 1: Price > 200 EMA with buffer ===
    if price > row["ema200"] * (1 + buffer):
        # Enter if not already long
        if position == 0:
            shares = capital / price
            position = shares
            capital = 0
            entry_price = price
            regime_mode = True
            regime_entry_time = timestamp  # Track when we entered
            buy_times.append((timestamp, price))
            print(f"REGIME ENTRY at {timestamp}: price={price:.2f}, 200EMA={row['ema200']:.2f}")

    # === REGIME 2: Price < 200 EMA with buffer AND minimum hold time ===
    elif price < row["ema200"] * (1 - buffer):
        # Exit regime 1 position if it exists AND we've held long enough
        if regime_mode and position > 0:
            time_in_trade_hours = (timestamp - regime_entry_time).total_seconds() / 3600
            
            if time_in_trade_hours >= min_hold_hours:
                pnl = ((price - entry_price) / entry_price) * 100
                capital = position * price
                position = 0
                entry_price = 0
                regime_mode = False
                sell_times.append((timestamp, price))
                print(f"REGIME EXIT at {timestamp}: price={price:.2f}, 200EMA={row['ema200']:.2f}, PnL={pnl:.2f}, Hours={time_in_trade_hours:.1f}")

    # === Active trading strategy ===
    # === Entry Condition ===
    if position == 0 and trade_count_this_hour < max_trades_per_hour:
        # Trend confirmation filters
        trend_confirmed = (
            row["ema21_slope"] > 0 and
            row["vwap_distance"] >= 0.5 * row["atr"]
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
                shares = (capital * position_size) / price
                position = shares
                capital -= shares * price
                entry_price = price
                take_profit = entry_price + take_profit_multiplier * row["atr"]
                stop_loss = entry_price - atr_multiplier * row["atr"]
                regime_mode = False
                trade_count_this_hour += 1
                active_trades += 1
                buy_times.append((timestamp, price))

    # === Exit Condition (Active Strategy Only) ===
    if position > 0 and not regime_mode:
        exit_trade = (
            price <= stop_loss or
            price >= take_profit or
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
            take_profit = 0
            sell_times.append((timestamp, price))

    # Track equity
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

print("===== Realistic Backtest Summary =====")
print(f"Final Capital: ${equity_curve[-1]:.2f}")
print(f"Number of Trades: {num_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")

print(f"\n=== Trade Breakdown ===")
print(f"Regime Trades: {regime_trades}")
print(f"Active Strategy Trades: {active_trades}")
print(f"Total Trades: {num_trades}")

# === Save equity curve ===
df["equity"] = equity_curve
df.to_csv("data/xlk_backtest_realistic.csv", index=False)

# === Plot equity curve ===
plt.figure(figsize=(14,7))
plt.plot(df["timestamp"], equity_curve, label="Equity Curve", color="blue")
if buy_times:
    plt.scatter(*zip(*buy_times), color="green", marker="^", s=50, label="Buy")
if sell_times:
    plt.scatter(*zip(*sell_times), color="red", marker="v", s=50, label="Sell")
plt.title("XLK Strategy Equity Curve with Buy/Sell Points (Realistic)")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.show()