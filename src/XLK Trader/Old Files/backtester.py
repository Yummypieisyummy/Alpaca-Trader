import pandas as pd

# Load signals
df = pd.read_csv("data/xlk_signals.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === Parameters ===
initial_capital = 10000  # starting cash
position_size = 0.1      # fraction of capital per trade (10%)
stop_loss_pct = 0.005    # 0.5% stop loss
take_profit_pct = 0.01   # 1% take profit

# Initialize variables
capital = initial_capital
position = 0
entry_price = 0
equity_curve = []

for idx, row in df.iterrows():
    price = row["close"]
    signal = row["signal"]

    # --- Entry ---
    if signal == 1 and position == 0:
        cash_to_use = capital * position_size
        position = cash_to_use / price
        capital -= cash_to_use  # remove cash used for the position
        entry_price = price
    
    # --- Exit ---
    if position > 0:
        pnl_pct = (price - entry_price) / entry_price
    
        if signal == -1 or pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
            capital += position * price  # cash from selling
            position = 0
            entry_price = 0
    

    # Track total equity
    total_equity = capital + (position * price if position > 0 else 0)
    equity_curve.append(total_equity)

# Add equity curve to DataFrame
df["equity"] = equity_curve

# Output results
df.to_csv("data/xlk_backtest.csv", index=False)
print(f"Final capital: ${equity_curve[-1]:.2f}")
print("Backtest complete!")