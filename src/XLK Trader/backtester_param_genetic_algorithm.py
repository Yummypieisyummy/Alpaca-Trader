import pandas as pd
import numpy as np
import random
from copy import deepcopy

# === Load data ===
df = pd.read_csv("data/xlk_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === Precompute static indicators ===
df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["ema21_slope"] = df["ema21"].diff()
if "atr" not in df.columns:
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
df["vwap_distance"] = df["close"] - df["vwap"]

# === Parameter ranges for GA ===
param_space = {
    "ema_short": range(5, 10),            # slightly faster short EMA
    "ema_mid": range(20, 25),             # faster mid EMA for trend capture
    "ema_long": range(40, 55),            # keep long EMA smooth
    "atr_multiplier": np.linspace(1.8, 2.2, 5),
    "take_profit_multiplier": np.linspace(2.5, 3.0, 6),
    "rsi_low": range(45, 50),
    "rsi_high": range(60, 66),
    "position_size": np.linspace(0.4, 0.5, 3),
    "volume_ratio_thresh": np.linspace(1.5, 2.0, 6)
}

# === Backtest function returning annualized return ===
def backtest_fitness(params, df, initial_capital=100, years=3):
    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    last_trade_hour = None
    trade_count_this_hour = 0
    max_trades_per_hour = 4

    for idx, row in df.iterrows():
        price = row["close"]
        timestamp = row["timestamp"]
        current_hour = timestamp.floor('h')

        if last_trade_hour != current_hour:
            trade_count_this_hour = 0
            last_trade_hour = current_hour

        # Entry condition
        if position == 0 and trade_count_this_hour < max_trades_per_hour:
            trend_confirmed = (row["ema21_slope"] > 0) and (row["vwap_distance"] >= 0.5 * row["atr"])
            buy_cond = (
                trend_confirmed and
                (row["ema9"] > row["ema21"] > row["ema50"]) and
                (price > row["ema21"]) and
                (price > row["vwap"]) and
                (row["volume_ratio"] > params["volume_ratio_thresh"]) and
                (params["rsi_low"] <= row["rsi"] <= params["rsi_high"])
            )
            if buy_cond:
                cash_to_use = capital * params["position_size"]
                position = cash_to_use / price
                capital -= cash_to_use
                entry_price = price
                take_profit = entry_price + params["take_profit_multiplier"] * row["atr"]
                stop_loss = entry_price - params["atr_multiplier"] * row["atr"]
                trade_count_this_hour += 1

        # Exit condition
        if position > 0:
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

    # Final equity and annualized return
    final_equity = capital + (position * df["close"].iloc[-1] if position > 0 else 0)
    annualized_return = (final_equity / initial_capital) ** (1 / years) - 1

    # Fitness: reward annualized growth, lightly penalize drawdown
    equity_series = pd.Series([initial_capital] + np.cumsum(trades))
    drawdowns = (equity_series.cummax() - equity_series) / equity_series.cummax()
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
    fitness_score = annualized_return * 1000 - max_drawdown * 10  # tune weighting if desired

    return fitness_score

# === GA helpers ===
def random_params():
    return {k: random.choice(v) for k, v in param_space.items()}

def mutate_params(params, mutation_rate=0.2):
    new_params = deepcopy(params)
    for k in new_params:
        if random.random() < mutation_rate:
            new_params[k] = random.choice(param_space[k])
    return new_params

def crossover_params(parent1, parent2):
    child = {}
    for k in parent1:
        child[k] = random.choice([parent1[k], parent2[k]])
    return child

# === GA runner ===
def run_ga(df, generations=10, population_size=40):
    population = [random_params() for _ in range(population_size)]
    for gen in range(generations):
        fitness_scores = [backtest_fitness(p, df) for p in population]
        # Sort population by fitness descending
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        population = sorted_pop[:population_size // 2]

        # Refill population via crossover + mutation
        children = []
        while len(children) + len(population) < population_size:
            parents = random.sample(population, 2)
            child = crossover_params(parents[0], parents[1])
            child = mutate_params(child)
            children.append(child)
        population += children

        # Report best fitness
        best_fitness = backtest_fitness(population[0], df)
        print(f"Generation {gen+1} Best Fitness: {best_fitness:.2f}")

    # Return best parameter set
    final_fitness = [backtest_fitness(p, df) for p in population]
    best_idx = np.argmax(final_fitness)
    return population[best_idx]

# === Run GA ===
best_params = run_ga(df, generations=8, population_size=20)
print("=== Optimized Parameters for 3-Year Growth ===")
for k, v in best_params.items():
    print(f"{k}: {v}")