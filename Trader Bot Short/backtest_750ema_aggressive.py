"""
AGGRESSIVE 200-EMA BOT BACKTEST (1-MINUTE)
============================================
High-frequency bidirectional trading strategy.
Target: 15-25% annual return through:
- 200-bar EMA (faster signals vs 750-bar)
- Tight 0.2% entry buffer
- 0.75% / 1.5% scale-outs
- 24/5 trading (no market hour restrictions)
- Mean reversion (short) opportunities
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class BacktestAggressiveBot:
    """Backtest for aggressive 200-EMA bot"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.entry_prices = {}
        self.peak_capitals = {}
        
        # Aggressive strategy metrics
        self.win_rate = 0.45
        self.avg_win = 25.0
        self.avg_loss = 12.0
        self.profit_factor = self.avg_win / self.avg_loss
    
    def calculate_slope(self, data, periods=20):
        if len(data) < periods:
            return 0.0
        recent = data.iloc[-periods:].values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def calculate_atr_percentile(self, bars, periods=14, lookback=100):
        if len(bars) < periods:
            return 1.0, 50.0
        
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean()
        
        current_atr = atr.iloc[-1]
        if np.isnan(current_atr):
            current_atr = 1.0
        
        atr_hist = atr.tail(lookback).dropna()
        if len(atr_hist) < 2:
            percentile = 50.0
        else:
            percentile = (atr_hist < current_atr).sum() / len(atr_hist) * 100
        
        return current_atr, percentile
    
    def calculate_position_size(self, capital, atr, atr_percentile, slope, current_drawdown, equity_high):
        """Aggressive Kelly Criterion"""
        kelly_fraction = (self.profit_factor * self.win_rate - (1 - self.win_rate)) / self.profit_factor
        # Use full Kelly + 20% boost for aggressive strategy
        base_position = capital * (kelly_fraction * 1.2)
        
        # Volatility: more aggressive in calm markets
        if atr_percentile > 75:
            base_position *= 0.8
        elif atr_percentile < 25:
            base_position *= 1.5  # Really boost on low volatility
        
        # Slope momentum
        if slope > 0.005:
            base_position *= 1.3
        elif slope < -0.005:
            base_position *= 1.2
        
        # Growth bonus
        if capital > equity_high * 1.05:
            base_position *= 1.25
        
        # Drawdown protection
        if current_drawdown < -0.10:
            base_position *= 0.6
        elif current_drawdown < -0.20:
            base_position *= 0.3
        
        return max(base_position, 0.0)
    
    def backtest_symbol(self, symbol, df):
        """Run aggressive backtest on one symbol"""
        df = df.copy().reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            print(f"  ERROR: Missing columns")
            return None
        
        print(f"\n{symbol} Aggressive Backtest:")
        print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Bars: {len(df)}")
        print(f"  Pre-calculating indicators...")
        
        # FAST 200-EMA instead of 750
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['ema_500'] = df['close'].ewm(span=500, adjust=False).mean()
        
        # Slope on recent 10 bars (tight momentum)
        def calc_slope(series, periods=10):
            if len(series) < periods:
                return 0.0
            recent = series.values[-periods:]
            x = np.arange(len(recent))
            return np.polyfit(x, recent, 1)[0]
        
        df['slope_10'] = df['ema_200'].rolling(10).apply(lambda x: calc_slope(x, 10) if len(x) >= 10 else 0.0, raw=False)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        df['atr_percentile'] = df['atr'].rolling(100).apply(
            lambda x: (x.iloc[-1] > x[:-1]).sum() / len(x[:-1]) * 100 if len(x) > 1 else 50.0, raw=False
        )
        
        print(f"  Starting backtest loop ({len(df)} bars)...")
        
        trades = []
        position_shares = 0
        entry_price = None
        entry_idx = None
        partial_exits = {"first_exit": False, "second_exit": False}
        
        # Start from bar 200 (need full EMA)
        for i in range(200, len(df)):
            price = df.loc[i, 'close']
            timestamp = df.loc[i, 'timestamp']
            ema_200 = df.loc[i, 'ema_200']
            slope_10 = df.loc[i, 'slope_10']
            atr = df.loc[i, 'atr']
            atr_percentile = df.loc[i, 'atr_percentile']
            
            if pd.isna(ema_200) or pd.isna(slope_10) or pd.isna(atr):
                continue
            
            # Update peak
            if symbol not in self.peak_capitals:
                self.peak_capitals[symbol] = self.capital
            else:
                self.peak_capitals[symbol] = max(self.peak_capitals[symbol], self.capital)
            
            current_drawdown = (self.capital - self.peak_capitals[symbol]) / self.peak_capitals[symbol] if self.peak_capitals[symbol] > 0 else 0
            
            # AGGRESSIVE: Tight buffer for fast entries
            tight_buffer = 0.002  # 0.2%
            entry_threshold = 0.003  # Very loose threshold
            
            # LONG: Above EMA + positive slope
            long_signal = (price > ema_200 * (1 + tight_buffer)) and (slope_10 > entry_threshold)
            
            # SHORT: Below EMA + negative slope (mean reversion)
            short_signal = (price < ema_200 * (1 - tight_buffer)) and (slope_10 < -entry_threshold)
            
            # ENTRY
            if long_signal and position_shares == 0:
                position_dollars = self.calculate_position_size(
                    self.capital, atr, atr_percentile, slope_10, current_drawdown, self.peak_capitals[symbol]
                )
                if position_dollars > 0 and position_dollars <= self.capital * 0.80:
                    shares = int(position_dollars / price)
                    if shares > 0:
                        cost = shares * price
                        self.capital -= cost
                        position_shares = shares
                        entry_price = price
                        entry_idx = i
                        partial_exits = {"first_exit": False, "second_exit": False}
            
            # EXITS
            elif position_shares > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
                
                # Scale-out #1: Exit 40% at +0.75%
                if not partial_exits["first_exit"] and pnl_pct > 0.75:
                    shares_to_sell = int(position_shares * 0.4)
                    if shares_to_sell > 0:
                        self.capital += shares_to_sell * price
                        position_shares -= shares_to_sell
                        partial_exits["first_exit"] = True
                
                # Scale-out #2: Exit another 40% at +1.5%
                elif not partial_exits["second_exit"] and pnl_pct > 1.5:
                    shares_to_sell = int(position_shares * 0.667)
                    if shares_to_sell > 0:
                        self.capital += shares_to_sell * price
                        position_shares -= shares_to_sell
                        partial_exits["second_exit"] = True
                
                # Stop loss at -0.5%
                elif pnl_pct < -0.5 and position_shares > 0:
                    exit_price = price
                    pnl = position_shares * (exit_price - entry_price)
                    self.capital += position_shares * exit_price
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position_shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_idx,
                        'type': 'STOP_LOSS'
                    })
                    
                    position_shares = 0
                    entry_price = None
                
                # Reverse signal
                elif short_signal and position_shares > 0:
                    exit_price = price
                    pnl = position_shares * (exit_price - entry_price)
                    self.capital += position_shares * exit_price
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position_shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_idx,
                        'type': 'REVERSE'
                    })
                    
                    position_shares = 0
                    entry_price = None
            
            # Update equity
            if position_shares > 0:
                self.equity = self.capital + position_shares * price
            else:
                self.equity = self.capital
            
            self.peak_equity = max(self.peak_equity, self.equity)
            
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1}/{len(df)} bars...")
        
        print(f"  Completed {len(df)} bars")
        
        # Close final position
        if position_shares > 0:
            final_price = df.loc[df.index[-1], 'close']
            pnl = position_shares * (final_price - entry_price)
            pnl_pct = (final_price - entry_price) / entry_price * 100
            self.capital += position_shares * final_price
            
            trades.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position_shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'bars_held': len(df) - entry_idx,
                'type': 'END_OF_DATA'
            })
        
        return trades


def main():
    data_dir = Path("data")
    bt = BacktestAggressiveBot(initial_capital=10000)
    
    print("=" * 70)
    print("AGGRESSIVE 200-EMA BOT BACKTEST (1-MINUTE)")
    print("=" * 70)
    print(f"Initial Capital: ${bt.initial_capital:,.2f}")
    
    all_trades = []
    
    symbols_files = [
        ("NVDA", "nvda_1min_2024-2025_cleaned.csv"),
        ("TSLA", "tsla_1min_2024-2025_cleaned.csv"),
        ("XLK", "xlk_1min_2024-2025_cleaned.csv"),
    ]
    
    for symbol, filename in symbols_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\nLoading {filepath}...")
            df = pd.read_csv(filepath)
            trades = bt.backtest_symbol(symbol, df)
            if trades:
                all_trades.extend(trades)
                wins = sum(1 for t in trades if t['pnl'] > 0)
                print(f"  Trades: {len(trades)}")
                print(f"  Win Rate: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
        else:
            print(f"\n{symbol} data not found at {filepath}")
    
    # Results
    print("\n" + "=" * 70)
    print("AGGRESSIVE BOT RESULTS")
    print("=" * 70)
    print(f"Final Capital: ${bt.capital:,.2f}")
    print(f"Final Equity: ${bt.equity:,.2f}")
    print(f"Peak Equity: ${bt.peak_equity:,.2f}")
    
    total_return = (bt.equity - bt.initial_capital) / bt.initial_capital * 100
    max_drawdown = (bt.peak_equity - bt.equity) / bt.peak_equity * 100 if bt.peak_equity > 0 else 0
    
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {len(all_trades)}")
    
    if all_trades:
        wins = sum(1 for t in all_trades if t['pnl'] > 0)
        losses = sum(1 for t in all_trades if t['pnl'] <= 0)
        avg_win = np.mean([t['pnl'] for t in all_trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in all_trades if t['pnl'] <= 0]) if losses > 0 else 0
        total_pnl = sum(t['pnl'] for t in all_trades)
        
        print(f"\nWins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {abs(sum(t['pnl'] for t in all_trades if t['pnl'] > 0) / sum(t['pnl'] for t in all_trades if t['pnl'] < 0)):.2f}" if any(t['pnl'] < 0 for t in all_trades) else "All wins!")
    
    # Save
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        output_file = "backtest_aggressive_200ema_results.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("vs Baseline (6.73%): This aggressive version aims for 15-25%")
    print("=" * 70)


if __name__ == "__main__":
    main()
