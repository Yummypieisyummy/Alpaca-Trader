"""
200-EMA AGGRESSIVE BOT - FINAL TUNING
=====================================
Single change only: Widen scale-out targets to let trades breathe

v1 targets: +0.75%/+1.5%  (capture quick moves, many stops)
Final targets: +1.0%/+2.0% (let winners run, reduce stops)

Everything else identical:
- 0.003 entry threshold
- 200-EMA
- -0.5% stop (same tight stop)
- 24/5 trading
- 9% risk per trade

Expected: Fewer stops, better profit factor, similar-or-better returns
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class BacktestFinalTuning:
    """200-EMA final tuning - just adjust scale-out targets"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.peak_capitals = {}
    
    def backtest_symbol(self, symbol, df):
        df = df.copy().reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        print(f"\n{symbol} Final Tuning Backtest:")
        print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Bars: {len(df)}")
        print(f"  Pre-calculating indicators...")
        
        # 200-bar EMA (fast intraday)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 10-bar slope (momentum)
        df['slope'] = df['close'].diff(10)
        
        print(f"  Starting backtest ({len(df)} bars)...")
        
        trades = []
        position_shares = 0
        entry_price = None
        entry_idx = None
        partial_exits = {"first_exit": False}
        
        if symbol not in self.peak_capitals:
            self.peak_capitals[symbol] = self.capital
        
        for i in range(200, len(df)):
            price = df.loc[i, 'close']
            timestamp = df.loc[i, 'timestamp']
            ema_200 = df.loc[i, 'ema_200']
            slope = df.loc[i, 'slope']
            
            if pd.isna(ema_200):
                continue
            
            # Update peak capital
            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], self.capital)
            
            # ORIGINAL AGGRESSIVE SIGNALS (no change)
            long_signal = (price > ema_200 * 1.002) and (slope > 0)
            short_signal = (price < ema_200 * 0.998) and (slope < 0)
            
            # === ENTRY ===
            if long_signal and position_shares == 0:
                position_dollars = self.capital * 0.09  # 9% risk per trade
                shares = int(position_dollars / price)
                if shares > 0 and shares * price <= self.capital * 0.90:
                    cost = shares * price
                    self.capital -= cost
                    position_shares = shares
                    entry_price = price
                    entry_idx = i
                    partial_exits = {"first_exit": False}
            
            # === EXITS ===
            elif position_shares > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
                
                # Scale-out #1: Exit 40% at +1.0% (was +0.75%)
                if not partial_exits["first_exit"] and pnl_pct > 1.0:
                    shares_to_sell = int(position_shares * 0.4)
                    if shares_to_sell > 0:
                        self.capital += shares_to_sell * price
                        position_shares -= shares_to_sell
                        partial_exits["first_exit"] = True
                
                # Scale-out #2: Exit remaining 40% at +2.0% (was +1.5%)
                elif pnl_pct > 2.0 and position_shares > 0:
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
                        'type': 'TARGET'
                    })
                    
                    position_shares = 0
                    entry_price = None
                
                # Tight stop: -0.5% (same)
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
                        'type': 'STOP'
                    })
                    
                    position_shares = 0
                    entry_price = None
                
                # Reverse signal
                elif short_signal and position_shares > 0:
                    exit_price = price
                    pnl = position_shares * (exit_price - entry_price)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
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
    bt = BacktestFinalTuning(initial_capital=10000)
    
    print("=" * 70)
    print("200-EMA AGGRESSIVE BOT - FINAL TUNING")
    print("=" * 70)
    print(f"Initial Capital: ${bt.initial_capital:,.2f}")
    print("\nSingle Change:")
    print("  • Scale-out targets: +0.75%/+1.5% → +1.0%/+2.0%")
    print("  • Rationale: Let winners run, reduce early stops, improve PF")
    print("  • Everything else: IDENTICAL to v1 (200-EMA, 0.003, -0.5% stop)")
    
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
            print(f"\n{symbol} data not found")
    
    # Results
    print("\n" + "=" * 70)
    print("FINAL TUNING RESULTS")
    print("=" * 70)
    print(f"Final Capital: ${bt.capital:,.2f}")
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
        
        print(f"\nWins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")
        
        if any(t['pnl'] < 0 for t in all_trades):
            pos_pnl = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
            neg_pnl = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
            if neg_pnl > 0:
                pf = pos_pnl / neg_pnl
                print(f"Profit Factor: {pf:.2f}")
        
        # Trade type breakdown
        print(f"\nExit Types:")
        for exit_type in ['TARGET', 'STOP', 'REVERSE']:
            type_trades = [t for t in all_trades if t['type'] == exit_type]
            if type_trades:
                type_wins = sum(1 for t in type_trades if t['pnl'] > 0)
                pct = type_wins/len(type_trades)*100 if len(type_trades) > 0 else 0
                print(f"  {exit_type}: {len(type_trades)} trades, {pct:.1f}% win rate")
    
    # Save
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        output_file = "backtest_200ema_final_tuning_results.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON:")
    print("=" * 70)
    print(f"  750-EMA Baseline:         6.73%  | 103 trades   | 43% win")
    print(f"  200-EMA v1 (+0.75%/+1.5%): 13.59%  | 939 trades   | 24.9% win")
    print(f"  200-EMA FinalTune (+1%/+2%): {total_return:6.2f}%  | {len(all_trades):3d} trades | {wins/(wins+losses)*100:.1f}% win")
    print("=" * 70)


if __name__ == "__main__":
    main()
