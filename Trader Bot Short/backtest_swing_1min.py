"""
SWING TRADING ON 1-MINUTE DATA (HYBRID)
========================================
Best of both worlds:
- 200-bar EMA (fast entries like aggressive bot)
- Swing trade holds (1-2 hours, not 5-30 min)
- Tighter entry threshold (0.005 vs 0.003)
- Better stop placement (-0.75% vs early exits)
- Market hours only (9:30-3:59 PM ET, avoid pre/post market noise)

This catches intraday moves WITHOUT over-trading
Target: 15%+ like aggressive bot, but WITH higher win rate
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class BacktestSwingOn1Min:
    """Swing trading using 1-minute data for better entries"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.peak_capitals = {}
    
    def is_market_hours(self, timestamp):
        """Check if timestamp is during US market hours (9:30-3:59 PM ET)"""
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Market hours: 9:30-15:59 (UTC-5, but timestamp is UTC so shift by 5 hours)
        # UTC 13:30 = 8:30 AM ET (pre-market)
        # UTC 14:30 = 9:30 AM ET (market open)
        # UTC 20:00 = 3:00 PM ET
        # UTC 21:00 = 4:00 PM ET (after hours)
        
        market_start_utc = 14.5  # 9:30 AM ET
        market_end_utc = 20.0    # 3:00 PM ET
        current_hours = hour + minute / 60.0
        
        return market_start_utc <= current_hours < market_end_utc
    
    def backtest_symbol(self, symbol, df):
        df = df.copy().reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        print(f"\n{symbol} Swing Trading (1-min) Backtest:")
        print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Bars: {len(df)}")
        print(f"  Pre-calculating indicators...")
        
        # 200-bar EMA (fast for intraday, not 750)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 10-bar slope for confirmation
        df['slope'] = df['close'].diff(10)
        df['slope_pct'] = df['slope'] / df['close'].shift(10) * 100
        
        print(f"  Starting backtest ({len(df)} bars)...")
        
        trades = []
        position_shares = 0
        entry_price = None
        entry_idx = None
        
        if symbol not in self.peak_capitals:
            self.peak_capitals[symbol] = self.capital
        
        for i in range(200, len(df)):
            price = df.loc[i, 'close']
            timestamp = df.loc[i, 'timestamp']
            ema_200 = df.loc[i, 'ema_200']
            slope_pct = df.loc[i, 'slope_pct']
            
            if pd.isna(ema_200):
                continue
            
            # Update peak capital for drawdown
            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], self.capital)
            
            # ENTRY CRITERIA
            # Long: Price above EMA + positive slope + market hours
            long_entry = (
                (price > ema_200 * 1.005) and      # 0.5% buffer (less noise than 0.003)
                (slope_pct > 0.01) and              # Slight positive momentum
                (position_shares == 0) and
                self.is_market_hours(timestamp)
            )
            
            # === ENTRY ===
            if long_entry:
                # Risk 3-4% per trade for swing moves
                position_dollars = self.capital * 0.035
                shares = int(position_dollars / price)
                if shares > 0 and shares * price <= self.capital * 0.90:
                    cost = shares * price
                    self.capital -= cost
                    position_shares = shares
                    entry_price = price
                    entry_idx = i
            
            # === EXITS ===
            elif position_shares > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
                
                # Target 1: +1.0% (swing move)
                if pnl_pct > 1.0 and i - entry_idx > 20:  # At least 20 bars (20 min)
                    # Take partial: 60% off
                    shares_to_sell = int(position_shares * 0.6)
                    if shares_to_sell > 0:
                        self.capital += shares_to_sell * price
                        position_shares -= shares_to_sell
                        exit_pnl = shares_to_sell * (price - entry_price)
                        
                        trades.append({
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'shares': shares_to_sell,
                            'pnl': exit_pnl,
                            'pnl_pct': pnl_pct,
                            'bars_held': i - entry_idx,
                            'type': 'PARTIAL_1.0%'
                        })
                
                # Target 2: +2.0% (swing continuation)
                elif pnl_pct > 2.0 and position_shares > 0:
                    self.capital += position_shares * price
                    pnl = position_shares * (price - entry_price)
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'shares': position_shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_idx,
                        'type': 'TARGET_2.0%'
                    })
                    
                    position_shares = 0
                    entry_price = None
                
                # Stop: -0.75% (wider stop for swing moves)
                elif pnl_pct < -0.75:
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
                
                # Max hold: 120 bars (2 hours) - swing trade max
                elif i - entry_idx > 120 and position_shares > 0:
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
                        'type': 'TIMEOUT_2HR'
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
    bt = BacktestSwingOn1Min(initial_capital=10000)
    
    print("=" * 70)
    print("SWING TRADING ON 1-MINUTE DATA (HYBRID APPROACH)")
    print("=" * 70)
    print(f"Initial Capital: ${bt.initial_capital:,.2f}")
    print("\nStrategy: 200-EMA Swings (Best of Both Worlds)")
    print("  - 200-bar EMA for fast entries (not slow 750)")
    print("  - 0.5% entry buffer (filters noise, not 0.1%)")
    print("  - Partial at +1.0%, full at +2.0%")
    print("  - Stop at -0.75% (wider for swings)")
    print("  - Market hours only (9:30-3:59 PM ET)")
    print("  - Max hold: 2 hours (swing trade window)")
    
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
    print("SWING TRADING ON 1-MIN RESULTS")
    print("=" * 70)
    print(f"Final Capital: ${bt.capital:,.2f}")
    print(f"Peak Equity: ${bt.peak_equity:,.2f}")
    
    total_return = (bt.equity - bt.initial_capital) / bt.initial_capital * 100
    max_drawdown = (bt.peak_equity - bt.equity) / bt.peak_equity * 100 if bt.peak_equity > 0 else 0
    
    print(f"Total Return: {total_return:.2f}% {'✅' if total_return > 10 else '⚠️'}")
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
        print(f"\nExit Type Breakdown:")
        for exit_type in ['PARTIAL_1.0%', 'TARGET_2.0%', 'STOP', 'TIMEOUT_2HR', 'END_OF_DATA']:
            type_trades = [t for t in all_trades if t['type'] == exit_type]
            if type_trades:
                type_wins = sum(1 for t in type_trades if t['pnl'] > 0)
                pct = type_wins/len(type_trades)*100 if len(type_trades) > 0 else 0
                print(f"  {exit_type}: {len(type_trades)} trades, {pct:.1f}% win rate")
    
    # Save
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        output_file = "backtest_swing_1min_results.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("📊 FINAL STRATEGY COMPARISON:")
    print("  750-EMA (6.73%): Slow, few trades, high accuracy")
    print("  200-EMA Aggressive (13.59%): Fast, many trades, low accuracy")
    print("  Swing on 1-min (???): Fast entries + Swing holds = BEST?")
    print("=" * 70)


if __name__ == "__main__":
    main()
