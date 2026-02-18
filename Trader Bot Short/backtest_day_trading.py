"""
DAY TRADING BOT BACKTEST (1-MINUTE)
====================================
Intraday scalping strategy with:
- 50-bar EMA (fast intraday trend)
- RSI confirmation (30-70 range)
- Tight -0.25% stops
- Scale-out at +0.5% and +1.0%
- 5-30 minute holds

Target: 15-25% return with 60%+ win rate
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class BacktestDayTradingBot:
    """Day trading bot on 1-minute data"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.peak_capitals = {}
        
        # Day trading metrics
        self.win_rate = 0.60
        self.avg_win = 12.0
        self.avg_loss = 8.0
        self.profit_factor = self.avg_win / self.avg_loss  # 1.5x
    
    def calculate_rsi(self, data, periods=14):
        if len(data) < periods + 1:
            return 50.0
        deltas = data.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        avg_gain = gains.rolling(window=periods).mean().iloc[-1]
        avg_loss = losses.rolling(window=periods).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi if not np.isnan(rsi) else 50.0
    
    def calculate_position_size(self, capital):
        """Day trading: Aggressive sizing with tight stops"""
        # With -0.25% stop and 60% win rate, can use 8-10% per trade
        return capital * 0.09  # 9% risk per trade
    
    def backtest_symbol(self, symbol, df):
        df = df.copy().reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        print(f"\n{symbol} Day Trading Backtest:")
        print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Bars: {len(df)}")
        print(f"  Pre-calculating indicators...")
        
        # 50-bar EMA for intraday
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI 14 - proper calculation
        deltas = df['close'].diff()
        gains = deltas.where(deltas > 0, 0).rolling(window=14).mean()
        losses = -deltas.where(deltas < 0, 0).rolling(window=14).mean()
        rs = gains / losses
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50.0, inplace=True)
        
        print(f"  Starting backtest ({len(df)} bars)...")
        
        trades = []
        position_shares = 0
        entry_price = None
        entry_idx = None
        partial_exits = {"first_exit": False}
        
        if symbol not in self.peak_capitals:
            self.peak_capitals[symbol] = self.capital
        
        for i in range(50, len(df)):
            price = df.loc[i, 'close']
            timestamp = df.loc[i, 'timestamp']
            ema_50 = df.loc[i, 'ema_50']
            rsi = df.loc[i, 'rsi']
            
            if pd.isna(ema_50) or pd.isna(rsi):
                continue
            
            # Update peak capital
            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], self.capital)
            current_drawdown = (self.capital - self.peak_capitals[symbol]) / self.peak_capitals[symbol] if self.peak_capitals[symbol] > 0 else 0
            
            # DAY TRADING SIGNALS
            # Entry: Price crosses above EMA + RSI not overbought
            long_signal = (price > ema_50 * 1.001) and (rsi < 65)  # Tight 0.1% buffer
            
            # Entry: Price crosses below EMA + RSI not oversold (mean reversion)
            short_signal = (price < ema_50 * 0.999) and (rsi > 35)
            
            # === ENTRY ===
            if long_signal and position_shares == 0:
                position_dollars = self.calculate_position_size(self.capital)
                if position_dollars > 0 and position_dollars <= self.capital * 0.90:
                    shares = int(position_dollars / price)
                    if shares > 0:
                        cost = shares * price
                        self.capital -= cost
                        position_shares = shares
                        entry_price = price
                        entry_idx = i
                        partial_exits = {"first_exit": False}
            
            # === EXITS ===
            elif position_shares > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
                
                # Scale-out #1: Exit 60% at +0.5%
                if not partial_exits["first_exit"] and pnl_pct > 0.5:
                    shares_to_sell = int(position_shares * 0.6)
                    if shares_to_sell > 0:
                        self.capital += shares_to_sell * price
                        position_shares -= shares_to_sell
                        partial_exits["first_exit"] = True
                
                # Scale-out #2: Exit remaining 40% at +1.0%
                elif pnl_pct > 1.0 and position_shares > 0:
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
                
                # Tight stop: -0.25%
                elif pnl_pct < -0.25 and position_shares > 0:
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
                
                # Max hold: 30 bars (30 minutes)
                elif i - entry_idx > 30 and position_shares > 0:
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
                        'type': 'TIMEOUT'
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
    bt = BacktestDayTradingBot(initial_capital=10000)
    
    print("=" * 70)
    print("DAY TRADING BOT BACKTEST (1-MINUTE)")
    print("=" * 70)
    print(f"Initial Capital: ${bt.initial_capital:,.2f}")
    print("\nStrategy: 50-EMA + RSI confirmation")
    print("  - Tight 0.1% entry buffer")
    print("  - Scale-out at +0.5% / +1.0%")
    print("  - Stop at -0.25%")
    print("  - Max hold: 30 minutes")
    
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
    print("DAY TRADING BOT RESULTS")
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
        total_pnl = sum(t['pnl'] for t in all_trades)
        
        print(f"\nWins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")
        
        if any(t['pnl'] < 0 for t in all_trades):
            pf = sum(t['pnl'] for t in all_trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
            print(f"Profit Factor: {pf:.2f}")
        
        # Trade type breakdown
        print(f"\nExit Types:")
        for exit_type in ['TARGET', 'STOP', 'TIMEOUT', 'REVERSE']:
            type_trades = [t for t in all_trades if t['type'] == exit_type]
            if type_trades:
                type_wins = sum(1 for t in type_trades if t['pnl'] > 0)
                print(f"  {exit_type}: {len(type_trades)} trades, {type_wins/len(type_trades)*100:.1f}% win rate")
    
    # Save
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        output_file = "backtest_day_trading_results.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("  Baseline (750-EMA, market hours): 6.73% return, 103 trades, 43% win rate")
    print("  Aggressive (200-EMA, loose): 13.59% return, 939 trades, 24.9% win rate")
    print("  Day Trading (50-EMA, tight): [See above]")
    print("=" * 70)


if __name__ == "__main__":
    main()
