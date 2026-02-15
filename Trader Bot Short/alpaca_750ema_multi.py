"""
ALPACA MULTI-SYMBOL 750 EMA BOT
================================
Adaptive dual-layer trading bot for NVDA, TSLA, XLK simultaneously.

Layer 1 (Macro): 750 EMA regime detection
Layer 2 (Tactical): 4-layer dynamic position sizing

Usage:
    python alpaca_750ema_multi.py --mode paper --symbols NVDA TSLA XLK
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Trading Bot v1"))

from alpaca_trader import AlpacaTrader

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_750ema_multi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlpacaMultiSymbolBot:
    """
    Multi-symbol adaptive dual-layer trading bot
    
    Trades NVDA, TSLA, XLK using 750 EMA regime detection
    """
    
    def __init__(self, symbols: List[str] = None, mode: str = "paper"):
        if symbols is None:
            symbols = ["NVDA", "TSLA", "XLK"]
        
        self.symbols = symbols
        self.mode = mode
        
        # Initialize traders for each symbol
        self.traders = {sym: AlpacaTrader(symbol=sym, mode=mode) for sym in symbols}
        
        # State per symbol
        self.entry_prices = {sym: None for sym in symbols}
        self.entry_times = {sym: None for sym in symbols}
        self.peak_capitals = {sym: None for sym in symbols}
        
        logger.info(f"AlpacaMultiSymbolBot initialized for {symbols} in {mode} mode")
    
    def calculate_slope(self, data: pd.Series, periods: int = 20) -> float:
        """Calculate slope of data using linear regression"""
        if len(data) < periods:
            return 0.0
        
        recent = data.tail(periods).values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def calculate_atr(self, bars: pd.DataFrame, periods: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean().iloc[-1]
        return atr if not np.isnan(atr) else 0.0
    
    def calculate_position_size(
        self,
        capital: float,
        atr: float,
        slope: float,
        current_drawdown: float,
        equity_high: float
    ) -> float:
        """
        Calculate position size in DOLLARS using 4-layer risk management
        
        Layer 1: Base risk per trade (2.5% volatility-adjusted)
        Layer 2: Regime strength multiplier
        Layer 3: Equity curve acceleration
        Layer 4: Drawdown throttle
        """
        # Layer 1: Base risk (2.5% of capital, divided by volatility)
        risk_per_trade = 0.025
        base_position = capital * risk_per_trade / (atr * 1.8) if atr > 0 else capital * risk_per_trade
        
        # Layer 2: Regime strength multiplier
        strong_threshold = 0.010
        weak_threshold = 0.001
        
        if slope > strong_threshold:
            base_position *= 1.4  # 40% more aggressive
        elif slope < weak_threshold:
            base_position *= 0.7  # 30% less aggressive
        
        # Layer 3: Equity curve acceleration
        equity_high_threshold = capital * 1.05
        if capital > equity_high_threshold:
            base_position *= 1.2  # 20% boost on new highs
        
        # Layer 4: Drawdown throttle
        drawdown_limit = -0.20
        if current_drawdown < drawdown_limit:
            base_position *= 0.6  # 40% reduction when underwater
        
        return max(base_position, 0.0)
    
    def run(self, check_interval: int = 300, days_back: int = 5):
        """
        Main trading loop - checks signals every check_interval seconds
        
        Args:
            check_interval (int): Seconds between signal checks (default 300 = 5 min)
            days_back (int): Days of historical data to fetch
        """
        logger.info("=" * 70)
        logger.info("ALPACA MULTI-SYMBOL 750 EMA BOT STARTED")
        logger.info("=" * 70)
        
        # Get initial account info
        trader = self.traders[self.symbols[0]]
        account_info = trader.get_account_info()
        logger.info(f"Starting Capital: ${account_info['equity']:.2f}")
        logger.info(f"Cash Available: ${account_info['cash']:.2f}")
        logger.info(f"Buying Power: ${account_info['buying_power']:.2f}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Check Interval: {check_interval} seconds")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                timestamp = datetime.now()
                
                # Get fresh account info
                account = trader.get_account_info()
                current_capital = float(account['equity'])
                
                logger.info(f"\n[Iteration {iteration}] {timestamp}")
                logger.info(f"Account Equity: ${current_capital:,.2f}")
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        sym_trader = self.traders[symbol]
                        
                        # Get price and position
                        price = sym_trader.get_latest_price()
                        position = sym_trader.get_position()
                        
                        # Load historical data
                        bars = sym_trader.get_historical_bars(timeframe="5Min", days_back=days_back)
                        
                        if bars is None or len(bars) < 750:
                            logger.warning(f"{symbol}: Not enough data ({len(bars) if bars is not None else 0} bars)")
                            continue
                        
                        # Calculate indicators
                        ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                        slope = self.calculate_slope(bars['close'], periods=20)
                        atr = self.calculate_atr(bars, periods=14)
                        
                        # Calculate drawdown
                        if self.peak_capitals[symbol] is None:
                            self.peak_capitals[symbol] = current_capital
                        else:
                            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], current_capital)
                        
                        current_drawdown = (current_capital - self.peak_capitals[symbol]) / self.peak_capitals[symbol] if self.peak_capitals[symbol] > 0 else 0
                        
                        # Regime detection
                        entry_threshold = 0.005
                        exit_threshold = -0.005
                        buffer = 0.005
                        
                        regime_entry = (price > ema_750 * (1 + buffer)) and (slope > entry_threshold)
                        regime_exit = (price < ema_750 * (1 - buffer)) and (slope < exit_threshold)
                        
                        logger.info(
                            f"\n{symbol}: Price ${price:.2f} | EMA750 ${ema_750:.2f} | Slope {slope:.6f} | ATR ${atr:.2f}"
                        )
                        
                        # === ENTRY LOGIC ===
                        if regime_entry and not position:
                            position_dollars = self.calculate_position_size(
                                current_capital, atr, slope, current_drawdown, self.peak_capitals[symbol]
                            )
                            
                            logger.info(f"{symbol} [ENTRY SIGNAL] Regime ON | Position Size: ${position_dollars:.2f}")
                            order = sym_trader.place_buy_order_dollars(position_dollars)
                            
                            if order:
                                self.entry_prices[symbol] = price
                                self.entry_times[symbol] = timestamp
                                logger.info(f"{symbol} ✓ BUY ORDER PLACED: ${position_dollars:.2f}")
                        
                        # === EXIT LOGIC ===
                        elif regime_exit and position:
                            logger.info(f"{symbol} [EXIT SIGNAL] Regime OFF | Closing position of {position['shares']} shares")
                            order = sym_trader.close_position()
                            
                            if order:
                                pnl = position['unrealized_pl']
                                pnl_pct = position['unrealized_plpc']
                                logger.info(f"{symbol} ✓ SELL ORDER PLACED: PnL = ${pnl:.2f} ({pnl_pct:.2f}%)")
                        
                        # Log position state
                        if position:
                            logger.info(
                                f"{symbol} [POSITION] {position['shares']} shares | "
                                f"Avg: ${position['avg_fill_price']:.2f} | "
                                f"Value: ${position['market_value']:.2f} | "
                                f"PnL: ${position['unrealized_pl']:.2f}"
                            )
                        else:
                            logger.info(f"{symbol} [NO POSITION] Ready for entry")
                    
                    except Exception as e:
                        logger.error(f"{symbol} error: {e}", exc_info=True)
                
                # Wait for next check
                logger.info(f"\nSleeping {check_interval} seconds...")
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
            self._closeout()
        
        except Exception as e:
            logger.error(f"[ERROR] {e}", exc_info=True)
            self._closeout()
    
    def _closeout(self):
        """Close all positions and export logs"""
        logger.info("Closing all positions...")
        for symbol in self.symbols:
            try:
                position = self.traders[symbol].get_position()
                if position and position['shares'] > 0:
                    self.traders[symbol].close_position()
                    logger.info(f"{symbol}: Position closed")
            except Exception as e:
                logger.error(f"{symbol} close error: {e}")
        
        # Export trade logs
        for symbol in self.symbols:
            try:
                filename = f"trade_log_{symbol}.json"
                self.traders[symbol].export_trade_log(filename)
            except:
                pass
        
        logger.info("Bot shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "TSLA", "XLK"])
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()
    
    bot = AlpacaMultiSymbolBot(symbols=args.symbols, mode=args.mode)
    bot.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
