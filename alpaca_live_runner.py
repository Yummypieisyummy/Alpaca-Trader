"""
ALPACA LIVE TRADING BOT
=======================
Real-time trading bot with dual-layer architecture and 4-layer risk management

Connects to Alpaca API for paper/live trading on TSLA

Usage:
    from alpaca_live_runner import AlpacaLiveBot
    bot = AlpacaLiveBot()
    bot.run(check_interval=300)  # Check signals every 5 minutes
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from alpaca_trader import AlpacaTrader

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlpacaLiveBot:
    """
    Adaptive dual-layer trading bot for Alpaca
    
    Layer 1 (Macro): 750EMA regime detection
    Layer 2 (Tactical): 4-layer dynamic position sizing
    """
    
    def __init__(self, symbol="TSLA", mode="paper"):
        """Initialize the bot"""
        self.trader = AlpacaTrader(symbol=symbol, mode=mode)
        self.symbol = symbol
        self.mode = mode
        self.entry_price = None
        self.entry_time = None
        self.peak_capital = None
        
        logger.info(f"AlpacaLiveBot initialized for {symbol} in {mode} mode")
    
    def calculate_slope(self, data, periods=20):
        """
        Calculate slope of data using linear regression
        
        Args:
            data: Pandas Series
            periods: Look-back periods
        
        Returns:
            Slope value (positive = uptrend, negative = downtrend)
        """
        if len(data) < periods:
            return 0.0
        
        recent = data.tail(periods).values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def calculate_atr(self, bars, periods=14):
        """
        Calculate Average True Range
        
        Args:
            bars: DataFrame with 'high', 'low', 'close'
            periods: ATR lookback period
        
        Returns:
            ATR value
        """
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean().iloc[-1]
        return atr
    
    def calculate_position_size(self, capital, atr, slope, current_drawdown, equity_high):
        """
        Calculate position size in DOLLARS using 4-layer risk management
        
        Layer 1: Base risk per trade (2.5% volatility-adjusted)
        Layer 2: Regime strength multiplier
        Layer 3: Equity curve acceleration
        Layer 4: Drawdown throttle
        
        Args:
            capital: Current account equity
            atr: Average True Range
            slope: EMA slope
            current_drawdown: Current drawdown ratio
            equity_high: Peak equity for acceleration
        
        Returns:
            Position size in dollars
        """
        # Layer 1: Base risk (2.5% of capital, divided by volatility)
        risk_per_trade = 0.025  # 2.5%
        base_position = capital * risk_per_trade / (atr * 1.8)
        
        logger.debug(f"Layer 1 (Base): ${base_position:.2f}")
        
        # Layer 2: Regime strength multiplier
        strong_threshold = 0.010
        weak_threshold = 0.001
        
        if slope > strong_threshold:
            base_position *= 1.4  # 40% more aggressive
            logger.debug(f"Layer 2 (Strong): {base_position:.2f} (slope {slope:.6f})")
        elif slope < weak_threshold:
            base_position *= 0.7  # 30% less aggressive
            logger.debug(f"Layer 2 (Weak): ${base_position:.2f} (slope {slope:.6f})")
        else:
            logger.debug(f"Layer 2 (Normal): ${base_position:.2f} (slope {slope:.6f})")
        
        # Layer 3: Equity curve acceleration
        equity_high_threshold = capital * 1.05  # 5% new high
        if capital > equity_high_threshold:
            base_position *= 1.2  # 20% boost on new highs
            logger.debug(f"Layer 3 (Acceleration): ${base_position:.2f} (new high)")
        else:
            logger.debug(f"Layer 3 (Normal): ${base_position:.2f}")
        
        # Layer 4: Drawdown throttle
        drawdown_limit = -0.20  # -20%
        if current_drawdown < drawdown_limit:
            base_position *= 0.6  # 40% reduction when underwater
            logger.debug(f"Layer 4 (Throttle): ${base_position:.2f} (DD {current_drawdown:.2%})")
        else:
            logger.debug(f"Layer 4 (Normal): ${base_position:.2f}")
        
        return base_position
    
    def run(self, check_interval=300):
        """
        Main trading loop - checks signals every check_interval seconds
        
        Args:
            check_interval (int): Seconds between signal checks (default 300 = 5 min)
        """
        logger.info("=" * 60)
        logger.info("ALPACA LIVE TRADING BOT STARTED")
        logger.info("=" * 60)
        
        account_info = self.trader.get_account_info()
        logger.info(f"Starting Capital: ${account_info['equity']:.2f}")
        logger.info(f"Cash Available: ${account_info['cash']:.2f}")
        logger.info(f"Buying Power: ${account_info['buying_power']:.2f}")
        logger.info(f"Check Interval: {check_interval} seconds")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                timestamp = datetime.now()
                
                # Get fresh data
                account = self.trader.get_account_info()
                current_capital = float(account['equity'])
                price = self.trader.get_latest_price()
                position = self.trader.get_position()
                
                # Load latest bars
                bars = self.trader.get_historical_bars(timeframe="5Min", days_back=5)
                
                # Calculate indicators
                ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                slope = self.calculate_slope(bars['close'], periods=20)
                atr = self.calculate_atr(bars, periods=14)
                
                # Calculate drawdown
                if not hasattr(self, 'peak_capital') or self.peak_capital is None:
                    self.peak_capital = current_capital
                else:
                    self.peak_capital = max(self.peak_capital, current_capital)
                
                current_drawdown = (current_capital - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
                
                # Regime detection
                entry_threshold = 0.005
                exit_threshold = -0.005
                buffer = 0.005
                
                regime_entry = (price > ema_750 * (1 + buffer)) and (slope > entry_threshold)
                regime_exit = (price < ema_750 * (1 - buffer)) and (slope < exit_threshold)
                
                logger.info(f"\n[Iteration {iteration}] {timestamp}")
                logger.info(f"Price: ${price:.2f} | EMA750: ${ema_750:.2f} | Slope: {slope:.6f}")
                logger.info(f"ATR: ${atr:.2f} | Capital: ${current_capital:.2f} | DD: {current_drawdown:.2%}")
                
                # === ENTRY LOGIC ===
                if regime_entry and not position:
                    position_dollars = self.calculate_position_size(
                        current_capital, atr, slope, current_drawdown, self.peak_capital
                    )
                    
                    logger.info(f"[ENTRY SIGNAL] Regime ON | Position Size: ${position_dollars:.2f}")
                    order = self.trader.place_buy_order_dollars(position_dollars)
                    
                    if order:
                        self.entry_price = price
                        self.entry_time = timestamp
                        logger.info(f"✓ BUY ORDER PLACED: ${position_dollars:.2f}")
                
                # === EXIT LOGIC ===
                elif regime_exit and position:
                    logger.info(f"[EXIT SIGNAL] Regime OFF | Closing position of {position['shares']} shares")
                    order = self.trader.close_position()
                    
                    if order:
                        pnl = position['unrealized_pl']
                        pnl_pct = position['unrealized_plpc']
                        logger.info(f"✓ SELL ORDER PLACED: PnL = ${pnl:.2f} ({pnl_pct:.2f}%)")
                
                # Log current state
                if position:
                    logger.info(f"[POSITION] {position['shares']} shares | Avg: ${position['avg_fill_price']:.2f} | Value: ${position['market_value']:.2f} | PnL: ${position['unrealized_pl']:.2f}")
                else:
                    logger.info(f"[NO POSITION] Ready for next entry")
                
                # Wait for next check
                logger.info(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
            self.trader.export_trade_log()
            
        except Exception as e:
            logger.error(f"[ERROR] {e}", exc_info=True)
            self.trader.export_trade_log()


if __name__ == "__main__":
    bot = AlpacaLiveBot(symbol="TSLA", mode="paper")
    
    # TEST MODE (single iteration, no actual trading)
    logger.info("Running bot in test mode...")
    
    # Uncomment below to enable continuous trading:
    # bot.run(check_interval=300)  # Check every 5 minutes
